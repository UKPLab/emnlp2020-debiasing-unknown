import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
import numpy as np
import math


class ClfDistillLossFunction(nn.Module):
    """Torch classification debiasing loss function"""

    def forward(self, hidden, logits, bias, teach_probs, labels):
        """
        :param hidden: [batch, n_features] hidden features from the model
        :param logits: [batch, n_classes] logit score for each class
        :param bias: [batch, n_classes] log-probabilties from the bias for each class
        :param labels: [batch] integer class labels
        :return: scalar loss
        """
        raise NotImplementedError()


class Plain(ClfDistillLossFunction):
    def forward(self, hidden, logits, bias, teacher_probs, labels):
        return F.cross_entropy(logits, labels)

class FocalLoss(ClfDistillLossFunction):
    def __init__(self, gamma=1.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, hidden, logits, bias, teacher_probs, labels):

        logits = logits.float()  # In case we were in fp16 mode
        loss = F.cross_entropy(logits, labels, reduction='none')

        softmaxf = torch.nn.Softmax(dim=1)
        current_probs = softmaxf(logits).detach()
        one_hot_labels = torch.eye(logits.size(1)).cuda()[labels]
        weights = 1 - (one_hot_labels * current_probs).sum(1)
        weights = weights ** self.gamma

        return (weights * loss).sum() / weights.sum()


class CalibratedPlain(ClfDistillLossFunction):
    def __init__(self):
        super(CalibratedPlain, self).__init__()
        self.temp_param = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)

    def forward(self, hidden, logits, bias, teacher_probs, labels):
        return F.cross_entropy(logits / self.temp_param, labels)


class DistillLoss(ClfDistillLossFunction):
    def forward(self, hidden, logits, bias, teacher_probs, labels):
        softmaxf = torch.nn.Softmax(dim=1)
        probs = softmaxf(logits)

        example_loss = -(teacher_probs * probs.log()).sum(1)
        batch_loss = example_loss.mean()

        return batch_loss

class SmoothedDistillLoss(ClfDistillLossFunction):
    def forward(self, hidden, logits, bias, teacher_probs, labels):
        softmaxf = torch.nn.Softmax(dim=1)
        probs = softmaxf(logits)
        
        one_hot_labels = torch.eye(logits.size(1)).cuda()[labels]
        weights = (1 - (one_hot_labels * torch.exp(bias)).sum(1))
        weights = weights.unsqueeze(1).expand_as(teacher_probs)

        exp_teacher_probs = teacher_probs ** weights
        norm_teacher_probs = exp_teacher_probs / exp_teacher_probs.sum(1).unsqueeze(1).expand_as(teacher_probs)

        example_loss = -(norm_teacher_probs * probs.log()).sum(1)
        batch_loss = example_loss.mean()

        return batch_loss


class PermuteSmoothedDistillLoss(ClfDistillLossFunction):
    def __init__(self, num_class=3):
        super(PermuteSmoothedDistillLoss, self).__init__()
        self.num_class = num_class

    def forward(self, hidden, logits, bias, teacher_probs, labels):
        softmaxf = torch.nn.Softmax(dim=1)
        probs = softmaxf(logits)

        one_hot_labels = torch.eye(logits.size(1)).cuda()[labels]
        weights = (1 - (one_hot_labels * torch.exp(bias)).sum(1))
        weights = weights.unsqueeze(1).expand_as(teacher_probs)

        exp_teacher_probs = teacher_probs ** weights
        norm_teacher_probs = exp_teacher_probs / exp_teacher_probs.sum(1).unsqueeze(1).expand_as(teacher_probs)

        # hack for the permutation, may be very slow
        for i, argmax_id in enumerate(norm_teacher_probs.argmax(1).tolist()):
            nonmax_id = [x for x in range(self.num_class)]
            nonmax_id.remove(argmax_id)
            assert len(nonmax_id) == 2

            nonmax1, nonmax2 = norm_teacher_probs[i, nonmax_id[0]].item(), norm_teacher_probs[i, nonmax_id[1]].item()
            norm_teacher_probs[i, nonmax_id[0]] = nonmax2
            norm_teacher_probs[i, nonmax_id[1]] = nonmax1

        example_loss = -(norm_teacher_probs * probs.log()).sum(1)
        batch_loss = example_loss.mean()

        return batch_loss


class SmoothedReweightLoss(ClfDistillLossFunction):
    def forward(self, hidden, logits, bias, teacher_probs, labels):
        one_hot_labels = torch.eye(logits.size(1)).cuda()[labels]
        weights = (1 - (one_hot_labels * torch.exp(bias)).sum(1))
        weights = weights.unsqueeze(1).expand_as(teacher_probs)

        exp_teacher_probs = teacher_probs ** weights
        norm_teacher_probs = exp_teacher_probs / exp_teacher_probs.sum(1).unsqueeze(1).expand_as(teacher_probs)
        scaled_weights = (one_hot_labels * norm_teacher_probs).sum(1)

        loss = F.cross_entropy(logits, labels, reduction='none')

        return (scaled_weights * loss).sum() / scaled_weights.sum()


class LabelSmoothing(ClfDistillLossFunction):
    def __init__(self, num_class):
        super(LabelSmoothing, self).__init__()
        self.num_class = num_class

    def forward(self, hidden, logits, bias, teacher_probs, labels):
        softmaxf = torch.nn.Softmax(dim=1)
        probs = softmaxf(logits)

        one_hot_labels = torch.eye(logits.size(1)).cuda()[labels]
        alphas = (one_hot_labels * torch.exp(bias)).sum(1).unsqueeze(1).expand_as(one_hot_labels)
        target_probs = (1 - alphas) * one_hot_labels + alphas / self.num_class

        example_loss = -(target_probs * probs.log()).sum(1)
        batch_loss = example_loss.mean()

        return batch_loss

class ThetaSmoothedDistillLoss(ClfDistillLossFunction):
    def __init__(self, theta):
        super(ThetaSmoothedDistillLoss, self).__init__()
        self.theta = theta
    
    def forward(self, hidden, logits, bias, teacher_probs, labels):
        softmaxf = torch.nn.Softmax(dim=1)
        probs = softmaxf(logits)

        one_hot_labels = torch.eye(logits.size(1)).cuda()[labels]
        weights = (1 - (one_hot_labels * torch.exp(bias)).sum(1))
        weights = weights.unsqueeze(1).expand_as(teacher_probs) + self.theta

        exp_teacher_probs = teacher_probs ** weights
        norm_teacher_probs = exp_teacher_probs / exp_teacher_probs.sum(1).unsqueeze(1).expand_as(teacher_probs)

        example_loss = -(norm_teacher_probs * probs.log()).sum(1)
        batch_loss = example_loss.mean()
        
        return batch_loss


class ReweightBaseline(ClfDistillLossFunction):
    def forward(self, hidden, logits, bias, teacher_probs, labels):
        logits = logits.float()  # In case we were in fp16 mode
        loss = F.cross_entropy(logits, labels, reduction='none')
        one_hot_labels = torch.eye(logits.size(1)).cuda()[labels]
        weights = 1 - (one_hot_labels * torch.exp(bias)).sum(1)

        return (weights * loss).sum() / weights.sum()


class ReweightByTeacher(ClfDistillLossFunction):
    def forward(self, hidden, logits, bias, teacher_probs, labels, theta=1.0):
        logits = logits.float()  # In case we were in fp16 mode
        loss = F.cross_entropy(logits, labels, reduction='none')
        one_hot_labels = torch.eye(logits.size(1)).cuda()[labels]

        weights = 1 - (one_hot_labels * teacher_probs).sum(1)
        # weights = weights ** theta
        
        return (weights * loss).sum() / weights.sum()


class BiasProductByTeacher(ClfDistillLossFunction):
    def forward(self, hidden, logits, bias, teacher_probs, labels):
        logits = logits.float()  # In case we were in fp16 mode
        logits = F.log_softmax(logits, 1)
        teacher_logits = torch.log(teacher_probs)
        return F.cross_entropy(logits + teacher_logits, labels)

class BiasProductByTeacherAnnealed(ClfDistillLossFunction):
    def __init__(self, max_theta=1.0, min_theta=0.8,
                 total_steps=12272, num_epochs=3):
        super().__init__()
        self.max_theta = max_theta
        self.min_theta = min_theta
        self.num_train_optimization_steps = total_steps
        self.num_epochs = num_epochs
        self.current_step = 0

    def get_current_theta(self):
        linspace_theta = np.linspace(self.max_theta, self.min_theta,
                                     self.num_train_optimization_steps+self.num_epochs)
        current_theta = linspace_theta[self.current_step]
        self.current_step += 1
        return current_theta

    def forward(self, hidden, logits, bias, teacher_probs, labels):
        logits = logits.float()  # In case we were in fp16 mode
        logits = F.log_softmax(logits, 1)

        current_theta = self.get_current_theta()
        denom = (teacher_probs ** current_theta).sum(1).unsqueeze(1).expand_as(teacher_probs)
        scaled_probs = (teacher_probs ** current_theta) / denom

        teacher_logits = torch.log(scaled_probs)
        return F.cross_entropy(logits + teacher_logits, labels)


class ReweightByTeacherAnnealed(ClfDistillLossFunction):
    def __init__(self, max_theta=1.0, min_theta=0.8,
                 total_steps=12272, num_epochs=3):
        super().__init__()
        self.max_theta = max_theta
        self.min_theta = min_theta
        self.num_train_optimization_steps = total_steps
        self.num_epochs = num_epochs
        self.current_step = 0

    def get_current_theta(self):
        linspace_theta = np.linspace(self.max_theta, self.min_theta,
                                     self.num_train_optimization_steps+self.num_epochs)
        current_theta = linspace_theta[self.current_step]
        self.current_step += 1
        return current_theta


    def forward(self, hidden, logits, bias, teacher_probs, labels):
        logits = logits.float()  # In case we were in fp16 mode
        loss = F.cross_entropy(logits, labels, reduction='none')
        one_hot_labels = torch.eye(logits.size(1)).cuda()[labels]

        weights = 1 - (one_hot_labels * teacher_probs).sum(1)

        current_theta = self.get_current_theta()
        weights = weights ** current_theta

        return (weights * loss).sum() / weights.sum()


class SmoothedDistillLossAnnealed(ClfDistillLossFunction):
    def __init__(self, max_theta=1.0, min_theta=0.8,
                 total_steps=12272, num_epochs=3):
        super().__init__()
        self.max_theta = max_theta
        self.min_theta = min_theta
        self.num_train_optimization_steps = total_steps
        self.num_epochs = num_epochs
        self.current_step = 0

    def get_current_theta(self):
        linspace_theta = np.linspace(self.max_theta, self.min_theta,
                                     self.num_train_optimization_steps+self.num_epochs)
        current_theta = linspace_theta[self.current_step]
        self.current_step += 1
        return current_theta

    def forward(self, hidden, logits, bias, teacher_probs, labels):
        softmaxf = torch.nn.Softmax(dim=1)
        probs = softmaxf(logits)

        bias_probs = torch.exp(bias)
        current_theta = self.get_current_theta()
        denom = (bias_probs ** current_theta).sum(1).unsqueeze(1).expand_as(bias_probs)
        scaled_bias_probs = (bias_probs ** current_theta) / denom

        one_hot_labels = torch.eye(logits.size(1)).cuda()[labels]
        weights = (1 - (one_hot_labels * scaled_bias_probs).sum(1))
        weights = weights.unsqueeze(1).expand_as(teacher_probs)

        exp_teacher_probs = teacher_probs ** weights
        norm_teacher_probs = exp_teacher_probs / exp_teacher_probs.sum(1).unsqueeze(1).expand_as(teacher_probs)

        example_loss = -(norm_teacher_probs * probs.log()).sum(1)
        batch_loss = example_loss.mean()

        return batch_loss


class BiasProductBaseline(ClfDistillLossFunction):
    def forward(self, hidden, logits, bias, teacher_probs, labels):
        logits = logits.float()  # In case we were in fp16 mode
        logits = F.log_softmax(logits, 1)
        return F.cross_entropy(logits + bias.float(), labels)

class LearnedMixinBaseline(ClfDistillLossFunction):

    def __init__(self, penalty):
        super().__init__()
        self.penalty = penalty
        self.bias_lin = torch.nn.Linear(768, 1)

    def forward(self, hidden, logits, bias, teacher_probs, labels):
        logits = logits.float()  # In case we were in fp16 mode
        logits = F.log_softmax(logits, 1)

        factor = self.bias_lin.forward(hidden)
        factor = factor.float()
        factor = F.softplus(factor)

        bias = bias * factor

        bias_lp = F.log_softmax(bias, 1)
        entropy = -(torch.exp(bias_lp) * bias_lp).sum(1).mean(0)

        loss = F.cross_entropy(logits + bias, labels) + self.penalty * entropy
        return loss

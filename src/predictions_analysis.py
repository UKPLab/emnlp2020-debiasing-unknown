from tqdm import tqdm
import json
import numpy as np
import random
import math
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

from sklearn.manifold import TSNE

plt.style.use('ggplot')
NLI_LABELS = ["non_entailment_0", "entailment", "non_entailment_1"]
NLI_DATASETS = ["HANS", "MNLI_TRAIN", "MNLI_DEV", "MNLI_OVERLAP"]

def visualize_predictions(all_probs: np.ndarray, all_labels: np.ndarray, conf_plot_file: str):

    width = 0.1
    num_bins = 10
    all_bins = [[] for x in range(num_bins)]
    all_bins_stat = [[0, 0, 0.] for _ in range(num_bins)] # cor_num, num, conf_accum

    pred_probs = all_probs.max(axis=1)
    pred_labels = all_probs.argmax(axis=1)
    # correct_probs = []
    for pred_prob, pred_label, true_label in zip(pred_probs, pred_labels, all_labels):
        which_bins = math.floor(pred_prob / width)
        which_bins = min(9, which_bins)  # in case the pred prob is 1.0

        all_bins[which_bins].append((pred_prob, pred_label, true_label))

        all_bins_stat[which_bins][1] += 1
        all_bins_stat[which_bins][2] += pred_prob

        if pred_label == true_label:
            all_bins_stat[which_bins][0] += 1

    all_bins_acc = []
    for bin in all_bins_stat:
        if bin[1] == 0:
            all_bins_acc.append(0.)
        else:
            all_bins_acc.append(float(bin[0])/ bin[1])
    all_bins_conf = []
    for bin in all_bins_stat:
        if bin[1] == 0:
            all_bins_conf.append(0.)
        else:
            all_bins_conf.append(bin[2] / bin[1])

    all_nums = [x[1] for x in all_bins_stat]
    ECE_bin = [(all_nums[x]/sum(all_nums)) * abs(all_bins_acc[x] - all_bins_conf[x]) for x in range(len(all_bins_acc))]

    fracts = [x/sum(all_nums) for x in all_nums]
    acc_fracts = [x*y for x, y in zip(fracts, all_bins_acc)]

    objects = ('0.', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9')
    y_pos = np.arange(len(objects))

    plt.bar(y_pos, fracts, align='center', alpha=0.5, label="dataset fraction")
    plt.bar(y_pos, acc_fracts, align='center', alpha=0.5, label="correctly predicted")
    plt.xticks(y_pos, objects)
    plt.ylabel('percentage of dataset')
    plt.title('predictions confidence plot')
    plt.legend(loc="upper right", fontsize=12)

    plt.savefig(conf_plot_file)
    plt.show()
    plt.close()

    return sum(ECE_bin), all_bins_acc, all_bins_conf, all_nums


def visualize_penultimate(all_labels: np.ndarray, all_is_correct: np.ndarray,
                          all_penultimates: np.ndarray, all_dataset_ids: np.ndarray,
                          plot_file: str=None, sample=-1):
    if sample > 0:
        rand_idx = np.random.randint(all_penultimates.shape[0], size=sample)
        all_labels = all_labels[rand_idx]
        all_is_correct = all_is_correct[rand_idx]
        all_penultimates = all_penultimates[rand_idx, :]
    
    tsne = TSNE(n_components=2, random_state=555, perplexity=50, n_iter=5000)
    print("computing the tsne reduction...")
    all_penultimates = tsne.fit_transform(all_penultimates)

    plt.figure(figsize=(12, 10))

    ax = plt.gca()

    marker_types = ["D", "s", "o", "^"]
    color_types = [next(ax._get_lines.prop_cycler)['color'],
                   next(ax._get_lines.prop_cycler)['color'],
                   next(ax._get_lines.prop_cycler)['color']]
    for dat_ix in np.unique(all_dataset_ids):
        for label in np.unique(all_labels):
            cur_penultimates = []
            cur_incor_penultimates = []
            for cur_label, cur_is_correct, cur_dataset_id, cur_penultimate in zip(all_labels, all_is_correct,
                                                                                  all_dataset_ids, all_penultimates):
                # if dat_ix == 0 and label == 1 and cur_dataset_id == 0:
                #     import pdb; pdb.set_trace()
                if cur_label == label and cur_dataset_id == dat_ix:
                    cur_penultimates.append(np.expand_dims(cur_penultimate, 0))
                    if cur_is_correct == 0:
                        cur_incor_penultimates.append(np.expand_dims(cur_penultimate, 0))

            # if dat_ix == 0:
            #     continue

            if len(cur_penultimates) == 0:
                # import pdb; pdb.set_trace()
                continue

            cur_penultimates = np.concatenate(cur_penultimates, axis=0)

            label_str = NLI_LABELS[label]
            dataset_str = NLI_DATASETS[dat_ix]
            
            size = plt.rcParams['lines.markersize']**2
            if dat_ix == 3:
                size = size * 2

            if len(cur_incor_penultimates) > 0:
                cur_incor_penultimates = np.concatenate(cur_incor_penultimates, axis=0)
                if dat_ix == 0:
                    plt.scatter(cur_incor_penultimates[:, 0], cur_incor_penultimates[:, 1],
                                alpha=0.8, marker='x', color=color_types[label], s=size)
                else:
                    plt.scatter(cur_incor_penultimates[:, 0], cur_incor_penultimates[:, 1],
                                alpha=0.8, marker='*', color=color_types[label], s=size)

            plt.scatter(cur_penultimates[:, 0], cur_penultimates[:, 1],
                        label=dataset_str+"_"+label_str, alpha=0.5, marker=marker_types[dat_ix],
                        color=color_types[label], s=size)

    plt.legend()

    plt.savefig(plot_file)
    plt.show()
    plt.close()

    return all_penultimates

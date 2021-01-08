# Towards Debiasing NLU Models from Unknown Biases
> **Abstract:** NLU models often exploit biased features
to achieve high dataset-specific performance
without properly learning the intended task.
Recently proposed debiasing methods are
shown to be effective in mitigating this tendency. However, these methods rely on a
major assumption that the type of biased features is known a-priori, which limits their application to many NLU tasks and datasets. In
this work, we present the first step to bridge
this gap by introducing a self-debiasing framework that prevents models from mainly utilizing biases without knowing them in advance.
The proposed framework is general and complementary to the existing debiasing methods.
We show that the proposed framework allows
these existing methods to retain the improvement on the challenge datasets (i.e., sets of examples designed to expose models’ reliance
to biases) without specifically targeting certain biases. Furthermore, the evaluation suggests that applying the framework results in
improved overall robustness. 

The repository contains the code to reproduce our work in debiasing NLU models without prior information on biases.
We provide 3 runs of experiment that are shown in our paper:
1. Debias [MNLI](https://www.nyu.edu/projects/bowman/multinli/paper.pdf) model from syntactic bias and evaluate on 
[HANS](https://arxiv.org/abs/1902.01007) as the out-of-distribution data using example reweighting.
2. Debias [MNLI](https://www.nyu.edu/projects/bowman/multinli/paper.pdf) model from syntactic bias and evaluate on 
[HANS](https://arxiv.org/abs/1902.01007) as the out-of-distribution data using product of expert.
3. Debias [MNLI](https://www.nyu.edu/projects/bowman/multinli/paper.pdf) model from syntactic bias and evaluate on 
[HANS](https://arxiv.org/abs/1902.01007) as the out-of-distribution data using confidence regularization.

## Requirements
The code requires python >= 3.6 and pytorch >= 1.1.0.

Additional required dependencies can be found in `requirements.txt`.
Install all requirements by running:
```bash
pip install -r requirements.txt
```


## Data
Our experiments use MNLI dataset version provided by GLUE benchmark.
Download the file from <a href="https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FMNLI.zip?alt=media&token=50329ea1-e339-40e2-809c-10c40afff3ce" target="_blank">here</a>, 
and unzip under the directory ``./dataset`` 
The dataset directory should be structured as the following:
```bash
└── dataset 
    └── MNLI
        ├── train.tsv
        ├── dev_matched.tsv
        ├── dev_mismatched.tsv
        ├── dev_mismatched.tsv
```


## Running the experiments
For each evaluation setting, use the `--mode` arguments to
set the appropriate loss function. Choose the annealed version of the loss function for reproducing the annealed results.

To reproduce our result on MNLI ⮕ HANS, run the following:

```
cd src/
CUDA_VISIBLE_DEVICES=9 python train_distill_bert.py \
  --output_dir ../experiments_self_debias_mnli_seed111/bert_reweighted_sampled2K_teacher_seed111_annealed_1to08 \
  --do_train --do_eval --mode reweight_by_teacher_annealed \
  --custom_teacher ../teacher_preds/mnli_trained_on_sample2K_seed111.json --seed 111 --which_bias hans
```

## Expected results

Results on the MNLI ⮕ HANS setting without annealing:

|Mode|Seed|MNLI-m|MNLI-mm|HANS avg.|
|-----|----|---|---|---|
|None|111|84.57|84.72|62.04|
|reweighting|111|81.8|82.3|72.1|
|PoE|111|81.5|81.1|70.3|
|conf-reg|222|83.7|84.1|68.7|
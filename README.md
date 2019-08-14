# Adversarial VQA

Adversarial regularization is a powerful technique for mitigating learned biases. This repo contains code for an ongoing project applying adversarial regularization to VQA.

![Alt text](info/advreg_viz.png?raw=true "adversarial regularization visualization")

## Pythia

This repo is forked from [Pythia](https://github.com/facebookresearch/pythia). Pythia is a modular framework for Visual Question Answering research, which formed the basis for the winning entry to the VQA Challenge 2018 from Facebook AI Research (FAIR)’s A-STAR team. See their [paper](https://arxiv.org/abs/1807.09956) for more details.

## Running the code

This repo is a fork of Pythia, so can use the commands provided in the [Pythia README](https://github.com/facebookresearch/pythia/blob/master/README.md) to train and evaluate the model. It may also be helpful to consult the original [version](https://github.com/facebookresearch/pythia/blob/8a0fbcf5eb0cce00ae82ef411697378c0a286dc7/README.md) of the Pythia README from the time when this code was forked.

## Setting up the VQA-CP dataset

### Download the data
The VQA-CP dataset is available at <https://www.cc.gatech.edu/~aagrawal307/vqa-cp/>. You can use this [script](https://github.com/gabegrand/adversarial-vqa/blob/master/data_prep/vqa_v2.0_cp/download_vqa_2.0_cp.sh) to download VQA-CP v2. The [`make_trainval_split.py`](https://github.com/gabegrand/adversarial-vqa/blob/master/data_prep/vqa_v2.0_cp/make_trainval_split.py) script will generate the train / valid / test split used in our paper.

### Preprocessing
Instructions for preprocessing can be found at <https://github.com/gabegrand/adversarial-vqa/blob/master/data_prep/data_preprocess.md>.

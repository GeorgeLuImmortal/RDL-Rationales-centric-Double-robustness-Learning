# A Rationale-Centric Framework for Human-in-the-loop Machine Learning

This repository is associated with paper [A Rationale-Centric Framework for Human-in-the-loop Machine Learning](https://arxiv.org/abs/2203.12918) (to be pulished in ACL 2022)

![overview](./plots/overview.png)

## Usage

### Dependencies
Tested Python 3.6, and requiring the following packages, which are available via PIP:

* Required: [numpy >= 1.19.5](http://www.numpy.org/)
* Required: [scikit-learn >= 0.21.1](http://scikit-learn.org/stable/)
* Required: [pandas >= 1.1.5](https://pandas.pydata.org/)
* Required: [torch >= 1.9.0](https://pytorch.org/)
* Required: [transformers >= 4.8.2](https://huggingface.co/transformers/)
* Required: [datasets>=1.14.0](https://huggingface.co/docs/datasets/index)
* Required: [nltk>=3.6.5](https://www.nltk.org/)



### Generate static semi-factual augmented examples by replacing non-rationales

See _static_semi_factual_generation.ipynb_ 

### Generate false rationales augmented data

Run _IMDb_step1_generate_false_rationales_position.py_
Then see _IMDb_generate_false_rationales_examples.ipynb_

### Generate missing rationales augmented data

Run _IMDb_step1_generate_missing_rationales_examples.py_

other baselines training and prediction scripts are in progress ...





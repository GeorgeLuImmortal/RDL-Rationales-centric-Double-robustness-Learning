# A Rationale-Centric Framework for Human-in-the-loop Machine Learning

This repository is associated with paper [A Rationale-Centric Framework for Human-in-the-loop Machine Learning](https://arxiv.org/abs/2203.12918) (to be pulished in ACL 2022)

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


### Step 1. Data Processing

The first step is encoding raw text data into different high-dimensional vectorised representations. The raw text data should be stored in directory "raw_corpora/", each dataset should have its individual directory, for example, the "longer_moviereview/" directory under folder "raw_corpora/". The input corpus of documents should consist of plain text files stored in csv format (two files for one corpus, one for documents belong to class A and one for documents for class B) with a columan named as __text__. It should be noted that the csv file must be named in the format _#datasetname_neg_text.csv_ or _#datasetname_pos_text.csv_. Each row corresponding to one document in that corpus, the format can be refered to the csv file in the sample directory "raw_corpora/longer_moviereview/". Then we can start preprocessing text data and converting them into vectors by:

    python encode_text.py -d dataset_name -t encoding_methods
    
The options of -t are `hbm` (corresponding to the sentence representation generated by the token-level RoBERTa encoder in the paper), `roberta-base` and `fasttext`, for example `-t roberta-base,fasttext` means encoding documents by RoBERTa and FastText respectively. The encoded documents are stored in directory "dataset/", while the FastText document representations are stored in "datasets/fasttext/" and other representations are stored in "datasets/roberta-base/". It should be noted that the sentence representations for hbm is suffixed by ".pt" and the document representations generated by RoBERTa are suffixed by ".csv"(average all tokens to represent a document) or "\_cls.csv" (using classifier token "\<s\>" to represent a document). Due to the upload file size limit, we did not upload sample ".pt" files but you can generate yours. For encoding by FastText, you need to download the pretrained FastText model in advance (see __Dependencies__).

### Step 2. 


### Step 3. 


### Step 4.


### Step 5. 


### Step 6.


## Setup of experiments in the paper


1.step0: semi-facutal augmented
2.step0: duplication
3.step0: random replacement




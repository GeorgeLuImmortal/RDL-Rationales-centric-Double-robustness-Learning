#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ['PYTHONHASHSEED'] = str(2019)
os.environ['TRANSFORMERS_CACHE'] = 'D:\\python_pkg_data\\huggingface\\transformers'

from tqdm import tqdm_notebook,tqdm

import numpy as np 
np.random.seed(2019)
import random
random.seed(2019)

import torch
torch.manual_seed(2019)
torch.cuda.manual_seed_all(2019)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from sklearn.utils import shuffle
import transformers

from datasets import load_metric,load_dataset,Value
import csv


import nltk
nltk.data.path.append('D:\\python_pkg_data\\nltk_data')

import pandas as pd
import json
import glob
import math
from optparse import OptionParser







args = {
    'ori_test_dir':'./datasets/IMDb/orig/test.tsv',
    'gpu_device':0,
    'tokenizer': transformers.AutoTokenizer.from_pretrained('roberta-base'),
#     'tokenizer': transformers.AutoTokenizer.from_pretrained('bert-base-cased'),
    'dataset_cache_dir':"D:\\python_pkg_data\\huggingface\\Datasets", ## local directory for datasets
#     'train_random_seed':[2024,2025,2026,2027,2028],
    'num_per_class': 25,                                              ## number of examples per class for initial training set
    
}






metric = load_metric("accuracy")

 ## take a list of strings as optional arguments
def list_callback(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def import_data(directory):
    
    data_pos = []
    data_neg = []
    with open(directory,errors='ignore') as file:
        file = csv.reader(file, delimiter="\t")
        for idx,row in enumerate(file):
            if idx!=0:
                if row[0] == 'Negative':
                    data_neg.append({'idx':idx,'text':row[1],'label':0})
                else:
                    data_pos.append({'idx':idx,'text':row[1],'label':1})
            
    return data_neg,data_pos

class CustomerDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)





# ood yelp testing set

def yelp_tokenize_function(examples):
    return args['tokenizer'](examples["text"], padding="max_length", truncation=True)



yelp_raw_datasets = load_dataset("yelp_polarity",cache_dir=args['dataset_cache_dir'])

## get testing set and dataloader format processing
yelp_tokenized_datasets = yelp_raw_datasets['test'].map(yelp_tokenize_function, batched=True)
yelp_tokenized_datasets = yelp_tokenized_datasets.remove_columns(["text"])
yelp_tokenized_datasets = yelp_tokenized_datasets.rename_column("label", "labels")
yelp_tokenized_datasets.set_format("torch")


## construct testing dataloader
yelp_test_dataloader = torch.utils.data.DataLoader(yelp_tokenized_datasets, batch_size=64,shuffle=False)

## testing set statistics
yelp_labels = [yelp_tokenized_datasets.__getitem__(i)['labels'].item() for i in range(len(yelp_tokenized_datasets))]
    
print('Yelp testing data statistics -----------------------')
print(f'Num of examples {len(yelp_labels)}')
print(np.unique(yelp_labels), np.bincount(yelp_labels))
print('----------------------------------------------------')


# #  Start testing

if __name__ == "__main__":

    parser = OptionParser(usage='usage: -n num_per_example -r random_seed')
    parser.add_option("-n","--num_per_example", action="store", type="int", dest="num_per_example", help="number of augmented example per review", default = '3')
    parser.add_option('-r', '--random_seeds', type='string', action='callback',dest='random_seeds',callback=list_callback,default=['2019'])

    (options, _) = parser.parse_args()

    num_per_example = int(options.num_per_example)
   

    dir_dict = {
            'step0_human':{'result_dir':'og_results','model_dir':'./IMDb_og_human_trainer'},
            'step0_rs':{'result_dir':'og_results','model_dir':'./IMDb_rs_trainer'},
            'step0_dp':{'result_dir':'og_results','model_dir':'./IMDb_dp_trainer'},
            'step1_human':{'result_dir':'og_results','model_dir':'./step1_LR_5e-6/IMDb_og_human_trainer_step1'},
            'step1_hybrid':{'result_dir':'og_results','model_dir':'./step1_hybrid_LR_5e-6/IMDb_og_human_trainer_hybrid_step1'},
            'step1_ow':{'result_dir':'ow_results','model_dir':'./step1_baseline_LR1.25e-5/new_IMDb_trainer_step1'},
            'all':{'result_dir':'ow_results','model_dir':'./step1_all/IMDb_all_trainer'},
            'step1_CF':{'result_dir':'ow_results','model_dir':'./step1_hovy/IMDb_CF_step1'},
            'step1_CF_all':{'result_dir':'ow_results','model_dir':'./IMDb_CF_all_step1'},
            'step1_auto_CF_all':{'result_dir':'ow_results','model_dir':'./IMDb_AUTO_CF_all_step1'}
           }

## options are ow-without augmentation,dp-duplicate,og-over generalisation,rs-random replace
    method = 'step1_auto_CF_all'


    train_seed = [int(number) for number in options.random_seeds]
    num_per_class = args['num_per_class']

    for seed in train_seed:
    
        # model_dir = glob.glob(f"{dir_dict[method]['model_dir']}_{seed}_{num_per_class}_{num_per_example}/checkpoint*")[0]
        # model_dir = glob.glob(f"{dir_dict[method]['model_dir']}_{seed}_{num_per_class}/checkpoint*")[0]
        model_dir = glob.glob(f"{dir_dict[method]['model_dir']}_{seed}/checkpoint*")[0]
        print(model_dir)
        model = transformers.AutoModelForSequenceClassification.from_pretrained(model_dir).cuda(args['gpu_device'])

    #     Yelp testing

        metric= load_metric("accuracy")
        model.eval()
        for batch in tqdm(yelp_test_dataloader):
            batch = {k: v.cuda(args['gpu_device']) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])

        yelp_result = metric.compute()
        print(f"{seed}- Yelp acc: {yelp_result['accuracy']}")

        del logits
        del predictions
        del outputs
        del metric




        result_df = pd.DataFrame({'Seed':seed,'Yelp':[yelp_result['accuracy']]})

        result_df = result_df.set_index('Seed')

        # if os.path.exists(f"{dir_dict[method]['result_dir']}/{method}_{num_per_class}_{num_per_example}_yelp.csv"):
        #     result_df.to_csv(f"{dir_dict[method]['result_dir']}/{method}_{num_per_class}_{num_per_example}_yelp.csv",mode='a',header=False)
        # else:
        #     result_df.to_csv(f"{dir_dict[method]['result_dir']}/{method}_{num_per_class}_{num_per_example}_yelp.csv")

        # del model

        if os.path.exists(f"{dir_dict[method]['result_dir']}/{method}_{num_per_class}_yelp.csv"):
            result_df.to_csv(f"{dir_dict[method]['result_dir']}/{method}_{num_per_class}_yelp.csv",mode='a',header=False)
        else:
            result_df.to_csv(f"{dir_dict[method]['result_dir']}/{method}_{num_per_class}_yelp.csv")

        del model


         




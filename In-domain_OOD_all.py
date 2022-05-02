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





# # In domain testing dataset


test_data_neg,test_data_pos = import_data(args['ori_test_dir'])
test_data = test_data_neg +test_data_pos

test_texts = [doc['text'] for doc in test_data]
test_labels = [doc['label'] for doc in test_data]

test_encodings = args['tokenizer'](test_texts, truncation=True, padding=True)
test_dataset = CustomerDataset(test_encodings, test_labels)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=128,shuffle=True)

## testing set statistics
print('IMDb in-domain testing data statistics -----------------------')
print(f'Num of examples {len(test_labels)}')
print(np.unique(test_labels), np.bincount(test_labels))
print('----------------------------------------------------')


# # Semeval Testing Dataset




with open('./datasets/OOD_Data/2017_English_final/2017_English_final/GOLD/Subtasks_BD/twitter-2016test-BD.txt', 'r') as in_file:
    stripped = (line.strip() for line in in_file)
    lines = (line.split("\t") for line in stripped if line)
    with open('./twitter-2017.csv', 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(('id', 'name','label','sentence'))
        writer.writerows(lines)





test_ori = pd.read_csv('./twitter-2017.csv')
test_sentences = test_ori['sentence'].values
test_labels = test_ori['label'].values

semeval_texts = []
semeval_test_labels = []

pos_num = 0

for idx,label in tqdm(enumerate(test_labels)):
    if label == 'negative':
        semeval_texts.append(test_sentences[idx])
        semeval_test_labels.append(0)
    if label == 'positive'and pos_num<2339:
        pos_num += 1
        semeval_texts.append(test_sentences[idx])
        semeval_test_labels.append(1)

print('Semeval testing data statistics -----------------------')
print(f'Num of examples {len(semeval_test_labels)}')
print(np.unique(semeval_test_labels), np.bincount(semeval_test_labels))
print('----------------------------------------------------')

semeval_test_encodings = args['tokenizer'](semeval_texts, truncation=True, padding=True)
semeval_test_dataset = CustomerDataset(semeval_test_encodings, semeval_test_labels)
semeval_test_dataloader = torch.utils.data.DataLoader(semeval_test_dataset, batch_size=128,shuffle=True)





# # SST-2 testing dataset


def sst_tokenize_function(examples):
    tokenized_batch = args['tokenizer'](examples["sentence"], padding="max_length", truncation=True)
    tokenized_batch["label"] = [round(label) for label in examples["label"]]
    return tokenized_batch 


sst_raw_datasets = load_dataset("sst",cache_dir=args['dataset_cache_dir'])

## get testing set and dataloader format processing
sst_tokenized_datasets = sst_raw_datasets['test'].map(sst_tokenize_function, batched=True)
sst_tokenized_datasets = sst_tokenized_datasets.remove_columns(["sentence"])
sst_tokenized_datasets = sst_tokenized_datasets.remove_columns(["tokens"])
sst_tokenized_datasets = sst_tokenized_datasets.remove_columns(["tree"])
sst_tokenized_datasets = sst_tokenized_datasets.rename_column("label", "labels")
sst_tokenized_datasets.set_format("torch")



## use the full testing set
sst_new_features = sst_tokenized_datasets.features.copy()
sst_new_features["labels"] = Value('int64')
sst_tokenized_datasets = sst_tokenized_datasets.cast(sst_new_features)

## construct testing dataloader
sst_test_dataloader = torch.utils.data.DataLoader(sst_tokenized_datasets, batch_size=64,shuffle=False)

## testing set statistics
sst_labels = [sst_tokenized_datasets.__getitem__(i)['labels'].item() for i in range(len(sst_tokenized_datasets))]

print('SST-2 testing data statistics -----------------------')
print(f'Num of examples {len(sst_labels)}')
print(np.unique(sst_labels), np.bincount(sst_labels))
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
            'step1_missing':{'result_dir':'og_results','model_dir':'./step1_missing_LR_5e-6/IMDb_og_human_trainer_missing_step1'},
            'step1_ow':{'result_dir':'ow_results','model_dir':'./step1_baseline_LR1.25e-5/new_IMDb_trainer_step1'},
            'all':{'result_dir':'ow_results','model_dir':'./step1_all/IMDb_all_trainer'},
            'step1_CF':{'result_dir':'ow_results','model_dir':'./IMDb_CF_step1'},
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
        model.eval()

        # indomain testing

        metric = load_metric("accuracy")
        for batch in tqdm(test_dataloader):
            batch = {k: v.cuda(args['gpu_device']) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])

        result = metric.compute()
        print(f"{seed} in-domain acc: {result['accuracy']}")

        del logits
        del predictions
        del outputs
        del metric
              
        # Semeval testing

        metric = load_metric("accuracy")
        model.eval()
        for batch in tqdm(semeval_test_dataloader):
            batch = {k: v.cuda(args['gpu_device']) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])

        semeval_result = metric.compute()
        print(f"{seed} Semeval acc: {semeval_result['accuracy']}")

        del logits
        del predictions
        del outputs
        del metric



        # SST-2 testing

        metric= load_metric("accuracy")
        model.eval()
        for batch in tqdm(sst_test_dataloader):
            batch = {k: v.cuda(args['gpu_device']) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])

        sst_result = metric.compute()
        print(f"{seed}- SST2 acc: {sst_result['accuracy']}")

        del logits
        del predictions
        del outputs
        del metric



        result_df = pd.DataFrame({'Seed':seed,'In-domain':[result['accuracy']],'Semeval':[semeval_result['accuracy']],
                                  'SST-2':[sst_result['accuracy']]})
        result_df = result_df.set_index('Seed')

        # if os.path.exists(f"{dir_dict[method]['result_dir']}/{method}_{num_per_class}_{num_per_example}.csv"):
        #     result_df.to_csv(f"{dir_dict[method]['result_dir']}/{method}_{num_per_class}_{num_per_example}.csv",mode='a',header=False)
        # else:
        #     result_df.to_csv(f"{dir_dict[method]['result_dir']}/{method}_{num_per_class}_{num_per_example}.csv")

        if os.path.exists(f"{dir_dict[method]['result_dir']}/{method}_{num_per_class}.csv"):
            result_df.to_csv(f"{dir_dict[method]['result_dir']}/{method}_{num_per_class}.csv",mode='a',header=False)
        else:
            result_df.to_csv(f"{dir_dict[method]['result_dir']}/{method}_{num_per_class}.csv")

        del model


         




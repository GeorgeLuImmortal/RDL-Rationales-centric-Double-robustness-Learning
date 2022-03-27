#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
os.environ['PYTHONHASHSEED'] = str(2019)
os.environ['TRANSFORMERS_CACHE'] = 'D:\\python_pkg_data\\huggingface\\transformers'

import json
from tqdm import tqdm_notebook

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
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

import ast
import glob
import shutil

import importlib
os.environ["WANDB_DISABLED"] = "true"
from optparse import OptionParser


# In[ ]:


args = {
    'ori_train_dir':'./datasets/IMDb/orig/train.tsv',
    'ori_dev_dir':'./datasets/IMDb/orig/dev.tsv',
    'gpu_device':0,
    'tokenizer': transformers.AutoTokenizer.from_pretrained('roberta-base'),
    'dataset_cache_dir':"D:\\python_pkg_data\\huggingface\\Datasets", ## local directory for datasets
    'num_per_class': 25,                                              ## number of examples per class for initial training set
    'save_dir': './SF_results/IMDb_step0_sf_trainer',                                  ##directory for saving models
    'new_save_dir': './FR_results/IMDb_step0_fr_trainer',
}


# In[ ]:


metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

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

def import_paired_data(directory,original_texts):
    paired_data = {}
    with open(directory,errors='ignore') as file:
        file = csv.reader(file, delimiter="\t")
        for idx,row in enumerate(file):
            if idx!=0:
                if row[2] not in paired_data.keys():
                    paired_data[row[2]] = []

                    if row[1] in original_texts:
                        paired_data[row[2]].append({'text':row[1],'label':0 if row[0]=='Negative'else 1,'ori_flag':1})
                    else:
                        paired_data[row[2]].append({'text':row[1],'label':0 if row[0]=='Negative'else 1,'ori_flag':0})
                else:
                    if row[1] in original_texts:
                        paired_data[row[2]].append({'text':row[1],'label':0 if row[0]=='Negative'else 1,'ori_flag':1})
                    else:
                        paired_data[row[2]].append({'text':row[1],'label':0 if row[0]=='Negative'else 1,'ori_flag':0})
                        
    return paired_data


# In[ ]:


def construct_training_eval(args,random_seed,num_per_example=7): 
    
    new_save_dir = f"{args['new_save_dir']}_{random_seed}_{args['num_per_class']}_{num_per_example}"
    print(new_save_dir)
    if not os.path.exists(new_save_dir):
        os.mkdir(new_save_dir)
    
    ## load false-rationales augmented data
    augmented_data_dir =  f"{args['save_dir']}_{random_seed}_{args['num_per_class']}_{num_per_example}/false_rationales_augmented_step1.json"
    print(augmented_data_dir)
    with open(augmented_data_dir, "r") as file_name:
        augmented_data = json.load(file_name)
              
    train_keys = list(augmented_data.keys())
    print('Training example index',train_keys)  
              
    with open(f"{new_save_dir}/keys.txt", "w") as fp:
        for k in train_keys:
            fp.write(str(k) +"\n")
  

  
    ## import train data
    IMDb_data = {}

    with open(args['ori_train_dir'],errors='ignore') as file:
        file = csv.reader(file, delimiter="\t")
        for idx,row in enumerate(file):
            if len(row)>0:

                if row[0] == 'Negative':
                    IMDb_data[row[2]] = {'text':row[1],'label':0}
                else:
                    IMDb_data[row[2]] = {'text':row[1],'label':1}

    dev_data_neg,dev_data_pos = import_data(args['ori_dev_dir'])
    dev_data = dev_data_neg +dev_data_pos

                          
    ## magnify with false-rationales augmented data
    train_texts = []
    train_labels = []
                                   
    for key in train_keys:
        label = IMDb_data[key]['label']
        train_texts.append(IMDb_data[key]['text'])
        train_labels.append(label)
    
        candidates = augmented_data[key]['candidates']
#         sample_index = shuffle([i for i in range(len(candidates))],random_state=args['train_random_seed'])
#         candidates = np.array(candidates)[sample_index][:num_per_example]
        candidates = np.array(candidates)[:num_per_example]

        train_texts = train_texts + list(candidates)
        train_labels = train_labels + [label]*len(candidates)
                                   
    eval_texts = [doc['text'] for doc in dev_data]
    eval_labels = [doc['label'] for doc in dev_data]

    train_encodings = args['tokenizer'](train_texts, truncation=True, padding=True)
    eval_encodings = args['tokenizer'](eval_texts, truncation=True, padding=True)

    print('IMDb training data statistics -----------------------')
    print(np.unique(train_labels),np.bincount(train_labels))
    print('IMDb eval data statistics -----------------------')
    print(np.unique(eval_labels),np.bincount(eval_labels))

    train_dataset = CustomerDataset(train_encodings, train_labels)
    eval_dataset = CustomerDataset(eval_encodings, eval_labels)
    
    return train_dataset, eval_dataset


def fine_tune(args,pre_trained_model,train_dataset,eval_dataset,random_seed,num_per_example):

    new_save_dir = f"{args['new_save_dir']}_{random_seed}_{args['num_per_class']}_{num_per_example}"

    training_args = transformers.TrainingArguments(new_save_dir,per_device_train_batch_size=4,per_device_eval_batch_size=16, 
                                      evaluation_strategy="epoch",num_train_epochs = 20.0,
                                      save_strategy='epoch',overwrite_output_dir=True,logging_strategy='epoch',load_best_model_at_end = True,
                                     metric_for_best_model='loss',learning_rate=5e-6)


    trainer = transformers.Trainer(
        model=pre_trained_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks = [transformers.EarlyStoppingCallback(early_stopping_patience=5)]
    )

    ## remove logging file
    if os.path.exists("lossoutput.txt"):
        os.remove('lossoutput.txt')

    trainer.train()

    ## record best performing model based on eval accuracy

    loggings = {}

    with open('./lossoutput.txt','r') as file:
        for line in file.readlines():
            line = ast.literal_eval(line)
            if 'eval_accuracy' in line.keys():
                loggings[line['step']] = {'eval_accuracy':line['eval_accuracy'],'eval_loss':line['eval_loss']}
                
    
                

    ##  sort by eval_acc and eval_loss descending order
              
    dicts = [{k: v} for (k,v) in loggings.items()]
    dicts.sort(key=lambda d: (-list(d.values())[0]['eval_accuracy'], -list(d.values())[0]['eval_loss'],))
    
    best_step = list(dicts[0].keys())[0]
    best_acc =dicts[0][best_step]['eval_accuracy']

    # remove the rest models
    
    for idx,directory in enumerate(glob.glob(f"{new_save_dir}/*")):
        if os.path.basename(directory)[11:] != str(best_step):
            try:
                shutil.rmtree(directory)
            except Exception:
                print(f"Exception {directory}")                             
                                             
    with open(f"{new_save_dir}/loggings.json", "w") as file_name:
        json.dump(loggings, file_name)
            
    del pre_trained_model
    del trainer
    
            
    return best_acc, best_step


if __name__ == "__main__":


    parser = OptionParser(usage='usage: -n num_per_example -r random_seed')
    parser.add_option("-n","--num_per_example", action="store", type="int", dest="num_per_example", help="number of augmented example per review", default = '7')
    parser.add_option("-r","--random_seed", action="store", type="int", dest="random_seed", help="random seed for initialisation", default = '2019')
   
    (options, _) = parser.parse_args()

    num_per_example = int(options.num_per_example)
    random_seed = int(options.random_seed)
    # learning_rate = float(options.learning_rate)
    # print(f"learning rate is {learning_rate}")

    ## construct training/evaluation set
    train_dataset, eval_dataset = construct_training_eval(args,num_per_example=num_per_example,random_seed=random_seed)


    # In[ ]:


    if 'model' in globals() or 'model' in locals():
        del model
    old_dir = f"{args['save_dir']}_{random_seed}_{args['num_per_class']}_{num_per_example}/checkpoint*"
    model_dir = glob.glob(old_dir)[0]
    print(model_dir)
                          
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=2).cuda(args['gpu_device'])

    best_acc, best_step = fine_tune(args,model,train_dataset,eval_dataset,random_seed,num_per_example)

# del model


# In[ ]:





# In[ ]:





# In[ ]:





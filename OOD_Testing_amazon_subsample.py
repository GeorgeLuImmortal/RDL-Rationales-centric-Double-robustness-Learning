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


# In[2]:


dir_dict = {'ow':{'result_dir':'ow_results','model_dir':'IMDb_trainer'},
           'dp':{'result_dir':'dp_results','model_dir':'IMDb_dp_trainer'},
           'rs':{'result_dir':'rs_results','model_dir':'IMDb_rs_trainer'},
           'og':{'result_dir':'og_results','model_dir':'IMDb_og_trainer'},
            'BERT_ow':{'result_dir':'ow_results','model_dir':'IMDb_trainer_BERT'},
            'BERT_og':{'result_dir':'og_results','model_dir':'IMDb_og_trainer_BERT'},
            'step1_ow':{'result_dir':'ow_results','model_dir':'Step1_IMDb_trainer'},
            'step1_og':{'result_dir':'og_results','model_dir':'Step1_IMDb_og_trainer'},
            'step1_rs':{'result_dir':'rs_results','model_dir':'Step1_IMDb_rs_trainer'},
            'step1_dp':{'result_dir':'dp_results','model_dir':'Step1_IMDb_dp_trainer'},
            'step1_auto_CF_all':{'result_dir':'ow_results','model_dir':'./IMDb_AUTO_CF_all_step1'}
           }

## options are ow-without augmentation,dp-duplicate,og-over generalisation,rs-random replace
method = 'step1_auto_CF_all'

args = {
    'ori_test_dir':'./datasets/IMDb/orig/test.tsv',
    'gpu_device':0,
    'tokenizer': transformers.AutoTokenizer.from_pretrained('roberta-base'),
    # 'tokenizer': transformers.AutoTokenizer.from_pretrained('bert-base-cased'),
    'dataset_cache_dir':"D:\\python_pkg_data\\huggingface\\Datasets", ## local directory for datasets
    'num_per_class': 25,                                              ## number of examples per class for initial training set
    'result_dir':dir_dict[method]['result_dir'],
    'model_dir': dir_dict[method]['model_dir']
}


# In[3]:



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


# In[4]:


if __name__ == "__main__":

    parser = OptionParser(usage='usage: -n num_per_example -r random_seed')
    parser.add_option("-n","--num_per_example", action="store", type="int", dest="num_per_example", help="number of augmented example per review", default = '3')
    parser.add_option('-r', '--random_seeds', type='string', action='callback',dest='random_seeds',callback=list_callback,default=['2019','2020'])

    (options, _) = parser.parse_args()

    num_per_example = int(options.num_per_example)
    train_seed = [int(number) for number in options.random_seeds]


# # Amazon testing dataset

# In[5]:


    beauty = []
    for line in tqdm(open('./datasets/OOD_Data/All_Beauty.json/All_Beauty.json')):
        beauty.append(json.loads(line))
    beauty_df = pd.DataFrame(pd.DataFrame(beauty),columns=['overall','reviewText'])
        
    software = []
    for line in tqdm(open('./datasets/OOD_Data/Software.json/Software.json')):
        software.append(json.loads(line))
    software_df = pd.DataFrame(pd.DataFrame(software),columns=['overall','reviewText'])
        
    fashion = []
    for line in tqdm(open('./datasets/OOD_Data/AMAZON_FASHION.json/AMAZON_FASHION.json')):
        fashion.append(json.loads(line))
    fashion_df = pd.DataFrame(pd.DataFrame(fashion),columns=['overall','reviewText'])
        
    magazine = []
    for line in tqdm(open('./datasets/OOD_Data/Magazine_Subscriptions.json/Magazine_Subscriptions.json')):
        magazine.append(json.loads(line))
    magazine_df = pd.DataFrame(pd.DataFrame(magazine),columns=['overall','reviewText'])
        
    giftcards = []
    for line in tqdm(open('./datasets/OOD_Data/Gift_Cards.json/Gift_Cards.json')):
        giftcards.append(json.loads(line))
    giftcards_df = pd.DataFrame(pd.DataFrame(giftcards),columns=['overall','reviewText'])
        
    appliances = []
    for line in tqdm(open('./datasets/OOD_Data/Appliances.json/Appliances.json')):
        appliances.append(json.loads(line))
    applicances_df = pd.DataFrame(pd.DataFrame(appliances),columns=['overall','reviewText'])




    pdList = [software_df, 
              fashion_df, 
              applicances_df,
              beauty_df, 
              magazine_df, 
              giftcards_df] 

    amazon_df = pd.concat(pdList)
    print(len(amazon_df))




    amazon_texts = []
    amazon_test_labels = []

    neg_num = 0
    pos_num = 0

    for score,text in tqdm(zip(amazon_df['overall'].values,amazon_df['reviewText'].values)):
        
        if (score == 1.0) or (score == 2.0):
            if not isinstance(text,float) and neg_num <= 470569:
                neg_num += 1
                amazon_texts.append(text)
                amazon_test_labels.append(0)
            else:
                pass

        if (score == 4.0) or (score == 5.0):
            
            if not isinstance(text,float) and pos_num < 470569:
                pos_num+=1
                amazon_texts.append(text)
                amazon_test_labels.append(1)
            else:
                pass
                          
            
    print('Amazon testing data statistics -----------------------')
    print(f'Num of examples {len(amazon_test_labels)}')
    print(np.unique(amazon_test_labels), np.bincount(amazon_test_labels))
    print('----------------------------------------------------')


    # In[15]:


    from sklearn.model_selection import StratifiedShuffleSplit    

    sss = StratifiedShuffleSplit(n_splits=1, 
                                 test_size=.99, random_state=2019)  


    # In[16]:


    train_indexs = []
    test_indexs = []
    for train_index, test_index in sss.split(amazon_texts, amazon_test_labels):
        train_indexs.append(train_index)


    # In[17]:


    subsample_amazon_texts = [amazon_texts[i] for i in train_indexs[0]]
    subsample_amazon_labels = [amazon_test_labels[i] for i in train_indexs[0]]

    print('Amazon testing data statistics -----------------------')
    print(f'Num of examples {len(subsample_amazon_labels)}')
    print(np.unique(subsample_amazon_labels), np.bincount(subsample_amazon_labels))
    print('----------------------------------------------------')


    # In[18]:


    amazon_test_encodings = args['tokenizer'](subsample_amazon_texts, truncation=True, padding=True)


    # In[20]:


    amazon_test_dataset = CustomerDataset(amazon_test_encodings, subsample_amazon_labels)
    amazon_test_dataloader = torch.utils.data.DataLoader(amazon_test_dataset, batch_size=64,shuffle=True)


    # In[ ]:





    # In[13]:


    del amazon_df,software_df, fashion_df, applicances_df,beauty_df, magazine_df, giftcards_df


    # #  Start testing

    # In[21]:


    
    num_per_class = args['num_per_class']

    for seed in train_seed:
        
        # model_dir = glob.glob(f"{args['model_dir']}_{seed}_{num_per_class}/checkpoint*")[0]
        model_dir = glob.glob(f"{args['model_dir']}_{seed}/checkpoint*")[0]
        print(model_dir)
        model = transformers.AutoModelForSequenceClassification.from_pretrained(model_dir).cuda(args['gpu_device'])
        model.eval()


        # Amazon testing

        metric= load_metric("accuracy")
        model.eval()
        for batch in tqdm(amazon_test_dataloader):
            batch = {k: v.cuda(args['gpu_device']) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])

        amazon_result = metric.compute()
        print(f"{seed}- Amazon acc: {amazon_result['accuracy']}")

        del logits
        del predictions
        del outputs
        del metric

        result_df = pd.DataFrame({'Seed':seed,'Amazon':[amazon_result['accuracy']]})
        result_df = result_df.set_index('Seed')

        if os.path.exists(f"{args['result_dir']}/{method}_{num_per_class}_amazon.csv"):
            result_df.to_csv(f"{args['result_dir']}/{method}_{num_per_class}_amazon.csv",mode='a',header=False)
        else:
            result_df.to_csv(f"{args['result_dir']}/{method}_{num_per_class}_amazon.csv")

        del model


# In[ ]:





# In[ ]:





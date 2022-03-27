#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ['PYTHONHASHSEED'] = str(2019)
os.environ['TRANSFORMERS_CACHE'] = 'D:\\python_pkg_data\\huggingface\\transformers'

import json
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
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

import ast
import glob
import shutil

import importlib
from torch.utils.data import DataLoader
from torch.nn import Softmax
from termcolor import colored
from itertools import groupby
from operator import itemgetter
import html
from IPython.core.display import display, HTML
import more_itertools as mit
from optparse import OptionParser


# In[2]:


args = {
    'ori_train_dir':'./datasets/IMDb/orig/train.tsv',
    'gpu_device':0,
    'tokenizer': transformers.AutoTokenizer.from_pretrained('roberta-base'),
    'dataset_cache_dir':"D:\\python_pkg_data\\huggingface\\Datasets", ## local directory for datasets
    # 'train_random_seed':2019,                                        ## random seed for subsampling training set
    'num_per_class': 25,                                              ## number of examples per class for initial training set
    'save_dir': './SF_results/IMDb_step0_sf_trainer',                                   ##directory for saving models
    'num_per_step':50,                                                ##num labelled data per step
    'num_per_example':7,
    'labelled_pos': './datasets/IMDb/human_labelled/positives.json',
    'labelled_neg':'./datasets/IMDb/human_labelled/negatives.json',
    'supplement': './datasets/IMDb/human_labelled/supplement_rationales.tsv',
    'al_dir': './AL_results/AL_step0_IMDb_trainer'
}


# In[3]:


metric = load_metric("accuracy")

def html_escape(text):
    return html.escape(text)

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

def visualise_rationales(original,rationale_spans,rationale_pos,visualise_all=False):
    
    if visualise_all:
        highlighted = []
        for idx,term in enumerate(word_tokenize(original)):
            if idx in rationale_pos:
                highlighted.append(colored(term,'blue'))
            else:
                highlighted.append(term)
            
        return TreebankWordDetokenizer().detokenize(highlighted)
                
    else:
        highlights = []
        for span in rationale_spans:
            highlighted = []
            for idx,term in enumerate(word_tokenize(original)):
                if idx in span:
                    highlighted.append(colored(term,'blue'))
                else:
                    highlighted.append(term)
                    
            highlights.append(TreebankWordDetokenizer().detokenize(highlighted))
        
        return highlights
    
def visualise_rationales_html(original,rationale_spans,rationale_pos,visualise_all=False):
    
    if visualise_all:
        highlighted = []
        for idx,term in enumerate(word_tokenize(original)):
            if idx in rationale_pos:
                highlighted.append('<font color="blue">' + html_escape(term) + '</font>')
            else:
                highlighted.append(term)
            
        return TreebankWordDetokenizer().detokenize(highlighted)
                
    else:
        highlights = []
        for span in rationale_spans:
            highlighted = []
            for idx,term in enumerate(word_tokenize(original)):
                if idx in span:
                    highlighted.append('<font color="blue">' + html_escape(term) + '</font>')
                else:
                    highlighted.append(term)
                    
            highlights.append(TreebankWordDetokenizer().detokenize(highlighted))
        
        return highlights
    

def detect_rationale_spans(non_rationale_pos,text_length,max_length=1):
    
    rationale_spans = []
    
    rationale_pos = list(set([i for i in range(text_length)])-set(non_rationale_pos))
    rationale_pos.sort()
 
    for k, g in groupby(enumerate(rationale_pos),lambda ix : ix[0] - ix[1]): 
        span = list(map(itemgetter(1), g))
        if len(span) <= max_length:
            rationale_spans.append(span)
    
    
      
    return rationale_spans, rationale_pos

def identify_important_terms(token_text):
    candidates = []
    remove_terms = {}
    count = 0
    for idx,token in enumerate(token_text):
        duplicate = token_text.copy()
        remove_terms[count] = {'terms':duplicate[idx],'start_token':idx,'end_token':idx+1}
        del duplicate[idx]
        count += 1
        candidates.append(TreebankWordDetokenizer().detokenize(duplicate))

    for idx,token in enumerate(token_text[:-1]):
        duplicate = token_text.copy()
        remove_terms[count] = {'terms':duplicate[idx:idx+2],'start_token':idx,'end_token':idx+2}
        del duplicate[idx:idx+2]
        count += 1
        candidates.append(TreebankWordDetokenizer().detokenize(duplicate))

    for idx,token in enumerate(token_text[:-2]):
        duplicate = token_text.copy()
        remove_terms[count] = {'terms':duplicate[idx:idx+3],'start_token':idx,'end_token':idx+3}
        del duplicate[idx:idx+3]
        count += 1
        candidates.append(TreebankWordDetokenizer().detokenize(duplicate))


    candidates.append(text)
    
    return candidates, remove_terms


def get_rationale_spans(model,text,label,topk=15):
    
#     text = imdb_texts[example_idx]
#     label = imdb_labels[example_idx]
    token_text = word_tokenize(text)

    candidates, remove_terms = identify_important_terms(token_text)

    candidates_label = [label]*len(candidates)

    candidates_encodings = args['tokenizer'](candidates, truncation=True, padding=True)
    candidates_dataset = CustomerDataset(candidates_encodings, candidates_label)
    candidates_dataloader = DataLoader(candidates_dataset, batch_size=32, shuffle=False)

    model.eval()
    output_logits = []
    for batch in tqdm(candidates_dataloader):
        batch = {k: v.cuda(args['gpu_device']) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        output_logits.append(logits)

    outputs = torch.cat(output_logits)
    sm = Softmax(dim=1)
    outputs = sm(outputs)

    results = {}
    for idx,score in enumerate(outputs[:-1]):
        changes = abs(float(outputs[idx][label]-outputs[-1][label]))
        results[idx]=changes

    token_id = list(dict(sorted(results.items(), key=lambda item: item[1],reverse=True)).keys())

    inferred_spans = []
    for ids in token_id[:topk]:
#         print(ids, remove_terms[ids]['terms'])
        span = [i for i in range(remove_terms[ids]['start_token'],remove_terms[ids]['end_token'])]
        inferred_spans.append(span)

    inferred_pos = []
    for span in inferred_spans:
        for number in span:
            inferred_pos.append(number)


    inferred_pos = list(set(inferred_pos))
    
    return inferred_pos


# In[4]:

if __name__ == "__main__":


    parser = OptionParser(usage='usage: -n num_per_example -r random_seed')
    parser.add_option("-r","--random_seed", action="store", type="int", dest="random_seed", help="random seed for initialisation", default = '2019')
   
    (options, _) = parser.parse_args()


    random_seed = int(options.random_seed)
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










    pos_rationales = json.load(open(args['labelled_pos'], 'r'))
    neg_rationales = json.load(open(args['labelled_neg'], 'r'))

    rationale_spans = {}
    for item in neg_rationales:
        key = list(item.keys())[1]
        index = list(item.keys())[1][9:-4]
        rationale_spans[index] = item[key]
        

    for item in pos_rationales:
        key = list(item.keys())[1]
        index = list(item.keys())[1][9:-4]
        rationale_spans[index] = item[key]

    rationale_positions = {}

    train_keys = list(rationale_spans.keys())

    for key in train_keys:
        doc_positions = []
        positions = rationale_spans[key]
        for span in positions:
            start = span['start_token']
            end = span['end_token']
            doc_positions = doc_positions +[i for i in range(start,end)]
            
        rationale_positions[key] = doc_positions

    len(rationale_positions)





    supplement_rationales = {}
    with open(args['supplement'],'r') as file:
        file = csv.reader(file, delimiter='\t')
        for idx,row in enumerate(file):
            supplement_rationales[row[0]] = ast.literal_eval(row[1])




    rationale_positions.update(supplement_rationales)


    # # Most uncertainty examples




    train_keys_dir = f"{args['al_dir']}_{random_seed}_{args['num_per_class']}/keys.txt"
    new_keys_dir = f"{args['al_dir']}_{random_seed}_{args['num_per_class']}/new_keys.txt"

    all_keys = []
    with open(train_keys_dir,'r') as file:
        for key in file.readlines():
            all_keys.append(key[:-1])
            
    with open(new_keys_dir,'r') as file:
        for key in file.readlines():
            all_keys.append(key[:-1])





    model_dir = glob.glob(f"{args['save_dir']}_{random_seed}_{args['num_per_class']}_{args['num_per_example']}/checkpoint*")[0]
    print(f"previous model {model_dir}")
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=2).cuda(args['gpu_device'])


    


    augmented_data ={}
    for example_idx in tqdm(all_keys[:]):
        
        
        
        text = IMDb_data[example_idx]['text']
        label = IMDb_data[example_idx]['label']
        token_text = word_tokenize(text)
        
        print(example_idx, 'negative' if label==0 else 'positive')
        
        generated_rationales_spans = get_rationale_spans(model,text,label)
        
        
        
        highlighted = visualise_rationales(text,_,generated_rationales_spans,visualise_all=True)
        print(highlighted)



        rationale_pos = rationale_positions[example_idx]
        highlighted = visualise_rationales(text,_,rationale_pos,visualise_all=True)
        print(highlighted)


        missing_rationales = list(set(rationale_pos) - set(generated_rationales_spans))
        missing_rationales.sort()
        missing_rationale_pos = [list(group) for group in mit.consecutive_groups(missing_rationales)]
        missing_rationale_token = [TreebankWordDetokenizer().detokenize(token_text[row[0]:row[-1]+1]) for row in missing_rationale_pos]

        selected_sentence = []
        sentences  = sent_tokenize(text)
        for idx,sent in enumerate(sentences):
            for token in missing_rationale_token:
                if token in sent:
                    selected_sentence.append(sent)
                    break

        augmented_data[example_idx] = {'candidates':selected_sentence,'label':label}



    output_dir =  f"{args['save_dir']}_{random_seed}_{args['num_per_class']}_{args['num_per_example']}/missing_rationales_augmented_step1.json"




    with open(output_dir, "w") as file_name:
        json.dump(augmented_data, file_name)








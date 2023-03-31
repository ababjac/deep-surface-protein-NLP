#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
import torch
import pickle

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split


# In[9]:


PATH='/lustre/isaac/proj/UTK0196/deep-surface-protein-data/'


# In[10]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Tells the model we need to use the GPU


# In[11]:


df = pd.read_csv(PATH+'M0059E_training_set.tsv', delimiter=',', header=0)


# In[ ]:


#df = df.sample(115000, random_state=1097253) #random set
#df = df[(df['percent.identity'] >= 74.5) & (df['percent.identity'] < 89.6)] #middle set
#df = df[(df['percent.identity'] >= 89.6) | (df['percent.identity'] < 74.5)] #edge set


# In[4]:


#df.columns


# In[5]:


#df.shape


# In[6]:


#df


# In[16]:


df_train, df_test = train_test_split(df, test_size=0.2, random_state=1234)
df_train, df_val = train_test_split(df_train, test_size=0.1, random_state=1234)


# In[17]:


# Load one split:
#split_no = 1
#inds = pickle.load(open('./splits/splits.pkl', 'rb'))

#for train_inds, test_inds in inds[split_no]:
#    train_set = df.iloc[train_inds,:]
#    test_set = df.iloc[test_inds,:]
    
#    train_seqs_list = train_set['surf.sequence'].tolist() + train_set['deep.sequence'].tolist()
#    train_seqs_labels = np.concatenate([np.zeros(train_set.shape[0]), np.ones(train_set.shape[0])])
surf_series_train = df_train['surf.sequence']
deep_series_train = df_train['deep.sequence']

surf_series_val = df_val['surf.sequence']
deep_series_val = df_val['deep.sequence']

surf_series_test = df_test['surf.sequence']
deep_series_test = df_test['deep.sequence']

classification_df_train = pd.DataFrame({'text' : surf_series_train.append(deep_series_train, ignore_index=True), 'label' : [0]*surf_series_train.size+[1]*deep_series_train.size})
classification_df_val = pd.DataFrame({'text' : surf_series_val.append(deep_series_val, ignore_index=True), 'label' : [0]*surf_series_val.size+[1]*deep_series_val.size})
classification_df_test = pd.DataFrame({'text' : surf_series_test.append(deep_series_test, ignore_index=True), 'label' : [0]*surf_series_test.size+[1]*deep_series_test.size})


# In[18]:


def overlap_sequence(seq, word_length, overlap):
    if overlap >= word_length:
        print('Overlap must be less than word length')
        return
    
    for i in range(0, len(seq)-overlap, word_length-overlap):
        yield seq[i:i+word_length]
        
def get_overlap_array(seq, word_length=4, overlap=1):
    return np.array(list(overlap_sequence(seq, word_length, overlap)))

def get_overlap_string(seq, word_length=4, overlap=1):
    return ' '.join(list(overlap_sequence(seq, word_length, overlap)))

def compute_metrics(epred):
    # Computes metrics from specialized output from huggingface

    preds = np.exp(epred[0]) / np.sum(np.exp(epred[0]), axis = 0)
    labels = epred[1]

    metrics = {}
    metrics['auprc'] = average_precision_score(labels, preds[:,1])
    metrics['auroc'] = roc_auc_score(labels, preds[:,1])

    return metrics


# In[19]:


classification_df_train['text'] = classification_df_train['text'].transform(get_overlap_string)
classification_df_val['text'] = classification_df_val['text'].transform(get_overlap_string)
classification_df_test['text'] = classification_df_test['text'].transform(get_overlap_string)
med_len = int(np.median([len(elem) for elem in classification_df_train['text']]))
#classification_df


# In[20]:


ds_train = Dataset.from_pandas(classification_df_train)
ds_val = Dataset.from_pandas(classification_df_val)
ds_test = Dataset.from_pandas(classification_df_test)


# In[21]:


tokenizer = AutoTokenizer.from_pretrained('tokenizers/AA-overlap-4_1', model_max_length=med_len, padding_side='left', truncation_side='right')


# In[22]:


tokenized_ds_train = ds_train.map(lambda d : tokenizer(d['text'], truncation=True, padding=True), batched=True)
tokenized_ds_val = ds_val.map(lambda d : tokenizer(d['text'], truncation=True, padding=True), batched=True)
tokenized_ds_test = ds_test.map(lambda d : tokenizer(d['text'], truncation=True, padding=True), batched=True)


# In[24]:


#init_splits = tokenized_ds.train_test_split(test_size=0.2)

#tmp = init_splits['train']
#test_ds = init_splits['test']

#splits = tmp.train_test_split(test_size=0.1)
#train_ds = splits['train']
#val_ds = splits['test']


# In[23]:


model = AutoModelForSequenceClassification.from_pretrained('BERT-random/checkpoint-14500', num_labels=2)


# In[25]:


training_args = TrainingArguments(
    output_dir='./custom-model-overlap-4_1',
    learning_rate=2e-4,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds_train,
    eval_dataset=tokenized_ds_val,
    tokenizer=tokenizer,
    #data_collator=data_collator,
)


# In[26]:


trainer.train()


# In[28]:


trainer.evaluate()


# In[29]:


out = trainer.predict(test_dataset=tokenized_ds_test)


# In[30]:


scores = compute_metrics(out)
with open('./results/BERT-custom-4_1-scores.txt','w') as data: 
      data.write(str(scores))
print(scores)


# In[ ]:


#trainer.save_pretrained('./models/initial')


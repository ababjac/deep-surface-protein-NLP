#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
import torch
import pickle
import gc
import os

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

PATH='/lustre/isaac/proj/UTK0196/deep-surface-protein-data/'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


def compute_metrics(epred):
    # Computes metrics from specialized output from huggingface

    preds = np.exp(epred[0]) / np.sum(np.exp(epred[0]), axis = 0)
    labels = epred[1]

    metrics = {}
    metrics['auprc'] = average_precision_score(labels, preds[:,1])
    metrics['auroc'] = roc_auc_score(labels, preds[:,1])

    return metrics





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Tells the model we need to use the GPU

print('Reading data...')
df = pd.read_csv(PATH+'M0059E_training_set.tsv', delimiter=',', header=0)

split_no = 1
inds = pickle.load(open('./splits/splits.pkl', 'rb'))
RUN = 1
#for train_inds, test_inds in inds[split_no]:
train_inds, test_inds = inds[split_no][RUN]
train_set = df.iloc[train_inds,:]
test_set = df.iloc[test_inds,:]

train_set, val_set = train_test_split(train_set, test_size=0.1, random_state=1234)

train_seqs_list = train_set['surf.sequence'].tolist() + train_set['deep.sequence'].tolist()
train_seqs_labels = np.concatenate([np.zeros(train_set.shape[0], dtype=int), np.ones(train_set.shape[0], dtype=int)])

val_seqs_list = val_set['surf.sequence'].tolist() + val_set['deep.sequence'].tolist()
val_seqs_labels = np.concatenate([np.zeros(val_set.shape[0], dtype=int), np.ones(val_set.shape[0], dtype=int)])

test_seqs_list = test_set['surf.sequence'].tolist() + test_set['deep.sequence'].tolist()
test_seqs_labels = np.concatenate([np.zeros(test_set.shape[0], dtype=int), np.ones(test_set.shape[0], dtype=int)])

classification_df_train = pd.DataFrame({'text' : train_seqs_list, 'label' : train_seqs_labels})
classification_df_val = pd.DataFrame({'text' : val_seqs_list, 'label' : val_seqs_labels})
classification_df_test = pd.DataFrame({'text' : test_seqs_list, 'label' : test_seqs_labels})

ds_train = Dataset.from_pandas(classification_df_train)
ds_val = Dataset.from_pandas(classification_df_val)
ds_test = Dataset.from_pandas(classification_df_test)

del classification_df_train
del classification_df_val
del classification_df_test

print('Tokenizing...')
tokenizer = AutoTokenizer.from_pretrained('./checkpoint-final', model_max_length=1024)

tokenized_ds_train = ds_train.map(lambda d : tokenizer(d['text'], truncation=True, padding=True), batched=True)
tokenized_ds_val = ds_val.map(lambda d : tokenizer(d['text'], truncation=True, padding=True), batched=True)
tokenized_ds_test = ds_test.map(lambda d : tokenizer(d['text'], truncation=True, padding=True), batched=True)

del ds_train
del ds_val
del ds_test

gc.collect()
torch.cuda.empty_cache()
    
print('Building Model...')
model = AutoModelForSequenceClassification.from_pretrained('./checkpoint-final')

training_args = TrainingArguments(
    output_dir='./models/drbert-test_{}'.format(RUN),
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=50,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds_train,
    eval_dataset=tokenized_ds_val,
    tokenizer=tokenizer,
)


print('Training...')
trainer.train()
trainer.evaluate()
out = trainer.predict(test_dataset=tokenized_ds_test)

scores = compute_metrics(out)
with open('./results/drbert-test_{}.txt'.format(RUN),'w') as data: 
    data.write(str(scores))

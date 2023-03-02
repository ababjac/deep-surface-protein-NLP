#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import torch

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import roc_auc_score, average_precision_score


# In[2]:


PATH='/lustre/isaac/proj/UTK0196/deep-surface-protein-data/'
HOMEDIR='/lustre/isaac/scratch/ababjac/deep-surface-protein-NLP/'


# In[3]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Tells the model we need to use the GPU

# In[4]:


df = pd.read_csv(PATH+'M0059E_training_set.tsv', delimiter=',', header=0)


# In[ ]:


#df = df.sample(115000, random_state=1097253) #random set
#df = df[(df['percent.identity'] >= 74.5) & (df['percent.identity'] < 89.6)] #middle set
df = df[(df['percent.identity'] >= 89.6) | (df['percent.identity'] < 74.5)] #edge set


# In[4]:


#df.columns


# In[5]:


#df.shape


# In[6]:


#df


# In[5]:


surf_series = df['surf.sequence']
deep_series = df['deep.sequence']

classification_df = pd.DataFrame({'text' : surf_series.append(deep_series, ignore_index=True), 'label' : [0]*surf_series.size+[1]*deep_series.size})
#classification_df


# In[6]:


def overlap_sequence(seq, word_length, overlap):
    if overlap >= word_length:
        print('Overlap must be less than word length')
        return
    
    for i in range(0, len(seq)-overlap, word_length-overlap):
        yield seq[i:i+word_length]
        
def get_overlap_array(seq, word_length=5, overlap=2):
    return np.array(list(overlap_sequence(seq, word_length, overlap)))

def get_overlap_string(seq, word_length=5, overlap=2):
    return ' '.join(list(overlap_sequence(seq, word_length, overlap)))

def compute_metrics(epred):
    # Computes metrics from specialized output from huggingface

    preds = np.exp(epred[0]) / np.sum(np.exp(epred[0]), axis = 0)
    labels = epred[1]

    metrics = {}
    metrics['auprc'] = average_precision_score(labels, preds[:,1])
    metrics['auroc'] = roc_auc_score(labels, preds[:,1])

    return metrics


# In[7]:


classification_df['text'] = classification_df['text'].transform(get_overlap_string)
#classification_df


# In[8]:


ds = Dataset.from_pandas(classification_df)


# In[9]:


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


# In[10]:


tokenized_ds = ds.map(lambda d : tokenizer(d['text'], truncation=True), batched=True)


# In[11]:


init_splits = tokenized_ds.train_test_split(test_size=0.2)

tmp = init_splits['train']
test_ds = init_splits['test']

splits = tmp.train_test_split(test_size=0.1)
train_ds = splits['train']
val_ds = splits['test']


# In[12]:


model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)


# In[15]:


training_args = TrainingArguments(
    output_dir=HOMEDIR+'/BERT-edges',
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    #data_collator=data_collator,
)


# In[16]:


trainer.train()


# In[17]:


trainer.evaluate()


# In[18]:


out = trainer.predict(test_dataset=test_ds)


# In[19]:


scores = compute_metrics(out)
with open(HOMEDIR+'BERT-edges-test.txt','w') as data: 
      data.write(str(scores))


# In[ ]:


#trainer.save_pretrained('./models/initial')


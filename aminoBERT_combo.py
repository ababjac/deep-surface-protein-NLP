import pandas as pd
import numpy as np
import torch
import pickle

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

PATH='/lustre/isaac/proj/UTK0196/deep-surface-protein-data/'



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




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Tells the model we need to use the GPU

print('Reading data...')
df = pd.read_csv(PATH+'M0059E_training_set.tsv', delimiter=',', header=0)

# Load one split:
split_no = 1
inds = pickle.load(open('./splits/splits.pkl', 'rb'))
RUN = 0
for train_inds, test_inds in inds[split_no]:
    train_set = df.iloc[train_inds,:]
    test_set = df.iloc[test_inds,:]
    
    train_set, val_set = train_test_split(train_set, test_size=0.1, random_state=1234)
    
    train_seqs_list = train_set['surf.sequence'].tolist() + train_set['deep.sequence'].tolist()
    train_seqs_labels = np.concatenate([np.zeros(train_set.shape[0]), np.ones(train_set.shape[0])])
    
    val_seqs_list = val_set['surf.sequence'].tolist() + val_set['deep.sequence'].tolist()
    val_seqs_labels = np.concatenate([np.zeros(val_set.shape[0]), np.ones(val_set.shape[0])])
    
    test_seqs_list = test_set['surf.sequence'].tolist() + test_set['deep.sequence'].tolist()
    test_seqs_labels = np.concatenate([np.zeros(test_set.shape[0]), np.ones(test_set.shape[0])])
    
    classification_df_train = pd.DataFrame({'text' : train_seqs_list, 'label' : train_seqs_labels})
    classification_df_val = pd.DataFrame({'text' : val_seqs_list, 'label' : val_seqs_labels})
    classification_df_test = pd.DataFrame({'text' : test_seqs_list, 'label' : test_seqs_labels})

    #classification_df_train['text'] = classification_df_train['text'].transform(get_overlap_string)
    #classification_df_val['text'] = classification_df_val['text'].transform(get_overlap_string)
    #classification_df_test['text'] = classification_df_test['text'].transform(get_overlap_string)
    #med_len = int(np.median([len(elem) for elem in classification_df_train['text']]))

    ds_train = Dataset.from_pandas(classification_df_train)
    ds_val = Dataset.from_pandas(classification_df_val)
    ds_test = Dataset.from_pandas(classification_df_test)


    print('Tokenizing...')
    tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D') #ESM tokenizer

    tokenized_ds_train = ds_train.map(lambda d : tokenizer(d['text'], truncation=True, padding=True), batched=True)
    tokenized_ds_val = ds_val.map(lambda d : tokenizer(d['text'], truncation=True, padding=True), batched=True)
    tokenized_ds_test = ds_test.map(lambda d : tokenizer(d['text'], truncation=True, padding=True), batched=True)

    print('Building Model...')
    model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2) #standard pretrained bert

    training_args = TrainingArguments(
        output_dir='./models/custom-model-combo_{}'.format(RUN),
        learning_rate=2e-4,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
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
    with open('./results/BERT-custom-combo-scores_{}.txt'.format(RUN),'w') as data: 
        data.write(str(scores))

    RUN += 1

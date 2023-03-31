import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

dpath = '/lustre/isaac/proj/UTK0196/deep-surface-protein-data/M0059E_training_set.tsv'
df = pd.read_csv(dpath)

def main():
    # Make 5 10-fold cross validations
    print(df.shape)

    inds = np.arange(df.shape[0])

    ind_lists = []

    for _ in range(5):
        all_inds = list(KFold(n_splits = 10, shuffle = True, random_state = 1234).split(inds))
        ind_lists.append(all_inds)

    pickle.dump(ind_lists, open('./splits/splits.pkl', 'wb'))

def test_load():
    inds = pickle.load(open('./splits/splits.pkl', 'rb'))
    print(len(inds))

def example_load_data():
    dpath = '/lustre/isaac/proj/UTK0196/deep-surface-protein-data/M0059E_training_set.tsv'
    df = pd.read_csv(dpath)

    # Load one split:
    split_no = 1
    inds = pickle.load(open('./splits/splits.pkl', 'rb'))

    for train_inds, test_inds in inds[split_no]:
        train_set = df.iloc[train_inds,:]
        test_set = df.iloc[test_inds,:]
        train_seqs_list = train_set['surf.sequence'].tolist() + train_set['deep.sequence'].tolist()
        train_seqs_labels = np.concatenate([np.zeros(train_set.shape[0]), np.ones(train_set.shape[0])])
        print('list', len(train_seqs_list))
        print('labels', train_seqs_labels.shape)
        


if __name__ == '__main__':
    #main()
    #test_load()
    example_load_data()

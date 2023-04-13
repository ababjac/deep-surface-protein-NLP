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
        all_inds = list(KFold(n_splits = 5, shuffle = True, random_state = 1234).split(inds))
        ind_lists.append(all_inds)

    pickle.dump(ind_lists, open('./splits/splits.pkl', 'wb'))

def test_load():
    inds = pickle.load(open('./splits/splits.pkl', 'rb'))
    print('len', len(inds))

def example_load_data():
    dpath = '/lustre/isaac/proj/UTK0196/deep-surface-protein-data/M0059E_training_set.tsv'
    df = pd.read_csv(dpath)

    # Load one split:
    split_no = 0
    inds = pickle.load(open('splits.pkl', 'rb'))

    for train_inds, test_inds in inds[split_no]:
        train_set = df.iloc[train_inds,:]
        test_set = df.iloc[test_inds,:]
        train_seqs_list = train_set['surf.sequence'].tolist() + train_set['deep.sequence'].tolist()
        train_seqs_labels = np.concatenate([np.zeros(train_set.shape[0]), np.ones(train_set.shape[0])])
        print('train list', len(train_seqs_list))
        print('train labels', train_seqs_labels.shape)

        test_seqs_list = test_set['surf.sequence'].tolist() + test_set['deep.sequence'].tolist()
        test_seqs_labels = np.concatenate([np.zeros(test_set.shape[0]), np.ones(test_set.shape[0])])
        print('test list', len(test_seqs_list))
        print('test labels', test_seqs_labels.shape)
        print('-' * 30)
        


if __name__ == '__main__':
    main()
    #test_load()
    #example_load_data()

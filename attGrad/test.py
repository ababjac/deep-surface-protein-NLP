import numpy as np

data = np.load('attr_dict_esm_sp=2.npy', allow_pickle=True)

data_dict = data.item()

for d in data_dict:
    key = d
    val = data_dict[key]
    
    means = np.mean(val[0], axis=1)[1:-1]

    #print('sequence length: ', len(key))
    #print('attributions length: ', len(means))
    print(means)

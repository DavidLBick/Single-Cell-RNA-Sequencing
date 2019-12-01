import torch
import torch.utils.data as Data
import os
import pdb
import pandas as pd
import numpy as np
import config
import matplotlib.pyplot as plt

print("Begin dataloading.py")

class GeneDataset(Data.Dataset):
    def __init__(self, features, labels,
                 label_idx_to_str, label_str_to_idx):
        super(GeneDataset, self).__init__()
        self.features = features
        self.labels = labels
        self.label_idx_to_str = label_idx_to_str
        self.label_str_to_idx = label_str_to_idx

    def __getitem__(self, index):
        # change behavior for np.array rather than pd DataFrame
        if isinstance(arr, np.ndarray):
            feature = self.features[index,:]
        else:
            feature = self.features.iloc[index,:]
            
        label_str = self.labels[index]
        label_idx = self.label_str_to_idx[label_str]

        return np.array(feature), label_idx

    def __len__(self):
        return self.features.shape[0]


################
### DATASETS ###
################
def get_dataset(is_train):
    # we have to get the labels from the HDFStore as before
    # but now we need to get the features from a numpy array
    # that Jakob is saving which is the result of the gOMP
    if is_train:
        store = pd.HDFStore(config.TRAIN_DATA_PATH)
    else:
        store = pd.HDFStore(config.TEST_DATA_PATH)

    #features = store['rpkm'] # (21389, 20499)
    labels = store['labels'] # (21389,)

    features = np.load(config.TRAIN_DATA_NP_ARRAY)

    return GeneDataset(features, labels,
                       label_idx_to_str, label_str_to_idx)

# the labels mapping from index to string is based off all the data
store = pd.HDFStore(config.DATA_PATH + "all_data.h5")
labels = store['labels']
uniques = np.sort(np.unique(labels))
label_idx_to_str = dict()
label_str_to_idx = dict()
for idx, str in enumerate(uniques):
    label_idx_to_str[idx] = str
    label_str_to_idx[str] = idx

train_dataset = get_dataset(is_train=True)
train_loader = Data.DataLoader(train_dataset,
                               batch_size = config.BATCH_SIZE,
                               shuffle = True,
                               drop_last = True)

test_dataset = get_dataset(is_train=False)
test_loader = Data.DataLoader(test_dataset,
                             batch_size = config.BATCH_SIZE,
                             shuffle = True,
                             drop_last = True)

print("dataloading.py done!")

import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np

def normalize_data(train_data, valid_data, test_data):
    """Normalize features wrt. mean and std of training data."""
    _, seq_len, input_dim = train_data.shape
    train_data_reshape = np.reshape(train_data, (-1, input_dim))
    test_data_reshape = np.reshape(test_data, (-1, input_dim))
    features_mean = np.mean(train_data_reshape, axis=0)
    features_std = np.std(train_data_reshape, axis=0)
    train_data_reshape = (train_data_reshape - features_mean) / features_std
    test_data_reshape = (test_data_reshape - features_mean) / features_std
    train_data = np.reshape(train_data_reshape, (-1, seq_len, input_dim))
    test_data = np.reshape(test_data_reshape, (-1, seq_len, input_dim))
    if valid_data is not None:
        valid_data_reshape = np.reshape(valid_data, (-1, input_dim))
        valid_data_reshape = (valid_data_reshape - features_mean) / features_std
        valid_data = np.reshape(valid_data_reshape, (-1, seq_len, input_dim))
    return train_data, valid_data, test_data

class MyDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).long()
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        return len(self.data)
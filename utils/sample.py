from utils.conf import Configuration
import numpy as np
import torch
from torch.utils.data import Dataset


class TSData(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def getSamples(conf:Configuration):
    dim_series = conf.getEntry('dim_series')
    train_samples, val_samples = sample(conf)
    train_samples = train_samples.view([-1, 1, dim_series])
    val_samples = val_samples.view([-1, 1, dim_series])
    return train_samples, val_samples


def normalize_sequence(sequence):
    mean = np.mean(sequence)
    std = np.std(sequence)
    if std == 0:
        return np.zeros_like(sequence)
    return (sequence - mean) / std


def sample(conf:Configuration):
    data_path = conf.getEntry('data_path')
    train_path = conf.getEntry('train_path')
    train_indices_path = conf.getEntry('train_indices_path')
    val_indices_path = conf.getEntry('val_indices_path')
    val_path = conf.getEntry('val_path')
    
    dim_series = conf.getEntry('dim_series')
    train_size = conf.getEntry('train_size')    
    val_size = conf.getEntry('val_size')
    data_size = conf.getEntry('data_size')
    
    train_samples_indices = np.random.randint(0, data_size, size = train_size, dtype=np.int64)
    val_samples_indices = np.random.randint(0, data_size, size = val_size, dtype=np.int64)
    
    train_samples_indices.tofile(train_indices_path)
    val_samples_indices.tofile(val_indices_path)
    
    loaded = []
    for index in train_samples_indices:
        sequence = np.fromfile(data_path, dtype=np.float32, count=dim_series, offset=4 * dim_series * index)
        if not np.isnan(np.sum(sequence)):
            sequence = normalize_sequence(sequence)
            loaded.append(sequence)
            
    train_samples = np.asarray(loaded, dtype=np.float32)
    train_samples.tofile(train_path)
    train_samples = torch.from_numpy(train_samples)
    
    loaded = []
    for index in val_samples_indices:
        sequence = np.fromfile(data_path, dtype=np.float32, count=dim_series, offset=4 * dim_series * index)
        if not np.isnan(np.sum(sequence)):
            sequence = normalize_sequence(sequence)
            loaded.append(sequence)
    
    val_samples = np.asarray(loaded, dtype=np.float32)
    val_samples.tofile(val_path)
    val_samples = torch.from_numpy(val_samples)
    
    return train_samples, val_samples
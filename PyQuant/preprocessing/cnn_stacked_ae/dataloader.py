import os
import modin.pandas as pd
import numpy as np
import torch
import PyQuant.path as path
from numba import jit
from torch.utils.data import DataLoader, Dataset

@jit
def window_function(x:np.ndarray, size:int):
    ret = np.zeros(shape=(x.shape[0]-size+1, size))
    for i in range(len(x) - size + 1):
        ret[i] = x[i:i+size]
    return ret

def load_price_data(direc):
    fnames = os.listdir(direc)
    file_n = len(fnames)
    ret = [0]*file_n*3
    idx = 0
    for i in range(file_n):
        df = pd.read_csv(direc + fnames[i])
        ret[idx] = df["시가"].to_numpy()
        ret[idx+1] = df["고가"].to_numpy()
        ret[idx+2] = df["저가"].to_numpy()
        idx += 3
    return ret

def load_data(device=torch.device("cpu"), window_size=100):
    price_data = []
    for price_path in path.price_data_paths:
        price_data.extend(load_price_data(price_path))
    price_data = list(filter(lambda x: x.shape[0] >= window_size, price_data))
    price_data = list(map(lambda x: window_function(x, window_size), price_data))
    price_data = np.vstack(price_data)
    price_data = torch.tensor(price_data, dtype=torch.float32).to(device)
    return price_data.unsqueeze(1)

class StockCAEDataSet(Dataset):
    def __init__(self, data, device=None):
        self.device = device
        if not device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert data.shape.__len__() == 3, "3dim tensor required"
        self.data = data
        self.data = torch.nn.functional.normalize(self.data, 2)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return self.data[item], self.data[item]

def get_dataloader(device=None, batch_size=64, train_size=0.8, window_size=100):
    if not device: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data(device, window_size=window_size)
    train_idx = int(len(data) * train_size)
    train_set = StockCAEDataSet(data[:train_idx], device=device)
    test_set = StockCAEDataSet(data[train_idx:], device=device)
    return DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True), \
           DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

if __name__ == '__main__':
    print(load_data())




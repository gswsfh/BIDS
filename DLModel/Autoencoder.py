
import time

import numpy as np
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random

from tqdm import tqdm

from datasetProcess import readDataFromJson, readDataFromNp


seed=42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
"""-----------------------------------------------------------------------------------"""
def get_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device

class SimpleAE(nn.Module):
    def __init__(self,input_size,hidden_sizes):
        super().__init__()
        self.linears=list()
        self.embeddings=nn.Linear(input_size,hidden_sizes[0])
        self.hidden=nn.Sequential()
        for i in range(len(hidden_sizes)-1):
            self.hidden.append(nn.Linear(hidden_sizes[i],hidden_sizes[i+1]))
            self.hidden.append(nn.ReLU())
        self.relu=nn.ReLU()
        self.fn=nn.Linear(hidden_sizes[-1],input_size)

    def forward(self, x):
        x=self.relu(self.embeddings(x))
        x=self.hidden(x)
        x=self.fn(x)
        return x

def normalize_cols(X, method='zscore'):
    if method == 'zscore':
        return (X - X.mean(axis=0)) / X.std(axis=0)
    elif method == 'minmax':
        return (X - X.min(axis=0)) / (np.ptp(X, axis=0) + 1e-12)
    else:
        raise ValueError('method must be zscore or minmax')

class CICIDS2017Dataset(Dataset):
    def __init__(self):
        time1=time.time()
        self.data=readDataFromNp('filepath')
        time2=time.time()
        self.data[self.data[:, -1] > 1, -1] = 1
        self.data[:,:-1]=normalize_cols(self.data[:,:-1],method = 'minmax')


    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def sample_benign(X, k, seed=42):

    np.random.seed(seed)

    mask0 = X[:, -1] == 0
    idx0  = np.where(mask0)[0]
    idx_rest = np.where(~mask0)[0]

    if k > len(idx0):
        raise ValueError(f'Only {len(idx0)} label==0, can not draw {k}.')

    pick0 = np.random.choice(idx0, k, replace=False)
    X0_sample = X[pick0]


    remain_idx = np.setdiff1d(np.arange(len(X)), pick0)
    X_rest = X[remain_idx]

    return X0_sample, X_rest

if __name__ == '__main__':
    train_batch_size = 20480
    test_batch_size = train_batch_size
    learning_rate = 0.001
    epochs = 30
    benign_select_num = 100 * 10000
    step = 1  # 0 train,1 detect,2 generate
    delta = 0.2

    device = get_device()
    dataset = readDataFromNp("filepath")
    print("Data length:", len(dataset))
    print("Benign length:", len(dataset[dataset[:, -1] == 0]))
    print("Malicious length:", len(dataset[dataset[:, -1] != 0]))
    dataset[dataset[:, -1] > 1, -1] = 1
    dataset[:, :-1] = np.where(dataset[:, :-1] < -1, -1, dataset[:, :-1])
    dataset[:, :-1] = normalize_cols(dataset[:, :-1], method='minmax')
    feaselect_colIndex = [0, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 34, 35, 38, 39, 40, 52, 55, 66, 67, -1]
    dataset = dataset[:, feaselect_colIndex]
    trainDataset, testDataset = sample_benign(dataset, benign_select_num)
    train_loader = DataLoader(trainDataset, batch_size=train_batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(testDataset, batch_size=test_batch_size, shuffle=False)
    oneSample = dataset[0]
    input_size = len(oneSample) - 1
    output_size = input_size
    hidden_size = [16,8,16]
    model = SimpleAE(input_size, hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if step == 0:
        loss_MSE = torch.nn.MSELoss(reduction='mean')
        train_loss = list()
        train_epochs_loss = list()
        for epoch in range(epochs):
            model.train()
            train_epoch_loss = list()
            for idx, data in tqdm(enumerate(train_loader, 0)):
                data_x = data[:, :-1].to(torch.float32).to(device)
                predict = model(data_x)
                optimizer.zero_grad()
                loss_mse = loss_MSE(predict, data_x)
                loss = loss_mse
                loss.backward()
                optimizer.step()
                train_epoch_loss.append(loss.item())
                train_loss.append(loss.item())
            print("epoch:", epoch, "loss:", np.average(train_epoch_loss))
            train_epochs_loss.append(np.average(train_epoch_loss))
            torch.save(model.state_dict(), "network.pth")
    elif step == 1:
        criterion = nn.CrossEntropyLoss(reduction='none')
        model.load_state_dict(torch.load("network.pth"))
        labels = list()
        predicts = list()
        benign_loss = list()
        malic_loss = list()
        for idx, data in tqdm(enumerate(test_loader, 0)):
            data_x = data[:, :-1].to(torch.float32).to(device)
            idx_labels = data[:, -1].to(torch.int32).tolist()
            labels.extend(idx_labels)
            predict= model(data_x)
            loss_mse = ((predict - data_x) ** 2).sum(dim=1)
            loss = loss_mse
            predicts.extend((loss > delta).int().tolist())
            benign_loss.extend(loss[data[:, -1] == 0].tolist())
            malic_loss.extend(loss[data[:, -1] == 1].tolist())

        acc = accuracy_score(labels, predicts)
        pre = precision_score(labels, predicts)
        rec = recall_score(labels, predicts)
        f1 = f1_score(labels, predicts)

        print("acc：", acc)
        print("pre：", pre)
        print("rec：", rec)
        print("f1：", f1)
    elif step == 2:
        model.load_state_dict(torch.load("network.pth"))
        normaldata = readDataFromJson("normalParams.json")
        normaldata["min"] = np.array(normaldata["min"])
        normaldata["ptp"] = np.array(normaldata["ptp"])
        feaselect_colIndex = feaselect_colIndex[:-1]
        normaldata["min"] = normaldata["min"][feaselect_colIndex]
        normaldata["ptp"] = normaldata["ptp"][feaselect_colIndex]
        feanames = [item for i, item in enumerate(cicids2017fea) if i in feaselect_colIndex]
        z = torch.randn(10, laten_size).to(device)
        output = model.decoder(z).cpu().detach().numpy()
        outputRaw = inverse_normalize_cols(output, normaldata).tolist()

        for i, item in enumerate(outputRaw):
            print("---------------------------")
            for fea, val in zip(feanames, item):
                print(f"{fea}:{val}")

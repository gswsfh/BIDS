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

class Encoder(nn.Module):
    def __init__(self,input_size,hidden_size,latent_size):
        super(Encoder,self).__init__()
        self.linear = torch.nn.Linear(input_size, hidden_size)
        self.mu = torch.nn.Linear(hidden_size, latent_size)
        self.sigma = torch.nn.Linear(hidden_size, latent_size)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        x=self.relu(self.linear(x))
        mu=self.mu(x)
        sigma=self.sigma(x)
        return mu, sigma

class Decoder(nn.Module):
    def __init__(self,latent_size,hidden_size,output_size):
        super(Decoder,self).__init__()
        self.linear1=torch.nn.Linear(latent_size,hidden_size)
        self.linear2=torch.nn.Linear(hidden_size,output_size)
        self.relu = torch.nn.ReLU()
    def forward(self,x):
        x=self.relu(self.linear1(x))
        x=self.relu(self.linear2(x))
        return x

class SimpleVAE(nn.Module):
    def __init__(self, input_size, output_size, latent_size, hidden_size):
        super(SimpleVAE,self).__init__()
        self.encoder = Encoder(input_size, hidden_size, latent_size)
        self.decoder = Decoder(latent_size, hidden_size, output_size)

    def forward(self, x):
        mu, sigma = self.encoder(x)
        eps = torch.randn_like(sigma)
        z = mu + eps * torch.exp(0.5 * sigma)
        re_x = self.decoder(z)
        return re_x, mu, sigma

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
        self.data=readDataFromNp('/mnt/storage/fuhao/codes/SSH/WWW26/datasets/CICIDS2017/processed/CICDIS2017.np.npy')
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

    train_batch_size=20480
    test_batch_size=train_batch_size
    learning_rate=0.001
    epochs=50
    benign_select_num=2000000
    step = 1  # 0 train,1detection
    delta=1050

    device = get_device()
    dataset = readDataFromNp("filepath")
    dataset[dataset[:, -1] > 1, -1] = 1
    dataset[:, :-1] = normalize_cols(dataset[:, :-1], method='minmax')
    trainDataset,testDataset=sample_benign(dataset,benign_select_num)
    train_loader = DataLoader(trainDataset, batch_size=train_batch_size, shuffle=True,drop_last=False)
    test_loader = DataLoader(testDataset, batch_size=test_batch_size, shuffle=False)
    oneSample=dataset[0]
    input_size=len(oneSample)-1
    output_size=input_size
    laten_size = 16
    hidden_size=64
    model=SimpleVAE(input_size,output_size,laten_size,hidden_size).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
    loss_KLD = lambda mu, sigma: -0.5 * torch.sum(1 + torch.log(sigma ** 2) - mu.pow(2) - sigma ** 2)

    if step==0:
        loss_MSE = torch.nn.MSELoss(reduction = 'sum')

        train_loss=list()
        train_epochs_loss=list()
        for epoch in range(epochs):
            model.train()
            train_epoch_loss=list()
            for idx,data in tqdm(enumerate(train_loader,0)):
                data_x=data[:,:-1].to(torch.float32).to(device)
                predict,mu,sigma=model(data_x)
                optimizer.zero_grad()
                loss_mse=loss_MSE(predict,data_x)
                loss_kld=loss_KLD(mu,sigma)
                loss=loss_kld+loss_mse
                loss.backward()
                optimizer.step()
                train_epoch_loss.append(loss.item())
                train_loss.append(loss.item())
            print("epoch:",epoch,"loss:",np.average(train_epoch_loss))
            train_epochs_loss.append(np.average(train_epoch_loss))
            torch.save(model.state_dict(),"network.pth")
    elif step==1:
        criterion = nn.CrossEntropyLoss(reduction='none')
        model.load_state_dict(torch.load("network.pth"))
        labels=list()
        predicts=list()
        for idx, data in tqdm(enumerate(test_loader, 0)):
            data_x = data[:, :-1].to(torch.float32).to(device)
            labels.extend(data[:, -1].to(torch.int32).tolist())
            predict, mu, sigma = model(data_x)
            loss_mse = ((predict - data_x) ** 2).sum(dim=1)
            loss_kld = loss_KLD(mu, sigma)
            loss = loss_kld + loss_mse
            predicts.extend((loss > delta).int().tolist())
        acc=accuracy_score(labels,predicts)
        pre=precision_score(labels,predicts)
        rec=recall_score(labels,predicts)
        f1=f1_score(labels,predicts)

        print("acc：",acc)
        print("pre：",pre)
        print("rec：",rec)
        print("f1：",f1)

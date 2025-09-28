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
from datasetProcess import readDataFromJson, readDataFromNp, saveDataToJson
import torch.nn.functional as F


seed=42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def get_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device

class Encoder(nn.Module):
    def __init__(self,input_size,hidden_size,latent_size,n_att=2):
        super(Encoder,self).__init__()
        self.linear = torch.nn.Linear(input_size, hidden_size)
        self.mu = torch.nn.Linear(hidden_size, latent_size)
        self.sigma = torch.nn.Linear(hidden_size, latent_size)
        self.relu = torch.nn.ReLU()
        self.att_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_att):
            self.att_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.Tanh()
                )
            )
            self.norms.append(nn.LayerNorm(hidden_size))
    def forward(self, x):
        x=self.relu(self.linear(x))
        for att, norm in zip(self.att_layers, self.norms):
            u = att(x)
            scores = torch.bmm(u.unsqueeze(2), u.unsqueeze(1))
            attn = F.softmax(scores, dim=-1)
            out = torch.bmm(attn, x.unsqueeze(2)).squeeze(2)
            x = norm(x + out)

        mu=self.mu(x)
        sigma=self.sigma(x)
        return mu, sigma

class Decoder(nn.Module):
    def __init__(self,latent_size,hidden_size,output_size,n_att=2):
        super(Decoder,self).__init__()
        self.linear1=torch.nn.Linear(latent_size,hidden_size)
        self.linear2=torch.nn.Linear(hidden_size,output_size)
        self.relu = torch.nn.ReLU()

        self.att_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_att):
            self.att_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.Tanh()
                )
            )
            self.norms.append(nn.LayerNorm(hidden_size))
    def forward(self,x):
        x=self.relu(self.linear1(x))
        for att, norm in zip(self.att_layers, self.norms):
            u = att(x)
            scores = torch.bmm(u.unsqueeze(2), u.unsqueeze(1))
            attn = F.softmax(scores, dim=-1)
            out = torch.bmm(attn, x.unsqueeze(2)).squeeze(2)
            x = norm(x + out)
        x=self.linear2(x)
        return x

class FeatureVAE(nn.Module):
    def __init__(self, input_size, output_size, latent_size, hidden_size,n_att=2):
        super(FeatureVAE,self).__init__()
        self.encoder = Encoder(input_size, hidden_size, latent_size,n_att)
        self.decoder = Decoder(latent_size, hidden_size, output_size,n_att)

    def forward(self, x):
        mu, sigma = self.encoder(x)
        eps = torch.randn_like(sigma)
        z = mu + eps * torch.exp(0.5 * sigma)
        re_x = self.decoder(z)
        return re_x, mu, sigma

def normalize_cols(X, method='zscore', save_params=True):
    params = {'method': method}

    if method == 'zscore':
        mean = X.mean(axis=0)
        std  = X.std(axis=0)
        params.update({'mean': mean.tolist(), 'std': std.tolist()})
        X_norm = (X - mean) / (std + 1e-12)

    elif method == 'minmax':
        min_ = X.min(axis=0)
        ptp_ = np.ptp(X, axis=0)
        params.update({'min': min_.tolist(), 'ptp': ptp_.tolist()})
        X_norm = (X - min_) / (ptp_ + 1e-12)

    else:
        raise ValueError("method must be 'zscore' or 'minmax'")

    if save_params:
        saveDataToJson( params,"normalParams.json")

    return X_norm

def inverse_normalize_cols(X_norm, params):
    method = params['method']

    if method == 'zscore':
        mean = np.array(params['mean'])
        std  = np.array(params['std'])
        return X_norm * std + mean

    elif method == 'minmax':
        min_ = np.array(params['min'])
        ptp_ = np.array(params['ptp'])
        return X_norm * ptp_ + min_

    else:
        raise ValueError("params['method'] must be 'zscore' or 'minmax'")

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

cicids2017fea=[ "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets","Total Length of Fwd Packets",
                "Total Length of Bwd Packets", "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
                "Fwd Packet Length Std","Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
                "Bwd Packet Length Std","Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std",
                "Flow IAT Max", "Flow IAT Min","Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max",
                "Fwd IAT Min","Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
                "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length",
                "Bwd Header Length","Fwd Packets/s", "Bwd Packets/s", "Min Packet Length", "Max Packet Length",
                "Packet Length Mean", "Packet Length Std", "Packet Length Variance",'FIN Flag Count',
                "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
                "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size",
                "Avg Bwd Segment Size", "Fwd Header Length","Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk",
                "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk","Bwd Avg Bulk Rate","Subflow Fwd Packets",
                "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes","Init_Win_bytes_forward", "Init_Win_bytes_backward",
                "act_data_pkt_fwd", "min_seg_size_forward","Active Mean", "Active Std", "Active Max", "Active Min","Idle Mean", "Idle Std", "Idle Max", "Idle Min"]

if __name__ == '__main__':

    train_batch_size = 20480
    test_batch_size = train_batch_size
    learning_rate = 0.001
    epochs = 80
    benign_select_num = 200*10000
    step = 1  # 0 train,1 detect,2 generate
    delta = 0.2

    device = get_device()
    dataset = readDataFromNp("filepath")
    print("Data length：",len(dataset))
    print("benign length：",len(dataset[dataset[:,-1]==0]))
    print("malicious length：", len(dataset[dataset[:,-1] != 0]))
    dataset[dataset[:, -1] > 1, -1] = 1
    dataset[:, :-1] = np.where(dataset[:, :-1] < -1, -1, dataset[:, :-1])
    dataset[:, :-1] = normalize_cols(dataset[:, :-1], method='minmax')
    feaselect_colIndex = [0,2,3,4,5,6,7,8,10,11,12,34,35,38,39,40,52,55,66,67,-1]
    dataset = dataset[:, feaselect_colIndex]
    trainDataset, testDataset = sample_benign(dataset, benign_select_num)
    train_loader = DataLoader(trainDataset, batch_size=train_batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(testDataset, batch_size=test_batch_size, shuffle=False)
    oneSample = dataset[0]
    input_size = len(oneSample) - 1
    output_size = input_size
    hidden_size = int(input_size/2)
    laten_size = int(hidden_size/2)
    att_layer_size = 2
    model = FeatureVAE(input_size, output_size, laten_size, hidden_size,att_layer_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    kld_loss = lambda mu, sigma: torch.mean(-0.5 * torch.sum(1 + sigma - mu ** 2 - sigma.exp(), dim=1), dim=0)


    if step == 0:
        loss_MSE = torch.nn.MSELoss(reduction='mean')
        train_loss = list()
        train_epochs_loss = list()
        for epoch in range(epochs):
            model.train()
            train_epoch_loss = list()
            for idx, data in tqdm(enumerate(train_loader, 0)):
                data_x = data[:, :-1].to(torch.float32).to(device)
                predict, mu, sigma = model(data_x)
                optimizer.zero_grad()
                loss_mse = loss_MSE(predict, data_x)
                loss_kld = kld_loss(mu, sigma)
                loss = loss_kld + loss_mse
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
        benign_loss=list()
        malic_loss=list()
        for idx, data in tqdm(enumerate(test_loader, 0)):
            data_x = data[:, :-1].to(torch.float32).to(device)
            idx_labels=data[:, -1].to(torch.int32).tolist()
            labels.extend(idx_labels)
            predict, mu, sigma = model(data_x)
            loss_mse = ((predict - data_x) ** 2).sum(dim=1)
            loss_kld = kld_loss(mu, sigma)
            loss = loss_kld + loss_mse
            predicts.extend((loss > delta).int().tolist())
            benign_loss.extend(loss[data[:, -1]==0].tolist())
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
        normaldata=readDataFromJson("normalParams.json")
        normaldata["min"]=np.array(normaldata["min"])
        normaldata["ptp"]=np.array(normaldata["ptp"])
        feaselect_colIndex=feaselect_colIndex[:-1]
        normaldata["min"]=normaldata["min"][feaselect_colIndex]
        normaldata["ptp"]=normaldata["ptp"][feaselect_colIndex]
        feanames=[item for i,item in enumerate(cicids2017fea) if i in feaselect_colIndex]
        z=torch.randn(10, laten_size).to(device)
        output=model.decoder(z).cpu().detach().numpy()
        outputRaw=inverse_normalize_cols(output,normaldata).tolist()

        for i,item in enumerate(outputRaw):
            print("---------------------------")
            print(f"Sample：{i}")
            for fea,val in zip(feanames,item):
                print(f"{fea}:{val}")



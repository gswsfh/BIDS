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

class SimpleCNN(nn.Module):
    def __init__(self,input_size,num_classes):
        super().__init__()
        self.linear_first=nn.Linear(input_size, input_size)
        self.linear_middle = nn.Linear(input_size, input_size)
        self.linear_middle2 = nn.Linear(input_size, input_size)
        self.linear_last=nn.Linear(input_size, num_classes)
        self.relu=nn.ReLU()

    def forward(self, x):
        x =self.relu(self.linear_first(x))
        x = self.relu(self.linear_middle(x))
        x = self.relu(self.linear_middle2(x))
        x =self.linear_last(x)
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

if __name__ == '__main__':
    train_batch_size=20480
    test_batch_size=train_batch_size
    learning_rate=0.001
    epochs=30
    step = 1  # 0 train, 1 predict

    device = get_device()
    dataset=CICIDS2017Dataset()
    test_rate=0.2
    train_len=int(len(dataset)*(1-test_rate))
    test_len=len(dataset)-train_len
    train_dataset, test_dataset = random_split(
        dataset=dataset,
        lengths=[train_len, test_len]
    )
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True,drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    oneSample=train_dataset[0]
    fea_num=len(oneSample)-1
    label_mnum=2

    model=SimpleCNN(fea_num,label_mnum).to(device)
    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)


    if step==0:
        train_loss=list()
        train_epochs_loss=list()
        for epoch in range(epochs):
            model.train()
            train_epoch_loss=list()
            for idx,data in tqdm(enumerate(train_loader,0)):
                data_x=data[:,:-1].to(torch.float32).to(device)
                data_y=data[:,-1].to(torch.long).to(device)
                predict=model(data_x)
                optimizer.zero_grad()
                loss=criterion(predict,data_y)
                loss.backward()
                optimizer.step()
                train_epoch_loss.append(loss.item())
                train_loss.append(loss.item())
            print("epoch:",epoch,"loss:",np.average(train_epoch_loss))
            train_epochs_loss.append(np.average(train_epoch_loss))
            torch.save(model.state_dict(),"simpleCNN.pth")
    elif step==1:
        model.load_state_dict(torch.load("simpleCNN.pth"))
        labels=list()
        predicts=list()
        for idx, data in tqdm(enumerate(test_loader, 0)):
            data_x = data[:, :-1].to(torch.float32).to(device)
            labels.extend(data[:, -1].to(torch.int32).tolist())
            predict = model(data_x)
            _,predict = predict.max(dim=1)
            predicts.extend(predict.cpu().detach().numpy().tolist())
        acc=accuracy_score(labels,predicts)
        pre=precision_score(labels,predicts)
        rec=recall_score(labels,predicts)
        f1=f1_score(labels,predicts)

        print("acc：",acc)
        print("pre：",pre)
        print("rec：",rec)
        print("f1：",f1)

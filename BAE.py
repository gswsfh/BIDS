
import gc
import os
import time
import numpy as np
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pandas.core.indexers import length_of_indexer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from datasetProcess import readDataFromJson, readDataFromNp, saveDataToJson, saveDataToNp
import torch.nn.functional as F


# seed=11542 # cicids2017
seed=42 # cicids2018

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
        self.leftDelta = torch.nn.Linear(hidden_size, latent_size)
        self.rightDelta = torch.nn.Linear(hidden_size, latent_size)
        self.fea_select=torch.nn.Linear(hidden_size, latent_size)
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

        deltaLeft=self.leftDelta(x)
        deltaRight=self.rightDelta(x)

        return deltaLeft, deltaRight

class Decoder(nn.Module):
    def __init__(self,latent_size,hidden_size,output_size,n_att=2):
        super(Decoder,self).__init__()
        self.linear1=torch.nn.Linear(hidden_size,output_size)
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
        for att, norm in zip(self.att_layers, self.norms):
            u = att(x)
            scores = torch.bmm(u.unsqueeze(2), u.unsqueeze(1))
            attn = F.softmax(scores, dim=-1)
            out = torch.bmm(attn, x.unsqueeze(2)).squeeze(2)
            x = norm(x + out)
        x=self.linear1(x)
        return x

class CirVAE(nn.Module):
    def __init__(self, input_size, output_size, latent_size, hidden_size,n_att=2):
        super(CirVAE,self).__init__()
        self.encoder = Encoder(input_size, hidden_size, latent_size,n_att)
        self.decoder = Decoder(latent_size, hidden_size, output_size,n_att)
        self.bound=5e-6

    def forward(self, x,feaIndex):
        b,f=x.shape
        maxDelta=0.5
        delta_left, delta_right = self.encoder(x)
        delta_left=torch.sigmoid(delta_left)*maxDelta
        delta_right=torch.sigmoid(delta_right)*maxDelta
        eps = torch.rand_like(delta_left)
        z=x.clone()
        z.scatter_(1, feaIndex,z.gather(dim=1, index=feaIndex)-delta_left+ eps*(delta_left+delta_right))
        re_x = self.decoder(z)
        return re_x, delta_left, delta_right

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

def sample_benign(X, k, labelIndex=-1,seed=42):
    np.random.seed(seed)

    mask0 = X[:, labelIndex] == 0
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

def calNumPerBind(x,lower,upper):
    lower_exp = lower.unsqueeze(1)
    upper_exp = upper.unsqueeze(1)
    x_exp = x.unsqueeze(0)
    lowerRaw=x_exp - lower_exp
    lowerRaw[lowerRaw==0.5] = 0.0000001
    in_lower = F.sigmoid(lowerRaw) - 0.5
    upperRaw = upper_exp - x_exp
    upperRaw[upperRaw == 0.5] = 0.0000001
    bound=upperRaw*lowerRaw
    count = (bound>0).all(dim=-1).sum(-1).float().mean().item()
    return count


def areaLossPerBindMultiArea(x,lower,upper,eps: float = 1e-8):
    diff = (upper - lower).clamp_min(eps)
    diff[diff==1e-6] = 1
    volume = diff.prod(dim=-1)
    lower_exp = lower.unsqueeze(1)
    upper_exp = upper.unsqueeze(1)
    x_exp = x.unsqueeze(0)
    in_lower=x_exp - lower_exp
    in_upper = upper_exp - x_exp
    membership_score = in_lower * in_upper
    inside_per_point,_ = membership_score.min(dim=-1)
    memnum=F.sigmoid(inside_per_point).sum(dim=-1)

    density=-memnum/volume
    return density


def selectSampleByFeaIndex(x,featureSelect):
    B,FEA=x.shape
    index_total = torch.arange(FEA).unsqueeze(0).expand(B, -1).to(device)
    index_rest = [row_a[~torch.isin(row_a, row_b)] for row_a, row_b in zip(index_total, featureSelect)]
    res=list()
    for i,item in enumerate(index_rest):
        restX=x[:,item]
        mask=(restX == restX[i]).all(dim=-1)
        selected_rows = x[mask]
        sel_result = selected_rows[:, featureSelect[i]]
        res.append(sel_result)
    return res


def areaLossPerBind(x,delta_lower,delta_upper,feat_idx,eps: float = 1e-8):
    testDATA=selectSampleByFeaIndex(x,feat_idx)

    B,FEA=x.shape
    m = feat_idx.shape[1]

    row_idx = torch.arange(FEA, device=device).view(1, FEA).expand(B, -1)
    selected_mask = (row_idx.unsqueeze(2) == feat_idx.unsqueeze(1)).any(dim=2)
    unselected_mask = ~selected_mask

    R=FEA-m
    x_rest = torch.zeros(B, R, dtype=x.dtype, device=device)
    for i in range(B):
        x_rest[i] = x[i, unselected_mask[i]]

    match = (x_rest.unsqueeze(1) == x_rest.unsqueeze(0)).all(dim=2)

    x_sel = torch.gather(x, 1, feat_idx)
    cand = x_sel.unsqueeze(0).expand(B, -1, -1)


    lower = (x_sel - delta_lower).unsqueeze(1)
    upper = (x_sel + delta_upper).unsqueeze(1)
    volume = (delta_lower + delta_upper).clamp_min(eps).prod(dim=1)

    in_lower = cand - lower
    in_upper = upper - cand
    membership = (in_lower * in_upper).prod(dim=2)

    membership = membership.where(match, torch.tensor(0., device=device))

    len_in_lower,_ = in_lower.min(dim=-1)
    len_in_upper,_ = in_upper.min(dim=-1)
    len_in_lower=len_in_lower.where(match, torch.tensor(0., device=device))
    len_in_upper = len_in_upper.where(match, torch.tensor(0., device=device))
    len_in_lower=len_in_lower.mean(dim=-1)
    len_in_upper=len_in_upper.mean(dim=-1)
    density = -membership.sum(dim=1) + (len_in_lower+ len_in_upper)*10000
    return density


def check_in_jdg_areas_rtree_optimized(data, jdgAreas):
    import rtree
    N, _, D = jdgAreas.shape
    props = rtree.index.Property()
    props.dimension = D
    idx = rtree.index.Index(properties=props)

    for i in tqdm(range(N)):
        min_coords = jdgAreas[i, 0, :]
        max_coords = jdgAreas[i, 1, :]
        idx.insert(i, tuple(min_coords) + tuple(max_coords))
    query_bboxes = np.column_stack([data, data]).reshape(-1, 2 * D)
    results = np.zeros(len(data), dtype=np.bool_)

    for i, bbox in tqdm(enumerate(query_bboxes)):
        if next(idx.intersection(bbox), None) is not None:
            results[i] = True
    return results


def most_redundant_dim_per_sample(X: np.ndarray):
    import  xxhash
    N, D = X.shape
    cnt = np.zeros((N, D), dtype=np.int32)
    slices = [list(range(i))+list(range(i+1, D)) for i in range(D)]

    for d in range(D):
        row19 = X[:, slices[d]]
        keys  = np.array([xxhash.xxh64(r.tobytes()).intdigest()
                          for r in row19], dtype=np.uint64)

        order = keys.argsort(kind='stable')
        keys_sorted = keys[order]

        start = 0
        for end in range(1, N+1):
            if end == N or keys_sorted[end] != keys_sorted[start]:
                cnt[order[start:end], d] = end - start - 1
                start = end
    return cnt.argmax(axis=1)

def most_redundant_k_dims_per_sample(X: np.ndarray, k: int = 3):
    import xxhash
    N, D = X.shape
    cnt = np.zeros((N, D), dtype=np.int32)
    slices = [list(range(i)) + list(range(i + 1, D)) for i in range(D)]

    for d in range(D):
        row19 = X[:, slices[d]]
        keys = np.array([xxhash.xxh64(r.tobytes()).intdigest()
                         for r in row19], dtype=np.uint64)
        order = keys.argsort(kind='stable')
        keys_sorted = keys[order]

        start = 0
        for end in range(1, N + 1):
            if end == N or keys_sorted[end] != keys_sorted[start]:
                cnt[order[start:end], d] = end - start - 1
                start = end

    topk = np.full((N, k), -1, dtype=np.int32)
    for i in range(N):
        cols = cnt[i].argsort()[::-1]
        _, idx = np.unique(cols, return_index=True)
        uniq = cols[np.sort(idx)][:k]
        topk[i, :len(uniq)] = uniq

    return topk

if __name__ == '__main__':
    train_batch_size = 1024
    test_batch_size = train_batch_size
    learning_rate = 0.01
    epochs = 30
    step = 4 # 0train, 1 Detected by the model, 2generate, 3 Rule-based detection, 4Visualization (rules), 5Visualization (models)
    continue_train=True
    laten_size = 1
    att_layer_size = 4
    device = get_device()
    dataset = readDataFromNp("filepath")
    benign_select_num = len(dataset[dataset[:, -1] == 0])
    feaselect_colIndex=[i for i in range(len(dataset[0]))]
    dataset = dataset[:, feaselect_colIndex]
    optindexfilepath="optindex"
    optindexRegen=True
    optindexLen=laten_size
    if os.path.exists(optindexfilepath+".npy") and not optindexRegen:
        optFeaIndex=readDataFromNp(optindexfilepath+".npy")
    else:
        optFeaIndex=most_redundant_k_dims_per_sample(dataset[:,:-1],optindexLen)
        saveDataToNp(optFeaIndex,optindexfilepath)
    dataset = np.hstack((dataset, optFeaIndex))

    dataset[dataset[:, -1-optindexLen] > 1, -1-optindexLen] = 1

    dataset[:, :-1-optindexLen] = np.where(dataset[:, :-1-optindexLen] < -1, -1, dataset[:, :-1-optindexLen])

    dataset[:, :-1-optindexLen] = normalize_cols(dataset[:, :-1-optindexLen], method='zscore')

    trainDataset, testDataset = sample_benign(dataset, benign_select_num,-1-optindexLen)

    trainDataset = trainDataset[np.lexsort([trainDataset[:,-1-optindexLen-i] for i in range(trainDataset.shape[-1]-optindexLen)])]

    trainDataset_tensor = torch.from_numpy(trainDataset)
    trainDataset_tensor = TensorDataset(trainDataset_tensor)
    train_loader = DataLoader(trainDataset_tensor, batch_size=train_batch_size, shuffle=False, drop_last=False)
    testDataset_tensor=torch.from_numpy(testDataset)
    testDataset_tensor_tensor = TensorDataset(testDataset_tensor)
    test_loader = DataLoader(testDataset, batch_size=test_batch_size, shuffle=False)
    totalDataset_tensor = torch.from_numpy(dataset)
    totalDataset_tensor= TensorDataset(totalDataset_tensor)
    totaldata_loader=DataLoader(totalDataset_tensor, batch_size=train_batch_size, shuffle=False, drop_last=False)
    oneSample = dataset[0]
    input_size = len(oneSample) - 1-optindexLen
    output_size = input_size
    hidden_size = input_size

    model = CirVAE(input_size, output_size, laten_size, hidden_size,att_layer_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_MSE = torch.nn.MSELoss(reduction='none')

    if step == 0:
        if continue_train:
            model.load_state_dict(torch.load("network.pth"))
        train_loss = list()
        train_epochs_loss = list()
        for epoch in range(epochs):
            model.train()
            train_epoch_loss = list()
            train_mse_loss = list()
            train_area_loss=list()
            area_total=list()
            ent_total=list()
            for idx, data in tqdm(enumerate(train_loader, 0)):
                data=data[0]
                data_x = data[:, :-1-optindexLen].to(torch.float32).to(device)
                feaIndex=data[:, -optindexLen:].to(torch.int).to(device)
                predict, deltaL, deltaR= model(data_x,feaIndex)
                optimizer.zero_grad()
                loss_mse = loss_MSE(predict, data_x).sum(dim=-1)
                loss_area=areaLossPerBind(data_x, deltaL, deltaR,feaIndex)
                loss=(loss_mse+loss_area).mean()
                loss.backward()
                optimizer.step()
                train_epoch_loss.append(loss.item())
                train_mse_loss.append(loss_mse.mean().item())
                train_area_loss.append(loss_area.mean().item())
                train_loss.append(loss.item())
                area_total.append((deltaL+deltaR).prod(dim=-1).mean().item())
            print("epoch:", epoch, "loss:", np.average(train_epoch_loss),"mse_loss:",np.average(train_mse_loss),"area_loss:",np.average(train_area_loss),"area_total:",np.average(area_total))
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
            predict, deltaL, deltaR = model(data_x)
            optimizer.zero_grad()
            loss_mse = loss_MSE(predict, data_x).sum(dim=-1)
            loss_area = areaLoss(data_x, data_x-deltaL, data_x+deltaR)
            loss = loss_mse + loss_area
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
            for fea,val in zip(feanames,item):
                print(f"{fea}:{val}")
    elif step == 3:
        model.load_state_dict(torch.load("network.pth"))
        jdgAreas=list()
        for idx, data in tqdm(enumerate(train_loader, 0)):
            data=data[0]
            data_x = data[:, :-1-optindexLen].to(torch.float32).to(device)
            feaIndex=data[:, -optindexLen:].to(torch.int).to(device)
            _, deltaL, deltaR= model(data_x,feaIndex)
            areas_lower=data_x.clone()-3e-5
            areas_lower.scatter_(1, feaIndex, areas_lower.gather(-1,feaIndex)-deltaL)
            areas_upper = data_x.clone()+3e-5
            areas_upper.scatter_(1, feaIndex, areas_upper.gather(-1,feaIndex)+deltaR)


            jdgAreas.extend(list(zip(areas_lower.tolist(),areas_upper.tolist())))

        jdgAreas=np.unique(jdgAreas,axis=0)


        labels = list()
        predicts = list()

        dataset_test=dataset

        predicts=~check_in_jdg_areas_rtree_optimized(dataset_test[:,:-1-optindexLen],jdgAreas)
        labels=dataset_test[:,-1-optindexLen]

        acc = accuracy_score(labels, predicts)
        pre = precision_score(labels, predicts)
        rec = recall_score(labels, predicts)
        f1 = f1_score(labels, predicts)
        auc = roc_auc_score(labels, predicts)

        print("acc:",acc,"pre:",pre,"rec:",rec,"f1:",f1,"auc:",auc)
    elif step ==4:
        model.load_state_dict(torch.load("network.pth"))
        jdgAreas = list()
        for idx, data in tqdm(enumerate(train_loader, 0)):
            data = data[0]
            data_x = data[:, :-1 - optindexLen].to(torch.float32).to(device)
            feaIndex = data[:, -optindexLen:].to(torch.int).to(device)
            _, deltaL, deltaR = model(data_x, feaIndex)
            areas_lower = data_x.clone() - 3e-5
            areas_lower.scatter_(1, feaIndex, areas_lower.gather(-1, feaIndex) - deltaL)
            areas_upper = data_x.clone() + 3e-5
            areas_upper.scatter_(1, feaIndex, areas_upper.gather(-1, feaIndex) + deltaR)

            jdgAreas.extend(list(zip(areas_lower.tolist(), areas_upper.tolist())))

        jdgAreas = np.unique(jdgAreas, axis=0)
        jdgAreas = jdgAreas.reshape(-1, jdgAreas.shape[-1])
        data=np.concatenate((dataset[:,:-1-optindexLen],jdgAreas),axis=0)
        downFea = TSNE(n_components=2)
        dataPCA = downFea.fit_transform(data)

        c=["b"  if item==0 else "r" for item in dataset[:,-1-optindexLen]]
        plt.figure(figsize=(10, 10))
        plt.scatter(dataPCA[:, 0], dataPCA[:, 1],
                    s=5,c=c+["g"]*len(jdgAreas),marker="*"
                 )

        labels = ['benign', 'malicious', 'rule area']
        colors = ['b', 'r', 'g']
        marker = ['*', '*', '*']

        from matplotlib.lines import Line2D

        legend_elements = [Line2D([0], [0], marker=m, color='w',
                                  markerfacecolor=c, markersize=10, linestyle='None')
                           for m, c in zip(marker, colors)]

        plt.legend(legend_elements, labels)

        plt.axis('off')
        plt.savefig("rulesPca.png",dpi=600,bbox_inches='tight',pad_inches=0.0)
    elif step ==5:
        model.load_state_dict(torch.load("network.pth"))
        res = list()
        label=list()
        for idx, data in tqdm(enumerate(totaldata_loader, 0)):
            data = data[0]
            data_x = data[:, :-1 - optindexLen].to(torch.float32).to(device)
            data_y=data[:,-1 - optindexLen].numpy().tolist()
            label.extend(data_y)
            feaIndex = data[:, -optindexLen:].to(torch.int).to(device)
            pred, _, _ = model(data_x, feaIndex)
            res.append(pred.cpu().detach().numpy())

        res = np.concatenate(res,axis=0)
        downFea = TSNE(n_components=2)
        dataPCA = downFea.fit_transform(res)
        c=["b"  if item==0 else "r" for item in label]
        plt.figure(figsize=(10, 10))
        plt.scatter(dataPCA[:, 0], dataPCA[:, 1],
                    s=5,c=c,marker="*"
                 )

        labels = ['benign', 'malicious']
        colors = ['b', 'r']
        marker = ['*', '*']

        from matplotlib.lines import Line2D

        legend_elements = [Line2D([0], [0], marker=m, color='w',
                                  markerfacecolor=c, markersize=10, linestyle='None')
                           for m, c in zip(marker, colors)]

        plt.legend(legend_elements, labels)
        plt.axis('off')
        plt.savefig("modelPca.png",dpi=600,bbox_inches='tight',pad_inches=0.0)




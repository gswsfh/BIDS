import json
import os
import time
from collections import Counter

import pandas as pd
from tqdm import tqdm
import numpy as np

def saveDataToJson(data,filepath):
    with open(filepath,'w',encoding='utf-8') as f:
        json.dump(data,f,ensure_ascii=False)

def readDataFromJson(filepath):
    with open(filepath,'r',encoding='utf-8') as f:
        data = json.load(f)
        return data

def saveDataToNp(data,filepath):
    np.save(filepath,data)

def readDataFromNp(filepath):
    return np.load(filepath)

def readCICIDS2017(datadir,savedatapath):
    if os.path.exists(savedatapath):
        return readDataFromJson(savedatapath)
    res=list()
    label=dict()

    files=[os.path.join(datadir,item) for item in os.listdir(datadir)]
    df_all = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    labels, uniques=pd.factorize(df_all[' Label'])
    df_all[' Label'] = labels
    df_all.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_all.fillna(-1, inplace=True)
    res=df_all.values.tolist()
    saveDataToJson(res,savedatapath)
    saveDataToNp(res,savedatapath[:-5])
    saveDataToJson(uniques.tolist(),savedatapath[:-5]+"Dict.json")
    return res

def readCICIDS2017WithIp(datadir,savedatapath):
    if os.path.exists(savedatapath):
        return readDataFromJson(savedatapath)
    res=list()
    label=dict()

    files=[os.path.join(datadir,item) for item in os.listdir(datadir)]
    df_all = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    labels, uniques=pd.factorize(df_all[' Label'])
    df_all[' Label'] = labels
    df_all.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_all.fillna(-1, inplace=True)
    res=df_all.values.tolist()
    saveDataToJson(res,savedatapath)
    saveDataToNp(res,savedatapath[:-5])
    saveDataToJson(uniques.tolist(),savedatapath[:-5]+"Dict.json")
    return res

def readCICIDS2018(datadir,savedatapath):
    if os.path.exists(savedatapath):
        return readDataFromJson(savedatapath)
    res=list()
    label=dict()
    col_select=['Dst Port', 'Flow Duration', 'Tot Fwd Pkts',
       'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max',
       'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std',
       'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Bwd Pkt Len Mean',
       'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean',
       'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Tot',
       'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
       'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max',
       'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags',
       'Bwd URG Flags', 'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s',
       'Bwd Pkts/s', 'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean',
       'Pkt Len Std', 'Pkt Len Var', 'FIN Flag Cnt', 'SYN Flag Cnt',
       'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt',
       'CWE Flag Count', 'ECE Flag Cnt', 'Down/Up Ratio', 'Pkt Size Avg',
       'Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Fwd Byts/b Avg',
       'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg',
       'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg', 'Subflow Fwd Pkts',
       'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts',
       'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Act Data Pkts',
       'Fwd Seg Size Min', 'Active Mean', 'Active Std', 'Active Max',
       'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min', 'Label']
    dtypes=['int64', 'int64', 'int64', 'int64', 'int64', 'float64', 'int64', 'int64', 'float64', 'float64', 'int64', 'int64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'float64', 'float64', 'int64', 'int64', 'float64', 'float64', 'float64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'float64', 'float64', 'float64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'object']
    col_dtype={k:v for k,v in zip(col_select,dtypes)}
    files=[os.path.join(datadir,item) for item in os.listdir(datadir)]
    df_all = pd.concat((pd.read_csv(f,usecols=col_select,dtype=col_dtype,na_values=['', 'Nan', 'Infinity']) for f in files), ignore_index=True)
    labels, uniques=pd.factorize(df_all['Label'])
    df_all['Label'] = labels
    df_all.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_all.fillna(-1, inplace=True)
    res=df_all.values.tolist()
    saveDataToJson(res,savedatapath)
    saveDataToNp(res,savedatapath[:-5])
    saveDataToJson(uniques.tolist(),savedatapath[:-5]+"Dict.json")
    return res


def dataAnalysis(data,label):
    feature_name=[
        " Destination Port",
        " Flow Duration",
        " Total Fwd Packets",
        " Total Backward Packets",
        "Total Length of Fwd Packets",
        " Total Length of Bwd Packets",
        " Fwd Packet Length Max",
        " Fwd Packet Length Min",
        " Fwd Packet Length Mean",
        "  Fwd Packet Length Std",
        "Bwd Packet Length Max",
        " Bwd Packet Length Min",
        " Bwd Packet Length Mean",
        " Bwd Packet Length Std",
        "Flow Bytes/s",
        " Flow Packets/s",
        " Flow IAT Mean",
        " Flow IAT Std",
        " Flow IAT Max",
        " Flow IAT Min",
        "Fwd IAT Total",
        " Fwd IAT Mean",
        " Fwd IAT Std",
        " Fwd IAT Max",
        " Fwd IAT Min",
        "Bwd IAT Total",
        " Bwd IAT Mean",
        " Bwd IAT Std",
        " Bwd IAT Max",
        " Bwd IAT Min",
        "Fwd PSH Flags",
        " Bwd PSH Flags",
        " Fwd URG Flags",
        " Bwd URG Flags",
        " Fwd Header Length",
        " Bwd Header Length",
        "Fwd Packets/s",
        " Bwd Packets/s",
        " Min Packet Length",
        " Max Packet Length",
        " Packet Length Mean",
        " Packet Length Std",
        " Packet Length Variance",
        "FIN Flag Count",
        " SYN Flag Count",
        " RST Flag Count",
        " PSH Flag Count",
        " ACK Flag Count",
        " URG Flag Count",
        " CWE Flag Count",
        " ECE Flag Count",
        " Down/Up Ratio",
        " Average Packet Size",
        " Avg Fwd Segment Size",
        " Avg Bwd Segment Size",
        " Fwd Header Length",
        "Fwd Avg Bytes/Bulk",
        " Fwd Avg Packets/Bulk",
        " Fwd Avg Bulk Rate",
        " Bwd Avg Bytes/Bulk",
        " Bwd Avg Packets/Bulk",
        "Bwd Avg Bulk Rate",
        "Subflow Fwd Packets",
        " Subflow Fwd Bytes",
        " Subflow Bwd Packets",
        " Subflow Bwd Bytes",
        "Init_Win_bytes_forward",
        " Init_Win_bytes_backward",
        " act_data_pkt_fwd",
        " min_seg_size_forward",
        "Active Mean",
        " Active Std",
        " Active Max",
        " Active Min",
        "Idle Mean",
        " Idle Std",
        " Idle Max",
        " Idle Min",
    ]
    x_num,ynum=data.shape
    assert ynum-1==len(feature_name)
    if label is None:
        res=""
        for i in range(ynum-1):
            data_i=data[:,i]
            res+= f"|{feature_name[i]}|"+"|".join([str(np.max(data_i).tolist()), str(np.min(data_i).tolist()),str(np.mean(data_i).tolist()),str(np.median(data_i).tolist()),str(np.std(data_i).tolist()),str(np.percentile(data_i, 25).tolist()),str(np.percentile(data_i, 50).tolist()),str(np.percentile(data_i, 75).tolist())])+"|"
            res+="\n"
        return res
    else:
        res=list()
        for i in range(ynum - 1):
            data_i = data[:, i]
            line_res=f"|{feature_name[i]}|"+f"{label}|"+"|".join([str(np.max(data_i).tolist()), str(np.min(data_i).tolist()),str(np.mean(data_i).tolist()),str(np.median(data_i).tolist()),str(np.std(data_i).tolist()),str(np.percentile(data_i, 25).tolist()),str(np.percentile(data_i, 50).tolist()),str(np.percentile(data_i, 75).tolist())])+"|"
            res.append(line_res)
        return res

def dataAnalysisBody():
    data = np.array(readCICIDS2017("dir",
                                   "filename"))
    category = readDataFromJson("filename")
    category_col = data[:, -1]
    unique_categories = np.unique(category_col)
    split_data = {}
    for cat in unique_categories:
        cat = int(cat.tolist())
        split_data[cat] = data[category_col == cat]

    total_res = list()
    for cat, subset in split_data.items():
        print("***" * 6)
        print(f"Catagory：{category[cat]}, processing")
        res = dataAnalysis(subset, category[cat])
        total_res.append(res)

    print_res = ""
    fea_num = len(total_res[0])
    for fea_i in range(fea_num):
        for labelSet in total_res:
            data = labelSet[fea_i]
            print_res += data
            print_res += "\n"
    print(print_res)

def drawFeaNum(data,title):
    import matplotlib.pyplot as plt
    count=dict(Counter(data))
    new_sys1 = sorted(count.items(), key=lambda d: d[0], reverse=False)
    x=[item[0] for item in new_sys1]
    y=[item[1] for item in new_sys1]
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel("Vaule")
    plt.ylabel("Number")
    plt.show()

def drawFeaNumContrast(datas,title):
    import matplotlib.pyplot as plt
    count0=dict(Counter(datas[0]))
    new_sys1 = sorted(count0.items(), key=lambda d: d[0], reverse=False)
    x0=[item[0] for item in new_sys1]
    y0=[item[1] for item in new_sys1]
    count1 = dict(Counter(datas[1]))
    new_sys2 = sorted(count1.items(), key=lambda d: d[0], reverse=False)
    x1 = [item[0] for item in new_sys2]
    y1 = [item[1] for item in new_sys2]
    plt.plot(x0, y0,color="blue")
    plt.plot(x1, y1, color="red")
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Number")
    plt.show()




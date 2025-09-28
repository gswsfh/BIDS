
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from datasetProcess import readCICIDS2017, readDataFromNp
import matplotlib.pyplot as plt

def selectByLabel(data,labelids):
    mask = np.isin(data[:, -1], labelids)
    selected = data[mask]
    return selected


if __name__ == '__main__':
    import time
    time1=time.time()
    data=readDataFromNp("filepath")
    time2 = time.time()
    data_X=data[:,:-1]
    data_X = np.delete(data_X, 55, axis=1)
    data_Y=data[:,-1]
    data_Y = np.clip(data_Y, a_min=None, a_max=1)
    time3 = time.time()
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(data_X, data_Y)
    time4 = time.time()
    del data,data_X,data_Y
    data=readDataFromNp("/mnt/storage/fuhao/codes/SSH/WWW26/datasets/CICIDS2018/processed/CICDIS2018.npy")
    data_X = data[:, :-1]
    data_Y = data[:, -1]
    data_Y = np.clip(data_Y, a_min=None, a_max=1)
    YPre=clf.predict(data_X)
    time5 = time.time()
    acc=accuracy_score(data_Y, YPre)
    pre=precision_score(data_Y, YPre)
    rec=recall_score(data_Y, YPre)
    f1=f1_score(data_Y, YPre)

    print("accuracy",acc)
    print("precision",pre)
    print("recall",rec)
    print("f1",f1)

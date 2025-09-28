import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from datasetProcess import readCICIDS2017, readDataFromNp
import matplotlib.pyplot as plt
import matplotlib as mpl

if __name__ == '__main__':
    import time
    time1=time.time()
    data=readDataFromNp("/mnt/storage/fuhao/codes/SSH/WWW26/datasets/CICIDS2017/processed/CICDIS2017.np.npy")
    time2 = time.time()
    data_X=data[:,:-1]
    data_X = np.delete(data_X, 55, axis=1)
    data_X [data_X  < -1] = -1
    data_X=data_X[:,[52,13,35,0,39,65]]
    data_Y=data[:,-1]
    data_Y = np.clip(data_Y, a_min=None, a_max=1)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(data_X, data_Y, test_size=0.3)
    time3 = time.time()
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(Xtrain, Ytrain)
    time4 = time.time()
    YPre=clf.predict(Xtest)
    time5 = time.time()
    acc=accuracy_score(Ytest, YPre)
    pre=precision_score(Ytest, YPre)
    rec=recall_score(Ytest, YPre)
    f1=f1_score(Ytest, YPre)

    features=[" Destination Port"," Flow Duration"," Total Fwd Packets"," Total Backward Packets","Total Length of Fwd Packets"," Total Length of Bwd Packets"," Fwd Packet Length Max"," Fwd Packet Length Min"," Fwd Packet Length Mean","  Fwd Packet Length Std","Bwd Packet Length Max"," Bwd Packet Length Min"," Bwd Packet Length Mean"," Bwd Packet Length Std","Flow Bytes/s"," Flow Packets/s"," Flow IAT Mean"," Flow IAT Std"," Flow IAT Max"," Flow IAT Min","Fwd IAT Total"," Fwd IAT Mean"," Fwd IAT Std"," Fwd IAT Max"," Fwd IAT Min","Bwd IAT Total"," Bwd IAT Mean"," Bwd IAT Std"," Bwd IAT Max"," Bwd IAT Min","Fwd PSH Flags"," Bwd PSH Flags"," Fwd URG Flags"," Bwd URG Flags"," Fwd Header Length"," Bwd Header Length","Fwd Packets/s"," Bwd Packets/s"," Min Packet Length"," Max Packet Length"," Packet Length Mean"," Packet Length Std"," Packet Length Variance","FIN Flag Count"," SYN Flag Count"," RST Flag Count"," PSH Flag Count"," ACK Flag Count"," URG Flag Count"," CWE Flag Count"," ECE Flag Count"," Down/Up Ratio"," Average Packet Size", " Avg Fwd Segment Size"," Avg Bwd Segment Size","Fwd Avg Bytes/Bulk"," Fwd Avg Packets/Bulk"," Fwd Avg Bulk Rate"," Bwd Avg Bytes/Bulk"," Bwd Avg Packets/Bulk","Bwd Avg Bulk Rate","Subflow Fwd Packets"," Subflow Fwd Bytes"," Subflow Bwd Packets"," Subflow Bwd Bytes","Init_Win_bytes_forward"," Init_Win_bytes_backward"," act_data_pkt_fwd"," min_seg_size_forward","Active Mean"," Active Std"," Active Max"," Active Min","Idle Mean"," Idle Std"," Idle Max"," Idle Min"]
    feat_importance = clf.tree_.compute_feature_importances(normalize=True)
    for i in range(len(features)):
        print(f"{features[i]}：{feat_importance[i]}")

    print("feat importance = " + str(feat_importance))

    print("accuracy",acc)
    print("precision",pre)
    print("recall",rec)
    print("f1",f1)


    plt.figure(figsize=(200, 100))
    tree.plot_tree(clf, fontsize=10,filled=True, feature_names=features, class_names=["normal","attack"])
    plt.show()
    plt.savefig('tree.png', dpi=600)
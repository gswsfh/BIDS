import numpy as np
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC

from datasetProcess import readCICIDS2017, readDataFromNp
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import MinMaxScaler

def DT():
    import time
    time1 = time.time()
    data = readDataFromNp("filepath")
    time2 = time.time()
    data_X = data[:, :-1]
    data_X = np.delete(data_X, 55, axis=1)
    data_X[data_X < -1] = -1
    data_Y = data[:, -1]
    data_Y = np.clip(data_Y, a_min=None, a_max=1)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(data_X, data_Y, test_size=0.3)
    time3 = time.time()
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(Xtrain, Ytrain)
    time4 = time.time()
    YPre = clf.predict(Xtest)
    time5 = time.time()
    acc = accuracy_score(Ytest, YPre)
    pre = precision_score(Ytest, YPre)
    rec = recall_score(Ytest, YPre)
    f1 = f1_score(Ytest, YPre)

    print("accuracy", acc)
    print("precision", pre)
    print("recall", rec)
    print("f1", f1)


def Classify(clf = GaussianNB()):
    import time
    time1 = time.time()
    data = readDataFromNp("filepath")
    time2 = time.time()

    data_X = data[:, :-1]
    data_X = np.delete(data_X, 55, axis=1)

    data_X[data_X < -1] = -1

    scaler = MinMaxScaler()
    data_X = scaler.fit_transform(data_X)

    data_Y = data[:, -1]

    data_Y = np.clip(data_Y, a_min=None, a_max=1)

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(data_X, data_Y, test_size=0.3)
    time3 = time.time()
    clf = clf.fit(Xtrain, Ytrain)
    time4 = time.time()
    YPre = clf.predict(Xtest)
    time5 = time.time()
    acc = accuracy_score(Ytest, YPre)
    pre = precision_score(Ytest, YPre)
    rec = recall_score(Ytest, YPre)
    f1 = f1_score(Ytest, YPre)

    print("accuracy", acc)
    print("precision", pre)
    print("recall", rec)
    print("f1", f1)




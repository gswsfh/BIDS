import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans

from DeepLearning.autoencoder import normalize_cols
from datasetProcess import readDataFromNp, readCICIDS2017, readCICIDS2018
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

def CICIDS2017PCA(filepath):
    import time
    time1 = time.time()
    data = readDataFromNp(filepath)
    time2 = time.time()
    print("Read data time:", time2 - time1)
    print("Total data length:", len(data))
    data_X = data[:, :-1]
    data_Y = data[:, -1]
    data_Y = np.clip(data_Y, a_min=None, a_max=1)
    data_X = normalize_cols(data_X, method='minmax')
    pca = PCA(n_components=2)
    data_X_pca = pca.fit_transform(data_X)
    idx0 = data_Y == 0
    idx1 = data_Y == 1
    plt.scatter(data_X_pca[idx0, 0], data_X_pca[idx0, 1],
                c='steelblue', label='Catagory 0', edgecolor='k')
    plt.scatter(data_X_pca[idx1, 0], data_X_pca[idx1, 1],
                c='tomato', label='Catagory 1', edgecolor='k')
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.title('PCA Result')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

def CICIDS2017KmeasnPCA(filepath):
    import time
    from numpy import unique, where
    time1 = time.time()
    data = readDataFromNp(filepath)
    time2 = time.time()
    print("Read data time:", time2 - time1)
    print("Total data length:", len(data))
    data_X = data[:, :-1]
    data_Y = data[:, -1]
    data_Y = np.clip(data_Y, a_min=None, a_max=1)
    data_X = normalize_cols(data_X, method='minmax')
    kmeans = KMeans(7)
    yhat = kmeans.fit_predict(data_X)
    print(yhat)
    clusters = unique(yhat)
    pca = PCA(n_components=2)
    data_X_pca = pca.fit_transform(data_X)
    colors = ["r", "g", "b", "c", "m", "y", "k"]
    for cluster in clusters:
        row_ix = where(yhat == cluster)
        plt.scatter(data_X_pca[row_ix, 0], data_X_pca[row_ix, 1], color=colors[cluster], )
    plt.show()


if __name__ == '__main__':
    import time
    from numpy import unique,where
    time1 = time.time()
    data = readDataFromNp("filepath")
    time2 = time.time()
    data_X = data[:, :-1]
    data_Y = data[:, -1]
    data_X = normalize_cols(data_X, method='minmax')
    pca = PCA(n_components=2)
    data_X_pca = pca.fit_transform(data_X)
    clusters=unique(data_Y)
    ALL_COLOR_NAMES = ['green',
        'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure',
        'beige', 'bisque', 'black', 'blanchedalmond', 'blue',
        'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse',
        'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson',
        'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray',
        'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen',
        'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen',
        'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise',
        'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey',
        'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'fuchsia',
        'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'gray', 'greenyellow', 'grey', 'honeydew', 'hotpink',
        'indianred', 'indigo', 'ivory', 'khaki', 'lavender',
        'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue',
        'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray',
        'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon',
        'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey',
        'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen',
        'magenta', 'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid',
        'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen',
        'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream',
        'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace',
        'olive', 'olivedrab', 'orange', 'orangered', 'orchid',
        'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred',
        'papayawhip', 'peachpuff', 'peru', 'pink', 'plum',
        'powderblue', 'purple', 'red', 'rosybrown', 'royalblue',
        'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell',
        'sienna', 'silver', 'skyblue', 'slateblue', 'slategray',
        'slategrey', 'snow', 'springgreen', 'steelblue', 'tan',
        'teal', 'thistle', 'tomato', 'turquoise', 'violet',
        'wheat', 'white', 'whitesmoke', 'yellow', 'yellowgreen']
    handles = []
    labels = []
    for cluster in clusters:
        row_ix = where(data_Y == cluster)
        sc = plt.scatter(data_X_pca[row_ix, 0], data_X_pca[row_ix, 1],color=ALL_COLOR_NAMES[int(cluster)])
        handles.append(sc)
        labels.append(f'Catagory {int(cluster)}')

    plt.legend(handles=handles, labels=labels,
               bbox_to_anchor=(1.3, 1),
               loc='upper right',
               borderaxespad=0.)
    plt.tight_layout()
    plt.show()



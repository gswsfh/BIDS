import time
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from tqdm import tqdm

from datasetProcess import readDataFromJson

if __name__ == '__main__':
    time1=time.time()
    datapath="filepath"
    data=pd.read_csv(datapath,usecols=['Flow ID'," Label"])
    print("Data length：",len(data))
    labelsIndex=readDataFromJson("filepath")
    time2=time.time()
    print("Reading time:",time2-time1)
    windowssize=1000
    chunks = [data[i:i+windowssize] for i in range(0, len(data), windowssize)]
    step=0
    for chunk in tqdm(chunks):
        G = nx.DiGraph()
        for item in chunk.to_numpy().tolist():
            id,label=item[0],item[1]
            dstip,srcip,dstport,srcport,proto= id.split("-")
            G.add_edge(srcip,dstip,label=labelsIndex.index(label.strip()))
            for ip in (srcip, dstip):
                if 'node_label' not in G.nodes[ip]:
                    G.nodes[ip]['node_label'] = f"{ip}"

        pos = nx.shell_layout(G,scale=2)
        plt.figure(figsize=(10, 10))

        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=100)
        node_labels = nx.get_node_attributes(G, 'node_label')
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)

        nx.draw_networkx_edges(G, pos)
        edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

        plt.savefig(f"filepath")
        plt.close()
        step+=1
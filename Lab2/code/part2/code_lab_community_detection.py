"""
Graph Mining - ALTEGRAD - Oct 2023
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye
from random import randint
from sklearn.cluster import KMeans



############## Task 6
# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(G, k):
    
    ##################
    # your code here #
    A=nx.adjacency_matrix(G)
    Dinv = diags([1/G.degree(node) for node in G.nodes()])
    Lrw = eye(G.number_of_nodes())-Dinv @ A
    eigenvals,eigenvecs = eigs(Lrw,k=k,which='SR')
    eigenvecs=np.real(eigenvecs)
    kmeans=KMeans(n_clusters=k).fit(eigenvecs)
    clustering={}
    for i, node in enumerate(G.nodes()):
        clustering[node] = kmeans.labels_[i]
    ##################
    
    return clustering


############## Task 7

##################
# your code here #

G=nx.read_edgelist('code\datasets\CA-HepTh.txt',comments='#',delimiter='\t')
largest_cc=max(nx.connected_components(G),key=len)
subG = G.subgraph(largest_cc)
spectral_clustering_giant_cc = spectral_clustering(subG,50)

##################





############## Task 8
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    
    ##################
    # your code here #
    m=G.number_of_edges()
    clusters = np.unique(list(clustering.values()))
    modularity = 0
    #for cluster,label in clustering.items():
     #   modularity+=cluster.number_of_edges()/

    for cluster in clusters:
        nodes_cluster = [
            node for node in G.nodes() if clustering[node] == cluster
        ]
        cluster_graph = G.subgraph(nodes_cluster)
        lc = cluster_graph.number_of_edges()
        dc = sum([G.degree(node) for node in nodes_cluster])
        modularity += (lc / m) - (
            dc / (2 * m)
        ) ** 2
    ##################
    
    return modularity

############## Task 9

##################
# your code here #

modularity_spectral_clustering_giant_cc=modularity(subG, spectral_clustering_giant_cc)
print(
    "Modularity of the spectral clustering with 50 clusters:",
    modularity_spectral_clustering_giant_cc
)

random_clustering_giant_cc = {node: randint(0, 49) for node in subG.nodes()}

print(
    "Modularity of the random clustering with 50 clusters:",
    modularity(subG, random_clustering_giant_cc),
)


##################








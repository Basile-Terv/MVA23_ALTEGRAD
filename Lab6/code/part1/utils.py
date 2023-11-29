"""
Deep Learning on Graphs - ALTEGRAD - Nov 2023
"""

import networkx as nx
import numpy as np
import torch
from random import randint

def create_dataset(n_graphs=50,probs=[0.2,0.4]):
    Gs = list()
    y = list()

    ############## Task 1
    
    ##################
    # your code here #
    for i in range(n_graphs):
        for label, prob in enumerate(probs):
            n = randint(10, 20)
            Gs.append(nx.erdos_renyi_graph(n,prob))
            y.append(label)
    ##################

    return Gs, y


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

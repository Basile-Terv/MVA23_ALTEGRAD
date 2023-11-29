"""
Deep Learning on Graphs - ALTEGRAD - Nov 2023
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch

from models import GNN
from utils import sparse_mx_to_torch_sparse_tensor

# Initializes device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Hyperparameters
hidden_dim = 32
output_dim = 4
dropout = 0.0
neighbor_aggr = 'mean'
readout = 'mean'

############## Task 4
        
##################
# your code here #
cycle_graphs=list()
for i in range(10,20):
    cycle_graphs.append(nx.cycle_graph(i))
##################

############## Task 5
        
##################
# your code here #
# adjacency matrix of the dataset
adj_cycles = [nx.adjacency_matrix(gr) for gr in cycle_graphs]
adj_cycles = sp.block_diag(adj_cycles)
#Here we do not add the self-loops, we want a different weight matrix for the node itself
adj_cycles = sparse_mx_to_torch_sparse_tensor(adj_cycles).to(device)
# features of nodes of the dataset
features_cycles = torch.ones((adj_cycles.shape[0], 1), dtype=torch.float).to(device)
# vector that indicates the graph to which each node belongs
nb_nodes = [g.number_of_nodes() for g in cycle_graphs] 
idx_cycles = [idx*torch.ones(total_node, dtype=torch.long) for idx, total_node in enumerate(nb_nodes)]
idx_cycles = torch.cat(idx_cycles).to(device)
##################

############## Task 8
        
##################
# your code here #
input_dim=1
for neighbor_aggr in ['mean','sum']:
    for readout in ['mean','sum']:
        with torch.no_grad():
            model = GNN(input_dim, hidden_dim, output_dim, neighbor_aggr, readout, dropout).to(device)
            representations=model(features_cycles, adj_cycles, idx_cycles)
        print(f"{neighbor_aggr=}, {readout=} \n {representations.detach().cpu().numpy()}")
##################

############## Task 9
        
##################
# your code here #
G1=nx.union(nx.cycle_graph(3),nx.cycle_graph(3),rename=('C3_1','C3_22'))
G2 = nx.cycle_graph(6)

nx.draw_networkx(G1)
plt.show()
nx.draw_networkx(G2)
plt.show()
##################

############## Task 10
        
##################
# your code here #
# adjacency matrix of the dataset
adj_G1G2 = [nx.adjacency_matrix(gr) for gr in [G1,G2]]
adj_G1G2 = sp.block_diag(adj_G1G2)
#Here we do not add the self-loops, we want a different weight matrix for the node itself
adj_G1G2 = sparse_mx_to_torch_sparse_tensor(adj_G1G2).to(device)
# features of nodes of the dataset
features_G1G2 = torch.ones((adj_G1G2.shape[0], 1), dtype=torch.float).to(device)
# vector that indicates the graph to which each node belongs
nb_nodes = [g.number_of_nodes() for g in [G1,G2]] 
idx_G1G2 = [idx*torch.ones(total_node, dtype=torch.long) for idx, total_node in enumerate(nb_nodes)]
idx_G1G2 = torch.cat(idx_G1G2).to(device)
##################

############## Task 11
        
##################
# your code here #
readout='sum'
neighbor_aggr='sum'
model = GNN(input_dim, hidden_dim, output_dim, neighbor_aggr, readout, dropout).to(device)
representations=model(features_G1G2, adj_G1G2, idx_G1G2)
print(f"G_1 and G_2 representations \n {neighbor_aggr=}, {readout=} \n {representations.detach().cpu().numpy()}")
##################

############## Question 4
G3=nx.union_all([nx.cycle_graph(3),nx.cycle_graph(3),nx.cycle_graph(3)],rename=('C3_1','C3_2','C3_3'))
G4=nx.cycle_graph(9)

nx.draw_networkx(G3)
plt.show()
nx.draw_networkx(G4)
plt.show()

adj_G3G4 = [nx.adjacency_matrix(gr) for gr in [G3,G4]]
adj_G3G4 = sp.block_diag(adj_G3G4)
#Here we do not add the self-loops, we want a different weight matrix for the node itself
adj_G3G4 = sparse_mx_to_torch_sparse_tensor(adj_G3G4).to(device)
# features of nodes of the dataset
features_G3G4 = torch.ones((adj_G3G4.shape[0], 1), dtype=torch.float).to(device)
# vector that indicates the graph to which each node belongs
nb_nodes = [g.number_of_nodes() for g in [G3,G4]] 
idx_G3G4 = [idx*torch.ones(total_node, dtype=torch.long) for idx, total_node in enumerate(nb_nodes)]
idx_G3G4 = torch.cat(idx_G3G4).to(device)

model = GNN(input_dim, hidden_dim, output_dim, neighbor_aggr, readout, dropout).to(device)
representations=model(features_G3G4, adj_G3G4, idx_G3G4)
print(f"G_3 and G_4 representations \n {neighbor_aggr=}, {readout=} \n {representations.detach().cpu().numpy()}")
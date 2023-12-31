"""
Deep Learning on Graphs - ALTEGRAD - Nov 2023
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3, n_class):
        super(GNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, hidden_dim_3)
        self.fc4 = nn.Linear(hidden_dim_3, n_class)
        self.relu = nn.ReLU()

    def forward(self, x_in, adj, idx):
        
        ############## Task 2
    
        ##################
        # your code here #
        z0=self.relu(torch.mm(adj, self.fc1(x_in)))
        z1=self.relu(torch.mm(adj, self.fc2(z0)))
        ##################
        #Readout sums among nodes to have a graph-level prediction
        idx = idx.unsqueeze(1).repeat(1, z1.size(1))
        out = torch.zeros(torch.max(idx)+1, z1.size(1), device=x_in.device)
        out = out.scatter_add_(0, idx, z1) 
        #out here has nb of rows equal to nb of graphs in the minibatch
        ##################
        # your code here #
        out=self.fc4(self.relu(self.fc3(out)))
        ##################

        return F.log_softmax(out, dim=1)

"""
Learning on Sets and Graph Generative Models - ALTEGRAD - Nov 2023
"""

import torch
import torch.nn as nn

class DeepSets(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim):
        super(DeepSets, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        
        ############## Task 3
    
        ##################
        # your code here #
        x=self.embedding(x)
        x=self.tanh(self.fc1(x))
        x=torch.sum(x,dim=1)
        x=self.fc2(x)
        ##################
        
        return x.squeeze()


class LSTM(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        
        ############## Task 4
    
        ##################
        # your code here #
        x=self.embedding(x)
        lstm_output, (last_hidden_state, last_cell_state) = self.lstm(x)

        # Use the final hidden state for each sequence element
        x = self.fc(last_hidden_state[-1])
        ##################
        return x.squeeze()
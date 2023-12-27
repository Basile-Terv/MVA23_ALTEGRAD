"""
Learning on Sets and Graph Generative Models - ALTEGRAD - Nov 2023
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_absolute_error
import torch

from utils import create_test_dataset
from models import DeepSets, LSTM

# Initializes device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Hyperparameters
batch_size = 64
embedding_dim = 128
hidden_dim = 64

# Generates test data
X_test, y_test = create_test_dataset()
cards = [X_test[i].shape[1] for i in range(len(X_test))]
n_samples_per_card = X_test[0].shape[0]
n_digits = 11

# Retrieves DeepSets model
deepsets = DeepSets(n_digits, embedding_dim, hidden_dim).to(device)
print("Loading DeepSets checkpoint!")
checkpoint = torch.load('model_deepsets.pth.tar')
deepsets.load_state_dict(checkpoint['state_dict'])
deepsets.eval()

# Retrieves LSTM model
lstm = LSTM(n_digits, embedding_dim, hidden_dim).to(device)
print("Loading LSTM checkpoint!")
checkpoint = torch.load('model_lstm.pth.tar')
lstm.load_state_dict(checkpoint['state_dict'])
lstm.eval()

# Dict to store the results
results = {'deepsets': {'acc':[], 'mae':[]}, 'lstm': {'acc':[], 'mae':[]}}

for i in range(len(cards)):
    y_pred_deepsets = list()
    y_pred_lstm = list()
    for j in range(0, n_samples_per_card, batch_size):
        
        ############## Task 6
    
        ##################
        # your code here #
        #print("X_test[i][j:min(j+batch_size,n_samples_per_card)]=",X_test[i][j:min(j+batch_size,n_samples_per_card)])
        x_batch=torch.IntTensor(X_test[i][j:j+batch_size])
        #y_batch=torch.IntTensor(y_test[i][j:min(j+batch_size,n_samples_per_card)])
        #print("x_batch=",x_batch)
        # Forward pass for DeepSets
        with torch.no_grad():
            output_deepsets = deepsets(x_batch)
        
        # Forward pass for LSTM
        with torch.no_grad():
            output_lstm = lstm(x_batch)
        
        # Append predictions to the lists
        y_pred_deepsets.append(output_deepsets)
        y_pred_lstm.append(output_lstm)
        ##################
        
    y_pred_deepsets = torch.cat(y_pred_deepsets)
    y_pred_deepsets = y_pred_deepsets.detach().cpu().numpy()
    #print("y_pred_deepsets shape",y_pred_deepsets.shape)
    #print("y_pred_deepsets=",y_pred_deepsets)
    #print("i=",i)
    #print("y_test[i]=",y_test[i])
    #print("np.round(y_pred_deepsets).astype(int)",np.round(y_pred_deepsets).astype(int))
    #acc_deepsets = accuracy_score(y_test[i],np.round(y_pred_deepsets).astype(int))#your code here
    #print("np.rint(y_pred_deepsets)=",np.rint(y_pred_deepsets))
    acc_deepsets = accuracy_score(y_test[i],np.rint(y_pred_deepsets))
    mae_deepsets = mean_absolute_error(y_test[i],np.round(y_pred_deepsets).astype(int))#your code here
    results['deepsets']['acc'].append(acc_deepsets)
    results['deepsets']['mae'].append(mae_deepsets)
    
    y_pred_lstm = torch.cat(y_pred_lstm)
    y_pred_lstm = y_pred_lstm.detach().cpu().numpy()
    
    acc_lstm = accuracy_score(y_test[i],np.round(y_pred_lstm).astype(int))#your code here
    mae_lstm = mean_absolute_error(y_test[i],np.round(y_pred_lstm).astype(int))#your code here
    results['lstm']['acc'].append(acc_lstm)
    results['lstm']['mae'].append(mae_lstm)


############## Task 7
    
##################
# your code here #
# Plot accuracies
plt.plot(cards, results['deepsets']['acc'], label='DeepSets')
plt.plot(cards, results['lstm']['acc'], label='LSTM')

# Add labels and legend
plt.xlabel('Maximum Cardinality of Input Sets')
plt.ylabel('Accuracy')
plt.title('Accuracies of DeepSets and LSTM')
plt.legend()

plt.show()
##################
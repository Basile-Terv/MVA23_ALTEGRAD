"""
Deep Learning on Graphs - ALTEGRAD - Nov 2023
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye

from sklearn.linear_model import LogisticRegression
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import accuracy_score
from deepwalk import deepwalk


# Loads the karate network
G = nx.read_weighted_edgelist('code\data\karate.edgelist', delimiter=' ', nodetype=int, create_using=nx.Graph())
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

n = G.number_of_nodes()

# Loads the class labels
class_labels = np.loadtxt('code\data\karate_labels.txt', delimiter=',', dtype=np.int32)
idx_to_class_label = dict()
for i in range(class_labels.shape[0]):
    idx_to_class_label[class_labels[i,0]] = class_labels[i,1]

y = list()
for node in G.nodes():
    y.append(idx_to_class_label[node])

y = np.array(y)


############## Task 5
# Visualizes the karate network

##################
# your code here #
nx.draw_networkx(G,labels=idx_to_class_label,label='drawing of karate graph with class labels')
##################


############## Task 6
# Extracts a set of random walks from the karate network and feeds them to the Skipgram model
n_dim = 128
n_walks = 10
walk_length = 20
model = deepwalk(G, n_walks, walk_length, n_dim)# your code here

embeddings = np.zeros((n, n_dim))
for i, node in enumerate(G.nodes()):
    embeddings[i,:] = model.wv[str(node)]

idx = np.random.RandomState(seed=42).permutation(n)
idx_train = idx[:int(0.8*n)]
idx_test = idx[int(0.8*n):]

X_train = embeddings[idx_train,:]
X_test = embeddings[idx_test,:]

y_train = y[idx_train]
y_test = y[idx_test]


############## Task 7
# Trains a logistic regression classifier and use it to make predictions


##################
# your code here #
clf = LogisticRegression(solver='lbfgs')
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

print('DeepWalk_y_pred :', y_pred)
train_acc = accuracy_score(y_train, clf.predict(X_train))
test_acc = accuracy_score(y_test, y_pred)
print(f'Test accuracy: {test_acc}\n Train accuracy: {train_acc}')
##################


############## Task 8
# Generates spectral embeddings

##################
# your code here #

A=nx.adjacency_matrix(G)
Dinv = diags([1/G.degree(node) for node in G.nodes()])
Lrw = eye(G.number_of_nodes())-Dinv @ A
eigenvals,eigenvecs = eigs(Lrw,k=2,which='SR')
eigenvectors = np.real(eigenvecs)

#Lrw_dense = Lrw.toarray()
#spectral_embeddings = SpectralEmbedding(affinity=Lrw_dense,n_components=2)
#print(spectral_embeddings)

#spectral_embeddings = SpectralEmbedding(n_components=2, affinity='precomputed')
#X_train_transformed = spectral_embeddings.fit_transform(Lrw_dense)

# each row of eigenvectors corresponds to a node in the graph, 
# and the values in that row are the coordinates of the node 
# in the 2-dimensional embedding space.

#X_train_transformed = spectral_embeddings.fit_transform([idx_train])
#print('X_train_spectral_embeddings=',X_train_transformed)
#X_test = eigenvectors[idx_test]
#y_train = y[idx_train]
#y_test = y[idx_test]

X_train = eigenvectors[idx_train]
X_test = eigenvectors[idx_test]
y_train = y[idx_train]
y_test = y[idx_test]

clf = LogisticRegression(solver='lbfgs')
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

print('Spectral_y_pred :', y_pred)
train_acc = accuracy_score(y_train, clf.predict(X_train))
test_acc = accuracy_score(y_test, y_pred)
print(f'Test accuracy: {test_acc}\n Train accuracy: {train_acc}')

##################

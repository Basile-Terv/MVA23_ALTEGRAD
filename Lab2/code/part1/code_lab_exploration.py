"""
Graph Mining - ALTEGRAD - Oct 2023
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


############## Task 1

##################
# your code here #
G=nx.read_edgelist('code\datasets\CA-HepTh.txt',comments='#',delimiter='\t')
print('the graph has ', G.number_of_nodes(), 'nodes.')
print('the graph has ', G.number_of_edges(), 'edges.')
#number of nodes and edges is stored in the graph object 
# but not the number of connected components
print('is the graph directed',nx.is_directed(G))
##################

############## Task 2

##################
# your code here #

print('The graph has', nx.number_connected_components(G), 'connected components.')
largest_cc=max(nx.connected_components(G),key=len)
print('The largest connected component in the graph has', len(largest_cc), 'nodes.')

subG=G.subgraph(largest_cc)
print('the largest connected component in the graph has', subG.number_of_edges(), ' edges.')
print('This corresponds to ',subG.number_of_edges()/G.number_of_edges()," of the original graph's edges.")
print('This corresponds to ',subG.number_of_nodes()/G.number_of_nodes()," of the original graph's nodes.")
##################



############## Task 3
# Degree
degree_sequence = [G.degree(node) for node in G.nodes()]

##################
# your code here #

min = np.min(degree_sequence)
print('min degree of graph is ', min)
max = np.max(degree_sequence)
print('max degree of graph is ', max)
median=np.median(degree_sequence)
print('median degree of graph is ', median)
mean=np.mean(degree_sequence)
print('mean degree of graph is ', mean)

##################


############## Task 4

##################
# your code here #

degree_hist_list = nx.degree_histogram(G)

plt.plot(degree_hist_list)
plt.xlabel('degree')
plt.ylabel('frequency')
plt.show()

plt.loglog(degree_hist_list)
plt.xlabel('log(degree)')
plt.ylabel('log(frequency)')
plt.show()

##################



############## Task 5

##################
# your code here #

print('the global clustering coefficient of the HepTh graph is ',nx.transitivity(G))
# In documentation of transitivity function, it returns 3*nb of triangles/nb of nodes
# This must equal nb of closed triplets/nb of open triplets
# check origin of this 3 factor

##################









import pennylane as qml
from pennylane import qaoa
import numpy as np
import networkx as nx
import random

G = nx.Graph([(0, 1)])
for (u,v,w) in G.edges(data=True):
    w['weight'] = random.randint(0,10)

wire_matrix = np.array([[0, 1], [2, 3]])

node_ordering = {i:i for i in range(4)}

cost_h, mixer_h = qaoa.travelling_salesman(G, node_ordering, wire_matrix)

print(cost_h)
print("-------")
print(mixer_h)

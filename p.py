import pennylane as qml
from pennylane import qaoa
from networkx import Graph
from matplotlib import pyplot as plt

# Defines the wires and the graph on which MaxCut is being performed

wires = range(3)
graph = Graph([(0, 1), (1, 2), (2, 0)])

# Defines the QAOA cost and mixer Hamiltonians

cost_h, mixer_h = qaoa.min_vertex_cover(graph)

# Defines a layer of the QAOA ansatz, from the cost and mixer Hamiltonians

def qaoa_layer(gamma, alpha):
    qaoa.cost_layer(gamma, cost_h)
    qaoa.mixer_layer(alpha, mixer_h)

# Repeatedly applies layers of the QAOA ansatz

def circuit(params):

    for w in wires:
        qml.Hadamard(wires=w)

    qml.layer(qaoa_layer, 2, params[0], params[1])

# Defines the device and the QAOA cost function

dev = qml.device('default.qubit', wires=len(wires))
cost_function = qml.VQECost(circuit, cost_h, dev)

# Creates the optimizer

optimizer = qml.GradientDescentOptimizer()
steps = 30
params = [[0.7, 0.5], [1.5, -0.4]]

for i in range(30):
    params = optimizer.step(cost_function, params)
    print(i)

@qml.qnode(dev)
def dist_circuit(gamma, alpha):
    circuit([gamma, alpha])
    return qml.probs(wires=wires)

output = dist_circuit(params[0], params[1])

plt.bar(range(2**3), output)
plt.show()


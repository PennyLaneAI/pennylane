"""Qubit optimization example for the ProjectQ plugin.

In this demo, we perform rotation on one qubit, entangle it via a CNOT
gate to a second qubit, and then measure the second qubit projected onto PauliZ.
We then optimize the circuit such the resulting expectation value is 1.
"""
import openqml as qm
from openqml import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--backend", default='simulator', choices=['simulator', 'ibm'], help="backend to use")
parser.add_argument("--user", help="IBM Quantum Experience user name")
parser.add_argument("--password", help="IBM Quantum Experience password")
parser.add_argument("--optimizer", default="SGD", choices=qm.optimizer.OPTIMIZER_NAMES, help="optimizer to use")
args = parser.parse_args()

dev1 = qm.device('projectq.'+args.backend, wires=2, **vars(args))

@qm.qfunc(dev1)
def circuit(x, y, z):
    """Single quit rotation and CNOT

    Args:
        x (float): x rotation angle
        y (float): y rotation angle
        z (float): z rotation angle
    """
    qm.RZ(z, wires=[0])
    qm.RY(y, wires=[0])
    qm.RX(x, wires=[0])
    qm.CNOT(wires=[0, 1])
    return qm.expectation.PauliZ(wires=1)

circuit = qm.QNode(circuit, dev1)

def cost(weights):
    """Cost (error) function to be minimized.

    Args:
        weights (float): weights
    """
    return np.abs(circuit(*weights)-1)


# initialize x with random value
init_weights = np.random.randn(3)
o = qm.Optimizer(cost, init_weights, optimizer=args.optimizer)

# train the circuit
c = o.train(max_steps=100)

# print the results
print('Initial [x,y,z] rotation angles:', init_weights)
print('Optimized [x,y,z] rotation angles:', o.weights)
print('Circuit output at rotation angles:', circuit(*o.weights))
print('Circuit gradient at rotation angles:', qm.grad(circuit, o.weights)[0])
print('Cost gradient at optimized parameters:', qm.grad(cost, [o.weights, None]))

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
    """QNode"""
    qm.RZ(z, [0])
    qm.RY(y, [0])
    qm.RX(x, [0])
    qm.CNOT([0, 1])
    return qm.expectation.PauliZ(wires=1)

circuit = qm.QNode(circuit, dev1)

def cost(xyz, batched):
    """Cost (error) function to be minimized."""
    return np.abs(circuit(*xyz)-1)


# initialize x with random value
xyz0 = np.random.randn(3)
o = qm.Optimizer(cost, xyz0, optimizer=args.optimizer)

# train the circuit
c = o.train(max_steps=100)

# print the results
print('Initial rotation angles:', xyz0)
print('Optimized rotation angles:', o.weights)
print('Circuit output at rotation angles:', circuit(*o.weights))
print('Circuit gradient at rotation angles:', qm.grad(circuit, o.weights)[0])

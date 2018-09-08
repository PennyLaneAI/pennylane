"""Photon redirection example.

In this demo we optimize an optical quantum circuit to redirect a photon from mode 1 to mode 2.
"""
import openqml as qm
from openqml import numpy as np

dev1 = qm.device('strawberryfields.fock', wires=2, cutoff_dim=5)

@qm.qfunc(dev1)
def circuit(theta):
    """Beamsplitter with single photon incident on mode 1

    Args:
        theta (float): beamsplitter angle
    """
    qm.FockState(1, wires=0)
    qm.Beamsplitter(theta, 0, wires=[0, 1])
    qm.expectation.Fock(wires=1)

def cost(theta, batched):
    """Cost (error) function to be minimized.

    Args:
        theta (float): beamsplitter angle
    """
    return np.abs(circuit(theta)-1)

# initialize theta with random value
theta0 = np.random.randn(1)
o = qm.Optimizer(cost, theta0, optimizer='BFGS')

# train the circuit
c = o.train(max_steps=100)

# print the results
print('Initial beamsplitter angle:', theta0)
print('Optimized beamsplitter angle:', o.weights)
print('Circuit output at optimized angle:', circuit(*o.weights))
print('Circuit gradient at optimized angle:', qm.grad(circuit, o.weights)[0])

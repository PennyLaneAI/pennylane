"""Photon redirection example.

In this demo we optimize an optical quantum circuit to redirect a photon from mode 0 to mode 1.
"""
import openqml as qm
from openqml import numpy as np

dev = qm.device('strawberryfields.fock', wires=2, cutoff_dim=5)
#dev = qm.device('strawberryfields.gaussian', wires=2)


def circuit(theta):
    """Here we define the quantum circuit.

    Beamsplitter with a single photon incident on mode 0, measure the photon number in mode 1.

    Args:
        theta (float): beamsplitter angle
    """
    qm.FockState(1, wires=0)
    qm.Beamsplitter(theta, 0, wires=[0, 1])
    return qm.expectation.PhotonNumber(wires=1)

q = qm.QNode(circuit, dev)

# initialize theta with random value
theta0 = np.array([np.pi * 0.2]) # np.random.randn(1)

print('Initial beamsplitter angle: {} * pi'.format(theta0/np.pi))
print('initial val: ', q(*theta0))
print(q.grad_method_for_par)

grad_F = q.gradient(theta0, method='F')
print('initial grad_F: ', grad_F)
grad_A = q.gradient(theta0)
print('initial grad_A: ', grad_A)

import sys
sys.exit() # TODO remove when the gradients are fixed so that they agree

def cost(theta):
    """Cost (error) function to be minimized.

    Args:
        theta (array[float]): optimization parameters
    """
    return (q(theta[0]) -1) ** 2

# create the optimizer
o = qm.Optimizer(cost, theta0, optimizer='BFGS')
# train the circuit
c = o.train(max_steps=100)

# print the results
print('Initial beamsplitter angle: {} * pi'.format(theta0/np.pi))
print('Optimized beamsplitter angle: {} * pi'.format(o.weights/np.pi))
print('Circuit output at optimized angle:', q(o.weights))
print('Circuit gradient at optimized angle:', qm.grad(q, o.weights))

"""Photon redirection example.

In this demo we optimize an optical quantum circuit to redirect a photon from mode 1 to mode 2.
"""
import openqml as qm
from openqml import numpy as np

dev1 = qm.device('strawberryfields.fock', wires=2, cutoff_dim=5)

class Beamsplitter(qm.Beamsplitter):
    grad_method = 'F'

# @qm.qfunc(dev1)
def circuit(theta):
    """Beamsplitter with single photon incident on mode 1

    Args:
        theta (float): beamsplitter angle
    """
    qm.FockState(1, wires=0)
    # BUG: circuit gradient only working if BS set to `method='F'`
    # otherwise the circuit gradient is always 0 irrespective of theta
    Beamsplitter(theta, 0, wires=[0, 1])
    return qm.expectation.Fock(wires=1)

circuit = qm.QNode(circuit, dev1)

def cost(theta):
    """Cost (error) function to be minimized.

    Args:
        theta (float): beamsplitter angle
    """
    return np.abs(circuit(*theta)-1)

# initialize theta with random value
theta0 = np.random.randn(1)
o = qm.Optimizer(cost, theta0, optimizer='SGD')

# train the circuit
c = o.train(max_steps=100)

# import autograd
# grad = autograd.grad(cost, 0)
# print(grad([1.], None), circuit.gradient([1.], method='A'), circuit.gradient([1.], method='F'))

# print the results
print('Initial beamsplitter angle:', theta0)
print('Optimized beamsplitter angle:', o.weights)
print('Circuit output at optimized angle:', circuit(*o.weights))
print('Circuit gradient at optimized angle:', qm.grad(circuit, o.weights))

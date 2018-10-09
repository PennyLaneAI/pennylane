"""Photon redirection example.

In this demo we optimize an optical quantum circuit to redirect a photon from mode 0 to mode 1.
"""
import openqml as qm
from openqml import numpy as np

np.set_printoptions(precision=5)

# construct a device to run the circuit on
dev = qm.device('strawberryfields.fock', wires=2, cutoff_dim=5)

def circuit(theta):
    """Here we define the quantum circuit.

    Beamsplitter with a single photon incident on mode 0, measure the photon number in mode 1.

    Args:
        theta (float): beamsplitter angle
    """
    qm.FockState(1, wires=0)
    qm.Beamsplitter(theta, 0, wires=[0, 1])
    return qm.expectation.PhotonNumber(wires=1)

# a quantum node represents a combination of a circuit and a device that evaluates it
q = qm.QNode(circuit, dev)

# initialize theta with a random value
theta0 = np.random.randn(1)
print('Initial beamsplitter angle: {} * pi'.format(theta0/np.pi))
print('initial circuit output: ', q(*theta0))

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
print('Circuit output at optimized angle: {:.5g}'.format(q(o.weights)))
print('Circuit gradient at optimized angle: {:.5g}'.format(qm.grad(q, o.weights)))

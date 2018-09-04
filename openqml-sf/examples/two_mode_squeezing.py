"""Two mode squeezing example.

In this demo we optimize an optical quantum circuit such that the mean photon number at mode 1 is 1.
"""
import openqml as qm
from openqml import numpy as np

dev1 = qm.device('strawberryfields.gaussian', wires=2)

@qm.qfunc(dev1)
def circuit(alpha, r):
    """Two mode squeezing with PNR on mode 1

    Args:
        r (float): squeezing parameter
    """
    qm.Displacement(alpha, 0, wires=[0])
    qm.Squeezing(r, 0, wires=[0, 1])
    qm.expectation.Fock(wires=1)

def cost(weights, batched):
    """Cost (error) function to be minimized.

    Args:
        r (float): squeezing parameter
    """
    return np.abs(circuit(weights)-1)

# initialize r with random value
init_weights = np.random.randn(2)
o = qm.Optimizer(cost, init_weights, optimizer='Nelder-Mead')

# train the circuit
c = o.train(max_steps=100)

from autograd import grad

# print the results
print('Initial [alpha, r] parameters:', init_weights)
print('Optimized [alpha, r] parameter:', o.weights)
print('Circuit output at optimized parameters:', circuit(*o.weights))
print('Circuit gradient at optimized parameters:', qm.grad(circuit, [o.weights]))
print('Cost gradient at optimized parameters:', qm.grad(cost, [o.weights, None]))

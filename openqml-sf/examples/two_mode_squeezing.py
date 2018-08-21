"""Two mode squeezing example.

In this demo we optimize an optical quantum circuit such that the mean photon number at mode 1 is 1.
"""
import openqml as qm
from openqml import numpy as np

dev1 = qm.device('strawberryfields.gaussian', wires=2)

@qm.qfunc(dev1)
def circuit(r):
    """Two mode squeezing with PNR on mode 1

    Args:
        r (float): squeezing parameter
    """
    qm.Squeezing(r, 0, wires=[0, 1])
    qm.expectation.Fock(wires=1)

def cost(r, batched):
    """Cost (error) function to be minimized.

    Args:
        r (float): squeezing parameter
    """
    return np.abs(circuit(r)-1)

# initialize r with random value
r0 = np.random.randn(1)
o = qm.Optimizer(cost, r0, optimizer='Nelder-Mead')

# train the circuit
c = o.train(max_steps=100)

# print the results
print('Initial squeezing parameter:', r0)
print('Optimized squeezing parameter:', o.weights)
print('Circuit output at squeezing parameter:', circuit(*o.weights))
print('Circuit gradient at squeezing parameter:', qm.grad(circuit, *o.weights)[0])

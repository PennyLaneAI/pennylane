r"""
.. _barren_plateaus:

Barren plateaus in quantum neural networks
==========================================
*Author: Shahnawaz Ahmed (shahnawaz.ahmed95@gmail.com)*

In classical optimization, it is suggested that saddle
points, not local minima, provide a fundamental impediment
to rapid high-dimensional non-convex optimization
(Dauphin et al., 2014).

The problem of such barren plateaus manifests in a different
form in variational quantum circuits, which are at the heart
of techniques such as quantum neural networks or approximate
optimization e.g., QAOA (Quantum Adiabatic Optimization Algorithm)
which can be found in this `PennyLane QAOA tutorial
<https://pennylane.readthedocs.io/en/latest/tutorials/pennylane_run_qaoa_maxcut.html#qaoa-maxcut>`_.

While starting from a parameterized
random quantum circuit seems like a good unbiased choice if
we do not know the problem structure, McClean et al. (2018)
show that

*"for a wide class of reasonable parameterized quantum
circuits, the probability that the gradient along any
reasonable direction is non-zero to some fixed precision
is exponentially small as a function of the number
of qubits."*

Thus, randomly selected quantum circuits might not be the best
option to choose while implementing variational quantum
algorithms.


.. figure:: ../../examples/figures/barren_plateaus/surface.png
   :width: 90%
   :align: center
   :alt: surface

|

In this tutorial, we will show how randomized quantum circuits
face the problem of barren plateaus using PennyLane. We will
partly reproduce some of the findings in McClean et. al., 2018
with just a few lines of code.

.. note::

    **An initialization strategy to tackle barren plateaus**

    How do we avoid the problem of barren plateaus?
    In Grant et al. (2019), the authors present one strategy to
    tackle the barren plateau problem in randomized quantum circuits:

    *"The technique involves randomly selecting some of the initial
    parameter values, then choosing the remaining values so that
    the final circuit is a sequence of shallow unitary blocks that
    each evaluates to the identity. Initializing in this way limits
    the effective depth of the circuits used to calculate the first
    parameter update so that they cannot be stuck in a barren plateau
    at the start of training."*
"""

##############################################################################
# Exploring the barren plateau problem with PennyLane
# ---------------------------------------------------
#
# First, we import PennyLane, NumPy, and Matplotlib

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt


##################################################
# Next, we create a randomized variational circuit

num_qubits = 4
dev = qml.device("default.qubit", wires=num_qubits)
gate_set = [qml.RX, qml.RY, qml.RZ]
gate_sequence = {i: np.random.choice(gate_set) for i in range(num_qubits)}


def rand_circuit(params, random_gate_sequence=None, num_qubits=None):
    """A random variational quantum circuit.

    Args:
        params (array[float]): array of parameters
        random_gate_sequence (dict): a dictionary of random gates
        num_qubits (int): the number of qubits in the circuit

    Returns:
        float: the expectation value of the target observable
    """
    for i in range(num_qubits):
        qml.RY(np.pi / 4, wires=i)

    for i in range(num_qubits):
        random_gate_sequence[i](params[i], wires=i)

    for i in range(num_qubits - 1):
        qml.CZ(wires=[i, i + 1])

    H = np.zeros((2 ** num_qubits, 2 ** num_qubits))
    H[0, 0] = 1
    wirelist = [i for i in range(num_qubits)]
    return qml.expval(qml.Hermitian(H, wirelist))


############################################################
# Now we can compute the gradient and calculate the variance.
# While we only take 200 samples to allow the code to run in
# a reasonable amount of time, this can be increased
# for more accurate results.

qcircuit = qml.QNode(rand_circuit, dev)
grad = qml.grad(qcircuit, argnum=0)

num_samples = 200
grad_vals = []

for i in range(num_samples):
    params = np.random.uniform(0, 2 * np.pi, size=num_qubits)
    grad_vals.append(grad(params, random_gate_sequence=gate_sequence, num_qubits=num_qubits))

print("Variance of the gradient for {} samples: {}".format(num_samples, np.var(grad_vals)))


###########################################################
# Evaluate the gradient for more qubits
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We can repeat the above analysis with increasing number
# of qubits.


def generate_random_circuit(num_qubits):
    """
    Generates a random quantum circuit based on (McClean et. al., 2019).

    Args:
        num_qubits (int): the number of qubits in the circuit
    """
    dev = qml.device("default.qubit", wires=num_qubits)
    gate_set = [qml.RX, qml.RY, qml.RZ]
    random_gate_sequence = {i: np.random.choice(gate_set) for i in range(num_qubits)}
    qcircuit = qml.QNode(rand_circuit, dev)

    return qcircuit, random_gate_sequence


qubits = [2, 3, 4, 5, 6]
variances = []

num_samples = 200


for num_qubits in qubits:
    qcircuit, gate_sequence = generate_random_circuit(num_qubits)
    grad = qml.grad(qcircuit, argnum=0)
    grad_vals = []
    for i in range(num_samples):
        params = np.random.uniform(0, 2 * np.pi, size=num_qubits)
        grad_vals.append(grad(params, random_gate_sequence=gate_sequence, num_qubits=num_qubits))
    vargrad = np.var(grad_vals)
    variances.append(vargrad)
    print("Variance of the gradient for {} qubits: {}".format(num_qubits, vargrad))
    

variances = np.array(variances)
qubits = np.array(qubits)


# Fit the semilog plot to a straight line
p = np.polyfit(qubits, np.log(variances), 1)


# Plot the straight line fit to the semilog
plt.semilogy(qubits, variances, "o")
plt.semilogy(qubits, np.exp(p[0] * qubits + p[1]), "o-.",
             label="Slope {:3.2f}".format(p[0]))
plt.xlabel(r"N Qubits")
plt.ylabel(r"$\langle \partial \theta_{1, 1} E\rangle$ variance")
plt.legend()
plt.show()



##############################################################################
# This tutorial was generated using the following PennyLane version:

qml.about()


##############################################################################
# References
# ----------
#
# 1. Dauphin, Yann N., et al.,
#    Identifying and attacking the saddle point problem in high-dimensional non-convex
#    optimization. Advances in Neural Information Processing
#    systems (2014).
#
# 2. McClean, Jarrod R., et al.,
#    Barren plateaus in quantum neural network training landscapes.
#    Nature communications 9.1 (2018): 4812.
#
# 3. Grant, Edward, et al.
#    An initialization strategy for addressing barren plateaus in
#    parametrized quantum circuits. arXiv preprint arXiv:1903.05076 (2019).

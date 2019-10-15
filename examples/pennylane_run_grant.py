r"""
.. _grant:

An initialization strategy to address barren plateaus in quantum neural networks
================================================================================
*Author: Shahnawaz Ahmed (shahnawaz.ahmed95@gmail.com)*

A random variational quantum circuit suffers from the problem of barren plateaus
as we see in this `PennyLane tutorial
<https://pennylane.readthedocs.io/en/latest/tutorials/pennylane_run_barren_plateaus.html#barren_plateaus>`_.

One way to deal with such barren plateaus is proposed as
*"randomly selecting some of the initial
  parameter values, then choosing the remaining values so that
  the final circuit is a sequence of shallow unitary blocks that
  each evaluates to the identity. Initializing in this way limits
  the effective depth of the circuits used to calculate the first
  parameter update so that they cannot be stuck in a barren plateau
  at the start of training."*

In this tutorial we will make a simple quantum circuit implementing this
initialization strategy.

"""

##############################################################################
# Revisiting the barren plateau problem with PennyLane
# ---------------------------------------------------
#
# First, we import PennyLane, NumPy, and Matplotlib

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt


##################################################
# We create a randomized variational circuit

# Set a seed for reproducibility
np.random.seed(22)


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


# qubits = [2, 3, 4, 5, 6]
# variances = []

# num_samples = 200


# for num_qubits in qubits:
#     qcircuit, gate_sequence = generate_random_circuit(num_qubits)
#     grad = qml.grad(qcircuit, argnum=0)
#     grad_vals = []
#     for i in range(num_samples):
#         params = np.random.uniform(0, 2 * np.pi, size=num_qubits)
#         grad_vals.append(grad(params, random_gate_sequence=gate_sequence, num_qubits=num_qubits))
#     vargrad = np.var(grad_vals)
#     variances.append(vargrad)
    

# variances = np.array(variances)
# qubits = np.array(qubits)


# # Fit the semilog plot to a straight line
# p = np.polyfit(qubits, np.log(variances), 1)


# # Plot the straight line fit to the semilog
# plt.semilogy(qubits, variances, "o")
# plt.semilogy(qubits, np.exp(p[0] * qubits + p[1]), "o-.",
#              label="Slope {:3.2f}".format(p[0]))
# plt.xlabel(r"N Qubits")
# plt.ylabel(r"$\langle \partial \theta_{1, 1} E\rangle$ variance")
# plt.legend()
# plt.show()


##############################################################################
# Initialization with identity blocks
# -----------------------------------


def identity_block_circuit(params, random_gate_sequence=None, num_qubits=None):
    """A random variational quantum circuit with an initialization strategy
    using identity blocks

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

    # Identity blocks
    for i in range(num_qubits - 1):
        qml.CZ(wires=[i + 1, i])

    for i in range(num_qubits):
        random_gate_sequence[i](-params[i], wires=i)

    for i in range(num_qubits):
        qml.RY(-np.pi / 4, wires=i)


    H = np.zeros((2 ** num_qubits, 2 ** num_qubits))
    H[0, 0] = 1
    wirelist = [i for i in range(num_qubits)]
    return qml.expval(qml.Hermitian(H, wirelist))


def generate_random_identity_circuit(num_qubits):
    """
    Generates a random quantum circuit based on (McClean et. al., 2019).

    Args:
        num_qubits (int): the number of qubits in the circuit
    """
    dev = qml.device("default.qubit", wires=num_qubits)
    gate_set = [qml.RX, qml.RY, qml.RZ]
    random_gate_sequence = {i: np.random.choice(gate_set) for i in range(num_qubits)}
    qcircuit = qml.QNode(identity_block_circuit, dev)

    return qcircuit, random_gate_sequence


qubits = [2, 3, 4, 5, 6, 7]
variances = []

num_samples = 500


for num_qubits in qubits:
    qcircuit, gate_sequence = generate_random_identity_circuit(num_qubits)
    grad = qml.grad(qcircuit, argnum=0)
    grad_vals = []
    for i in range(num_samples):
        params = np.random.uniform(0, 2 * np.pi, size=num_qubits)
        grad_vals.append(grad(params, random_gate_sequence=gate_sequence, num_qubits=num_qubits))
    vargrad = np.var(grad_vals)
    variances.append(vargrad)
    

variances = np.array(variances)
qubits = np.array(qubits)


plt.semilogy(qubits, np.exp(p[0] * qubits + p[1]), "-",
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

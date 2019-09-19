"""
.. _qaoa_maxcut:

QAOA for MaxCut
===============

In this tutorial, we demonstrate how to sample joint qubit measurements from a
variational quantum circuit to solve a combinatorial optimization problem. Namely,
we implement the quantum approximate optimization algorithm (QAOA) for the MaxCut
problem as proposed by `Farhi, Goldstone, and Gutmann (2014) <https://arxiv.org/abs/1411.4028>__`
"""

##############################################################################
# Background
# ----------
#
# The MaxCut problem
# ~~~~~~~~~~~~~~~~~~
# The aim of MaxCut is to maximize the number of edges (yellow lines) in a graph that are "cut" by
# a given partition of the vertices (blue circles) into two sets as shown in the figure below.
# |
#
# .. figure:: ../../examples/figures/qaoa_maxcut_partition.png
#    :align: center
#    :scale: 65%
#    :alt: qaoa_operators
#
# |
# Let's say there are :math:`m` edges. We seek the partition :math:`z` of the vertices into two sets
# 0 and 1 which maximizes
#
# .. math::
#   C(z) = \sum_{\alpha=1}^{m}C_\alpha(z)
#
# where :math:`C_\alpha(z)=1` if :math:`z` places one vertex from the
# :math:`\alpha^\text{th}` edge in partition 0 and the other in partition 1, and :math:`C_\alpha(z)=0` otherwise.
# The goal of approximate optimization in this case is to find a partition :math:`z` which
# yields a value for :math:`C(z)` that is close to the maximum possible value.
#
# For instance,
# in the situation depicted in the figure above, the optimal value of :math:`C(z)` is
# 4. Such a situation can be represented by the bitstring :math:`z=1010\text{,}`
# indicating that the :math:`0^{\text{th}}` and :math:`2^{\text{nd}}` vertices are in one partition
# while the :math:`1^{\text{st}}` and :math:`3^{\text{rd}}` are in
# the other. (The inverse partition, :math:`z=0101` is of course equally valid.) In the following sections,
# we will represent partitions using computational basis states and use PennyLane to
# rediscover this optimal cut.
#
# A quantum circuit
# ~~~~~~~~~~~~~~~~~
# This section presents the operators used in the QAOA algorithm and their implementation using basic unitary gates.
# Firstly, if the partition is given by the computational basis states, we can represent the terms in the
# objective function as operators like so
#
# .. math::
#   C_\alpha = \frac{1}{2}\left(1-\sigma_{z}^j\sigma_{z}^k\right)
#
# where the :math:`\alpha\text{th}` edge is between qubits/vertices :math:`(j,k)`.
# :math:`C_\alpha` is 1 if and only if the :math:`i\text{th}` and :math:`j\text{th}`
# qubits have different z-axis measurement values, representing separate partitions.
# The objective function :math:`C` is now a diagonal operator with integer eigenvalues.
#
# QAOA starts with a uniform superposition over the :math:`n` bitstring basis states,
#
# .. math::
#   |+_{n}\rangle = \frac{1}{\sqrt{2^n}}\sum_{z=0}^{2^n-1} |z\rangle
#
#
# We aim to explore the space of bitstring states for a superposition which is likely to yield a
# large value for the :math:`C` operator upon performing a measurement in the computational basis.
# Using the :math:`2p` angle parameters
# :math:`\boldsymbol{\gamma} = \gamma_1\gamma_2...\gamma_p`, :math:`\boldsymbol{\beta} = \beta_1\beta_2...\beta_p`
# we perform a sequence of operations on our initial state:
#
# .. math::
#   |\boldsymbol{\gamma},\boldsymbol{\beta}\rangle = U_{B_p}U_{C_p}U_{B_{p-1}}U_{C_{p-1}}...U_{B_1}U_{C_1}|+_n\rangle
#
# where the operators have the explicit forms
#
# .. math::
#   U_{B_i} &= e^{-i\beta_iB} = \prod_{j=1}^n e^{-i\beta_i\sigma_x^j} \\
#   U_{C_i} &= e^{-i\gamma_iC} = \prod_{\text{edge (j,k)}} e^{-i\gamma_i(1-\sigma_z^j\sigma_z^k)/2}
#
# In other words, we make :math:`p` layers of our parametrized :math:`U_bU_C` gates.
# These can be implemented on a quantum circuit using the gates depicted below (up to an irrelevant constant
# that gets absorbed into the parameters).
# |
#
# .. figure:: ../../examples/figures/qaoa_operators.png
#    :align: center
#    :scale: 100%
#    :alt: qaoa_operators
#
# |
# Let :math:`\langle \boldsymbol{\gamma},
# \boldsymbol{\beta} | C | \boldsymbol{\gamma},\boldsymbol{\beta} \rangle` be the expectation of our objective.
# In the next section, we will use PennyLane to sample this expectation value and perform classical optimization
# over the parameters. This will specify a state :math:`|\boldsymbol{\gamma},\boldsymbol{\beta}\rangle` which is
# likely to yield an approximately optimal partition :math:`|z\rangle` upon performing a measurement in the
# computational basis.
# In the case of the graph shown above, we want to measure either 0101 or 1010 from our state.
#
# .. figure:: ../../examples/figures/qaoa_optimal_state.png
#   :align: center
#   :scale: 60%
#   :alt: optimal_state
#
#
# Implementing QAOA in PennyLane
# ------------------------------
#
# Imports and setup
# ~~~~~~~~~~~~~~~~~
#
# To get started, we import PennyLane along with the PennyLane-provided
# version of NumPy.


import pennylane as qml
from pennylane import numpy as np


##############################################################################
# Operators
# ~~~~~~~~~
# We specify the number of qubits (vertices) with ``n_wires`` and
# compose the unitary operators using the definitions
# above. :math:`U_B` operators act on each of the wires, while :math:`U_C`
# operators act on wires whose corresponding vertices are joined by an edge in
# the graph. We also define the graph using
# the list ``graph``, which contains the tuples of vertices defining
# each edge in the graph.

n_wires = 4
graph = [(0, 1), (0, 3), (1, 2), (2, 3)]

# unitary operator U_B with parameter beta
def U_B(beta):
    for wire in range(n_wires):
        qml.RX(2 * beta, wires=wire)


# unitary operator U_C with parameter gamma
def U_C(gamma):
    for edge in graph:
        wire1 = edge[0]
        wire2 = edge[1]
        qml.CNOT(wires=[wire1, wire2])
        qml.RZ(gamma, wires=wire2)
        qml.CNOT(wires=[wire1, wire2])


##############################################################################
# With some foresight, we notice that we will need a way to sample
# a measurement of multiple qubits in the computational basis, so we define
# a Hermitian operator to do this. The eigenvalues of the operator are
# the joint qubit measurement values in integer form.


def comp_basis_measurement(wires):
    n_wires = len(wires)
    return qml.Hermitian(np.diag(range(2 ** n_wires)), wires=wires)


##############################################################################
# Circuit
# ~~~~~~~
# Next, we create a quantum device with 4 qubits.

dev1 = qml.device("default.qubit", wires=n_wires, analytic=True, shots=1)

##############################################################################
# We also require a quantum node which will apply the operators according to the
# angle parameters, and return the expectation value of the observable
# :math:`\sigma_z^{j}\sigma_z^{k}` to be used in each term of the objective function later on. The
# argument ``edge`` specifies the chosen edge term in the objective function, :math:`(j,k)`.
# Once optimized, the same quantum node can be used for sampling an approximately optimal bitstring
# if executed with the ``edge`` keyword set to None.
#
#  We can also adjust the number of layers in the circuit :math:`p` using a keyword, `n_layers`.

pauli_z = [[1, 0], [0, -1]]
pauli_z_2 = np.kron(pauli_z, pauli_z)


@qml.qnode(dev1)
def circuit(params, edge=None, n_layers=1):
    # apply hadamards to get n qubit |+> state
    for wire in range(n_wires):
        qml.Hadamard(wires=wire)
    # p instances of unitary operators
    for i in range(n_layers):
        U_C(params[0][i])
        U_B(params[1][i])
    # during measurement phase we are evaluating
    # the circuit with optimized parameters
    if edge is None:
        return qml.sample(comp_basis_measurement(range(n_wires)))
    # during the optimization phase we are evaluating a term
    # in the objective using exp val
    return qml.expval(qml.Hermitian(pauli_z_2, wires=edge))


##############################################################################
# Optimization
# ~~~~~~~~~~~~
# Finally, we optimize the objective over the
# angle parameters :math:`\boldsymbol{\gamma}` (``params[0]``) and :math:`\boldsymbol{\beta}`
# (``params[1]``).
# and then sample the optimized
# circuit multiple times to yield a distribution of bitstrings. One of the optimal partitions
# (:math:`z=0101` or :math:`z=1010`) should be the most frequently sampled bitstring.


def qaoa_maxcut(n_layers=1):
    print("\np={:d}".format(n_layers))

    # initialize the parameters near zero
    init_params = 0.01 * np.random.rand(2, n_layers)

    # minimize negative the objective
    def objective(params):
        neg_obj = 0
        for edge in graph:
            # objective for the maxcut problem
            neg_obj -= 0.5 * (1 - circuit(params, edge=edge, n_layers=n_layers))
        return neg_obj

    # initialize optimizer
    opt = qml.AdagradOptimizer(stepsize=0.5)

    # optimize parameters in objective
    params = init_params
    steps = 30
    for i in range(steps):
        params = opt.step(objective, params)
        if (i + 1) % 5 == 0:
            print("Objective after step {:5d}: {: .7f}".format(i + 1, -objective(params)))

    # sample measured bitstrings 100 times
    bit_strings = []
    n_samples = 100
    for i in range(0, n_samples):
        bit_strings.append(int(circuit(params, edge=None, n_layers=n_layers)))

    # print optimal parameters and most frequently sampled bitstring
    counts = np.bincount(np.array(bit_strings))
    most_freq_bit_string = np.argmax(counts)
    print("Optimized (gamma,beta) vectors: {}".format(params))
    print("Most frequently sampled bit string is: {:04b}".format(most_freq_bit_string))

    return -objective(params), bit_strings


# perform qaoa on our graph with p=1,2 and
# keep the bitstring sample lists
bitstrings1 = qaoa_maxcut(n_layers=1)[1]
bitstrings2 = qaoa_maxcut(n_layers=2)[1]

##############################################################################
# In the case where we set `n_layers=2`, we recover the optimal
# objective function :math:`C=4`

##############################################################################
# Plotting the results
# --------------------
# We can plot the distribution of measurements we got from the optimized circuits. As
# expected for this graph, the partitions 0101 and 1010 are measured with the highest frequencies,
# and in the case where we set ``p=2`` we obtain one of the optimal partitions with 100% certainty.

import matplotlib.pyplot as plt

xticks = range(0, 16)
xtick_labels = list(map(lambda x: format(x, "04b"), xticks))
bins = np.arange(0, 17) - 0.5

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title("p=1")
plt.xlabel("bitstrings")
plt.ylabel("freq.")
plt.xticks(xticks, xtick_labels, rotation="vertical")
plt.hist(bitstrings1, bins=bins)
plt.subplot(1, 2, 2)
plt.title("p=2")
plt.xlabel("bitstrings")
plt.ylabel("freq.")
plt.xticks(xticks, xtick_labels, rotation="vertical")
plt.hist(bitstrings2, bins=bins)
plt.tight_layout()
plt.show()

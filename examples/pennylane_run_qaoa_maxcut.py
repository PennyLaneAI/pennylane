"""
.. _qaoa_maxcut:

QAOA for MaxCut
===============

In this tutorial we implement the quantum approximate optimization algorithm (QAOA) for the MaxCut
problem as proposed by `Farhi, Goldstone, and Gutmann (2014) <https://arxiv.org/abs/1411.4028>`__. First, we
give an overview of the MaxCut problem using a simple example, a graph with 4 vertices and 4 edges. We then
show how to find the maximum cut by running the QAOA algorithm using PennyLane.
"""

##############################################################################
# Background
# ----------
#
# The MaxCut problem
# ~~~~~~~~~~~~~~~~~~
# The aim of MaxCut is to maximize the number of edges (yellow lines) in a graph that are "cut" by
# a given partition of the vertices (blue circles) into two sets (see figure below).
#
# .. figure:: ../../examples/figures/qaoa_maxcut_partition.png
#    :align: center
#    :scale: 65%
#    :alt: qaoa_operators
#
# |
# Consider a graph with :math:`m` edges and :math:`n` vertices. We seek the partition
# :math:`z` of the vertices into two sets
# :math:`A` and :math:`B` which maximizes
#
# .. math::
#   C(z) = \sum_{\alpha=1}^{m}C_\alpha(z),
#
# where :math:`C` counts the number of edges cut. :math:`C_\alpha(z)=1` if :math:`z` places one vertex from the
# :math:`\alpha^\text{th}` edge in set :math:`A` and the other in set :math:`B`, and :math:`C_\alpha(z)=0` otherwise.
# Finding a cut which yields the maximum possible value of :math:`C` is an NP-complete problem, so our best hope for a
# polynomial-time algorithm lies in an approximate optimization.
# In the case of MaxCut, this means finding a partition :math:`z` which
# yields a value for :math:`C(z)` that is close to the maximum possible value.
#
# We can represent the assignment of vertices to set :math:`A` or :math:`B` using a bitstring,
# :math:`z=z_1...z_n` where :math:`z_i=0` if the :math:`i^\text{th}` vertex is in :math:`A` and
# :math:`z_i = 1` if it is in :math:`B`. For instance,
# in the situation depicted in the figure above the bitstring representation is :math:`z=0101\text{,}`
# indicating that the :math:`0^{\text{th}}` and :math:`2^{\text{nd}}` vertices are in :math:`A`
# while the :math:`1^{\text{st}}` and :math:`3^{\text{rd}}` are in
# :math:`B`. This assignment yields a value for the objective function (the number of yellow lines cut)
# :math:`C=4`, which turns out to be the maximum cut. In the following sections,
# we will represent partitions using computational basis states and use PennyLane to
# rediscover this maximum cut.
#
# .. note:: In the graph above, :math:`z=1010` could equally well serve as the maximum cut.
#
# A circuit for QAOA
# ~~~~~~~~~~~~~~~~~~~~
# This section describes implementing a circuit for QAOA using basic unitary gates to find approximate
# solutions to the MaxCut problem.
# Firstly, denoting the partitions using computational basis states :math:`|z\rangle`, we can represent the terms in the
# objective function as operators acting on these states
#
# .. math::
#   C_\alpha = \frac{1}{2}\left(1-\sigma_{z}^j\sigma_{z}^k\right),
#
# where the :math:`\alpha\text{th}` edge is between vertices :math:`(j,k)`.
# :math:`C_\alpha` has eigenvalue 1 if and only if the :math:`j\text{th}` and :math:`k\text{th}`
# qubits have different z-axis measurement values, representing separate partitions.
# The objective function :math:`C` can be considered a diagonal operator with integer eigenvalues.
#
# QAOA starts with a uniform superposition over the :math:`n` bitstring basis states,
#
# .. math::
#   |+_{n}\rangle = \frac{1}{\sqrt{2^n}}\sum_{z\in \{0,1\}^n} |z\rangle.
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
#   U_{B_l} &= e^{-i\beta_lB} = \prod_{j=1}^n e^{-i\beta_l\sigma_x^j}, \\
#   U_{C_l} &= e^{-i\gamma_lC} = \prod_{\text{edge (j,k)}} e^{-i\gamma_l(1-\sigma_z^j\sigma_z^k)/2}.
#
# In other words, we make :math:`p` layers of parametrized :math:`U_bU_C` gates.
# These can be implemented on a quantum circuit using the gates depicted below, up to an irrelevant constant
# that gets absorbed into the parameters.
#
# .. figure:: ../../examples/figures/qaoa_operators.png
#    :align: center
#    :scale: 100%
#    :alt: qaoa_operators
#
# |
# Let :math:`\langle \boldsymbol{\gamma},
# \boldsymbol{\beta} | C | \boldsymbol{\gamma},\boldsymbol{\beta} \rangle` be the expectation of the objective operator.
# In the next section, we will use PennyLane to perform classical optimization
# over the circuit parameters :math:`(\boldsymbol{\gamma}, \boldsymbol{\beta})`.
# This will specify a state :math:`|\boldsymbol{\gamma},\boldsymbol{\beta}\rangle` which is
# likely to yield an approximately optimal partition :math:`|z\rangle` upon performing a measurement in the
# computational basis.
# In the case of the graph shown above, we want to measure either 0101 or 1010 from our state since these correspond to
# the optimal partitions.
#
# .. figure:: ../../examples/figures/qaoa_optimal_state.png
#   :align: center
#   :scale: 60%
#   :alt: optimal_state
#
# |
# Qualitatively, QAOA tries to evolve the initial state into the plane of the
# :math:`|0101\rangle`, :math:`|1010\rangle` basis states (see figure above).
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
# above. :math:`U_B` operators act on individual wires, while :math:`U_C`
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
# We will need a way to sample
# a measurement of multiple qubits in the computational basis, so we define
# a Hermitian operator to do this. The eigenvalues of the operator are
# the qubit measurement values in integer form.


def comp_basis_measurement(wires):
    n_wires = len(wires)
    return qml.Hermitian(np.diag(range(2 ** n_wires)), wires=wires)


##############################################################################
# Circuit
# ~~~~~~~
# Next, we create a quantum device with 4 qubits.

dev = qml.device("default.qubit", wires=n_wires, analytic=True, shots=1)

##############################################################################
# We also require a quantum node which will apply the operators according to the
# angle parameters, and return the expectation value of the observable
# :math:`\sigma_z^{j}\sigma_z^{k}` to be used in each term of the objective function later on. The
# argument ``edge`` specifies the chosen edge term in the objective function, :math:`(j,k)`.
# Once optimized, the same quantum node can be used for sampling an approximately optimal bitstring
# if executed with the ``edge`` keyword set to ``None``. Additionally, we specify the number of layers
# (repeated applications of :math:`U_BU_C`) using the keyword ``n_layers``.

pauli_z = [[1, 0], [0, -1]]
pauli_z_2 = np.kron(pauli_z, pauli_z)


@qml.qnode(dev)
def circuit(gammas, betas, edge=None, n_layers=1):
    # apply Hadamards to get the n qubit |+> state
    for wire in range(n_wires):
        qml.Hadamard(wires=wire)
    # p instances of unitary operators
    for i in range(n_layers):
        U_C(gammas[i])
        U_B(betas[i])
    if edge is None:
        # measurement phase
        return qml.sample(comp_basis_measurement(range(n_wires)))
    # during the optimization phase we are evaluating a term
    # in the objective using expval
    return qml.expval(qml.Hermitian(pauli_z_2, wires=edge))


##############################################################################
# Optimization
# ~~~~~~~~~~~~
# Finally, we optimize the objective over the
# angle parameters :math:`\boldsymbol{\gamma}` (``params[0]``) and :math:`\boldsymbol{\beta}`
# (``params[1]``)
# and then sample the optimized
# circuit multiple times to yield a distribution of bitstrings. One of the optimal partitions
# (:math:`z=0101` or :math:`z=1010`) should be the most frequently sampled bitstring.
# We perform a maximization of :math:`C` by
# minimizing :math:`-C`, following the convention that optimizations are cast as minimizations
# in PennyLane.


def qaoa_maxcut(n_layers=1):
    print("\np={:d}".format(n_layers))

    # initialize the parameters near zero
    init_params = 0.01 * np.random.rand(2, n_layers)

    # minimize the negative of the objective function
    def objective(params):
        gammas = params[0]
        betas = params[1]
        neg_obj = 0
        for edge in graph:
            # objective for the MaxCut problem
            neg_obj -= 0.5 * (1 - circuit(gammas, betas, edge=edge, n_layers=n_layers))
        return neg_obj

    # initialize optimizer: Adagrad works well empirically
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
        bit_strings.append(int(circuit(params[0], params[1], edge=None, n_layers=n_layers)))

    # print optimal parameters and most frequently sampled bitstring
    counts = np.bincount(np.array(bit_strings))
    most_freq_bit_string = np.argmax(counts)
    print("Optimized (gamma, beta) vectors:\n{}".format(params))
    print("Most frequently sampled bit string is: {:04b}".format(most_freq_bit_string))

    return -objective(params), bit_strings


# perform qaoa on our graph with p=1,2 and
# keep the bitstring sample lists
bitstrings1 = qaoa_maxcut(n_layers=1)[1]
bitstrings2 = qaoa_maxcut(n_layers=2)[1]

##############################################################################
# In the case where we set ``n_layers=2``, we recover the optimal
# objective function :math:`C=4`

##############################################################################
# Plotting the results
# --------------------
# We can plot the distribution of measurements obtained from the optimized circuits. As
# expected for this graph, the partitions 0101 and 1010 are measured with the highest frequencies,
# and in the case where we set ``n_layers=2`` we obtain one of the optimal partitions with 100% certainty.

import matplotlib.pyplot as plt

xticks = range(0, 16)
xtick_labels = list(map(lambda x: format(x, "04b"), xticks))
bins = np.arange(0, 17) - 0.5

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title("n_layers=1")
plt.xlabel("bitstrings")
plt.ylabel("freq.")
plt.xticks(xticks, xtick_labels, rotation="vertical")
plt.hist(bitstrings1, bins=bins)
plt.subplot(1, 2, 2)
plt.title("n_layers=2")
plt.xlabel("bitstrings")
plt.ylabel("freq.")
plt.xticks(xticks, xtick_labels, rotation="vertical")
plt.hist(bitstrings2, bins=bins)
plt.tight_layout()
plt.show()

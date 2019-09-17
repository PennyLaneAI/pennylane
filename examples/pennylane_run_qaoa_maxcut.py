"""
.. _qaoa_maxcut:

Quantum Approximate Optimization for MaxCut
===========================================

In this tutorial, we implement a quantum approximate
optimization algorithm (QAOA) for the MaxCut problem in Pennylane. (See `Farhi,
Goldstone, and Gutmann (2014) <https://arxiv.org/abs/1411.4028>`__.) This
example demonstrates how to sample joint qubit measurements from a variational
quantum circuit to solve a combinatorial optimization problem. We encode
constraints for the MaxCut problem as diagonal operators and introduce... in the quantum
circuit.
"""

##############################################################################
# 1 Background
# -------------
#
# 1.1 The MaxCut problem
# ~~~~~~~~~~~~~~~~~~~~~~
# The aim of MaxCut is to maximize the number of edges in a graph that are "cut" by
# a given partition of the vertices into two sets as shown in the figure below.
#
# That is, given a graph
# with a set of :math:`n` vertices :math:`V=\{v_i\}` and :math:`m` edges
# :math:`E=\{(v_j,v_k)\}`, we seek the partition :math:`z` of :math:`V` into two sets
# :math:`S` and :math:`S'` which maximizes
#
# .. math::
#   C(z) = \sum_{\alpha=1}^{m}C_\alpha(z)
#
# where :math:`C_\alpha(z)` is 1 if :math:`z` places one vertex from the
# :math:`\alpha^\text{th}` edge in :math:`S` and the other in :math:`S'`, and is 0 otherwise.
# The goal of approximate optimization in this case is to find a partition :math:`z` which
# yields a value for :math:`C(z)` that is close to the maximum possible value.
#
# For instance,
# in the "ring" situation depicted in the figure above, the optimal value of :math:`C(z)` is
# 4. Such a situation can be represented by the bitstring :math:`z=1010\text{,}`
# indicating that the first and third bits are in one partition while the second and fourth are in
# the other. (The inverse partition, :math:`z=0101` is of course equally valid.) In the following section,
# we will represent partitions using computational basis states and use PennyLane and QAOA to
# rediscover this optimal partition of the "ring" scenario.
#
# 1.2 A quantum circuit
# ~~~~~~~~~~~~~~~~~~~~~~~
# This section presents the operators used in the QAOA algorithm and their implementation using basic unitary gates.
# Firstly, if the partition is given by the computational basis states, we can represent the terms in the
# cost function like so
#
# .. math::
#   C_\alpha = \frac{1}{2}\left(1-\sigma_{j}^z\sigma_{k}^z\right)
#
# where the :math:`\alpha\text{th}` edge is between qubits/vertices :math:`(j,k)`.
# :math:`C_\alpha` is 1 if and only if the :math:`i\text{th}` and :math:`j\text{th}`
# qubits have different values, representing separate partitions.
# The objective function :math:`C` is now a diagonal operator with integer eigenvalues.
#
# QAOA starts with a uniform superposition over the :math:`n` bitstring basis states,
#
# .. math::
#   |+_{n}\rangle = \frac{1}{\sqrt{2^n}}\sum_{z=0}^{2^n-1} |z\rangle
#
#
# The algorithm aims to explore the space of bitstring states for a superposition which is likely to yield a
# large value for the :math:`C` operator after performing a measurement in the computational basis.
# It starts with the state :math:`|+_{n}\rangle` and the parameters
# :math:`\boldsymbol{\gamma} = \gamma_1\gamma_2...\gamma_p`, :math:`\boldsymbol{\beta} = \beta_1\beta_2...\beta_p`
# and applies a sequence of operations yielding the new state
#
# .. math::
#   |\boldsymbol{\gamma},\boldsymbol{\beta}\rangle = U_{B_p}U_{C_p}U_{B_{p-1}}U_{C_{p-1}}...U_{B_1}U_{C_1}|+_n\rangle
#
# where :math:`\gamma_i \in [0,2\pi]\text{, }\beta_i \in [0,\pi]` and the operators have the explicit forms
#
# .. math::
#   U_{B_i} &= e^{-i\beta_iB} = \prod_{j=1}^n e^{-i\beta_i\sigma_x^j} \\
#   U_{C_i} &= e^{-i\gamma_iC} = \prod_{\text{edge (j,k)}} e^{-i\gamma_i(1-\sigma_z^j\sigma_z^k)/2}
#
# which can be implemented on a quantum circuit using the gates depicted below (up to an irrelevant constant
# that gets absorbed into the parameters).
#
#
# .. figure:: ../../examples/figures/qaoa_operators.png
#    :align: center
#    :scale: 100%
#    :alt: qaoa_operators
#
#
# Let :math:`\langle \boldsymbol{\gamma},
# \boldsymbol{\beta} | C | \boldsymbol{\gamma},\boldsymbol{\beta} \rangle` be our objective function. In the next
# section, we will use PennyLane to sample this expectation value and perform classical optimization over the
# parameters. This will specify a state :math:`|\boldsymbol{\gamma},\boldsymbol{\beta}\rangle` which is
# likely to yield an approximately optimal partition :math:`|z\rangle` upon performing a measurement on the qubits.
#
# 2 Implementing QAOA in PennyLane
# ----------------------------------
#
# 2.2 Imports and setup
# ~~~~~~~~~~~~~~~~~~~~~
#
# To get started, we import PennyLane along with the Pennylane-provided
# version of NumPy.


import pennylane as qml
from pennylane import numpy as np


##############################################################################
# 2.3 Operators
# ~~~~~~~~~~~~~
# We specify the number of qubits (vertices) with ``n_wires`` and
# compose the unitary operators using the definitions
# above. :math:`U_B` operators act on each of the wires, while :math:`U_c`
# operators act on wires whose corresponding vertices are joined by an edge in
# the graph. Each operator takes a parameter
# as an argument, which is specified when the function is
# called in the quantum circuit. We also define the graph using
# the list ``edges``, which contains the tuples of vertices defining
# each edge in the graph.

n_wires = 4
edges = [(0, 1), (0, 3), (1, 2), (2, 3)]

# unitary operator U_B with parameter beta
def U_B(beta):
    for wire in range(n_wires):
        qml.RX(2 * beta, wires=wire)


# unitary operator U_C with parameter gamma
def U_C(gamma):
    for clause in edges:
        wire1 = clause[0]
        wire2 = clause[1]
        qml.CNOT(wires=[wire1, wire2])
        qml.RZ(gamma, wires=wire2)
        qml.CNOT(wires=[wire1, wire2])

##############################################################################
# With some foresight, we also notice that we will need a way to sample
# a measurement of multiple qubits in the computational basis, so we define
# a Hermitian operator to do this. The eigenvalues of the operator are
# the joint qubit measurement values in integer form.

def comp_basis_measurement(wires):
    n_wires = len(wires)
    return qml.Hermitian(np.diag(range(2 ** n_wires)),wires=wires)


##############################################################################
# 2.4 Circuit
# ~~~~~~~~~~~~~
# Next, we create a quantum device with 4 qubits.

dev1 = qml.device('default.qubit', wires=n_wires)

##############################################################################
# We also require a quantum node which will apply the operators p times given the
# angle parameters, and return the expectation value of the observable
# :math:`\sigma_j^{z}\sigma_k^{z}` to be used in the cost function terms later on. The
# argument "edge" specifies the chosen edge :math:`(j,k)`.
#
# Once optimized, the same quantum node can be used for sampling an approximately optimal bitstring
# if executed with the ``edge`` keyword set to None.

pauli_z = [[1,0],[0,-1]]
pauli_z_2 = np.kron(pauli_z,pauli_z)

@qml.qnode(dev1)
def circuit(params, edge=None, p=1):
    # apply hadamards to get n qubit |+> state
    for wire in range(n_wires):
        qml.Hadamard(wires=wire)
    # p instances of unitary operators
    for i in range(p):
        U_C(params[1][i])
        U_B(params[0][i])
    # during measurement phase we are evaluating
    # the circuit with optimized parameters
    if edge is None:
        return qml.sample(comp_basis_measurement(range(n_wires)), n=1)
    # during the optimization phase we are evaluating a term
    # in the cost function using exp val
    return qml.expval(qml.Hermitian(pauli_z_2, wires=edge))

##############################################################################
# 2.5 Optimization
# ~~~~~~~~~~~~~
# Finally, we optimize the cost function over the
# angle parameters :math:`\boldsymbol{\gamma}` (``params[1]``) and :math:`\boldsymbol{\beta}`
# (``params[0]``)
# and then sample the optimized
# circuit multiple times to yield a distribution of bitstrings. One of the optimal partitions
# (z=0101 or z=1010) should be the most frequently sampled bitstring.


def qaoa_maxcut(p=1):
    # initialize the parameters near zero
    init_params = np.array([(0.01 * np.random.rand(p)),
                            (0.01 * np.random.rand(p))])

    # minimize negative the objective
    def objective(params):
        neg_F_p = 0
        for edge in edges:
            # objective for the maxcut problem
            neg_F_p -= 0.5 * (1 - circuit(params, edge=edge,p=p))
        return neg_F_p

    # initialize optimizer
    opt = qml.AdagradOptimizer(stepsize=0.5)

    steps = 30
    # optimize parameters in objective
    params = init_params
    for i in range(steps):
        params = opt.step(objective, params)
        if (i + 1) % 5 == 0:
            print('Objective after step {:5d}: {: .7f}'.format(i + 1, -objective(params), params))
    bit_strings = []
    n_samples = 100
    for i in range(0, n_samples):
        bit_strings.append(int(circuit(params, edge=None, p=p)))
    counts = np.bincount(np.array(bit_strings))
    most_freq_bit_string = np.argmax(counts)
    print('Optimized (beta,gamma) vectors: {}'.format(params))
    print('Most frequently sampled bit string is: {:04b}'.format(most_freq_bit_string))
    return -objective(params), bit_strings

bitstring1 = qaoa_maxcut(p=1)[1]
bitstring2 = qaoa_maxcut(p=2)[1]


##############################################################################
# 3 Plotting the results
# ~~~~~~~~~~~~~~~~~~~~~~
# We can plot the distribution of measurements we got from the above optimized circuit. As
# expected for this graph, the partitions 0101 and 1010 are measured with the highest frequencies,
# and in the case where we set ``p``=2 we obtain the optimal partitions with 100% certainty.

import matplotlib.pyplot as plt

xs = np.arange(16)

fig,(ax1,ax2) = plt.subplots(1,2,figsize=(8,4))
plt.subplot(1,2,1)
plt.title("p=1")
plt.hist(bitstring1)
plt.xlabel("bitstrings")
plt.ylabel("freq.")
plt.xticks(xs,map(lambda x : format(x,'04b'),xs),rotation='vertical')
plt.subplot(1,2,2)
plt.title("p=2")
plt.hist(bitstring2)
plt.xlabel("bitstrings")
plt.ylabel("freq.")
plt.xticks(xs,map(lambda x : format(x,'04b'),xs),rotation='vertical')
plt.tight_layout()
plt.show()

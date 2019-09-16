"""
.. _qaoa_maxcut:

Quantum Approximate Optimization for MaxCut
===========================================

In this tutorial, we show how to implement a quantum approximate
optimization algorithm (QAOA) for the MaxCut problem in Pennylane. (See `Farhi,
Goldstone, and Gutmann (2014) <https://arxiv.org/abs/1411.4028>`__.) This
example demonstrates how to sample joint qubit measurements from a variational
circuit to solve a combinatorial optimization problem. In doing so, we encode
constraints for the MaxCut problem as parametrized operations in the quantum
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
# More formally, given a graph
# with a set of :math:`n` vertices :math:`V=\{v_i\}` and :math:`m` edges
# :math:`E=\{(v_j,v_k)\}`, we seek the partition :math:`z` of :math:`V` into two sets
# :math:`S` and :math:`S'` which maximizes
#
# .. math::
#   C(z) = \sum_{\alpha=0}^{m}C_\alpha(z)
#
# where :math:`C_\alpha(z)` is 1 if :math:`z` places one vertex from the
# :math:`\alpha^\text{th}` edge in :math:`S` and the other in :math:`S'`, and is 0 otherwise.
# The goal of approximate optimization in this case is to find a partition :math:`z` which
# yields a value for :math:`C(z)` that is close to the maximum possible value.
#
# For instance,
# in the "ring" situation depicted in the figure above, the optimal value of :math:`C(z)` is
# 4, when the partition separates the vertices into the sets :math:`\{(0,2)\}` and
# :math:`\{(1,3)\}`. Such a situation can be represented by the bitstring :math:`z=1010\text{,}`
# indicating that the first and third bits are in one partition while the 2nd and fourth are in
# the other. (The inverse partition, :math:`z=0101` is of course equally valid.) In the following section,
# we will represent partitions using computational basis states and use PennyLane and QAOA to
# rediscover this optimal partition.
#
# 1.2 A quantum circuit
# ~~~~~~~~~~~~~~~~~~~~~~~
# QAOA starts with a uniform superposition over the :math:`n` bit basis states,
#
# .. math::
#   |s\rangle = \frac{1}{\sqrt{2^n}}\sum |z\rangle
#
# 
#
# 2 QAOA for MaxCut in PennyLane
# ------------------------------
#
# 2.2 Imports
# ~~~~~~~~~~~
#
# To get started, we import PennyLane along with the Pennylane-provided
# version of NumPy.


import pennylane as qml
from pennylane import numpy as np

##############################################################################
# Problem definition
# ~~~~~~~~~~~~~~~~~~
# We use the default qubit device and specify the number of qubits using
# "n_wires".

pauli_z = np.array([[1,0],[0,-1]])
pauli_z_2 = np.kron(pauli_z,pauli_z)


def qaoa_maxcut(n_wires=0,clauses=[],p=1,steps=30):
    """Apply quantum approximate optimization algorithm to the max cut problem.

        Solves the max cut optimization problem using the QAOA algorithm and returns
        the optimal max cut objective along with parameters for the quantum circuit.

        Args:
            n_wires (int): number of qubits in circuit (vertices)
            clauses (List[tuple(int)]): qubits in each max cut clause (edges in graph)
            p (int): parameter in QAOA algorithm
            steps (int): steps in optimization

        Returns:
            float: optimized expectation for objective function
            array(List[float]): optimized parameter vectors for the quantum circuit
        """
    # TODO: efficient (classical) implementation using relevant subgraphs
    # TODO: specify the quantum device
    # TODO: specify optimizer
    if(type(p)!=int or p < 1):
        raise ValueError("p must be a strictly positive integer")
    if(n_wires==0):
        raise ValueError("No qubits provided")
    if(clauses==[]):
        raise Exception("No clauses provided")

    # unitary operator U_B acting on each wire
    # according to parameter beta
    def U_B(beta):
        for wire in range(n_wires):
            qml.RX(2*beta, wires=wire)

    # unitary operator U_C with each term acting on
    # subset of wires with locality given by corresponding
    # clause
    def U_C(gamma):
        for clause in clauses:
            wire1 = clause[0]
            wire2 = clause[1]
            qml.CNOT(wires=[wire1,wire2])
            qml.RZ(gamma,wires=wire2)
            qml.CNOT(wires=[wire1,wire2])

    def comp_basis_measurement(wires):
        n_wires = len(wires)
        return qml.Hermitian(np.diag(range(2 ** n_wires)),wires=wires)

    dev1 = qml.device('default.qubit', wires=n_wires)

    # circuit with parameters beta = params[0] and
    # gamma = params[1], each of which is a np array
    @qml.qnode(dev1)
    def circuit(params,wires):
        # apply hadamards to get n qubit |+> state
        for wire in range(n_wires):
            qml.Hadamard(wires=wire)
        # p instances of unitary operators
        for i in range(p):
            U_C(params[1][i])
            U_B(params[0][i])
        # return exp val of Hermitian observable, 2 PauliZs
        return qml.expval(qml.Hermitian(pauli_z_2,wires=wires))

    # circuit to be run with optimal parameters in order to
    # sample a measurement in the computational basis, yielding
    # a bit string which is a candidate for an
    # approximately optimal solution
    @qml.qnode(dev1)
    def circuit2(params):
        # apply hadamards to get n qubit |+> state
        for wire in range(n_wires):
            qml.Hadamard(wires=wire)
        # p instances of unitary operators
        for i in range(p):
            U_C(params[1][i])
            U_B(params[0][i])
        # return sample in computational basis in integer form
        return qml.sample(comp_basis_measurement(range(n_wires)),n=1)

    # objective function whose parameters correspond to
    # those in quantum circuit
    def objective(params):
        neg_F_p = 0
        for wires in clauses:
            # objective for the maxcut problem
            neg_F_p -= 0.5*(1-circuit(params,wires))
        return neg_F_p

    # initialize the objective near zero
    init_params = np.array([(0.01 * np.random.rand(p)).tolist(),
                            (0.01 * np.random.rand(p)).tolist()])

    # initialize optimizer
    opt = qml.AdagradOptimizer(stepsize=0.5)

    # optimize parameters in objective
    params = init_params
    for i in range(steps):
        params = opt.step(objective, params)
        if (i + 1) % 5 == 0:
            print('Objective after step {:5d}: {: .7f}'.format(i + 1, -objective(params), params))

    print('Optimized (beta,gamma) vectors: {}'.format(params))
    F_p = -objective(params)
    bit_strings = []
    n_samples = 100
    for i in range(0,n_samples):
        bit_strings.append(int(circuit2(params)))
    counts = np.bincount(np.array(bit_strings))
    most_freq_bit_string = np.argmax(counts)
    print('Most frequently sampled bit string is: {:04b}'.format(most_freq_bit_string))
    # return optimal values of objective function, corresponding
    # parameters, and list of sampled bit strings in tuple
    return F_p, params, bit_strings




##############################################################################
# Plotting the distribution of bitstrings
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We can.

import matplotlib.pyplot as plt


##############################################################################
# Stuff.

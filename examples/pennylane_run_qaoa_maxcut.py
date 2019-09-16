"""
.. _qaoa_maxcut:

Quantum Approximate Optimization Algorithm
==========================================

This demo covers the.
"""

##############################################################################
# Imports
# -------
#
# As usual, we import PennyLane, the PennyLane-provided version of NumPy,
# and an optimizer.


import pennylane as qml
from pennylane import numpy as np


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


ring = [(0,1),(0,3),(1,2),(2,3)]

qaoa_output_1 = qaoa_maxcut(n_wires=4,clauses=ring,p=1,steps=30)
qaoa_output_2 = qaoa_maxcut(n_wires=4,clauses=ring,p=2,steps=30)

bit_strings_1 = qaoa_output_1[2]
bit_strings_2 = qaoa_output_2[2]
xs = np.arange(16)
bins = xs - 0.5

fig,(ax1,ax2) = plt.subplots(1,2)
plt.subplot(1,2,1)
plt.title("p=1")
plt.hist(bit_strings_1,bins)
plt.xlabel("bit strings")
plt.ylabel("freq.")
plt.xticks(xs,map(lambda x : format(x,'04b'),xs),rotation='vertical')
plt.subplot(1,2,2)
plt.title("p=2")
plt.hist(bit_strings_2,bins)
plt.xlabel("bit strings")
plt.ylabel("freq.")
plt.xticks(xs,map(lambda x : format(x,'04b'),xs),rotation='vertical')
plt.subplots_adjust(bottom=0.15)
plt.tight_layout()
plt.show()

##############################################################################
# Stuff.

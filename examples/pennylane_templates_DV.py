r"""
.. _templates_DV:

Single Layer Strongly Entangling Circuit
========================================

Parameterized quantum circuits generally consist of three parts:

- state preparation where classical data is embedded into a quantum state (see :ref:`Quantum embeddings <concept_embeddings>`) 
- parameterized gates that act on the said quantum state 
- measurement

.. important::

    For discrete-variable computing, PennyLane provides simplified circuit
    architecture templates using the quantum universality of single-qubit
    rotation gates and imprimitive 2-qubit gates.

PennyLane’s qubit circuit architecture (how the gates are placed) can
have two layouts:

1. Strongly entangling: Generic rotations followed by entangling gates
   are applied to all qubits. This prepares strongly entangled quantum
   states that can ideally explore the whole Hilbert space.

2. Random: Randomly chosen rotations and randomly placed entangling
   gates are applied to randomly chosen qubits.

For more details, see :mod:`pennylane.templates`. For now, let’s look at the strongly entangling circuits in detail. A single layer
of this circuit is shown below:

.. figure:: ../../examples/figures/slsec.png
    :align: center
    :width: 60%
    :target: javascript:void(0);

PennyLane’s :func:`~.StronglyEntanglingLayer` template can be used to
implement such a layer. It requires four input arguments:

1. ``weights``: a matrix containing the angles for all the rotation gates
2. ``wires``: all the qubits the gates will act upon in this circuit architecture
3. ``range``: the range of 2-qubit entangling gates
4. ``imprimitive``: the imprimitive 2-qubit gate to be used

To implement a circuit with multiple layers,
:func:`~.StronglyEntanglingLayers` template can be called from inside a ``QNode``.

.. note::

    When using any parameterized quantum circuit, circuit parameters need to
    be initialized before they can be optimized to converge to the solution.
    PennyLane provides ready-to-use templates that initialize all the
    required parameters corresponding to all the circuit architectures it
    offers. See :mod:`pennylane.init` for more details.

For this tutorial, we will implement the following simple strongly
entangling circuit with just one layer:

.. figure:: ../../examples/figures/slsec_example.png
    :align: center
    :width: 60%
    :target: javascript:void(0);

**What is this circuit doing?**

1. After the single qubit gates,
   :math:`|\psi\rangle=|0\rangle|0\rangle|0\rangle|0\rangle`
   is converted to
   :math:`|\psi\rangle=|+\rangle|0\rangle|0\rangle|+\rangle`

2. After the CNOTs, the state is

   .. math:: |\psi\rangle=\frac{1}{2}[|00\rangle|00\rangle + |01\rangle|11\rangle + |10\rangle|01\rangle + |11\rangle|10\rangle]

**Note**: This is a completely entangled state; subsystem qubits 0,1 has
been entangled with subsystem qubits 2,3.

"""

import pennylane as qml
from pennylane import numpy as np
from pennylane.templates.layers import StronglyEntanglingLayer

##############################################################################

n_wires = 4
pi = np.pi
    
dev = qml.device('default.qubit', wires = n_wires)
    
@qml.qnode(dev)
def circuit(angles, A=None, B=None):
    StronglyEntanglingLayer(angles, wires = range(n_wires), r=1)
    # measure the expectation value of Pauli-Z operator on subsystem qubits 0,1 and subsystem qubits 2,3
    return qml.expval.Hermitian(A, wires=[0,1]),qml.expval.Hermitian(B, wires=[2,3])

##############################################################################
#
# Note that we do not use initialization templates provided by PennyLane
# for this simple example. Instead, we can simply provide the angles that result in the Hadamard and
# Identity gates.

my_angles= np.array([[0, pi/2, 0],[0, 0, 0],[0, 0, 0],[0, pi/2, 0]])
print(my_angles)

##############################################################################

pauliz = np.array([[1, 0], [0, -1]])
pauli2 = np.kron(pauliz, pauliz)
pauli4 = np.kron(pauli2, pauli2)

##############################################################################

print(circuit(my_angles, A=pauli2, B=pauli2))

##############################################################################
#
# The output of ``circuit`` makes sense as when we measure just one subsystem
# (a mixed state) of a completely entangled state, the expectation value
# of Pauli-Z operator averages to zero.


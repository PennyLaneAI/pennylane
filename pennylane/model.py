# Copyright 2018 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=protected-access
r"""
Models
======

**Module name:** :mod:`pennylane.model`

.. currentmodule:: pennylane.model

This module provides a growing library of functions representing
common circuit architectures that can be used to easily build more complex
quantum machine learning models.

For example, you can define and call a circuit-centric quantum classifier
:cite:`schuld2018circuit` on an arbitrary number of wires and with an
arbitrary number of blocks in the following way:

.. code-block:: python

    import pennylane as qml
    from pennylane import numpy as np

    num_wires=4
    num_blocks=3
    dev = qml.device('default.qubit', wires=num_wires)

    @qml.qnode(dev)
    def circuit(weights, x=None):
        qml.BasisState(x, wires=range(num_wires))
        qml.model.CircuitCentricClassifier(weights, periodic=True, wires=range(num_wires))
        return qml.expval.PauliZ(0)

    weights=np.random.randn(num_blocks, num_wires, 3)
    print(circuit(weights, x=np.array(np.random.randint(0,1,num_wires))))


Summary
^^^^^^^

.. autosummary::
  variational_quantum_classifyer

Code details
^^^^^^^^^^^^
"""
import logging as log
import numpy as np
import pennylane as qml

from pennylane.ops import CNOT, Rot, Squeezing, Displacement, Kerr, Rotation, Beamsplitter

log.getLogger()

def CircuitCentricClassifier(weights, periodic=True, ranges=None, imprimitive_gate=CNOT, wires=None):
    """A circuit-centric classifier circuit.

    Constructs a circuit-centric quantum classifier :cite:`schuld2018circuit`
    with len(weights) blocks on the given wires with the provided weights.
    Each element of weights must be a an array of size len(wires)*3.

    Args:
        weights (array[float]): Number of blocks*len(wires)*3 array of weights
        periodic (bool): whether to use periodic boundary conditions when
                         applying imprimitive gates
        ranges (Sequence[int]): Ranges of the imprimitive gates in the
                                respective blocks
        imprimitive_gate (pennylane.ops.Operation): Imprimitive gate to use, defaults to CNOT
        wires (Sequence[int]): Wires the model should act on
    """
    if ranges is None:
        ranges = [1]*len(weights)
    for block_weights, block_range in zip(weights, ranges):
        CircuitCentricClassifierBlock(block_weights, r=block_range, periodic=periodic, imprimitive_gate=imprimitive_gate, wires=wires)


def CircuitCentricClassifierBlock(weights, periodic=True, r=1, imprimitive_gate=CNOT, wires=None):
    """An individual block of a circuit-centric classifier circuit.

    Args:
        weights (array[float]): len(wires)*3 array of weights
        periodic (bool): Whether to use periodic boundary conditions when
                         applying imprimitive gates
        r (Sequence[int]): Range of the imprimitive gates of this block
        imprimitive_gate (pennylane.ops.Operation): Imprimitive gate to use, defaults to CNOT
        wires (Sequence[int]): Wires the model should act on
    """
    for i, wire in enumerate(wires):
        Rot(weights[i, 0], weights[i, 1], weights[i, 2], wires=wire)

    num_wires = len(wires)
    for i in range(num_wires) if periodic else range(num_wires-1):
        imprimitive_gate(wires=[wires[i], wires[(i+r) % num_wires]])

def CVNeuralNet(weights, wires=None):
    """A CV Quantum Neural Network

    Implements the CV Quantum Neural Network (CVNN) architecture from
    :cite:`killoran2018continuous` for an arbitrary number of wires
    and layers.

    Args:
         weights (array[float]): Array of weights for each layer of the CV
                                 neural network
        wires (Sequence[int]): Wires the CVNN should act on
    """
    for layer_weights in weights:
        CVNeuralNetLayer(layer_weights, wires=wires)

def CVNeuralNetLayer(weights, wires=None):
    """A single layer of a CV Quantum Neural Network

    Args:
         weights (array[float]): Array of weights for this layer of the CV
                                 neural network
        wires (Sequence[int]): Wires the layer should act on
    """
    PhaselessLinearInterferometer(weights[0], wires=wires)
    for wire in wires:
        Squeezing(weights[1], 0., wires=wire)
    PhaselessLinearInterferometer(weights[2], wires=wires)
    for wire in wires:
        Displacement(weights[3], 0., wires=wire)
    for wire in wires:
        Kerr(weights[4], wires=wire)

def PhaselessLinearInterferometer(weights, wires=None):
    raise NotImplementedError("PhaselessLinearInterferometer not yet implemented")


def Interferometer(U, tollerance=11, wires=None):
    r"""Linear interferometer

    Implements a linear interferometer as sequence of beamsplitters and
    rotation gates by means of the Clements decomposition.

    Args:
        U (array): A len(wires) by len(wires) complex unitary matrix
        tollerance (int): The number of decimal places to use when determining whether a gate parameter obtained from the Clements decomposition is so close to trivial that the gate is effectively an Identity and can be skipped.
        wires (Sequence[int]): Wires the Interferometer should act on
    """
    BS1, BS2, R = _clements(U)

    for n, m, theta, phi, _ in BS1:
        if np.round(phi, tollerance) != 0:
            Rotation(phi, wires=[wires[n]])
        if np.round(theta, tollerance) != 0:
            Beamsplitter(theta, 0, wires=[wires[n], wires[m]])

    for n, expphi in enumerate(R):
        if np.round(expphi, tollerance) != 1.0:
            q = np.log(expphi).imag
            Rotation(q, wires=[wires[n]])

    for n, m, theta, phi, _ in reversed(BS2):
        if np.round(theta, tollerance) != 0:
            Beamsplitter(-theta, 0, wires=[wires[n], wires[m]])
        if np.round(phi, tollerance) != 0:
            Rotation(-phi, wires=wires[n])



def _clements(V):
    r"""Performs the Clements decomposition of a Unitary complex matrix.

    See Clements et al. Optica 3, 1460 (2016) [10.1364/OPTICA.3.001460] for more details.

    Args:
        V (array): Unitary matrix of size n_size

    Returns:
        tuple[array]: returns a tuple of the form ``(tilist,tlist,np.diag(localV))``
            where:

            * ``tilist``: list containing ``[n,m,theta,phi,n_size]`` of the Ti unitaries needed
            * ``tlist``: list containing ``[n,m,theta,phi,n_size]`` of the T unitaries needed
            * ``localV``: Diagonal unitary sitting sandwhiched by Ti's and the T's
    """
    def T(m, n, theta, phi, nmax):
        r"""The Clements T matrix"""
        mat = np.identity(nmax, dtype=np.complex128)
        mat[m, m] = np.exp(1j*phi)*np.cos(theta)
        mat[m, n] = -np.sin(theta)
        mat[n, m] = np.exp(1j*phi)*np.sin(theta)
        mat[n, n] = np.cos(theta)
        return mat

    def Ti(m, n, theta, phi, nmax):
        r"""The inverse Clements T matrix"""
        return np.transpose(T(m, n, theta, -phi, nmax))

    def nullTi(m, n, U):
        r"""Nullifies element m,n of U using Ti"""
        (nmax, mmax) = U.shape

        if U[m, n+1] == 0:
            thetar = np.pi/2
            phir = 0
        else:
            r = U[m, n] / U[m, n+1]
            thetar = np.arctan(np.abs(r))
            phir = np.angle(r)

        return [n, n+1, thetar, phir, nmax]

    def nullT(n, m, U):
        r"""Nullifies element n,m of U using T"""
        (nmax, mmax) = U.shape

        if U[n-1, m] == 0:
            thetar = np.pi/2
            phir = 0
        else:
            r = -U[n, m] / U[n-1, m]
            thetar = np.arctan(np.abs(r))
            phir = np.angle(r)

        return [n-1, n, thetar, phir, nmax]


    localV = V
    (nsize, nsize2) = localV.shape

    if nsize != nsize2:
        raise ValueError("V must be a square unitary matrix")

    tilist = []
    tlist = []
    for k, i in enumerate(range(nsize-2, -1, -1)):
        if k%2 == 0:
            for j in reversed(range(nsize-1-i)):
                tilist.append(nullTi(i+j+1, j, localV))
                localV = localV @ Ti(*tilist[-1])
        else:
            for j in range(nsize-1-i):
                tlist.append(nullT(i+j+1, j, localV))
                localV = T(*tlist[-1]) @ localV

    return tilist, tlist, np.diag(localV)

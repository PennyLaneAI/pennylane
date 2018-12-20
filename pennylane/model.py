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
QML Models
==========

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
    print(circuit(weights, x=np.array(np.random.randint(0, 1, num_wires))))


The handy :func:`Interferometer` function can be used to construct arbitrary interferometers in terms of elementary :class:`~.Beamsplitter` and :class:`~.Rotation` operations, by means of the scheme from :cite:`clements2016optimal`, specified either via the unitary transformation on the bosonic operators or in terms of lists of beamsplitter parameters.

The function :func:`CVNeuralNet` implements the continuous variable neural network architecture from :cite:`killoran2018continuous`. Provided with a suitable array of weights, such models can now be easily constructed and trained with PennyLane.

Summary
^^^^^^^

.. autosummary::
  CircuitCentricClassifier
  CircuitCentricClassifierBlock
  CVNeuralNet
  CVNeuralNetLayer
  Interferometer
  clements

Code details
^^^^^^^^^^^^
"""
import logging as log
import numpy as np

from pennylane.ops import CNOT, Rot, Squeezing, Displacement, Kerr, Rotation, Beamsplitter

log.getLogger()

def CircuitCentricClassifier(weights, periodic=True, ranges=None, imprimitive_gate=CNOT, wires=None):
    """pennylane.model.CircuitCentricClassifier(weights, periodic=True, ranges=None, imprimitive_gate=qml.CNOT, wires)
    A circuit-centric classifier circuit.

    Constructs a circuit-centric quantum classifier :cite:`schuld2018circuit`
     with ``len(weights)`` blocks on the given wires with the provided weights.
     Each element of weights must be a an array of size ``len(wires)*3``.

    Args:
        weights (array[float]): shape ``(len(weights), len(wires), 3)`` array of weights
        periodic (bool): whether to use periodic boundary conditions when
                         applying imprimitive gates
        ranges (Sequence[int]): Ranges of the imprimitive gates in the
                                respective blocks
        imprimitive_gate (pennylane.ops.Operation): Imprimitive gate to use, defaults to :class:`~.CNOT`
        wires (Sequence[int]): Wires the model should act on
    """
    if ranges is None:
        ranges = [1]*len(weights)
    for block_weights, block_range in zip(weights, ranges):
        CircuitCentricClassifierBlock(block_weights, r=block_range, periodic=periodic, imprimitive_gate=imprimitive_gate, wires=wires)


def CircuitCentricClassifierBlock(weights, periodic=True, r=1, imprimitive_gate=CNOT, wires=None):
    """pennylane.model.CircuitCentricClassifierBlock(weights, periodic=True, r=1, imprimitive_gate=qml.CNOT, wires)
    An individual block of a circuit-centric classifier circuit.

    Args:
        weights (array[float]): shape ``(len(wires), 3)`` array of weights
        periodic (bool): Whether to use periodic boundary conditions when
                         applying imprimitive gates
        r (Sequence[int]): Range of the imprimitive gates of this block
        imprimitive_gate (pennylane.ops.Operation): Imprimitive gate to use, defaults to :class:`~.CNOT`
        wires (Sequence[int]): Wires the model should act on
    """
    for i, wire in enumerate(wires):
        Rot(weights[i, 0], weights[i, 1], weights[i, 2], wires=wire)

    num_wires = len(wires)
    for i in range(num_wires) if periodic else range(num_wires-1):
        imprimitive_gate(wires=[wires[i], wires[(i+r) % num_wires]])

def CVNeuralNet(weights, wires=None):
    """pennylane.model.CVNeuralNet(weights, wires)
    A CV Quantum Neural Network

    Implements the CV Quantum Neural Network (CVQNN) architecture from
    :cite:`killoran2018continuous` for an arbitrary number of wires
    and layers.

    See :func:`CVNeuralNetLayer` for details of the expected format of
    input parameters.

    Args:
        weights (array[array]): Array of weights for each layer of the CV
                                neural network
        wires (Sequence[int]): Wires the CVQNN should act on
    """
    for layer_weights in weights:
        CVNeuralNetLayer(*layer_weights, wires=wires)

def CVNeuralNetLayer(theta1, phi1, s, theta2, phi2, r, k, tollerance=11, wires=None): #pylint: disable-msg=too-many-arguments
    """pennylane.model.CVNeuralNetLayer(theta1, phi1, s, theta2, phi2, r, k, tollerance=11, wires)
    A single layer of a CV Quantum Neural Network

    Implements a single layer from the the CV Quantum Neural Network (CVQNN)
    architecture of :cite:`killoran2018continuous` for an arbitrary number
    of wires and layers.

    .. note::

       The CV neural network architecture includes :class:`~.Kerr` operations. Make sure to use a suitable device, such as the :code:`strawberryfields.fock` device of the `PennyLane-SF <https://github.com/XanaduAI/pennylane-sf>`_ plugin.

    Args:
        theta1 (array[float]): length ``len(wires)*(len(wires)-1)/2`` array of transmittivity angles
        phi1 (array[float]): length ``len(wires)*(len(wires)-1)/2`` array of phase angles
        s (array[float]): length ``len(wires)`` arrays of squeezing amounts for :class:`~.Squeezing` operations
        theta2 (array[float]): length ``len(wires)*(len(wires)-1)/2`` array of transmittivity angles
        phi2 (array[float]): length ``len(wires)*(len(wires)-1)/2`` array of phase angles
        r (array[float]): length ``len(wires)`` arrays of displacement magnitudes for :class:`~.Displacement` operations
        k (array[float]): length ``len(wires)`` arrays of kerr parameters for :class:`~.Kerr` operations
        tollerance (int): The number of decimal places to use when determining whether a gate parameter obtained is so close to trivial that the gate is effectively an Identity and can be skipped.
        wires (Sequence[int]): Wires the layer should act on
    """
    Interferometer(theta=theta1, phi=phi1, tollerance=tollerance, wires=wires)
    for i, wire in enumerate(wires):
        Squeezing(s[i], 0., wires=wire)
    Interferometer(theta=theta2, phi=phi2, tollerance=tollerance, wires=wires)
    for i, wire in enumerate(wires):
        Displacement(r[i], 0., wires=wire)
    for i, wire in enumerate(wires):
        Kerr(k[i], wires=wire)


def Interferometer(*, theta=None, phi=None, U=None, tollerance=11, wires=None): #pylint: disable=too-many-branches
    r"""pennylane.model.Interferometer([theta, phi,| U,] tollerance=11, wires)
    General linear interferometer

    The instance can be specified in two ways:

    (i)  By providing ``len(wires)*(len(wires)-1)/2`` many angles via theta and phi each.
         In this case the interferometer is implemented with the scheme described in
         :cite:`clements2016optimal` (Fig. 1a). Beam splitters are numbered per layer
         from top to bottom and ``theta[i]`` and ``phi[i]`` are used as the parameters of the
         ith beam splitter.

    (ii) By providing a unitary matrix ``U``. The interferometer is then implemented as
         network of beam splitters and rotation gates determined from ``U`` by means of
         algorithm described in :cite:`clements2016optimal`.

    .. note::

       While constructing interferometers via their defining unitary transformation is handy, for automatic differentiation, optimization, and variational learning you are strongly advised to use the parametrization in terms of beam splitter angles.


    Args:
        theta (array): length ``len(wires)*(len(wires)-1)/2`` array of transmittivity angles
        phi (array): length ``len(wires)*(len(wires)-1)/2`` array of phase angles
        U (array): A shape ``(len(wires), len(wires))`` complex unitary matrix
        tollerance (int): The number of decimal places to use when determining whether a gate parameter obtained from the Clements decomposition is so close to trivial that the gate is effectively an Identity and can be skipped.
        wires (Sequence[int]): Wires the Interferometer should act on

    """

    if (theta is not None or phi is not None) and U is not None:
        raise ValueError("You must only specify either theta and phi, or U")
    elif theta is None and phi is None and U is None:
        raise ValueError("You must specify either theta and phi, or U")
    elif theta is None and phi is not None or theta is not None and phi is None:
        raise ValueError("If you specify theta you must also specify phi and vice versa")
    elif theta is not None and phi is not None:
        #loop over layers
        gate_num = 0
        for l in range(len(wires)):
            for n, (i, j) in enumerate(zip(wires[:-1], wires[1:])):
                #skip even or odd pairs depending on layer
                if (l+n)%2 == 1:
                    continue
                Beamsplitter(theta[gate_num], phi[gate_num], wires=[i, j])
                gate_num += 1

    elif U is not None:
        BS1, BS2, R = clements(U)
        for n, m, theta1, phi1, _ in BS1:
            if np.round(phi1, tollerance) != 0:
                Rotation(phi1, wires=[wires[n]])
            if np.round(theta1, tollerance) != 0:
                Beamsplitter(theta1, 0, wires=[wires[n], wires[m]])

        for n, expphi in enumerate(R):
            if np.round(expphi, tollerance) != 1.0:
                q = np.log(expphi).imag
                Rotation(q, wires=[wires[n]])

        for n, m, theta2, phi2, _ in reversed(BS2):
            if np.round(theta2, tollerance) != 0:
                Beamsplitter(-theta2, 0, wires=[wires[n], wires[m]])
            if np.round(phi2, tollerance) != 0:
                Rotation(-phi2, wires=wires[n])


def clements(V):
    r"""Performs the Clements decomposition of a Unitary complex matrix.

    See :cite:`clements2016optimal` for more details.

    Args:
        V (array): A shape ``(len(wires), len(wires))`` complex unitary matrix

    Returns:
        tuple[array]: returns a tuple of the form ``(tilist, tlist, np.diag(localV))``
            where:

            * ``tilist``: list containing ``[n, m, theta, phi, n_size]`` of the Ti unitaries needed
            * ``tlist``: list containing ``[n, m, theta, phi, n_size]`` of the T unitaries needed
            * ``localV``: Diagonal unitary sitting sandwhiched by Ti's and the T's
    """
    def T(m, n, theta, phi, nmax):
        r"""The Clements ``T`` matrix"""
        mat = np.identity(nmax, dtype=np.complex128)
        mat[m, m] = np.exp(1j*phi)*np.cos(theta)
        mat[m, n] = -np.sin(theta)
        mat[n, m] = np.exp(1j*phi)*np.sin(theta)
        mat[n, n] = np.cos(theta)
        return mat

    def Ti(m, n, theta, phi, nmax):
        r"""The inverse of the Clements ``T`` matrix"""
        return np.transpose(T(m, n, theta, -phi, nmax))

    def nullTi(m, n, U):
        r"""Nullifies element ``m, n`` of ``U`` using ``Ti``"""
        (nmax, _) = U.shape

        if U[m, n+1] == 0:
            thetar = np.pi/2
            phir = 0
        else:
            r = U[m, n] / U[m, n+1]
            thetar = np.arctan(np.abs(r))
            phir = np.angle(r)

        return [n, n+1, thetar, phir, nmax]

    def nullT(n, m, U):
        r"""Nullifies element ``n, m`` of ``U`` using ``T``"""
        (nmax, _) = U.shape

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

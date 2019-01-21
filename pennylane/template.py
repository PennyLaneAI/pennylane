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
QML Templates
=============

**Module name:** :mod:`pennylane.template`

.. currentmodule:: pennylane.template

This module provides a growing library of templates of common quantum
machine learning circuit architectures that can be used to easily build,
evaluate, and train more complex quantum machine learning models.

For example, you can construct a circuit-centric quantum
classifier with the architecture from :cite:`schuld2018circuit` on an arbitrary
number of wires and with an arbitrary number of blocks by using the
template :class:`StronglyEntanglingCircuit` in the following way:

.. code-block:: python

    import pennylane as qml
    from pennylane import numpy as np

    num_wires=4
    num_blocks=3
    dev = qml.device('default.qubit', wires=num_wires)

    @qml.qnode(dev)
    def circuit(weights, x=None):
        qml.BasisState(x, wires=range(num_wires))
        qml.template.StronglyEntanglingCircuit(weights, periodic=True, wires=range(num_wires))
        return qml.expval.PauliZ(0)

    weights=np.random.randn(num_blocks, num_wires, 3)
    print(circuit(weights, x=np.array(np.random.randint(0, 1, num_wires))))


The handy :func:`Interferometer` function can be used to construct arbitrary
interferometers in terms of elementary :class:`~.Beamsplitter` Operations,
by providing lists of transmittivity and phase angles. PennyLane can
then be used to easily differentiate and optimize these
parameters:

.. code-block:: python

    import pennylane as qml
    from pennylane import numpy as np

    num_wires=4
    dev = qml.device('default.gaussian', wires=num_wires)
    squeezings = np.random.rand(num_wires, 2)
    num_params = int(num_wires*(num_wires-1)/2)
    theta = np.random.uniform(0, 2*np.pi, num_params)
    phi = np.random.uniform(0, 2*np.pi, num_params)

    @qml.qnode(dev)
    def circuit(theta, phi):
        for wire in range(num_wires):
            qml.Squeezing(squeezings[wire][0], squeezings[wire][1], wires=wire)

        qml.template.Interferometer(theta=theta, phi=phi, wires=range(num_wires))
        return tuple(qml.expval.MeanPhoton(wires=wires) for wires in range(num_wires))

    print(qml.jacobian(circuit, 0)(theta, phi))

The function :func:`CVNeuralNet` implements the continuous-variable neural network architecture from :cite:`killoran2018continuous`.
Provided with a suitable array of weights, such neural networks can be easily constructed and trained with PennyLane.

Summary
^^^^^^^

.. autosummary::
  StronglyEntanglingCircuit
  StronglyEntanglingCircuitBlock
  CVNeuralNet
  CVNeuralNetLayer
  Interferometer

Code details
^^^^^^^^^^^^
"""
from pennylane.ops import CNOT, Rot, Squeezing, Displacement, Kerr, Beamsplitter


def StronglyEntanglingCircuit(weights, periodic=True, ranges=None, imprimitive_gate=CNOT, wires=None):
    """pennylane.template.StronglyEntanglingCircuit(weights, periodic=True, ranges=None, imprimitive_gate=qml.CNOT, wires)
    A strongly entangling circuit.

    Constructs the strongly entangling circuit used in the circuit-centric quantum
    classifier :cite:`schuld2018circuit`
    with ``len(weights)`` blocks on the given wires with the provided weights.
    Each element of weights must be a an array of size ``len(wires)*3``.

    Args:
        weights (array[float]): shape ``(len(weights), len(wires), 3)`` array of weights
        periodic (bool): whether to use periodic boundary conditions when
                         applying imprimitive gates
        ranges (Sequence[int]): Ranges of the imprimitive gates in the
                                respective blocks
        imprimitive_gate (pennylane.ops.Operation): Imprimitive gate to use, defaults to :class:`~.CNOT`
        wires (Sequence[int]): Wires the strongly entangling circuit should act on
    """
    if ranges is None:
        ranges = [1]*len(weights)
    for block_weights, block_range in zip(weights, ranges):
        StronglyEntanglingCircuitBlock(block_weights, r=block_range, periodic=periodic, imprimitive_gate=imprimitive_gate, wires=wires)


def StronglyEntanglingCircuitBlock(weights, periodic=True, r=1, imprimitive_gate=CNOT, wires=None):
    """pennylane.template.StronglyEntanglingCircuitBlock(weights, periodic=True, r=1, imprimitive_gate=qml.CNOT, wires)
    An individual block of a strongly entangling circuit.

    Args:
        weights (array[float]): shape ``(len(wires), 3)`` array of weights
        periodic (bool): whether to use periodic boundary conditions when
                         applying imprimitive gates
        r (Sequence[int]): range of the imprimitive gates of this block
        imprimitive_gate (pennylane.ops.Operation): Imprimitive gate to use, defaults to :class:`~.CNOT`
        wires (Sequence[int]): Wires the block should act on
    """
    for i, wire in enumerate(wires):
        Rot(weights[i, 0], weights[i, 1], weights[i, 2], wires=wire)

    num_wires = len(wires)
    for i in range(num_wires) if periodic else range(num_wires-1):
        imprimitive_gate(wires=[wires[i], wires[(i+r) % num_wires]])


def CVNeuralNet(weights, wires=None):
    """pennylane.template.CVNeuralNet(weights, wires)
    A CV Quantum Neural Network

    Implements the CV Quantum Neural Network (CVQNN) architecture from
    :cite:`killoran2018continuous` for an arbitrary number of wires
    and layers.

    See :func:`CVNeuralNetLayer` for details of the expected format of
    input parameters.

    Args:
        weights (array[array]): array of arrays of weights for each
                                layer of the CV neural network
        wires (Sequence[int]): wires the CVQNN should act on
    """
    for layer_weights in weights:
        CVNeuralNetLayer(*layer_weights, wires=wires)


def CVNeuralNetLayer(theta_1, phi_1, r, phi_r, theta_2, phi_2, a, phi_a, k, wires=None): #pylint: disable-msg=too-many-arguments
    """pennylane.template.CVNeuralNetLayer(theta_1, phi_1, s, theta_2, phi_2, r, k, wires)
    A single layer of a CV Quantum Neural Network

    Implements a single layer from the the CV Quantum Neural Network (CVQNN)
    architecture of :cite:`killoran2018continuous` for an arbitrary number
    of wires and layers.

    .. note::

       The CV neural network architecture includes :class:`~.Kerr` operations. Make sure to use a suitable device,
       such as the :code:`strawberryfields.fock` device of the `PennyLane-SF <https://github.com/XanaduAI/pennylane-sf>`_ plugin.

    Args:
        theta_1 (array[float]): length ``len(wires)*(len(wires)-1)/2`` array of transmittivity angles
        phi_1 (array[float]): length ``len(wires)*(len(wires)-1)/2`` array of phase angles
        r (array[float]): length ``len(wires)`` arrays of squeezing amounts for :class:`~.Squeezing` operations
        phi_r (array[float]): length ``len(wires)`` arrays of squeezing angles for :class:`~.Squeezing` operations
        theta_2 (array[float]): length ``len(wires)*(len(wires)-1)/2`` array of transmittivity angles
        phi_2 (array[float]): length ``len(wires)*(len(wires)-1)/2`` array of phase angles
        a (array[float]): length ``len(wires)`` arrays of displacement magnitudes for :class:`~.Displacement` operations
        phi_a (array[float]): length ``len(wires)`` arrays of displacement angles for :class:`~.Displacement` operations
        k (array[float]): length ``len(wires)`` arrays of kerr parameters for :class:`~.Kerr` operations
        wires (Sequence[int]): Wires the layer should act on
    """
    Interferometer(theta=theta_1, phi=phi_1, wires=wires)
    for i, wire in enumerate(wires):
        Squeezing(r[i], phi_r[i], wires=wire)

    Interferometer(theta=theta_2, phi=phi_2, wires=wires)

    for i, wire in enumerate(wires):
        Displacement(a[i], phi_a[i], wires=wire)

    for i, wire in enumerate(wires):
        Kerr(k[i], wires=wire)


def Interferometer(theta, phi, wires=None): #pylint: disable=too-many-branches
    r"""pennylane.template.Interferometer(theta, phi, wires)
    General linear interferometer.

    An instance is specified by providing ``len(wires)*(len(wires)-1)/2`` many
    transmittivity angles and the same number of phase angles. The interferometer
    is then implemented with
    the scheme described in :cite:`clements2016optimal` (Fig. 1a). Beamsplitters
    are numbered per layer from top to bottom and ``theta[i]`` and ``phi[i]`` are
    used as the parameters of the i-th beam splitter.

    Args:
        theta (array): length ``len(wires)*(len(wires)-1)/2`` array of transmittivity angles
        phi (array): length ``len(wires)*(len(wires)-1)/2`` array of phase angles
        wires (Sequence[int]): wires the Interferometer should act on
    """

    #loop over layers
    gate_num = 0
    for l in range(len(wires)):
        for n, (i, j) in enumerate(zip(wires[:-1], wires[1:])):
            #skip even or odd pairs depending on layer
            if (l+n)%2 == 1:
                continue
            Beamsplitter(theta[gate_num], phi[gate_num], wires=[i, j])
            gate_num += 1

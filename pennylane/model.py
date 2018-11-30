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
import pennylane as qml

from pennylane.ops import CNOT

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
        wires (Sequence[int]): Wires the model acts on
    """
    if ranges is None:
        ranges = [1]*len(weights)
    for block_weights, block_range in zip(weights, ranges):
        CircuitCentricClassifierBlock(block_weights, r=block_range, periodic=periodic, imprimitive_gate=imprimitive_gate, wires=wires)


def CircuitCentricClassifierBlock(weights, periodic=True, r=1, imprimitive_gate=qml.ops.CNOT, wires=None):
    """A block of a circuit-centric classifier circuit.

    Args:
        weights (array[float]): len(wires)*3 array of weights
        periodic (bool): Whether to use periodic boundary conditions when
                         applying imprimitive gates
        r (Sequence[int]): Range of the imprimitive gates of this block
        imprimitive_gate (pennylane.ops.Operation): Imprimitive gate to use, defaults to CNOT
        wires (Sequence[int]): Wires the model acts on
    """
    for i, wire in enumerate(wires):
        qml.ops.Rot(weights[i, 0], weights[i, 1], weights[i, 2], wires=wire)

    num_wires = len(wires)
    for i in range(num_wires) if periodic else range(num_wires-1):
        imprimitive_gate(wires=[wires[i], wires[(i+r) % num_wires]])

def CVNeuralNet(weights, wires=None):
    """A CV Quantum Neural Network

    Implemented the CV Quantum Neural Network architecture from
    :cite:`killoran2018continuous` for an arbitrary number of wires
    and layers.

    Args:
         weights (array[float]): Array of weights for each layer of the CV
                                 neural network
        wires (Sequence[int]): Wires the model acts on
    """
    for layer_weights in weights:
        CVNeuralNetLayer(layer_weights, wires=wires)

def CVNeuralNetLayer(weights, wires=None):
    PhaselessLinearInterferometer(weights[0], wires)
    for wire in wires:
        qml.Squeezing(weights[1], 0., wires=wire)
    PhaselessLinearInterferometer(weights[2], wires)
    for wire in wires:
        qml.Displacement(weights[3], 0., wires=wire)
    for wire in wires:
        qml.Kerr(weights[4], wires=wire)

def PhaselessLinearInterferometer(weights, wires=None):
    raise NotImplementedError("PhaselessLinearInterferometer not yet implemented")

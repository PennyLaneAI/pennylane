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

This module provides functions representing  circuits of common quantum
machine learning architectures to make it easy to use them as building blocks
for quantum machine learning models.

For example, you can define and call a variational quantum classifier on an
arbitrary number of wires and with an arbitrary number of layers in the
following way:

.. code-block:: python

    import pennylane as qml
    from pennylane import numpy as np

    num_wires=4
    num_layers=3
    dev = qml.device('default.qubit', wires=num_wires)

    @qml.qnode(dev)
    def circuit(weights, x=None):
        qml.BasisState(x, wires=range(num_wires))
        qml.model.VariationalClassifyer(weights, True, wires=range(num_wires))
        return qml.expval.PauliZ(0)

    np.random.seed(0)
    weights=np.random.randn(num_layers, num_wires, 3)
    print(circuit(weights, x=np.array(np.random.randint(0,1,num_wires))))


Summary
^^^^^^^

.. autosummary::
  variational_quantum_classifyer

Code details
^^^^^^^^^^^^
"""
import abc
import numbers
import logging as log

import autograd.numpy as np

from .qnode import QNode, QuantumFunctionError
from .utils import _flatten, _unflatten
from .variable import Variable

log.getLogger()

from pennylane.ops import *

def VariationalClassifyer(weights, periodic=True, wires=None):
    """A variational quantum classifier.

    Constructs the circuit of a variational quantum classifier with len(weights)
    layers on the given wires with the provided weights. Each element of
    weights must be a an array of size len(wires)*3.

    Args:
        weights (array[float]): layers*len(wires)*3 array of weights
        periodic (bool): whether to construct a periodic classifier circuit,
                         i.e., whether to apply a CNOT operation also between the
                         last and the first wire.
        wires (Sequence[int]): the wires the operation acts on
    """
    for layer_weights in weights:
        _variational_classifyer_layer(layer_weights, periodic, wires)


def _variational_classifyer_layer(weights, periodic=True, wires=None):
    """A single layer of a variational quantum classifier.

    Args:
        weights:
        periodic:
        wires:
    """
    for i, wire in enumerate(wires):
        Rot(weights[i, 0], weights[i, 1], weights[i, 2], wires=wire)

    num_wires = len(wires);
    for i in range(num_wires) if periodic else range(num_wires-1):
        CNOT(wires=[wires[i], wires[(i+1) % num_wires]])

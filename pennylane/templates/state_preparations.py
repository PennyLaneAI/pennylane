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
r"""
State Preparations
==========

**Module name:** :mod:`pennylane.templates.state_preparations`

.. currentmodule:: pennylane.templates.state_preparations

This module provides routines that prepare a given state using only
elementary gates.

Qubit architectures
-------------------

.. autosummary::

    BasisStatePreparation
    MöttönenStatePreparation

Code details
^^^^^^^^^^^^
"""
from collections.abc import Iterable

import pennylane as qml


def BasisStatePreparation(basis_state, wires):
    r"""
    Prepares a basis state on the given wires using a sequence of Pauli X gates.

    Args:
        basis_state (array): Input array of shape ``(N,)``, where N is the number of qubits,
            with :math:`N\leq n`
        wires (Sequence[int]): sequence of qubit indices that the template acts on
    """

    if not isinstance(wires, Iterable):
        raise ValueError(
            "Wires needs to be a list of wires that the embedding uses; got {}.".format(wires)
        )

    if not len(basis_state) == len(wires):
        raise ValueError(
            "Number of qubits must be equal to the number of wires, which is {}; "
            "got {}.".format(len(wires), len(basis_state))
        )

    if any([x not in [0, 1] for x in basis_state]):
        raise ValueError(
            "Basis state must only consist of 0s and 1s, got {}".format(basis_state)
        )

    for wire, state in zip(wires, basis_state):
        if state == 1:
            qml.PauliX(wire)

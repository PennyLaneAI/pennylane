# Copyright 2022 Xanadu Quantum Technologies Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Miscellaneous support for circuit cutting.
"""


import uuid

import pennylane as qml
from pennylane.operation import Operation


class MeasureNode(Operation):
    """Placeholder node for measurement operations"""

    num_wires = 1
    grad_method = None

    def __init__(self, *params, wires=None, id=None):
        id = id or str(uuid.uuid4())

        super().__init__(*params, wires=wires, id=id)


class PrepareNode(Operation):
    """Placeholder node for state preparations"""

    num_wires = 1
    grad_method = None

    def __init__(self, *params, wires=None, id=None):
        id = id or str(uuid.uuid4())

        super().__init__(*params, wires=wires, id=id)


def _prep_zero_state(wire):
    qml.Identity(wire)


def _prep_one_state(wire):
    qml.PauliX(wire)


def _prep_plus_state(wire):
    qml.Hadamard(wire)


def _prep_minus_state(wire):
    qml.PauliX(wire)
    qml.Hadamard(wire)


def _prep_iplus_state(wire):
    qml.Hadamard(wire)
    qml.S(wires=wire)


def _prep_iminus_state(wire):
    qml.PauliX(wire)
    qml.Hadamard(wire)
    qml.S(wires=wire)

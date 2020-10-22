# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fixtures and configuration for the collections package tests"""
import pytest

import pennylane as qml


@pytest.fixture
def qnodes(interface, tf_support, torch_support):
    """fixture returning some QNodes"""
    if interface == "torch" and not torch_support:
        pytest.skip("Skipped, no torch support")

    if interface == "tf" and not tf_support:
        pytest.skip("Skipped, no tf support")

    dev1 = qml.device("default.qubit", wires=2)
    dev2 = qml.device("default.qubit", wires=2)

    @qml.qnode(dev1, interface=interface)
    def qnode1(x):
        qml.RX(x[0], wires=0)
        qml.RY(x[1], wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.var(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

    @qml.qnode(dev2, interface=interface)
    def qnode2(x):
        qml.Hadamard(wires=0)
        qml.RX(x[0], wires=0)
        qml.RY(x[1], wires=1)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0)), qml.var(qml.PauliZ(1))

    return qnode1, qnode2

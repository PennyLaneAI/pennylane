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
"""
Unit tests for the :mod:`pennylane.collection` submodule.
"""
import numpy as np
import pytest

import pennylane as qml


def qnodes(interface):
    """Function returning some QNodes for a specific interface"""

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


def catch_warn_map(
    template,
    observables,
    device,
    measure="expval",
    interface="autograd",
    diff_method="best",
    **kwargs
):
    """Computes the map and catches the initial deprecation warning."""

    with pytest.warns(UserWarning, match="is deprecated"):
        res = qml.map(
            template,
            observables,
            device,
            measure=measure,
            interface=interface,
            diff_method=diff_method,
            **kwargs
        )
    return res


def catch_warn_QNodeCollection(qnodes):
    """Computes the apply and catches the initial deprecation warning."""

    with pytest.warns(UserWarning, match="is deprecated"):
        res = qml.QNodeCollection(qnodes)
    return res


def test_mape_errortemplate_not_callable():
    """Test that QNode collection map does not work with the new return system."""
    with pytest.raises(
        qml.QuantumFunctionError, match="QNodeCollections does not support the new return system."
    ):
        catch_warn_map(5, 0, 0)


def test_qnodes_error(tol):
    """Test that QNode collection does not work with the new return system."""
    qnode1, qnode2 = qnodes("autograd")
    with pytest.raises(
        qml.QuantumFunctionError, match="QNodeCollections does not support the new return system."
    ):
        catch_warn_QNodeCollection([qnode1, qnode2])

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
Unit tests for the :mod:`pennylane.qnode` decorator.
"""
# pylint: disable=protected-access,cell-var-from-loop
import numpy as np
import pytest

import pennylane as qml
from pennylane.qnodes import qnode, CVQNode, JacobianQNode, BaseQNode, QubitQNode


def test_create_qubit_qnode():
    """Test the decorator correctly creates Qubit QNodes"""
    dev = qml.device('default.qubit', wires=1)

    @qnode(dev)
    def circuit(a):
        qml.RX(a, wires=0)
        return qml.expval(qml.PauliZ(wires=0))

    assert isinstance(circuit, QubitQNode)
    assert hasattr(circuit, "jacobian")

def test_create_CV_qnode():
    """Test the decorator correctly creates Qubit QNodes"""
    dev = qml.device('default.gaussian', wires=1)

    @qnode(dev)
    def circuit(a):
        qml.Displacement(a, 0, wires=0)
        return qml.expval(qml.X(wires=0))

    assert isinstance(circuit, CVQNode)
    assert hasattr(circuit, "jacobian")


def test_fallback_Jacobian_qnode(monkeypatch):
    """Test the decorator fallsback to Jacobian QNode if it
    can't determine the device model"""
    dev = qml.device('default.gaussian', wires=1)
    dev._capabilities["model"] = None

    @qnode(dev)
    def circuit(a):
        qml.Displacement(a, 0, wires=0)
        return qml.expval(qml.X(wires=0))

    assert not isinstance(circuit, CVQNode)
    assert not isinstance(circuit, QubitQNode)
    assert isinstance(circuit, JacobianQNode)
    assert hasattr(circuit, "jacobian")


def test_torch_interface(skip_if_no_torch_support):
    """Test torch interface conversion"""
    dev = qml.device('default.qubit', wires=1)

    @qnode(dev, interface="torch")
    def circuit(a):
        qml.RX(a, wires=0)
        return qml.expval(qml.PauliZ(wires=0))

    assert circuit.interface == "torch"


def test_finite_diff_qubit_qnode():
    """Test that a finite-difference differentiable qubit QNode
    is correctly created when diff_method='finite-diff'"""
    dev = qml.device('default.qubit', wires=1)

    @qnode(dev, diff_method="finite-diff")
    def circuit(a):
        qml.RX(a, wires=0)
        return qml.expval(qml.PauliZ(wires=0))

    assert not isinstance(circuit, CVQNode)
    assert not isinstance(circuit, QubitQNode)
    assert isinstance(circuit, JacobianQNode)
    assert hasattr(circuit, "jacobian")


def test_tf_interface(skip_if_no_tf_support):
    """Test tf interface conversion"""
    dev = qml.device('default.qubit', wires=1)

    @qnode(dev, interface="tf")
    def circuit(a):
        qml.RX(a, wires=0)
        return qml.expval(qml.PauliZ(wires=0))

    assert circuit.interface == "tf"


def test_autograd_interface():
    """Test autograd interface conversion"""
    dev = qml.device('default.qubit', wires=1)

    @qnode(dev, interface="autograd")
    def circuit(a):
        qml.RX(a, wires=0)
        return qml.expval(qml.PauliZ(wires=0))

    assert circuit.interface == "autograd"


def test_no_interface():
    """Test no interface conversion"""
    dev = qml.device('default.qubit', wires=1)

    @qnode(dev, interface=None)
    def circuit(a):
        qml.RX(a, wires=0)
        return qml.expval(qml.PauliZ(wires=0))

    assert circuit.interface is None


def test_not_differentiable():
    """Test QNode marked as non-differentiable"""
    dev = qml.device('default.qubit', wires=1)

    @qnode(dev, interface=None, diff_method=None)
    def circuit(a):
        qml.RX(a, wires=0)
        return qml.expval(qml.PauliZ(wires=0))

    assert isinstance(circuit, BaseQNode)
    assert not isinstance(circuit, JacobianQNode)

    assert not hasattr(circuit, "interface")
    assert not hasattr(circuit, "jacobian")


def test_invalid_diff_method():
    """Test exception raised if an invalid diff
    method is provided"""
    dev = qml.device('default.qubit', wires=1)

    with pytest.raises(ValueError, match=r"Differentiation method \w+ not recognized"):
        @qnode(dev, interface=None, diff_method="test")
        def circuit(a):
            qml.RX(a, wires=0)
            return qml.expval(qml.PauliZ(wires=0))


def test_invalid_interface():
    """Test exception raised if an invalid interface
    is provided"""
    dev = qml.device('default.qubit', wires=1)

    with pytest.raises(ValueError, match=r"Interface \w+ not recognized"):
        @qnode(dev, interface="test")
        def circuit(a):
            qml.RX(a, wires=0)
            return qml.expval(qml.PauliZ(wires=0))

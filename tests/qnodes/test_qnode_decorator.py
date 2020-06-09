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
from pennylane.qnodes import qnode, CVQNode, JacobianQNode, BaseQNode, QubitQNode, ReversibleQNode
from pennylane.qnodes.jacobian import DEFAULT_STEP_SIZE_ANALYTIC, DEFAULT_STEP_SIZE


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

    # use monkeypatch to avoid setting class attributes
    with monkeypatch.context() as m:
        m.setitem(dev._capabilities, "model", "None")

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

step_sizes = [(True, DEFAULT_STEP_SIZE_ANALYTIC),
            (False, DEFAULT_STEP_SIZE)]


@pytest.mark.parametrize("analytic, step_size", step_sizes)
def test_finite_diff_qubit_qnode(analytic, step_size):
    """Test that a finite-difference differentiable qubit QNode
    is correctly created when diff_method='finite-diff' and analytic=True"""
    dev = qml.device('default.qubit', wires=1, analytic=analytic)

    @qnode(dev, diff_method="finite-diff")
    def circuit(a):
        qml.RX(a, wires=0)
        return qml.expval(qml.PauliZ(wires=0))

    assert not isinstance(circuit, CVQNode)
    assert not isinstance(circuit, QubitQNode)
    assert isinstance(circuit, JacobianQNode)
    assert hasattr(circuit, "jacobian")
    assert circuit.h == step_size
    assert circuit.order == 1


@pytest.mark.parametrize("order", [1, 2])
def test_setting_order(order):
    """Test that the order is correctly set and reset in a finite-difference QNode."""
    dev = qml.device('default.qubit', wires=1)

    @qnode(dev, diff_method="finite-diff", order=order)
    def circuit(a):
        qml.RX(a, wires=0)
        return qml.expval(qml.PauliZ(wires=0))

    assert circuit.order == order

    circuit.order = 1
    assert circuit.order == 1


def test_finite_diff_qubit_qnode_passing_step_size_through_decorator():
    """Test that a finite-difference differentiable qubit QNode is correctly
    created when diff_method='finite-diff' and the step size is set through the
    decorator."""
    step_size = 0.5
    new_step_size = 0.12345

    dev = qml.device('default.qubit', wires=1)

    @qnode(dev, diff_method="finite-diff", h=step_size)
    def circuit(a):
        qml.RX(a, wires=0)
        return qml.expval(qml.PauliZ(wires=0))

    assert not isinstance(circuit, CVQNode)
    assert not isinstance(circuit, QubitQNode)
    assert isinstance(circuit, JacobianQNode)
    assert hasattr(circuit, "jacobian")
    assert circuit.h == step_size

    circuit.h = new_step_size
    assert circuit.h == new_step_size


def test_reversible_diff_method():
    """Test that a ReversibleQNode can be created via the qnode decorator"""
    dev = qml.device('default.qubit', wires=1)

        @qnode(dev, diff_method="reversible")
        def circuit(a):
            qml.RX(a, wires=0)
            return qml.expval(qml.PauliZ(wires=0))

    assert isinstance(circuit, ReversibleQNode)

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

def test_classical_diff_method_unsupported():
    """Test exception raised if an the classical diff method is specified for a
    device that does not support it"""
    dev = qml.device('default.qubit', wires=1)

    with pytest.raises(ValueError, match=r"device does not support native computations with "
            "autodifferentiation frameworks"):

        @qnode(dev, diff_method="backprop")
        def circuit(a):
            qml.RX(a, wires=0)
            return qml.expval(qml.PauliZ(wires=0))

def test_device_diff_method_unsupported():
    """Test exception raised if an the device diff method is specified for a
    device that does not support it"""
    dev = qml.device('default.qubit', wires=1)

    with pytest.raises(ValueError, match=r"device does not provide a native method "
            "for computing the jacobian"):

        @qnode(dev, diff_method="device")
        def circuit(a):
            qml.RX(a, wires=0)
            return qml.expval(qml.PauliZ(wires=0))

def test_parameter_shift_diff_method_unsupported():
    """Test exception raised if an the device diff method is specified for a
    device that does not support it"""
    class DummyDevice(qml.plugins.DefaultQubit):

        @classmethod
        def capabilities(cls):
            return { "model": "NotSupportedModel"}


    dev = DummyDevice(wires=2)

    with pytest.raises(ValueError, match=r"The parameter shift rule is not available for devices with model"):

        @qnode(dev, diff_method="parameter-shift")
        def circuit(a):
            qml.RX(a, wires=0)
            return qml.expval(qml.PauliZ(wires=0))

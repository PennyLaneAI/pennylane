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
"""
Unit tests for the :mod:`pennylane.plugin.DefaultGaussian` device.
"""
# pylint: disable=protected-access,cell-var-from-loop
from pennylane import numpy as np
from scipy.linalg import block_diag

import pennylane as qml
from pennylane.expval import Identity, VarianceFactory, _qubit__all__, _cv__all__, __all__
from pennylane.qnode import QuantumFunctionError
from pennylane.plugins import DefaultQubit

import pytest


def test_identity_raises_exception_if_outside_qnode():
    """expval: Tests that proper exceptions are raised if we try to call
    Idenity outside a QNode."""
    with pytest.raises(QuantumFunctionError, match="can only be used inside a qfunc"):
        Identity(wires=0)


def test_identity_raises_exception_if_cannot_guess_device_type():
    """expval: Tests that proper exceptions are raised if Identity fails to guess
    whether on a device is CV or qubit."""
    dev = qml.device("default.qubit", wires=1)
    dev._expectation_map = {}

    @qml.qnode(dev)
    def circuit():
        return qml.expval.Identity(wires=0)

    with pytest.raises(
        QuantumFunctionError,
        match="Unable to determine whether this device supports CV or qubit",
    ):
        circuit()


def test_pass_positional_wires_to_expval(monkeypatch, capfd):
    """Tests whether the ability to pass wires as positional argument is retained"""
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def circuit():
        return qml.expval.Identity(0)

    with monkeypatch.context() as m:
        m.setattr(DefaultQubit, "pre_expval", lambda self: print(self.expval_queue))
        circuit()

    out, err = capfd.readouterr()
    assert "pennylane.expval.qubit.Identity object" in out


class TestVarianceFactory:
    """Tests for the variance factory"""

    def test_non_existent_observable(self):
        """Test exception raised if attribute does not exist"""
        var = VarianceFactory()
        with pytest.raises(AttributeError, match=r"'pennylane\.var' has no attribute 'test'"):
            var.test

        with pytest.raises(AttributeError, match=r"'pennylane\.var\.cv' has no attribute 'PauliX'"):
            var.cv.PauliX

        with pytest.raises(AttributeError, match=r"'pennylane\.var\.qubit' has no attribute 'X'"):
            var.qubit.X

    def test_loading_outside_qnode(self):
        """Test the loading of attributes outside of a QNode"""
        var = VarianceFactory()
        v1 = var.Identity(0, do_queue=False)
        assert isinstance(v1, qml.operation.Expectation)
        assert v1.return_type == 'variance'
        assert v1.__doc__.split("\n")[0] == "pennylane.var.Identity(wires)"

        v1 = var.PauliX(0, do_queue=False)
        assert isinstance(v1, qml.operation.Expectation)
        assert v1.return_type == 'variance'
        assert v1.__doc__.split("\n")[0] == "pennylane.var.PauliX(wires)"

    def test_loading_inside_qnode(self):
        """Test the loading of attributes inside a QNode"""
        var = VarianceFactory()
        dev = qml.device('default.gaussian', wires=2)

        def circuit():
            return var.Identity(0)

        qnode = qml.QNode(circuit, dev)
        qnode() # construct the QNode
        v1 = qnode.ev[0]
        assert isinstance(v1, qml.operation.Expectation)
        assert v1.return_type == 'variance'
        assert v1.__doc__.split("\n")[0] == "pennylane.var.Identity(wires)"

    def test_loading_from_nested_attribute(self):
        """Test the loading of nested attributes"""
        var = VarianceFactory()

        v1 = var.qubit.Hadamard(0, do_queue=False)
        assert isinstance(v1, qml.operation.Expectation)
        assert not isinstance(v1, qml.operation.CV)
        assert v1.return_type == 'variance'

        v1 = var.cv.MeanPhoton(0, do_queue=False)
        assert isinstance(v1, qml.operation.Expectation)
        assert isinstance(v1, qml.operation.CV)
        assert v1.return_type == 'variance'

    def test_ambiguous_device_type(self):
        """Tests that proper exceptions are raised if the VarianceFactory
        fails to guess whether on a device is CV or qubit."""
        dev = qml.device("default.qubit", wires=1)
        dev._expectation_map = {}

        @qml.qnode(dev)
        def circuit():
            return qml.var.Identity(wires=0)

        with pytest.raises(QuantumFunctionError, match="Unable to determine whether this device supports CV or qubit"):
            circuit()

    def test_dir_method(self):
        """Test that the __dir__ special method is properly filled"""
        var = VarianceFactory()
        assert var.__dir__() == __all__
        assert var.cv.__dir__() == _cv__all__
        assert var.qubit.__dir__() == _qubit__all__

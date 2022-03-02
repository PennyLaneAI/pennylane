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
"""Tests for caching executions of the quantum device."""
import pytest

import pennylane as qml
from pennylane.measurements import expval
from pennylane import qnode, QNode
from pennylane.devices import DefaultQubit
from pennylane import numpy as np


def qfunc(x, y):
    """Simple quantum function"""
    qml.RX(x, wires=0)
    qml.RX(y, wires=1)
    qml.CNOT(wires=[0, 1])
    return expval(qml.PauliZ(wires=1))


def get_device_with_caching(wires, cache):
    """Get a device that defines caching."""
    with pytest.warns(UserWarning, match="deprecated"):
        dev = qml.device("default.qubit", wires=wires, cache=cache)

    return dev


class TestCaching:
    """Tests for device caching"""

    def test_set_and_get(self):
        """Test that the cache attribute can be set and accessed"""
        dev = qml.device("default.qubit", wires=2)
        assert dev.cache == 0

        dev = get_device_with_caching(wires=2, cache=10)
        assert dev.cache == 10

    def test_no_caching(self, mocker):
        """Test that no caching occurs when the cache attribute is equal to zero"""
        dev = qml.device("default.qubit", wires=2, cache=0)
        qn = QNode(qfunc, dev)

        spy = mocker.spy(DefaultQubit, "apply")
        qn(0.1, 0.2)
        qn(0.1, 0.2)

        assert len(spy.call_args_list) == 2
        assert len(dev._cache_execute) == 0

    def test_caching(self, mocker):
        """Test that caching occurs when the cache attribute is above zero"""
        dev = get_device_with_caching(wires=2, cache=10)
        qn = QNode(qfunc, dev)

        qn(0.1, 0.2)
        spy = mocker.spy(DefaultQubit, "apply")
        qn(0.1, 0.2)

        spy.assert_not_called()
        assert len(dev._cache_execute) == 1

    def test_add_to_cache_execute(self):
        """Test that the _cache_execute attribute is added to when the device is executed"""
        dev = get_device_with_caching(wires=2, cache=10)
        qn = QNode(qfunc, dev)

        result = qn(0.1, 0.2)
        cache_execute = dev._cache_execute
        hashed = qn.qtape.graph.hash

        assert len(cache_execute) == 1
        assert hashed in cache_execute
        assert np.allclose(cache_execute[hashed], result)

    def test_fill_cache(self):
        """Test that the cache is added to until it reaches its maximum size (in this case 10),
        and then maintains that size upon subsequent additions."""
        dev = get_device_with_caching(wires=2, cache=10)
        qn = QNode(qfunc, dev)

        args = np.arange(20)

        for i, arg in enumerate(args[:10]):
            qn(0.1, arg)
            assert len(dev._cache_execute) == i + 1

        for arg in args[10:]:
            qn(0.1, arg)
            assert len(dev._cache_execute) == 10

    def test_drop_from_cache(self):
        """Test that the first entry of the _cache_execute dictionary is the first to be dropped
        from the dictionary once it becomes full"""
        dev = get_device_with_caching(wires=2, cache=2)
        qn = QNode(qfunc, dev)

        qn(0.1, 0.2)
        first_hash = list(dev._cache_execute.keys())[0]

        qn(0.1, 0.3)
        assert first_hash in dev._cache_execute
        qn(0.1, 0.4)
        assert first_hash not in dev._cache_execute

    def test_caching_multiple_values(self, mocker):
        """Test that multiple device executions with different params are cached and accessed on
        subsequent executions"""
        dev = get_device_with_caching(wires=2, cache=10)
        qn = QNode(qfunc, dev)

        args = np.arange(10)

        for arg in args[:10]:
            qn(0.1, arg)

        spy = mocker.spy(DefaultQubit, "apply")
        for arg in args[:10]:
            qn(0.1, arg)

        spy.assert_not_called()

    def test_backprop_error(self):
        """Test if an error is raised when caching is used with the backprop diff_method"""
        dev = get_device_with_caching(wires=2, cache=10)
        with pytest.raises(
            qml.QuantumFunctionError, match="Device caching is incompatible with the backprop"
        ):
            QNode(qfunc, dev, diff_method="backprop")

    def test_gradient_autograd(self, mocker):
        """Test that caching works when calculating the gradient using the autograd
        interface"""
        dev = get_device_with_caching(wires=2, cache=10)
        qn = QNode(qfunc, dev, interface="autograd")
        d_qnode = qml.grad(qn)
        args = np.array([0.1, 0.2], requires_grad=True)

        d_qnode(*args)
        spy = mocker.spy(DefaultQubit, "apply")
        d_qnode(*args)
        spy.assert_not_called()

    @pytest.mark.usefixtures("skip_if_no_tf_support")
    def test_gradient_tf(self, mocker):
        """Test that caching works when calculating the gradient using the TF interface"""
        import tensorflow as tf

        dev = get_device_with_caching(wires=2, cache=10)
        qn = QNode(qfunc, dev, interface="tf")
        args0 = tf.Variable(0.1)
        args1 = tf.Variable(0.2)

        with tf.GradientTape() as tape:
            res = qn(args0, args1)

        grad = tape.gradient(res, args0)
        assert grad is not None

        spy = mocker.spy(DefaultQubit, "apply")
        with tf.GradientTape() as tape:
            res = qn(args0, args1)

        tape.gradient(res, args0)
        spy.assert_not_called()

    @pytest.mark.usefixtures("skip_if_no_torch_support")
    def test_gradient_torch(self, mocker):
        """Test that caching works when calculating the gradient using the Torch interface"""
        import torch

        dev = get_device_with_caching(wires=2, cache=10)
        qn = QNode(qfunc, dev, interface="torch")
        args0 = torch.tensor(0.1, requires_grad=True)
        args1 = torch.tensor(0.2)

        res = qn(args0, args1)
        res.backward()
        assert args0.grad is not None

        spy = mocker.spy(DefaultQubit, "apply")
        res = qn(args0, args1)
        res.backward()
        spy.assert_not_called()

    def test_mutable_circuit(self, mocker):
        """Test that caching is compatible with circuit mutability. Caching should take place if
        the circuit and parameters are the same, and should not take place if the parameters are
        the same but the circuit different."""
        dev = get_device_with_caching(wires=2, cache=10)

        @qnode(dev)
        def qfunc(x, y, flag=1):
            if flag == 1:
                qml.RX(x, wires=0)
                qml.RX(y, wires=1)
            else:
                qml.RX(x, wires=1)
                qml.RX(y, wires=1)
            qml.CNOT(wires=[0, 1])
            return expval(qml.PauliZ(wires=1))

        spy = mocker.spy(DefaultQubit, "apply")
        qfunc(0.1, 0.2)
        qfunc(0.1, 0.2)
        assert len(spy.call_args_list) == 1

        qfunc(0.1, 0.2, flag=0)
        assert len(spy.call_args_list) == 2

    def test_classical_processing_in_circuit(self, mocker):
        """Test if caching is compatible with QNodes that include classical processing"""

        dev = get_device_with_caching(wires=2, cache=10)

        @qnode(dev)
        def qfunc(x, y):
            qml.RX(x**2, wires=0)
            qml.RX(x / y, wires=1)
            qml.CNOT(wires=[0, 1])
            return expval(qml.PauliZ(wires=1))

        spy = mocker.spy(DefaultQubit, "apply")
        qfunc(0.1, 0.2)
        qfunc(0.1, 0.2)
        assert len(spy.call_args_list) == 1

        qfunc(0.1, 0.3)
        assert len(spy.call_args_list) == 2

    def test_grad_classical_processing_in_circuit(self, mocker):
        """Test that caching is compatible with calculating the gradient in QNodes which contain
        classical processing"""

        dev = get_device_with_caching(wires=2, cache=10)

        @qnode(dev)
        def qfunc(x, y):
            qml.RX(x**2, wires=0)
            qml.RX(x / y, wires=1)
            qml.CNOT(wires=[0, 1])
            return expval(qml.PauliZ(wires=1))

        d_qfunc = qml.grad(qfunc)

        spy = mocker.spy(DefaultQubit, "apply")
        x, y = np.array(0.1, requires_grad=True), np.array(0.2, requires_grad=True)
        g = d_qfunc(x, y)
        calls1 = len(spy.call_args_list)
        d_qfunc(x, y)
        calls2 = len(spy.call_args_list)

        x, y = np.array(0.1, requires_grad=True), np.array(0.3, requires_grad=True)
        d_qfunc(x, y)
        calls3 = len(spy.call_args_list)

        assert calls1 == 5
        assert calls2 == 5
        assert calls3 == 10
        assert g is not None

    devs = [
        (qml.device("default.qubit", wires=2)),
        (get_device_with_caching(wires=2, cache=10)),
    ]

    @pytest.mark.parametrize("dev", devs)
    def test_different_return_type(self, dev):
        """Test that same circuit with different return type, returns different results"""
        wires = range(2)

        hamiltonian = np.array(
            [
                [-2.5623 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 - 0.1234j],
                [0.0 + 0.0j, -2.5623 + 0.0j, 0.0 + 0.1234j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 - 0.1234j, -2.5623 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.1234j, 0.0 + 0.0j, 0.0 + 0.0j, -2.5623 + 0.0j],
            ]
        )

        np.random.seed(172)
        params = np.random.randn(2, 2)

        @qml.qnode(dev, diff_method="parameter-shift")
        def expval_circuit(params):
            qml.templates.BasicEntanglerLayers(params, wires=wires, rotation=qml.RX)
            return qml.expval(qml.Hermitian(hamiltonian, wires=wires))

        @qml.qnode(dev, diff_method="parameter-shift")
        def var_circuit(params):
            qml.templates.BasicEntanglerLayers(params, wires=wires, rotation=qml.RX)
            return qml.var(qml.Hermitian(hamiltonian, wires=wires))

        assert expval_circuit(params) != var_circuit(params)

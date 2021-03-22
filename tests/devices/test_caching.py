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
import numpy as np
import pytest

import pennylane as qml
from pennylane.measure import expval
from pennylane import qnode, QNode
from pennylane.devices import DefaultQubit


def qfunc(x, y):
    """Simple quantum function"""
    qml.RX(x, wires=0)
    qml.RX(y, wires=1)
    qml.CNOT(wires=[0, 1])
    return expval(qml.PauliZ(wires=1))


class TestCaching:
    """Tests for device caching"""

    def test_set_and_get(self):
        """Test that the cache attribute can be set and accessed"""
        dev = qml.device("default.qubit", wires=2)
        assert dev.cache == 0

        dev = qml.device("default.qubit", wires=2, cache=10)
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
        dev = qml.device("default.qubit", wires=2, cache=10)
        qn = QNode(qfunc, dev)

        qn(0.1, 0.2)
        spy = mocker.spy(DefaultQubit, "apply")
        qn(0.1, 0.2)

        spy.assert_not_called()
        assert len(dev._cache_execute) == 1

    def test_add_to_cache_execute(self):
        """Test that the _cache_execute attribute is added to when the device is executed"""
        dev = qml.device("default.qubit", wires=2, cache=10)
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
        dev = qml.device("default.qubit", wires=2, cache=10)
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
        dev = qml.device("default.qubit", wires=2, cache=2)
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
        dev = qml.device("default.qubit", wires=2, cache=10)
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
        dev = qml.device("default.qubit", wires=2, cache=10)
        with pytest.raises(
            qml.QuantumFunctionError, match="Device caching is incompatible with the backprop"
        ):
            QNode(qfunc, dev, diff_method="backprop")

    def test_gradient_autograd(self, mocker):
        """Test that caching works when calculating the gradient using the autograd
        interface"""
        dev = qml.device("default.qubit", wires=2, cache=10)
        qn = QNode(qfunc, dev, interface="autograd")
        d_qnode = qml.grad(qn)
        args = [0.1, 0.2]

        d_qnode(*args)
        spy = mocker.spy(DefaultQubit, "apply")
        d_qnode(*args)
        spy.assert_not_called()

    @pytest.mark.usefixtures("skip_if_no_tf_support")
    def test_gradient_tf(self, mocker):
        """Test that caching works when calculating the gradient using the TF interface"""
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=2, cache=10)
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

        dev = qml.device("default.qubit", wires=2, cache=10)
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
        dev = qml.device("default.qubit", wires=3, cache=10)

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

        dev = qml.device("default.qubit", wires=3, cache=10)

        @qnode(dev)
        def qfunc(x, y):
            qml.RX(x ** 2, wires=0)
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

        dev = qml.device("default.qubit", wires=3, cache=10)

        @qnode(dev)
        def qfunc(x, y):
            qml.RX(x ** 2, wires=0)
            qml.RX(x / y, wires=1)
            qml.CNOT(wires=[0, 1])
            return expval(qml.PauliZ(wires=1))

        d_qfunc = qml.grad(qfunc)

        spy = mocker.spy(DefaultQubit, "apply")
        g = d_qfunc(0.1, 0.2)
        calls1 = len(spy.call_args_list)
        d_qfunc(0.1, 0.2)
        calls2 = len(spy.call_args_list)

        d_qfunc(0.1, 0.3)
        calls3 = len(spy.call_args_list)

        assert calls1 == 5
        assert calls2 == 5
        assert calls3 == 10
        assert g is not None

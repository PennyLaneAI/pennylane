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
"""Tests for caching executions of the quantum tape and QNode."""
import numpy as np
import pytest

import pennylane as qml
from pennylane.tape.measure import expval
from pennylane.tape.tapes import JacobianTape
from pennylane.tape import qnode
from pennylane.devices import DefaultQubit
from pennylane.devices.default_qubit_autograd import DefaultQubitAutograd


def get_tape(caching):
    """Creates a simple quantum tape"""
    with JacobianTape(caching=caching) as tape:
        qml.QubitUnitary(np.eye(2), wires=0)
        qml.RX(0.1, wires=0)
        qml.RX(0.2, wires=1)
        qml.CNOT(wires=[0, 1])
        expval(qml.PauliZ(wires=1))
    return tape


def get_qnode(caching, diff_method="finite-diff", interface="autograd"):
    """Creates a simple QNode"""
    dev = qml.device("default.qubit.autograd", wires=3)

    @qnode(dev, caching=caching, diff_method=diff_method, interface=interface)
    def qfunc(x, y):
        qml.RX(x, wires=0)
        qml.RX(y, wires=1)
        qml.CNOT(wires=[0, 1])
        return expval(qml.PauliZ(wires=1))

    return qfunc


class TestTapeCaching:
    """Tests for caching when using quantum tape"""

    def test_set_and_get(self):
        """Test that the caching attribute can be set and accessed"""
        tape = JacobianTape()
        assert tape.caching == 0

        tape = JacobianTape(caching=10)
        assert tape.caching == 10

    def test_no_caching(self, mocker):
        """Test that no caching occurs when the caching attribute is equal to zero"""
        dev = qml.device("default.qubit", wires=2)
        tape = get_tape(0)

        spy = mocker.spy(DefaultQubit, "execute")
        tape.execute(device=dev)
        tape.execute(device=dev)

        assert len(spy.call_args_list) == 2
        assert len(tape._cache_execute) == 0

    def test_caching(self, mocker):
        """Test that caching occurs when the caching attribute is above zero"""
        dev = qml.device("default.qubit", wires=2)
        tape = get_tape(10)

        tape.execute(device=dev)
        spy = mocker.spy(DefaultQubit, "execute")
        tape.execute(device=dev)

        spy.assert_not_called()
        assert len(tape._cache_execute) == 1

    def test_add_to_cache_execute(self):
        """Test that the _cache_execute attribute is added to when the tape is executed"""
        dev = qml.device("default.qubit", wires=2)
        tape = get_tape(10)

        result = tape.execute(device=dev)
        cache_execute = tape._cache_execute
        hashed = tape.graph.hash

        assert len(cache_execute) == 1
        assert hashed in cache_execute
        assert np.allclose(cache_execute[hashed], result)

    def test_fill_cache(self):
        """Test that the cache is added to until it reaches its maximum size (in this case 10),
        and then maintains that size upon subsequent additions."""
        dev = qml.device("default.qubit", wires=2)
        tape = get_tape(10)

        tape.trainable_params = {1}
        args = np.arange(20)

        for i, arg in enumerate(args[:10]):
            tape.execute(params=[arg], device=dev)
            assert len(tape._cache_execute) == i + 1

        for arg in args[10:]:
            tape.execute(params=[arg], device=dev)
            assert len(tape._cache_execute) == 10

    def test_drop_from_cache(self):
        """Test that the first entry of the _cache_execute dictionary is the first to be dropped
         from the dictionary once it becomes full"""
        dev = qml.device("default.qubit", wires=2)
        tape = get_tape(2)

        tape.trainable_params = {1}
        tape.execute(device=dev)
        first_hash = list(tape._cache_execute.keys())[0]

        tape.execute(device=dev, params=[0.2])
        assert first_hash in tape._cache_execute
        tape.execute(device=dev, params=[0.3])
        assert first_hash not in tape._cache_execute

    def test_caching_multiple_values(self, mocker):
        """Test that multiple device executions with different params are cached and accessed on
        subsequent executions"""
        dev = qml.device("default.qubit", wires=2)
        tape = get_tape(10)

        tape.trainable_params = {1}
        args = np.arange(10)

        for arg in args[:10]:
            tape.execute(params=[arg], device=dev)

        spy = mocker.spy(DefaultQubit, "execute")
        for arg in args[:10]:
            tape.execute(params=[arg], device=dev)

        spy.assert_not_called()


class TestQNodeCaching:
    """Tests for caching when using the QNode"""

    def test_set_and_get(self):
        """Test that the caching attribute can be set and accessed"""
        qnode = get_qnode(caching=0)
        assert qnode.caching == 0

        qnode = get_qnode(caching=10)
        assert qnode.caching == 10

    def test_backprop_error(self):
        """Test if an error is raised when caching is used with the backprop diff_method"""
        with pytest.raises(ValueError, match="Caching mode is incompatible"):
            get_qnode(caching=10, diff_method="backprop")

    def test_caching(self, mocker):
        """Test that multiple device executions with different params are cached and accessed on
        subsequent executions"""
        qnode = get_qnode(caching=10)
        args = np.arange(10)

        for arg in args:
            qnode(arg, 0.2)

        assert qnode.qtape.caching == 10

        spy = mocker.spy(DefaultQubitAutograd, "execute")
        for arg in args:
            qnode(arg, 0.2)

        spy.assert_not_called()

    def test_gradient_autograd(self, mocker):
        """Test that caching works when calculating the gradient method using the autograd
        interface"""
        qnode = get_qnode(caching=10, interface="autograd")
        d_qnode = qml.grad(qnode)
        args = [0.1, 0.2]

        d_qnode(*args)
        spy = mocker.spy(DefaultQubitAutograd, "execute")
        d_qnode(*args)
        spy.assert_not_called()

    @pytest.mark.usefixtures("skip_if_no_tf_support")
    def test_gradient_tf(self, mocker):
        """Test that caching works when calculating the gradient method using the TF interface"""
        import tensorflow as tf

        qnode = get_qnode(caching=10, interface="tf")
        args0 = tf.Variable(0.1)
        args1 = tf.Variable(0.2)

        with tf.GradientTape() as tape:
            res = qnode(args0, args1)

        grad = tape.gradient(res, args0)
        assert grad is not None

        spy = mocker.spy(DefaultQubitAutograd, "execute")
        with tf.GradientTape() as tape:
            res = qnode(args0, args1)

        tape.gradient(res, args0)
        spy.assert_not_called()

    @pytest.mark.usefixtures("skip_if_no_torch_support")
    def test_gradient_torch(self, mocker):
        """Test that caching works when calculating the gradient method using the Torch interface"""
        import torch

        qnode = get_qnode(caching=10, interface="torch")
        args0 = torch.tensor(0.1, requires_grad=True)
        args1 = torch.tensor(0.2)

        res = qnode(args0, args1)
        res.backward()
        assert args0.grad is not None

        spy = mocker.spy(DefaultQubitAutograd, "execute")
        res = qnode(args0, args1)
        res.backward()
        spy.assert_not_called()

    def test_mutable_circuit(self, mocker):
        """Test that caching is compatible with circuit mutability. Caching should take place if
        the circuit and parameters are the same, and should not take place if the parameters are
        the same but the circuit different."""
        dev = qml.device("default.qubit", wires=3)

        @qnode(dev, caching=10)
        def qfunc(x, y, flag=1):
            if flag == 1:
                qml.RX(x, wires=0)
                qml.RX(y, wires=1)
            else:
                qml.RX(x, wires=1)
                qml.RX(y, wires=1)
            qml.CNOT(wires=[0, 1])
            return expval(qml.PauliZ(wires=1))

        spy = mocker.spy(DefaultQubit, "execute")
        qfunc(0.1, 0.2)
        qfunc(0.1, 0.2)
        assert len(spy.call_args_list) == 1

        qfunc(0.1, 0.2, flag=0)
        assert len(spy.call_args_list) == 2

    def test_classical_processing_in_circuit(self, mocker):
        """Test if caching is compatible with QNodes that include classical processing"""

        dev = qml.device("default.qubit", wires=3)

        @qnode(dev, caching=10)
        def qfunc(x, y):
            qml.RX(x ** 2, wires=0)
            qml.RX(x / y, wires=1)
            qml.CNOT(wires=[0, 1])
            return expval(qml.PauliZ(wires=1))

        spy = mocker.spy(DefaultQubit, "execute")
        qfunc(0.1, 0.2)
        qfunc(0.1, 0.2)
        assert len(spy.call_args_list) == 1

        qfunc(0.1, 0.3)
        assert len(spy.call_args_list) == 2

    def test_grad_classical_processing_in_circuit(self, mocker):
        """Test that caching is compatible with calculating the gradient in QNodes which contain
        classical processing"""

        dev = qml.device("default.qubit", wires=3)

        @qnode(dev, caching=10)
        def qfunc(x, y):
            qml.RX(x ** 2, wires=0)
            qml.RX(x / y, wires=1)
            qml.CNOT(wires=[0, 1])
            return expval(qml.PauliZ(wires=1))

        d_qfunc = qml.grad(qfunc)

        spy = mocker.spy(DefaultQubit, "execute")
        g = d_qfunc(0.1, 0.2)
        calls1 = len(spy.call_args_list)
        d_qfunc(0.1, 0.2)
        calls2 = len(spy.call_args_list)
        assert calls1 == calls2

        d_qfunc(0.1, 0.3)
        calls3 = len(spy.call_args_list)
        assert calls3 == 2 * calls1

        assert g is not None

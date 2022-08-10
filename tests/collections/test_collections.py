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
import pytest
import numpy as np

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


class TestMap:
    """Test for mapping ansatz over observables or devices,
    to return a QNode collection"""

    def test_template_not_callable(self):
        """Test that an exception is correctly called if a
        template is not callable"""
        with pytest.raises(ValueError, match="template is not a callable"):
            qml.map(5, 0, 0)

    def test_mapping_over_observables(self):
        """Test that mapping over a list of observables produces
        a QNodeCollection with the correct QNodes, with a single
        device broadcast."""
        dev = qml.device("default.qubit", wires=1)
        obs_list = [qml.PauliX(0), qml.PauliY(0)]
        template = lambda x, wires: qml.RX(x, wires=0)

        qc = qml.map(template, obs_list, dev)

        assert len(qc) == 2

        # evaluate collection so that queue is populated
        qc(1)

        queue = qc[0].qtape.operations + qc[0].qtape.observables
        assert len(queue) == 2
        assert queue[0].name == "RX"
        assert queue[1].name == "PauliX"

        queue = qc[1].qtape.operations + qc[1].qtape.observables
        assert len(queue) == 2
        assert queue[0].name == "RX"
        assert queue[1].name == "PauliY"

    def test_mapping_over_observables_as_tuples(self):
        """Test that mapping over a tuple of observables produces
        a QNodeCollection with the correct QNodes, with a single
        device broadcast."""
        dev = qml.device("default.qubit", wires=1)
        obs_list = (qml.PauliX(0), qml.PauliY(0))
        template = lambda x, wires: qml.RX(x, wires=0)

        qc = qml.map(template, obs_list, dev)

        assert len(qc) == 2

        # evaluate collection so that queue is populated
        qc(1)

        queue = qc[0].qtape.operations + qc[0].qtape.observables
        assert len(queue) == 2
        assert queue[0].name == "RX"
        assert queue[1].name == "PauliX"

        queue = qc[1].qtape.operations + qc[1].qtape.observables
        assert len(queue) == 2
        assert queue[0].name == "RX"
        assert queue[1].name == "PauliY"

    def test_mapping_over_devices(self):
        """Test that mapping over a list of devices produces
        a QNodeCollection with the correct QNodes"""
        dev_list = [qml.device("default.qubit", wires=1), qml.device("default.qubit", wires=1)]

        obs_list = [qml.PauliX(0), qml.PauliY(0)]
        template = lambda x, wires: qml.RX(x, wires=0)

        qc = qml.map(template, obs_list, dev_list)

        assert len(qc) == 2

        # evaluate collection so that queue is populated
        qc(1)

        queue = qc[0].qtape.operations + qc[0].qtape.observables
        assert len(queue) == 2
        assert queue[0].name == "RX"
        assert queue[1].name == "PauliX"

        queue = qc[1].qtape.operations + qc[1].qtape.observables
        assert len(queue) == 2
        assert queue[0].name == "RX"
        assert queue[1].name == "PauliY"

        # test that device is not broadcast
        assert qc[0].device is not qc[1].device

    def test_mapping_over_measurements(self):
        """Test that mapping over a list of measurement types produces
        a QNodeCollection with the correct QNodes"""
        dev = qml.device("default.qubit", wires=1)

        obs_list = [qml.PauliX(0), qml.PauliY(0)]
        template = lambda x, wires: qml.RX(x, wires=0)

        qc = qml.map(template, obs_list, dev, measure=["expval", "var"])

        assert len(qc) == 2

        # evaluate collection so that queue is populated
        qc(1)

        queue = qc[0].qtape.operations + qc[0].qtape.observables
        assert len(queue) == 2
        assert queue[0].name == "RX"
        assert queue[1].name == "PauliX"
        assert queue[1].return_type == qml.measurements.Expectation

        queue = qc[1].qtape.operations + qc[1].qtape.observables
        assert len(queue) == 2
        assert queue[0].name == "RX"
        assert queue[1].name == "PauliY"
        assert queue[1].return_type == qml.measurements.Variance

    def test_invalid_observable(self):
        """Test that an invalid observable raises an exception"""
        dev = qml.device("default.qubit", wires=1)

        obs_list = [qml.PauliX(0), qml.S(wires=0)]
        template = lambda x, wires: qml.RX(x, wires=0)

        with pytest.raises(ValueError, match="Some or all observables are not valid"):
            qml.map(template, obs_list, dev, measure=["expval", "var"])

    def test_passing_kwargs(self):
        """Test that the step size and order used for the finite differences
        differentiation method were passed to the QNode instances using the
        keyword arguments."""
        dev = qml.device("default.qubit", wires=1)

        obs_list = [qml.PauliX(0), qml.PauliY(0)]
        template = lambda x, wires: qml.RX(x, wires=0)

        qc = qml.map(template, obs_list, dev, measure=["expval", "var"], h=123, order=2)

        qc(1)

        assert len(qc) == 2

        # Checking the h attribute which contains the step size
        assert qc[0].gradient_kwargs["h"] == 123
        assert qc[1].gradient_kwargs["h"] == 123

        # Checking that the order is set in each QNode
        assert qc[0].gradient_kwargs["order"] == 2
        assert qc[1].gradient_kwargs["order"] == 2


class TestApply:
    """Tests for the apply function"""

    @pytest.mark.autograd
    def test_apply_summation_autograd(self, tol):
        """Test that summation can be applied using autograd"""
        qnode1, qnode2 = qnodes("autograd")
        qc = qml.QNodeCollection([qnode1, qnode2])
        cost = qml.collections.apply(np.sum, qc)

        params = [0.5643, -0.45]
        res = cost(params)
        expected = np.sum(qc(params))

        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.torch
    def test_apply_summation_torch(self, tol):
        """Test that summation can be applied using torch"""
        import torch

        qnode1, qnode2 = qnodes("torch")
        qc = qml.QNodeCollection([qnode1, qnode2])
        cost = qml.collections.apply(torch.sum, qc)

        params = [0.5643, -0.45]
        res = cost(params)
        expected = torch.sum(qc(params))

        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.tf
    def test_apply_summation_tf(self, tol):
        """Test that summation can be applied using tf"""
        import tensorflow as tf

        qnode1, qnode2 = qnodes("tf")
        qc = qml.QNodeCollection([qnode1, qnode2])
        cost = qml.collections.apply(tf.reduce_sum, qc)

        params = [0.5643, -0.45]
        res = cost(params)
        expected = tf.reduce_sum(qc(params))

        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.jax
    def test_apply_summation_jax(self, tol):
        """Test that summation can be applied using jax"""
        import jax.numpy as jnp

        qnode1, qnode2 = qnodes("jax")
        qc = qml.QNodeCollection([qnode1, qnode2])
        cost = qml.collections.apply(jnp.sum, qc)

        params = [0.5643, -0.45]
        res = cost(params)
        expected = jnp.sum(qc(params))

        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.autograd
    def test_nested_apply_autograd(self, tol):
        """Test that nested apply can be done using autograd"""
        qnode1, qnode2 = qnodes("autograd")
        qc = qml.QNodeCollection([qnode1, qnode2])

        sinfn = np.sin
        sfn = np.sum

        cost = qml.collections.apply(sfn, qml.collections.apply(sinfn, qc))

        params = [0.5643, -0.45]
        res = cost(params)
        expected = sfn(sinfn(qc(params)))

        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.torch
    def test_nested_apply_torch(self, tol):
        """Test that nested apply can be done using torch"""
        import torch

        qnode1, qnode2 = qnodes("torch")
        qc = qml.QNodeCollection([qnode1, qnode2])

        sinfn = torch.sin
        sfn = torch.sum

        cost = qml.collections.apply(sfn, qml.collections.apply(sinfn, qc))

        params = [0.5643, -0.45]
        res = cost(params)
        expected = sfn(sinfn(qc(params)))

        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.tf
    def test_nested_apply_tf(self, tol):
        """Test that nested apply can be done using tf"""
        import tensorflow as tf

        qnode1, qnode2 = qnodes("tf")
        qc = qml.QNodeCollection([qnode1, qnode2])

        sinfn = tf.sin
        sfn = tf.reduce_sum

        cost = qml.collections.apply(sfn, qml.collections.apply(sinfn, qc))

        params = [0.5643, -0.45]
        res = cost(params)
        expected = sfn(sinfn(qc(params)))

        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.jax
    def test_nested_apply_jax(self, tol):
        """Test that nested apply can be done using jax"""
        import jax.numpy as jnp

        qnode1, qnode2 = qnodes("jax")
        qc = qml.QNodeCollection([qnode1, qnode2])

        sinfn = jnp.sin
        sfn = jnp.sum

        cost = qml.collections.apply(sfn, qml.collections.apply(sinfn, qc))

        params = [0.5643, -0.45]
        res = cost(params)
        expected = sfn(sinfn(qc(params)))

        assert np.allclose(res, expected, atol=tol, rtol=0)


class TestSum:
    """Tests for the sum function"""

    def test_apply_summation_vanilla(self, tol):
        """Test that summation can be applied"""
        qnode1, qnode2 = qnodes(None)
        qc = qml.QNodeCollection([qnode1, qnode2])
        cost = qml.sum(qc)

        params = [0.5643, -0.45]
        res = cost(params)
        expected = sum(qc[0](params) + qc[1](params))

        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.autograd
    def test_apply_summation_autograd(self, tol):
        """Test that summation can be applied with autograd"""
        qnode1, qnode2 = qnodes("autograd")
        qc = qml.QNodeCollection([qnode1, qnode2])
        cost = qml.sum(qc)

        params = [0.5643, -0.45]
        res = cost(params)
        expected = sum(qc[0](params) + qc[1](params))

        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.torch
    def test_apply_summation_torch(self, tol):
        """Test that summation can be applied with torch"""
        qnode1, qnode2 = qnodes("torch")
        qc = qml.QNodeCollection([qnode1, qnode2])
        cost = qml.sum(qc)

        params = [0.5643, -0.45]
        res = cost(params)
        expected = sum(qc[0](params) + qc[1](params))

        res = res.numpy()
        expected = expected.numpy()

        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.tf
    def test_apply_summation_tf(self, tol):
        """Test that summation can be applied with tf"""
        qnode1, qnode2 = qnodes("tf")
        qc = qml.QNodeCollection([qnode1, qnode2])
        cost = qml.sum(qc)

        params = [0.5643, -0.45]
        res = cost(params)
        expected = sum(qc[0](params) + qc[1](params))

        res = res.numpy()
        expected = expected.numpy()

        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.jax
    def test_apply_summation_jax(self, tol):
        """Test that summation can be applied with jax"""
        qnode1, qnode2 = qnodes("jax")
        qc = qml.QNodeCollection([qnode1, qnode2])
        cost = qml.sum(qc)

        params = [0.5643, -0.45]
        res = cost(params)
        expected = sum(qc[0](params) + qc[1](params))

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_unknown_interface(self, monkeypatch):
        """Test exception raised if the interface is unknown"""
        monkeypatch.setattr(qml.QNodeCollection, "interface", "invalid")
        dev = qml.device("default.qubit", wires=1)

        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        qnodes = [qml.QNode(circuit, dev) for i in range(4)]
        qc = qml.QNodeCollection(qnodes)
        with pytest.raises(ValueError, match="Unknown interface invalid"):
            qml.sum(qc)


class TestDot:
    """Tests for the sum function"""

    def test_dot_product_tensor_qnodes(self):
        """Test that the dot product of tensor.qnodes can be applied"""

        qnode1, qnode2 = qnodes(None)
        qc = qml.QNodeCollection([qnode1, qnode2])
        coeffs = [0.5, -0.1]

        # test the dot product of tensor, qnodes
        cost = qml.dot(coeffs, qc)

        params = [0.5643, -0.45]
        res = cost(params)

        qcval = qc(params)

        expected = np.dot(coeffs, qcval)
        np.testing.assert_allclose(res, expected)

    @pytest.mark.autograd
    def test_dot_product_tensor_qnodes_autograd(self):
        """Test that the dot product of tensor.qnodes can be applied using autograd"""

        qnode1, qnode2 = qnodes("autograd")
        qc = qml.QNodeCollection([qnode1, qnode2])
        coeffs = [0.5, -0.1]

        # test the dot product of tensor, qnodes
        cost = qml.dot(coeffs, qc)

        params = [0.5643, -0.45]
        res = cost(params)

        qcval = qc(params)

        expected = np.dot(coeffs, qcval)
        np.testing.assert_allclose(res, expected)

    @pytest.mark.torch
    def test_dot_product_tensor_qnodes_torch(self):
        """Test that the dot product of tensor.qnodes can be applied using torch."""
        import torch

        qnode1, qnode2 = qnodes("torch")
        qc = qml.QNodeCollection([qnode1, qnode2])
        coeffs = [0.5, -0.1]

        coeffs = torch.tensor(coeffs, dtype=torch.float64)

        # test the dot product of tensor, qnodes
        cost = qml.dot(coeffs, qc)

        params = [0.5643, -0.45]
        res = cost(params)

        qcval = qc(params)

        res = res.numpy()
        qcval = qcval.numpy()
        coeffs = coeffs.numpy()

        expected = np.dot(coeffs, qcval)
        np.testing.assert_allclose(res, expected)

    @pytest.mark.tf
    def test_dot_product_tensor_qnodes_tf(self):
        """Test that the dot product of tensor.qnodes can be applied using tf."""
        import tensorflow as tf

        qnode1, qnode2 = qnodes("tf")
        qc = qml.QNodeCollection([qnode1, qnode2])
        coeffs = [0.5, -0.1]

        coeffs = tf.cast(coeffs, dtype=tf.float64)

        # test the dot product of tensor, qnodes
        cost = qml.dot(coeffs, qc)

        params = [0.5643, -0.45]
        res = cost(params)

        qcval = qc(params)

        res = res.numpy()
        qcval = qcval.numpy()
        coeffs = coeffs.numpy()

        expected = np.dot(coeffs, qcval)
        np.testing.assert_allclose(res, expected)

    @pytest.mark.jax
    def test_dot_product_tensor_jax(self):
        """Test that the dot product of tensor.qnodes can be applied using all interfaces"""

        qnode1, qnode2 = qnodes("jax")
        qc = qml.QNodeCollection([qnode1, qnode2])
        coeffs = [0.5, -0.1]

        # test the dot product of tensor, qnodes
        cost = qml.dot(coeffs, qc)

        params = [0.5643, -0.45]
        res = cost(params)

        qcval = qc(params)

        expected = np.dot(coeffs, qcval)
        np.testing.assert_allclose(res, expected)

    def test_dot_product_qnodes_qnodes(self):
        """Test that the dot product of qnodes.qnodes can be applied"""

        qnode1, qnode2 = qnodes(None)
        qc1 = qml.QNodeCollection([qnode1, qnode2])
        qc2 = qml.QNodeCollection([qnode1, qnode2])

        # test the dot product of qnodes, qnodes
        cost = qml.dot(qc1, qc2)

        params = [0.5643, -0.45]
        res = cost(params)

        qc1val = qc1(params)
        qc2val = qc2(params)

        expected = np.dot(qc1val, qc2val)
        assert np.all(res == expected)

    @pytest.mark.autograd
    def test_dot_product_qnodes_qnodes_autograd(self):
        """Test that the dot product of qnodes.qnodes can be applied using autograd"""

        qnode1, qnode2 = qnodes("autograd")
        qc1 = qml.QNodeCollection([qnode1, qnode2])
        qc2 = qml.QNodeCollection([qnode1, qnode2])

        # test the dot product of qnodes, qnodes
        cost = qml.dot(qc1, qc2)

        params = [0.5643, -0.45]
        res = cost(params)

        qc1val = qc1(params)
        qc2val = qc2(params)

        expected = np.dot(qc1val, qc2val)
        assert np.all(res == expected)

    @pytest.mark.torch
    def test_dot_product_qnodes_qnodes_torch(self):
        """Test that the dot product of qnodes.qnodes can be applied using torch"""
        qnode1, qnode2 = qnodes("torch")
        qc1 = qml.QNodeCollection([qnode1, qnode2])
        qc2 = qml.QNodeCollection([qnode1, qnode2])

        # test the dot product of qnodes, qnodes
        cost = qml.dot(qc1, qc2)

        params = [0.5643, -0.45]
        res = cost(params)

        qc1val = qc1(params)
        qc2val = qc2(params)

        res = res.numpy()
        qc1val = qc1val.numpy()
        qc2val = qc2val.numpy()

        expected = np.dot(qc1val, qc2val)
        assert np.all(res == expected)

    @pytest.mark.tf
    def test_dot_product_qnodes_qnodes_tf(self):
        """Test that the dot product of qnodes.qnodes can be applied using tf"""
        qnode1, qnode2 = qnodes("tf")
        qc1 = qml.QNodeCollection([qnode1, qnode2])
        qc2 = qml.QNodeCollection([qnode1, qnode2])

        # test the dot product of qnodes, qnodes
        cost = qml.dot(qc1, qc2)

        params = [0.5643, -0.45]
        res = cost(params)

        qc1val = qc1(params)
        qc2val = qc2(params)

        res = res.numpy()
        qc1val = qc1val.numpy()
        qc2val = qc2val.numpy()

        expected = np.dot(qc1val, qc2val)
        assert np.all(res == expected)

    @pytest.mark.jax
    def test_dot_product_qnodes_qnodes_jax(self):
        """Test that the dot product of qnodes.qnodes can be applied using jax"""

        qnode1, qnode2 = qnodes("jax")
        qc1 = qml.QNodeCollection([qnode1, qnode2])
        qc2 = qml.QNodeCollection([qnode1, qnode2])

        # test the dot product of qnodes, qnodes
        cost = qml.dot(qc1, qc2)

        params = [0.5643, -0.45]
        res = cost(params)

        qc1val = qc1(params)
        qc2val = qc2(params)

        expected = np.dot(qc1val, qc2val)
        assert np.all(res == expected)

    def test_dot_product_qnodes_tensor(self):
        """Test that the dot product of qnodes.tensor can be applied"""

        qnode1, _ = qnodes(None)
        qc = qml.QNodeCollection([qnode1])
        coeffs = [0.5, -0.1]

        # test the dot product of qnodes, tensor
        cost = qml.dot(qc, coeffs)

        params = [0.5643, -0.45]
        res = cost(params)
        qcval = qc(params)

        expected = np.dot(qcval, coeffs)
        assert np.allclose(res, expected)

    @pytest.mark.autograd
    def test_dot_product_qnodes_tensor_autograd(self):
        """Test that the dot product of qnodes.tensor can be applied using autograd"""

        qnode1, _ = qnodes("autograd")
        qc = qml.QNodeCollection([qnode1])
        coeffs = [0.5, -0.1]

        # test the dot product of qnodes, tensor
        cost = qml.dot(qc, coeffs)

        params = [0.5643, -0.45]
        res = cost(params)
        qcval = qc(params)

        expected = np.dot(qcval, coeffs)
        assert np.allclose(res, expected)

    @pytest.mark.torch
    def test_dot_product_qnodes_tensor_torch(self):
        """Test that the dot product of qnodes.tensor can be applied using torch"""
        import torch

        qnode1, _ = qnodes("torch")
        qc = qml.QNodeCollection([qnode1])
        coeffs = [0.5, -0.1]

        coeffs = torch.tensor(coeffs, dtype=torch.float64)
        # test the dot product of qnodes, tensor
        cost = qml.dot(qc, coeffs)

        params = [0.5643, -0.45]
        res = cost(params)
        qcval = qc(params)

        res = res.numpy()
        qcval = qcval.numpy()
        coeffs = coeffs.numpy()

        expected = np.dot(qcval, coeffs)
        assert np.allclose(res, expected)

    @pytest.mark.tf
    def test_dot_product_qnodes_tensor_tf(self):
        """Test that the dot product of qnodes.tensor can be applied using tf"""
        import tensorflow as tf

        qnode1, _ = qnodes("tf")
        qc = qml.QNodeCollection([qnode1])
        coeffs = [0.5, -0.1]

        coeffs = tf.Variable(coeffs, dtype=tf.float64)
        # test the dot product of qnodes, tensor
        cost = qml.dot(qc, coeffs)

        params = [0.5643, -0.45]
        res = cost(params)
        qcval = qc(params)

        res = res.numpy()
        qcval = qcval.numpy()
        coeffs = coeffs.numpy()

        expected = np.dot(qcval, coeffs)
        assert np.allclose(res, expected)

    @pytest.mark.jax
    def test_dot_product_qnodes_tensor_jax(self):
        """Test that the dot product of qnodes.tensor can be applied using jax"""

        qnode1, _ = qnodes("jax")
        qc = qml.QNodeCollection([qnode1])
        coeffs = [0.5, -0.1]

        # test the dot product of qnodes, tensor
        cost = qml.dot(qc, coeffs)

        params = [0.5643, -0.45]
        res = cost(params)
        qcval = qc(params)

        expected = np.dot(qcval, coeffs)
        assert np.allclose(res, expected)

    def test_unknown_interface(self, monkeypatch):
        """Test exception raised if the interface is unknown"""
        monkeypatch.setattr(qml.QNodeCollection, "interface", "invalid")
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit1(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(dev)
        def circuit2(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        qc = qml.QNodeCollection([circuit1, circuit2])
        with pytest.raises(ValueError, match="Unknown interface invalid"):
            qml.dot([1, 2], qc)

    def test_mismatching_interface(self, monkeypatch):
        """Test exception raised if the interfaces don't match"""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev, interface=None)
        def circuit1(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(dev, interface="autograd")
        def circuit2(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        qc1 = qml.QNodeCollection([circuit1])
        qc2 = qml.QNodeCollection([circuit2])
        with pytest.raises(ValueError, match="have non-matching interfaces"):
            qml.dot(qc1, qc2)

    def test_no_qnodes(self):
        """Test exception raised if no qnodes are provided as arguments"""
        with pytest.raises(ValueError, match="At least one argument must be a QNodeCollection"):
            qml.dot([1, 2], [3, 4])

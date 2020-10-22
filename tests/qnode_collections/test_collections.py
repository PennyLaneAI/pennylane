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
from collections.abc import Sequence

import pytest
import numpy as np

import pennylane as qml

try:
    import torch
except ImportError as e:
    torch = None

try:
    import tensorflow as tf

    if tf.__version__[0] == "1":
        tf.enable_eager_execution()

    from tensorflow import Variable
except ImportError as e:
    tf = None
    Variable = None


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

        assert len(qc[0].ops) == 2
        assert qc[0].ops[0].name == "RX"
        assert qc[0].ops[1].name == "PauliX"

        assert len(qc[1].ops) == 2
        assert qc[1].ops[0].name == "RX"
        assert qc[1].ops[1].name == "PauliY"

        # test that device is broadcast
        assert qc[0].device is qc[1].device

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

        assert len(qc[0].ops) == 2
        assert qc[0].ops[0].name == "RX"
        assert qc[0].ops[1].name == "PauliX"

        assert len(qc[1].ops) == 2
        assert qc[1].ops[0].name == "RX"
        assert qc[1].ops[1].name == "PauliY"

        # test that device is broadcast
        assert qc[0].device is qc[1].device

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

        assert len(qc[0].ops) == 2
        assert qc[0].ops[0].name == "RX"
        assert qc[0].ops[1].name == "PauliX"

        assert len(qc[1].ops) == 2
        assert qc[1].ops[0].name == "RX"
        assert qc[1].ops[1].name == "PauliY"

        # test that device is not broadcast
        assert qc[0].device is not qc[1].device
        assert qc[0].device is dev_list[0]
        assert qc[1].device is dev_list[1]

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

        assert len(qc[0].ops) == 2
        assert qc[0].ops[0].name == "RX"
        assert qc[0].ops[1].name == "PauliX"
        assert qc[0].ops[1].return_type == qml.operation.Expectation

        assert len(qc[1].ops) == 2
        assert qc[1].ops[0].name == "RX"
        assert qc[1].ops[1].name == "PauliY"
        assert qc[1].ops[1].return_type == qml.operation.Variance

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
        assert qc[0].h == 123
        assert qc[1].h == 123

        # Checking that the order is set in each QNode
        assert qc[0].order == 2
        assert qc[1].order == 2


class TestApply:
    """Tests for the apply function"""

    @pytest.mark.parametrize("interface", ["autograd", "numpy", "torch", "tf"])
    def test_apply_summation(self, qnodes, interface, tf_support, torch_support, tol):
        """Test that summation can be applied using all interfaces"""
        if interface == "torch" and not torch_support:
            pytest.skip("Skipped, no torch support")

        if interface == "tf" and not tf_support:
            pytest.skip("Skipped, no tf support")

        qnode1, qnode2 = qnodes
        qc = qml.QNodeCollection([qnode1, qnode2])

        if interface == "tf":
            sfn = tf.reduce_sum
        elif interface == "torch":
            sfn = torch.sum
        else:
            sfn = np.sum

        cost = qml.apply(sfn, qc)

        params = [0.5643, -0.45]
        res = cost(params)
        expected = sfn(qc(params))

        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("interface", ["autograd", "numpy", "torch", "tf"])
    def test_nested_apply(self, qnodes, interface, tf_support, torch_support, tol):
        """Test that nested apply can be done using all interfaces"""
        if interface == "torch" and not torch_support:
            pytest.skip("Skipped, no torch support")

        if interface == "tf" and not tf_support:
            pytest.skip("Skipped, no tf support")

        qnode1, qnode2 = qnodes
        qc = qml.QNodeCollection([qnode1, qnode2])

        if interface == "tf":
            sinfn = tf.sin
            sfn = tf.reduce_sum
        elif interface == "torch":
            sinfn = torch.sin
            sfn = torch.sum
        else:
            sinfn = np.sin
            sfn = np.sum

        cost = qml.apply(sfn, qml.apply(sinfn, qc))

        params = [0.5643, -0.45]
        res = cost(params)
        expected = sfn(sinfn(qc(params)))

        assert np.allclose(res, expected, atol=tol, rtol=0)


class TestSum:
    """Tests for the sum function"""

    @pytest.mark.parametrize("interface", ["autograd", "numpy", "torch", "tf", None])
    def test_apply_summation(self, qnodes, interface, tf_support, torch_support, tol):
        """Test that summation can be applied using all interfaces"""
        if interface == "torch" and not torch_support:
            pytest.skip("Skipped, no torch support")

        if interface == "tf" and not tf_support:
            pytest.skip("Skipped, no tf support")

        qnode1, qnode2 = qnodes
        qc = qml.QNodeCollection([qnode1, qnode2])
        cost = qml.sum(qc)

        params = [0.5643, -0.45]
        res = cost(params)
        expected = sum(qc[0](params) + qc[1](params))

        if interface in ("tf", "torch"):
            res = res.numpy()
            expected = expected.numpy()

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

    @pytest.mark.parametrize("interface", ["autograd", "numpy", "torch", "tf", None])
    def test_dot_product_tensor_qnodes(self, qnodes, interface, tf_support, torch_support):
        """Test that the dot product of tensor.qnodes can be applied using all interfaces"""
        if interface == "torch" and not torch_support:
            pytest.skip("Skipped, no torch support")

        if interface == "tf" and not tf_support:
            pytest.skip("Skipped, no tf support")

        qnode1, qnode2 = qnodes
        qc = qml.QNodeCollection([qnode1, qnode2])
        coeffs = [0.5, -0.1]

        if interface == "torch":
            coeffs = torch.tensor(coeffs, dtype=torch.float64)

        if interface == "tf":
            coeffs = tf.cast(coeffs, dtype=tf.float64)

        # test the dot product of tensor, qnodes
        cost = qml.dot(coeffs, qc)

        params = [0.5643, -0.45]
        res = cost(params)

        qcval = qc(params)

        if interface in ("tf", "torch"):
            res = res.numpy()
            qcval = qcval.numpy()
            coeffs = coeffs.numpy()

        expected = np.dot(coeffs, qcval)
        assert np.all(res == expected)

    @pytest.mark.parametrize("interface", ["autograd", "numpy", "torch", "tf", None])
    def test_dot_product_qnodes_qnodes(self, qnodes, interface, tf_support, torch_support):
        """Test that the dot product of qnodes.qnodes can be applied using all interfaces"""
        if interface == "torch" and not torch_support:
            pytest.skip("Skipped, no torch support")

        if interface == "tf" and not tf_support:
            pytest.skip("Skipped, no tf support")

        qnode1, qnode2 = qnodes
        qc1 = qml.QNodeCollection([qnode1, qnode2])
        qc2 = qml.QNodeCollection([qnode1, qnode2])

        # test the dot product of qnodes, qnodes
        cost = qml.dot(qc1, qc2)

        params = [0.5643, -0.45]
        res = cost(params)

        qc1val = qc1(params)
        qc2val = qc2(params)

        if interface in ("tf", "torch"):
            res = res.numpy()
            qc1val = qc1val.numpy()
            qc2val = qc2val.numpy()

        expected = np.dot(qc1val, qc2val)
        assert np.all(res == expected)

    @pytest.mark.parametrize("interface", ["autograd", "numpy", "torch", "tf", None])
    def test_dot_product_qnodes_tensor(self, qnodes, interface, tf_support, torch_support):
        """Test that the dot product of qnodes.tensor can be applied using all interfaces"""
        if interface == "torch" and not torch_support:
            pytest.skip("Skipped, no torch support")

        if interface == "tf" and not tf_support:
            pytest.skip("Skipped, no tf support")

        qnode1, _ = qnodes
        qc = qml.QNodeCollection([qnode1])
        coeffs = [0.5, -0.1]

        if interface == "torch":
            coeffs = torch.tensor(coeffs, dtype=torch.float64)

        if interface == "tf":
            coeffs = tf.cast(coeffs, dtype=tf.float64)

        # test the dot product of qnodes, tensor
        cost = qml.dot(qc, coeffs)

        params = [0.5643, -0.45]
        res = cost(params)
        qcval = qc(params)

        if interface in ("tf", "torch"):
            res = res.numpy()
            qcval = qcval.numpy()
            coeffs = coeffs.numpy()

        expected = np.dot(qcval, coeffs)
        assert np.all(res == expected)

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

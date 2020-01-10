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
Unit tests for the :mod:`pennylane.cluster` submodule.
"""
from collections.abc import Sequence

import pytest
from pennylane import numpy as np

import pennylane as qml


try:
    import torch
except ImportError as e:
    pass


try:
    import tensorflow as tf

    if tf.__version__[0] == "1":
        print(tf.__version__)
        import tensorflow.contrib.eager as tfe

        tf.enable_eager_execution()
        Variable = tfe.Variable
    else:
        from tensorflow import Variable
except ImportError as e:
    pass


@pytest.fixture
def qnodes(interface):
    """fixture returning some QNodes"""
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
    to return a QNode cluster"""

    def test_template_not_callable(self):
        """Test that an exception is correctly called if a
        template is not callable"""
        with pytest.raises(ValueError, match="template is not a callable"):
            qml.map(5, 0, 0)

    def test_mapping_over_observables(self):
        """Test that mapping over a list of observables produces
        a QNodeCluster with the correct QNodes, with a single
        device broadcast."""
        dev = qml.device("default.qubit", wires=1)
        obs_list = [qml.PauliX(0), qml.PauliY(0)]
        template = lambda x, wires: qml.RX(x, wires=0)

        qc = qml.map(template, obs_list, dev)

        assert len(qc) == 2

        # evaluate cluster so that queue is populated
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
        a QNodeCluster with the correct QNodes, with a single
        device broadcast."""
        dev = qml.device("default.qubit", wires=1)
        obs_list = (qml.PauliX(0), qml.PauliY(0))
        template = lambda x, wires: qml.RX(x, wires=0)

        qc = qml.map(template, obs_list, dev)

        assert len(qc) == 2

        # evaluate cluster so that queue is populated
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
        a QNodeCluster with the correct QNodes"""
        dev_list = [qml.device("default.qubit", wires=1), qml.device("default.qubit", wires=1)]

        obs_list = [qml.PauliX(0), qml.PauliY(0)]
        template = lambda x, wires: qml.RX(x, wires=0)

        qc = qml.map(template, obs_list, dev_list)

        assert len(qc) == 2

        # evaluate cluster so that queue is populated
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
        a QNodeCluster with the correct QNodes"""
        dev = qml.device("default.qubit", wires=1)

        obs_list = [qml.PauliX(0), qml.PauliY(0)]
        template = lambda x, wires: qml.RX(x, wires=0)

        qc = qml.map(template, obs_list, dev, measure=["expval", "var"])

        assert len(qc) == 2

        # evaluate cluster so that queue is populated
        qc(1)

        assert len(qc[0].ops) == 2
        assert qc[0].ops[0].name == "RX"
        assert qc[0].ops[1].name == "PauliX"
        assert qc[0].ops[1].return_type == qml.operation.Expectation

        assert len(qc[1].ops) == 2
        assert qc[1].ops[0].name == "RX"
        assert qc[1].ops[1].name == "PauliY"
        assert qc[1].ops[1].return_type == qml.operation.Variance

    def test_invalid_obserable(self):
        """Test that an invalid observable raises an exception"""
        dev = qml.device("default.qubit", wires=1)

        obs_list = [qml.PauliX(0), qml.S(wires=0)]
        template = lambda x, wires: qml.RX(x, wires=0)

        with pytest.raises(ValueError, match="Some or all observables are not valid"):
            qml.map(template, obs_list, dev, measure=["expval", "var"])


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
        qc = qml.QNodeCluster([qnode1, qnode2])

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
        qc = qml.QNodeCluster([qnode1, qnode2])

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
        qc = qml.QNodeCluster([qnode1, qnode2])
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
        monkeypatch.setattr(qml.QNodeCluster, "interface", "invalid")
        dev = qml.device("default.qubit", wires=1)

        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        qnodes = [qml.QNode(circuit, dev) for i in range(4)]
        qc = qml.QNodeCluster(qnodes)
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
        qc = qml.QNodeCluster([qnode1, qnode2])
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
        qc1 = qml.QNodeCluster([qnode1, qnode2])
        qc2 = qml.QNodeCluster([qnode1, qnode2])

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
        qc = qml.QNodeCluster([qnode1])
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
        monkeypatch.setattr(qml.QNodeCluster, "interface", "invalid")
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit1(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(dev)
        def circuit2(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        qc = qml.QNodeCluster([circuit1, circuit2])
        with pytest.raises(ValueError, match="Unknown interface invalid"):
            qml.dot([1, 2], qc)

    def test_no_qnodes(self):
        """Test exception raised if no qnodes are provided as arguments"""
        with pytest.raises(ValueError, match="At least one argument must be a QNodeCluster"):
            qml.dot([1, 2], [3, 4])


class TestQNodeCluster:
    """Tests for the QNodeCluster class"""

    def test_empty_init(self):
        """Test that an empty QNode cluster can be initialized"""
        qc = qml.QNodeCluster()
        assert qc.qnodes == []
        assert len(qc) == 0

    def test_init_with_qnodes(self):
        """Test that an QNode cluster can be initialized with QNodes"""
        dev = qml.device("default.qubit", wires=1)

        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        qnodes = [qml.QNode(circuit, dev) for i in range(4)]
        qc = qml.QNodeCluster(qnodes)

        assert qc.qnodes == qnodes
        assert len(qc) == 4

    @pytest.mark.parametrize("interface", ["autograd", "numpy", "torch", "tf"])
    def test_interface_property(self, interface, tf_support, torch_support):
        """Test that the interface property correctly
        resolves interfaces from the internal QNodes"""
        if interface == "torch" and not torch_support:
            pytest.skip("Skipped, no torch support")

        if interface == "tf" and not tf_support:
            pytest.skip("Skipped, no tf support")

        qc = qml.QNodeCluster()
        assert qc.interface is None

        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        dev = qml.device("default.qubit", wires=1)
        qnodes = [qml.QNode(circuit, dev, interface=interface) for i in range(4)]
        qc = qml.QNodeCluster(qnodes)

        if interface == "numpy":
            # Note: the "numpy" interface is deprecated, and
            # now resolves to "autograd"
            interface = "autograd"

        assert qc.interface == interface

    def test_append_qnode(self):
        """Test that a QNode is correctly appended"""
        qc = qml.QNodeCluster()
        assert qc.qnodes == []

        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        dev = qml.device("default.qubit", wires=1)
        qnode = qml.QNode(circuit, dev)
        qc.append(qnode)

        assert qc.qnodes == [qnode]

    def test_extend_qnodes(self):
        """Test that a list of QNodes is correctly appended"""
        qc = qml.QNodeCluster()
        assert qc.qnodes == []

        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        dev = qml.device("default.qubit", wires=1)
        qnodes = [qml.QNode(circuit, dev) for i in range(4)]
        qc.extend(qnodes)

        assert qc.qnodes == [] + qnodes

    def test_extend_multiple_interface_qnodes(self):
        """Test that an error is returned if QNodes with differing
        interfaces are attempted to be added to a QNodeCluster"""
        qc = qml.QNodeCluster()

        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        dev = qml.device("default.qubit", wires=1)
        qnodes = [
            qml.QNode(circuit, dev, interface="autograd"),
            qml.QNode(circuit, dev, interface=None),
        ]

        with pytest.raises(ValueError, match="do not all use the same interface"):
            qc.extend(qnodes)

    def test_extend_interface_mistmatch(self):
        """Test that an error is returned if QNodes with a differing
        interface to the QNode cluster are appended"""
        qc = qml.QNodeCluster()

        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        dev = qml.device("default.qubit", wires=1)
        qnode1 = qml.QNode(circuit, dev, interface="autograd")
        qnode2 = qml.QNode(circuit, dev, interface=None)

        qc.extend([qnode1])

        with pytest.raises(ValueError, match="Interface mismatch"):
            qc.extend([qnode2])

    def test_indexing(self):
        """Test that indexing into the QNodeCluster correctly works"""

        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        dev = qml.device("default.qubit", wires=1)
        qnodes = [qml.QNode(circuit, dev) for i in range(4)]

        qc = qml.QNodeCluster(qnodes)
        assert qc[2] == qnodes[2]

    def test_sequence(self):
        """Test that the QNodeCluster is a sequence type"""
        qc = qml.QNodeCluster()
        assert isinstance(qc, Sequence)

    @pytest.mark.parametrize("interface", ["autograd", "numpy"])
    def test_eval_autograd(self, qnodes):
        """Test correct evaluation of the QNodeCluster using
        the Autograd interface"""
        qnode1, qnode2 = qnodes
        qc = qml.QNodeCluster([qnode1, qnode2])
        params = [0.5643, -0.45]

        res = qc(params)
        expected = np.vstack([qnode1(params), qnode2(params)])
        assert np.all(res == expected)

    @pytest.mark.parametrize("interface", ["autograd", "numpy"])
    def test_grad_autograd(self, qnodes):
        """Test correct gradient of the QNodeCluster using
        the Autograd interface"""
        qnode1, qnode2 = qnodes

        params = [0.5643, -0.45]
        qc = qml.QNodeCluster([qnode1, qnode2])
        cost_qc = lambda params: np.sum(qc(params))
        grad_qc = qml.grad(cost_qc, argnum=0)

        cost_expected = lambda params: np.sum(qnode1(params) + qnode2(params))
        grad_expected = qml.grad(cost_expected, argnum=0)

        res = grad_qc(params)
        expected = grad_expected(params)

        assert np.all(res == expected)

    @pytest.mark.parametrize("interface", ["torch"])
    def test_eval_torch(self, qnodes, skip_if_no_torch_support):
        """Test correct evaluation of the QNodeCluster using
        the torch interface"""
        qnode1, qnode2 = qnodes
        qc = qml.QNodeCluster([qnode1, qnode2])
        params = [0.5643, -0.45]

        res = qc(params).numpy()
        expected = np.vstack([qnode1(params), qnode2(params)])
        assert np.all(res == expected)

    @pytest.mark.parametrize("interface", ["torch"])
    def test_grad_torch(self, qnodes, skip_if_no_torch_support):
        """Test correct gradient of the QNodeCluster using
        the torch interface"""
        qnode1, qnode2 = qnodes

        # calculate the gradient of the cluster using pytorch
        params = torch.autograd.Variable(torch.tensor([0.5643, -0.45]), requires_grad=True)
        qc = qml.QNodeCluster([qnode1, qnode2])
        cost = torch.sum(qc(params))
        cost.backward()
        res = params.grad.numpy()

        # calculate the gradient of the QNodes individually using pytorch
        params = torch.autograd.Variable(torch.tensor([0.5643, -0.45]), requires_grad=True)
        cost = torch.sum(qnode1(params) + qnode2(params))
        cost.backward()
        expected = params.grad.numpy()

        assert np.all(res == expected)

    @pytest.mark.parametrize("interface", ["tf"])
    def test_eval_tf(self, qnodes, skip_if_no_tf_support):
        """Test correct evaluation of the QNodeCluster using
        the tf interface"""
        qnode1, qnode2 = qnodes
        qc = qml.QNodeCluster([qnode1, qnode2])
        params = [0.5643, -0.45]

        res = qc(params).numpy()
        expected = np.vstack([qnode1(params), qnode2(params)])
        assert np.all(res == expected)

    @pytest.mark.parametrize("interface", ["tf"])
    def test_grad_tf(self, qnodes, skip_if_no_tf_support):
        """Test correct gradient of the QNodeCluster using
        the tf interface"""
        qnode1, qnode2 = qnodes

        # calculate the gradient of the cluster using tf
        params = Variable([0.5643, -0.45])
        qc = qml.QNodeCluster([qnode1, qnode2])

        with tf.GradientTape() as tape:
            tape.watch(params)
            cost = sum(qc(params))
            res = tape.gradient(cost, params).numpy()

        # calculate the gradient of the QNodes individually using tf
        params = Variable([0.5643, -0.45])

        with tf.GradientTape() as tape:
            tape.watch(params)
            cost = sum(qnode1(params) + qnode2(params))
            expected = tape.gradient(cost, params).numpy()

        assert np.all(res == expected)

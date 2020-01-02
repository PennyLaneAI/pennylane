# Copyright 2019 Xanadu Quantum Technologies Inc.

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


class TestQNodeCluster:
    """Tests for the QNodeCluster class"""

    @pytest.fixture
    def qnodes(self, interface):
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

    @pytest.mark.parametrize("interface", ["autograd", "torch", "tf"])
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
            qml.QNode(circuit, dev, interface=None)
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

    @pytest.mark.parametrize("interface", ["autograd"])
    def test_eval_autograd(self, qnodes):
        """Test correct evaluation of the QNodeCluster using
        the Autograd interface"""
        qnode1, qnode2 = qnodes
        qc = qml.QNodeCluster([qnode1, qnode2])
        params = [0.5643, -0.45]

        res = qc(params)
        expected = np.vstack([qnode1(params), qnode2(params)])
        assert np.all(res == expected)

    @pytest.mark.parametrize("interface", ["autograd"])
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

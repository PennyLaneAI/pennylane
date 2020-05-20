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
Unit tests for the :mod:`pennylane.QNodeCollection`
"""
from collections.abc import Sequence

import pytest
from pennylane import numpy as np

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


class TestConstruction:
    """Tests for the QNodeCollection construction"""

    def test_empty_init(self):
        """Test that an empty QNode collection can be initialized"""
        qc = qml.QNodeCollection()
        assert qc.qnodes == []
        assert len(qc) == 0

    def test_init_with_qnodes(self):
        """Test that a QNode collection can be initialized with QNodes"""
        dev = qml.device("default.qubit", wires=1)

        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        qnodes = [qml.QNode(circuit, dev) for i in range(4)]
        qc = qml.QNodeCollection(qnodes)

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

        qc = qml.QNodeCollection()
        assert qc.interface is None

        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        dev = qml.device("default.qubit", wires=1)
        qnodes = [qml.QNode(circuit, dev, interface=interface) for i in range(4)]
        qc = qml.QNodeCollection(qnodes)

        if interface == "numpy":
            # Note: the "numpy" interface is deprecated, and
            # now resolves to "autograd"
            interface = "autograd"

        assert qc.interface == interface

    def test_append_qnode(self):
        """Test that a QNode is correctly appended"""
        qc = qml.QNodeCollection()
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
        qc = qml.QNodeCollection()
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
        interfaces are attempted to be added to a QNodeCollection"""
        qc = qml.QNodeCollection()

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

    def test_extend_interface_mismatch(self):
        """Test that an error is returned if QNodes with a differing
        interface to the QNode collection are appended"""
        qc = qml.QNodeCollection()

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
        """Test that indexing into the QNodeCollection correctly works"""

        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        dev = qml.device("default.qubit", wires=1)
        qnodes = [qml.QNode(circuit, dev) for i in range(4)]

        qc = qml.QNodeCollection(qnodes)
        assert qc[2] == qnodes[2]

    def test_sequence(self):
        """Test that the QNodeCollection is a sequence type"""
        qc = qml.QNodeCollection()
        assert isinstance(qc, Sequence)


@pytest.mark.parametrize("parallel", [False, True])
class TestEvalation:
    """Tests for the QNodeCollection evaluation"""

    @pytest.mark.parametrize("interface", ["autograd", "numpy"])
    def test_eval_autograd(self, qnodes, parallel, interface):
        """Test correct evaluation of the QNodeCollection using
        the Autograd interface"""
        qnode1, qnode2 = qnodes
        qc = qml.QNodeCollection([qnode1, qnode2])
        params = [0.5643, -0.45]

        res = qc(params, parallel=parallel)
        expected = np.vstack([qnode1(params), qnode2(params)])
        assert np.all(res == expected)

    @pytest.mark.parametrize("interface", ["autograd", "numpy"])
    def test_grad_autograd(self, qnodes, parallel, interface):
        """Test correct gradient of the QNodeCollection using
        the Autograd interface"""
        qnode1, qnode2 = qnodes

        params = [0.5643, -0.45]
        qc = qml.QNodeCollection([qnode1, qnode2])
        cost_qc = lambda params: np.sum(qc(params))
        grad_qc = qml.grad(cost_qc, argnum=0)

        cost_expected = lambda params: np.sum(qnode1(params) + qnode2(params))
        grad_expected = qml.grad(cost_expected, argnum=0)

        res = grad_qc(params)
        expected = grad_expected(params)

        assert np.all(res == expected)

    @pytest.mark.parametrize("interface", ["torch"])
    def test_eval_torch(self, qnodes, skip_if_no_torch_support, parallel, interface):
        """Test correct evaluation of the QNodeCollection using
        the torch interface"""
        qnode1, qnode2 = qnodes
        qc = qml.QNodeCollection([qnode1, qnode2])
        params = [0.5643, -0.45]

        res = qc(params, parallel=parallel).numpy()
        expected = np.vstack([qnode1(params), qnode2(params)])
        assert np.all(res == expected)

    @pytest.mark.parametrize("interface", ["torch"])
    def test_grad_torch(self, qnodes, skip_if_no_torch_support, parallel, interface):
        """Test correct gradient of the QNodeCollection using
        the torch interface"""
        qnode1, qnode2 = qnodes

        # calculate the gradient of the collection using pytorch
        params = torch.autograd.Variable(torch.tensor([0.5643, -0.45]), requires_grad=True)
        qc = qml.QNodeCollection([qnode1, qnode2])
        cost = torch.sum(qc(params, parallel=parallel))
        cost.backward()
        res = params.grad.numpy()

        # calculate the gradient of the QNodes individually using pytorch
        params = torch.autograd.Variable(torch.tensor([0.5643, -0.45]), requires_grad=True)
        cost = torch.sum(qnode1(params) + qnode2(params))
        cost.backward()
        expected = params.grad.numpy()

        assert np.all(res == expected)

    @pytest.mark.parametrize("interface", ["tf"])
    def test_eval_tf(self, qnodes, skip_if_no_tf_support, parallel, interface):
        """Test correct evaluation of the QNodeCollection using
        the tf interface"""
        qnode1, qnode2 = qnodes
        qc = qml.QNodeCollection([qnode1, qnode2])
        params = [0.5643, -0.45]

        res = qc(params, parallel=parallel).numpy()
        expected = np.vstack([qnode1(params), qnode2(params)])
        assert np.all(res == expected)

    @pytest.mark.xfail(raises=AttributeError, reason="Dask breaks the TF gradient tape")
    @pytest.mark.parametrize("interface", ["tf"])
    def test_grad_tf(self, qnodes, skip_if_no_tf_support, parallel, interface):
        """Test correct gradient of the QNodeCollection using
        the tf interface"""
        qnode1, qnode2 = qnodes

        # calculate the gradient of the collection using tf
        params = Variable([0.5643, -0.45])
        qc = qml.QNodeCollection([qnode1, qnode2])

        with tf.GradientTape() as tape:
            tape.watch(params)

            if parallel:
                with pytest.warns(UserWarning):
                    cost = sum(qc(params, parallel=parallel))
            else:
                cost = sum(qc(params, parallel=parallel))

            # the gradient will be None
            res = tape.gradient(cost, params).numpy()

        # calculate the gradient of the QNodes individually using tf
        params = Variable([0.5643, -0.45])

        with tf.GradientTape() as tape:
            tape.watch(params)
            cost = sum(qnode1(params) + qnode2(params))
            expected = tape.gradient(cost, params).numpy()

        assert np.all(res == expected)

# Copyright 2020 Xanadu Quantum Technologies Inc.

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
Unit tests for the :mod:`pennylane` :class:`.PassthruQNode` class.
"""
import pytest
import numpy as np

try:
    import tensorflow as tf

    if tf.__version__[0] == "1":
        tf = None  # expt.tensornet.tf requires TF 2
    else:
        from tensorflow import Variable
except ImportError:
    tf = None

import pennylane as qml
from pennylane.qnodes import PassthruQNode, BaseQNode, JacobianQNode



# real data type used by the expt.tensornet.tf plugin (TensorFlow is strict about types)
R_DTYPE = tf.float64


@pytest.fixture(scope="function")
def mock_qnode(mock_device):
    """Simple PassthruQNode with default properties."""

    def circuit(x):
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliZ(0))

    node = PassthruQNode(circuit, mock_device)
    return node


class TestPassthruBasics:
    """Tests basic PassthruQNode properties."""

    def test_always_mutable(self, mock_qnode):
        """PassthruQNodes are always mutable."""
        assert mock_qnode.mutable

    def test_repr(self, mock_qnode):
        """String representation."""
        assert repr(mock_qnode) == "<PassthruQNode: device='mock_device', func=circuit, wires=2>"


@pytest.mark.skipif(tf is None, reason="TensorFlow 2.0 not found.")
@pytest.fixture(scope="function")
def tensornet_tf_device():
    return qml.device('expt.tensornet.tf', wires=2)


@pytest.mark.usefixtures("skip_if_no_tf_support")
class TestPassthruTF:
    """Test that TF objects can be successfully passed through to a TF simulator device, and back to user."""

    def test_arraylike_args(self, tensornet_tf_device, tol):
        """Tests that PassthruQNode can use array-like TF objects as positional arguments."""

        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.RX(2*x[1], wires=[1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        node = PassthruQNode(circuit, tensornet_tf_device)
        x = tf.Variable([1.1, 1.4], dtype=R_DTYPE)
        res = node(x)
        assert isinstance(res, tf.Tensor)
        assert res.shape == (2,)
        assert res.dtype == R_DTYPE

    def test_arraylike_keywordargs(self, tensornet_tf_device, tol):
        """Tests that qnodes use array-like TF objects as keyword-only arguments."""

        def circuit(*, x=None):
            qml.RX(x[0], wires=[0])
            qml.RX(2*x[1], wires=[1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        node = PassthruQNode(circuit, tensornet_tf_device)
        x = tf.Variable([1.1, 1.4], dtype=R_DTYPE)
        res = node(x=x)
        assert isinstance(res, tf.Tensor)
        assert res.shape == (2,)
        assert res.dtype == R_DTYPE

    def test_tensor_operations(self, tensornet_tf_device, tol):
        """Tests the evaluation of a PassthruQNode involving algebraic operations between tensor parameters,
        and TF functions acting on them."""

        def circuit(phi, theta):
            x = phi * theta
            qml.RX(x[0], wires=0)
            qml.RY(tf.cos(x[1]), wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.Hadamard(1))

        node = PassthruQNode(circuit, tensornet_tf_device)

        phi = tf.Variable(np.array([0.7, -1.2]), dtype=R_DTYPE)
        theta = tf.Variable(1.7, dtype=R_DTYPE)
        res = node(phi, theta)
        assert isinstance(res, tf.Tensor)
        assert res.shape == (2,)
        assert res.dtype == R_DTYPE

    def test_evaluate(self, tensornet_tf_device, tol):
        """Tests the evaluation of a PassthruQNode."""

        def circuit(x):
            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RX(x[2], wires=1)
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        x = np.array([-1.4, 0.2, 2.1])
        x_tf = tf.Variable(x, dtype=R_DTYPE)

        node = PassthruQNode(circuit, tensornet_tf_device)
        res = node(x_tf)

        # compare to a reference node
        ref_device = qml.device('default.qubit', wires=2)
        ref_node = BaseQNode(circuit, ref_device)
        ref_res = ref_node(x)

        assert isinstance(res, tf.Tensor)
        assert res.shape == (2,)
        assert res.dtype == R_DTYPE
        assert res.numpy() == pytest.approx(ref_res, abs=tol)

    @pytest.mark.parametrize("vectorize_jacobian", [True, False])
    def test_jacobian(self, tensornet_tf_device, vectorize_jacobian, tol):
        """Tests the computing of the Jacobian of a PassthruQNode using TensorFlow."""

        def circuit(phi, theta):
            qml.RX(phi[0], wires=0)
            qml.RY(phi[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.PhaseShift(theta, wires=0)
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.Hadamard(1))

        ph = np.array([0.7, -1.2])
        th = 1.7
        phi = tf.Variable(ph, dtype=R_DTYPE)
        theta = tf.Variable(th, dtype=R_DTYPE)

        node = PassthruQNode(circuit, tensornet_tf_device)
        # In TF 2, tf.GradientTape.jacobian comes with a vectorization option.
        with tf.GradientTape(persistent=not vectorize_jacobian) as tape:
            tape.watch([phi, theta])
            res = node(phi, theta)
        phi_grad, theta_grad = tape.jacobian(res, [phi, theta],
                                             unconnected_gradients=tf.UnconnectedGradients.ZERO,
                                             experimental_use_pfor=vectorize_jacobian)

        # compare to a reference Jacobian computed using finite differences
        ref_device = qml.device('default.qubit', wires=2)
        ref_node = JacobianQNode(circuit, ref_device)
        ref_jac = ref_node.jacobian([ph, th])

        assert isinstance(phi_grad, tf.Tensor)
        assert phi_grad.shape == (2, 2)
        assert phi_grad.dtype == R_DTYPE
        assert phi_grad.numpy() == pytest.approx(ref_jac[:, 0:2], abs=tol)

        assert isinstance(theta_grad, tf.Tensor)
        assert theta_grad.shape == (2,)
        assert theta_grad.dtype == R_DTYPE
        assert theta_grad.numpy() == pytest.approx(ref_jac[:, 2], abs=tol)

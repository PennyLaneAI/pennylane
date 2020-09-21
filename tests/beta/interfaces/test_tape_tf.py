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
"""Unit tests for the tf interface"""
import pytest

tf = pytest.importorskip("tensorflow", minversion="2.1")

import numpy as np

import pennylane as qml
from pennylane.beta.tapes import QuantumTape, qnode, QNode
from pennylane.beta.queuing import expval, var, sample, probs
from pennylane.beta.interfaces.tf import TFInterface


class TestTFQuantumTape:
    """Test the TensorFlow interface applied to a tape"""

    def test_interface_construction(self):
        """Test that the interface is correctly applied"""
        with TFInterface.apply(QuantumTape()) as tape:
            qml.RX(0.5, wires=0)
            expval(qml.PauliX(0))

        assert tape.interface == "tf"
        assert isinstance(tape, TFInterface)
        assert tape.__bare__ == QuantumTape
        assert tape.dtype is tf.float64

    def test_repeated_interface_construction(self):
        """Test that the interface is correctly applied multiple times"""
        with TFInterface.apply(QuantumTape()) as tape:
            qml.RX(0.5, wires=0)
            expval(qml.PauliX(0))

        assert tape.interface == "tf"
        assert isinstance(tape, TFInterface)
        assert tape.__bare__ == QuantumTape
        assert tape.dtype is tf.float64

        TFInterface.apply(tape, dtype=tf.float32)
        assert tape.interface == "tf"
        assert isinstance(tape, TFInterface)
        assert tape.__bare__ == QuantumTape
        assert tape.dtype is tf.float32

    def test_get_parameters(self):
        """Test that the get parameters function correctly sets and returns the
        trainable parameters"""
        a = tf.Variable(0.1)
        b = tf.constant(0.2)
        c = tf.Variable(0.3)
        d = 0.4

        with tf.GradientTape() as tape:
            with TFInterface.apply(QuantumTape()) as qtape:
                qml.Rot(a, b, c, wires=0)
                qml.RX(d, wires=1)
                qml.CNOT(wires=[0, 1])
                expval(qml.PauliX(0))

        assert qtape.trainable_params == {0, 2}
        assert np.all(qtape.get_parameters() == [a, c])

    def test_execution(self):
        """Test execution"""
        a = tf.Variable(0.1)
        dev = qml.device("default.qubit", wires=1)

        with tf.GradientTape() as tape:
            with TFInterface.apply(QuantumTape()) as qtape:
                qml.RY(a, wires=0)
                qml.RX(0.2, wires=0)
                expval(qml.PauliZ(0))

            assert qtape.trainable_params == {0}
            res = qtape.execute(dev)

        assert isinstance(res, tf.Tensor)
        assert res.shape == (1,)

    def test_jacobian(self, mocker, tol):
        """Test jacobian calculation"""
        spy = mocker.spy(QuantumTape, "jacobian")
        a = tf.Variable(0.1, dtype=tf.float64)
        b = tf.Variable(0.2, dtype=tf.float64)

        dev = qml.device("default.qubit", wires=2)

        with tf.GradientTape() as tape:
            with TFInterface.apply(QuantumTape()) as qtape:
                qml.RY(a, wires=0)
                qml.RX(b, wires=1)
                qml.CNOT(wires=[0, 1])
                expval(qml.PauliZ(0))
                expval(qml.PauliY(1))

            assert qtape.trainable_params == {0, 1}
            res = qtape.execute(dev)

        assert isinstance(res, tf.Tensor)
        assert res.shape == (2,)

        expected = [tf.cos(a), -tf.cos(a) * tf.sin(b)]
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = tape.jacobian(res, [a, b])
        expected = [[-tf.sin(a), tf.sin(a) * tf.sin(b)], [0, -tf.cos(a) * tf.cos(b)]]
        assert np.allclose(res, expected, atol=tol, rtol=0)

        spy.assert_called()

    def test_jacobian_dtype(self, tol):
        """Test calculating the jacobian with a different datatype. Here, we
        specify tf.float32, as opposed to the default value of tf.float64."""
        a = tf.Variable(0.1, dtype=tf.float32)
        b = tf.Variable(0.2, dtype=tf.float32)

        dev = qml.device("default.qubit", wires=2)

        with tf.GradientTape() as tape:
            with TFInterface.apply(QuantumTape(), dtype=tf.float32) as qtape:
                qml.RY(a, wires=0)
                qml.RX(b, wires=1)
                qml.CNOT(wires=[0, 1])
                expval(qml.PauliZ(0))
                expval(qml.PauliY(1))

            assert qtape.trainable_params == {0, 1}
            res = qtape.execute(dev)

        assert isinstance(res, tf.Tensor)
        assert res.shape == (2,)
        assert res.dtype is tf.float32

        res = tape.jacobian(res, [a, b])
        assert [r.dtype is tf.float32 for r in res]

    def test_jacobian_options(self, mocker, tol):
        """Test setting jacobian options"""
        spy = mocker.spy(QuantumTape, "numeric_pd")

        a = tf.Variable([0.1, 0.2])

        dev = qml.device("default.qubit", wires=1)

        with tf.GradientTape() as tape:
            with TFInterface.apply(QuantumTape()) as qtape:
                qml.RY(a[0], wires=0)
                qml.RX(a[1], wires=0)
                expval(qml.PauliZ(0))

            res = qtape.execute(dev)

        qtape.jacobian_options = {"h": 1e-8, "order": 2}
        tape.jacobian(res, a)

        for args in spy.call_args_list:
            assert args[1]["order"] == 2
            assert args[1]["h"] == 1e-8

    def test_reusing_quantum_tape(self, tol):
        """Test re-using a quantum tape by passing new parameters"""
        a = tf.Variable(0.1, dtype=tf.float64)
        b = tf.Variable(0.2, dtype=tf.float64)

        dev = qml.device("default.qubit", wires=2)

        with tf.GradientTape() as tape:

            with TFInterface.apply(QuantumTape()) as qtape:
                qml.RY(a, wires=0)
                qml.RX(b, wires=1)
                qml.CNOT(wires=[0, 1])
                expval(qml.PauliZ(0))
                expval(qml.PauliY(1))

            assert qtape.trainable_params == {0, 1}

            res = qtape.execute(dev)

        jac = tape.jacobian(res, [a, b])

        a = tf.Variable(0.54, dtype=tf.float64)
        b = tf.Variable(0.8, dtype=tf.float64)

        with tf.GradientTape() as tape:
            res2 = qtape.execute(dev, params=[2 * a, b])

        expected = [tf.cos(2 * a), -tf.cos(2 * a) * tf.sin(b)]
        assert np.allclose(res2, expected, atol=tol, rtol=0)

        jac2 = tape.jacobian(res2, [a, b])
        expected = [
            [-2 * tf.sin(2 * a), 2 * tf.sin(2 * a) * tf.sin(b)],
            [0, -tf.cos(2 * a) * tf.cos(b)],
        ]
        assert np.allclose(jac2, expected, atol=tol, rtol=0)

    def test_reusing_pre_constructed_quantum_tape(self, tol):
        """Test re-using a quantum tape that was previously constructed
        *outside of* a gradient tape, by passing new parameters"""
        a = tf.Variable(0.1, dtype=tf.float64)
        b = tf.Variable(0.2, dtype=tf.float64)

        dev = qml.device("default.qubit", wires=2)

        with TFInterface.apply(QuantumTape()) as qtape:
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            expval(qml.PauliZ(0))
            expval(qml.PauliY(1))

        with tf.GradientTape() as tape:
            qtape.set_parameters([a, b], trainable_only=False)
            qtape._update_trainable_params()
            assert qtape.trainable_params == {0, 1}
            res = qtape.execute(dev)

        jac = tape.jacobian(res, [a, b])

        a = tf.Variable(0.54, dtype=tf.float64)
        b = tf.Variable(0.8, dtype=tf.float64)

        with tf.GradientTape() as tape:
            res2 = qtape.execute(dev, params=[2 * a, b])

        expected = [tf.cos(2 * a), -tf.cos(2 * a) * tf.sin(b)]
        assert np.allclose(res2, expected, atol=tol, rtol=0)

        jac2 = tape.jacobian(res2, [a, b])
        expected = [
            [-2 * tf.sin(2 * a), 2 * tf.sin(2 * a) * tf.sin(b)],
            [0, -tf.cos(2 * a) * tf.cos(b)],
        ]
        assert np.allclose(jac2, expected, atol=tol, rtol=0)

    def test_classical_processing(self, tol):
        """Test classical processing within the quantum tape"""
        a = tf.Variable(0.1, dtype=tf.float64)
        b = tf.constant(0.2, dtype=tf.float64)
        c = tf.Variable(0.3, dtype=tf.float64)

        dev = qml.device("default.qubit", wires=1)

        with tf.GradientTape() as tape:
            with TFInterface.apply(QuantumTape()) as qtape:
                qml.RY(a * c, wires=0)
                qml.RZ(b, wires=0)
                qml.RX(c + c ** 2 + tf.sin(a), wires=0)
                expval(qml.PauliZ(0))

            assert qtape.trainable_params == {0, 2}
            assert qtape.get_parameters() == [a * c, c + c ** 2 + tf.sin(a)]
            res = qtape.execute(dev)

        res = tape.jacobian(res, [a, b, c])
        assert isinstance(res[0], tf.Tensor)
        assert res[1] is None
        assert isinstance(res[2], tf.Tensor)

    def test_no_trainable_parameters(self, tol):
        """Test evaluation if there are no trainable parameters"""
        dev = qml.device("default.qubit", wires=2)

        with tf.GradientTape() as tape:

            with TFInterface.apply(QuantumTape()) as qtape:
                qml.RY(0.2, wires=0)
                qml.RX(tf.constant(0.1), wires=0)
                qml.CNOT(wires=[0, 1])
                expval(qml.PauliZ(0))
                expval(qml.PauliZ(1))

            assert qtape.trainable_params == set()

            res = qtape.execute(dev)

        assert res.shape == (2,)
        assert isinstance(res, tf.Tensor)

    @pytest.mark.parametrize("U", [tf.constant([[0, 1], [1, 0]]), np.array([[0, 1], [1, 0]])])
    def test_matrix_parameter(self, U, tol):
        """Test that the TF interface works correctly
        with a matrix parameter"""
        a = tf.Variable(0.1, dtype=tf.float64)

        dev = qml.device("default.qubit", wires=2)

        with tf.GradientTape() as tape:

            with TFInterface.apply(QuantumTape()) as qtape:
                qml.QubitUnitary(U, wires=0)
                qml.RY(a, wires=0)
                expval(qml.PauliZ(0))

            assert qtape.trainable_params == {1}
            res = qtape.execute(dev)

        assert np.allclose(res, -tf.cos(a), atol=tol, rtol=0)

        res = tape.jacobian(res, a)
        assert np.allclose(res, tf.sin(a), atol=tol, rtol=0)

    def test_differentiable_expand(self, mocker, tol):
        """Test that operation and nested tapes expansion
        is differentiable"""
        mock = mocker.patch.object(qml.operation.Operation, "do_check_domain", False)

        class U3(qml.U3):
            def expand(self):
                tape = QuantumTape()
                theta, phi, lam = self.data
                wires = self.wires
                tape._ops += [
                    qml.Rot(lam, theta, -lam, wires=wires),
                    qml.PhaseShift(phi + lam, wires=wires),
                ]
                return tape

        qtape = QuantumTape()

        dev = qml.device("default.qubit", wires=1)
        a = np.array(0.1)
        p = tf.Variable([0.1, 0.2, 0.3], dtype=tf.float64)

        with tf.GradientTape() as tape:

            with qtape:
                qml.RX(a, wires=0)
                U3(p[0], p[1], p[2], wires=0)
                expval(qml.PauliX(0))

            qtape = TFInterface.apply(qtape.expand())

            assert qtape.trainable_params == {1, 2, 3, 4}
            assert [i.name for i in qtape.operations] == ["RX", "Rot", "PhaseShift"]
            assert np.all(qtape.get_parameters() == [p[2], p[0], -p[2], p[1] + p[2]])

            res = qtape.execute(device=dev)

        expected = tf.cos(a) * tf.cos(p[1]) * tf.sin(p[0]) + tf.sin(a) * (
            tf.cos(p[2]) * tf.sin(p[1]) + tf.cos(p[0]) * tf.cos(p[1]) * tf.sin(p[2])
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = tape.jacobian(res, p)
        expected = np.array(
            [
                tf.cos(p[1]) * (tf.cos(a) * tf.cos(p[0]) - tf.sin(a) * tf.sin(p[0]) * tf.sin(p[2])),
                tf.cos(p[1]) * tf.cos(p[2]) * tf.sin(a)
                - tf.sin(p[1])
                * (tf.cos(a) * tf.sin(p[0]) + tf.cos(p[0]) * tf.sin(a) * tf.sin(p[2])),
                tf.sin(a)
                * (tf.cos(p[0]) * tf.cos(p[1]) * tf.cos(p[2]) - tf.sin(p[1]) * tf.sin(p[2])),
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_probability_differentiation(self, tol):
        """Tests correct output shape and evaluation for a tape
        with multiple prob outputs"""

        dev = qml.device("default.qubit", wires=2)
        x = tf.Variable(0.543, dtype=tf.float64)
        y = tf.Variable(-0.654, dtype=tf.float64)

        with tf.GradientTape() as tape:
            with TFInterface.apply(QuantumTape()) as qtape:
                qml.RX(x, wires=[0])
                qml.RY(y, wires=[1])
                qml.CNOT(wires=[0, 1])
                probs(wires=[0])
                probs(wires=[1])

            res = qtape.execute(dev)

        expected = np.array(
            [
                [tf.cos(x / 2) ** 2, tf.sin(x / 2) ** 2],
                [(1 + tf.cos(x) * tf.cos(y)) / 2, (1 - tf.cos(x) * tf.cos(y)) / 2],
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = tape.jacobian(res, [x, y])
        expected = np.array(
            [
                [
                    [-tf.sin(x) / 2, tf.sin(x) / 2],
                    [-tf.sin(x) * tf.cos(y) / 2, tf.cos(y) * tf.sin(x) / 2],
                ],
                [
                    [0, 0],
                    [-tf.cos(x) * tf.sin(y) / 2, tf.cos(x) * tf.sin(y) / 2],
                ],
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_ragged_differentiation(self, tol):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""
        dev = qml.device("default.qubit", wires=2)
        x = tf.Variable(0.543, dtype=tf.float64)
        y = tf.Variable(-0.654, dtype=tf.float64)

        with tf.GradientTape() as tape:
            with TFInterface.apply(QuantumTape()) as qtape:
                qml.RX(x, wires=[0])
                qml.RY(y, wires=[1])
                qml.CNOT(wires=[0, 1])
                expval(qml.PauliZ(0))
                probs(wires=[1])

            res = qtape.execute(dev)

        expected = np.array(
            [tf.cos(x), (1 + tf.cos(x) * tf.cos(y)) / 2, (1 - tf.cos(x) * tf.cos(y)) / 2]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = tape.jacobian(res, [x, y])
        expected = np.array(
            [
                [-tf.sin(x), -tf.sin(x) * tf.cos(y) / 2, tf.cos(y) * tf.sin(x) / 2],
                [0, -tf.cos(x) * tf.sin(y) / 2, tf.cos(x) * tf.sin(y) / 2],
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_sampling(self):
        """Test sampling works as expected"""
        dev = qml.device("default.qubit", wires=2, shots=10)

        with tf.GradientTape() as tape:
            with TFInterface.apply(QuantumTape()) as qtape:
                qml.Hadamard(wires=[0])
                qml.CNOT(wires=[0, 1])
                sample(qml.PauliZ(0))
                sample(qml.PauliX(1))

            res = qtape.execute(dev)

        assert res.shape == (2, 10)
        assert isinstance(res, tf.Tensor)


class TestTFPassthru:
    """Test that the quantum tape works with a TF passthru
    device.

    These tests are very similar to the tests above, with three key differences:

    * We do **not** apply the TF interface. These tapes simply use passthru
      backprop, no custom gradient registration needed.

    * We do not test the trainable_params attribute. Since these tapes have no
      TF interface, the tape does not need to bookkeep which parameters
      are trainable; this is done by TF internally.

    * We use mock.spy to ensure that the tape's Jacobian method is not being called.
    """

    def test_execution(self):
        """Test execution"""
        a = tf.Variable(0.1)
        dev = qml.device("default.qubit.tf", wires=1)

        with tf.GradientTape() as tape:
            with QuantumTape() as qtape:
                qml.RY(a, wires=0)
                qml.RX(0.2, wires=0)
                expval(qml.PauliZ(0))

            res = qtape.execute(dev)

        assert isinstance(res, tf.Tensor)
        assert res.shape == (1,)

    def test_jacobian(self, mocker, tol):
        """Test jacobian calculation"""
        spy = mocker.spy(QuantumTape, "jacobian")
        a = tf.Variable(0.1, dtype=tf.float64)
        b = tf.Variable(0.2, dtype=tf.float64)

        dev = qml.device("default.qubit.tf", wires=2)

        with tf.GradientTape() as tape:
            with QuantumTape() as qtape:
                qml.RY(a, wires=0)
                qml.RX(b, wires=1)
                qml.CNOT(wires=[0, 1])
                expval(qml.PauliZ(0))
                expval(qml.PauliY(1))

            res = qtape.execute(dev)

        assert isinstance(res, tf.Tensor)
        assert res.shape == (2,)

        expected = [tf.cos(a), -tf.cos(a) * tf.sin(b)]
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = tape.jacobian(res, [a, b])
        expected = [[-tf.sin(a), tf.sin(a) * tf.sin(b)], [0, -tf.cos(a) * tf.cos(b)]]
        assert np.allclose(res, expected, atol=tol, rtol=0)

        spy.assert_not_called()

    def test_reusing_quantum_tape(self, mocker, tol):
        """Test re-using a quantum tape by passing new parameters"""
        spy = mocker.spy(QuantumTape, "jacobian")

        a = tf.Variable(0.1, dtype=tf.float64)
        b = tf.Variable(0.2, dtype=tf.float64)

        dev = qml.device("default.qubit.tf", wires=2)

        with tf.GradientTape() as tape:

            with QuantumTape() as qtape:
                qml.RY(a, wires=0)
                qml.RX(b, wires=1)
                qml.CNOT(wires=[0, 1])
                expval(qml.PauliZ(0))
                expval(qml.PauliY(1))

            res = qtape.execute(dev)

        jac = tape.jacobian(res, [a, b])

        a = tf.Variable(0.54, dtype=tf.float64)
        b = tf.Variable(0.8, dtype=tf.float64)

        with tf.GradientTape() as tape:
            res2 = qtape.execute(dev, params=[2 * a, b])

        expected = [tf.cos(2 * a), -tf.cos(2 * a) * tf.sin(b)]
        assert np.allclose(res2, expected, atol=tol, rtol=0)

        jac = tape.jacobian(res2, [a, b])
        expected = [
            [-2 * tf.sin(2 * a), 2 * tf.sin(2 * a) * tf.sin(b)],
            [0, -tf.cos(2 * a) * tf.cos(b)],
        ]
        assert np.allclose(jac, expected, atol=tol, rtol=0)

        spy.assert_not_called()

    def test_classical_processing(self, mocker, tol):
        """Test classical processing within the quantum tape"""
        spy = mocker.spy(QuantumTape, "jacobian")

        a = tf.Variable(0.1, dtype=tf.float64)
        b = tf.constant(0.2, dtype=tf.float64)
        c = tf.Variable(0.3, dtype=tf.float64)

        dev = qml.device("default.qubit.tf", wires=1)

        with tf.GradientTape() as tape:
            with QuantumTape() as qtape:
                qml.RY(a * c, wires=0)
                qml.RZ(b, wires=0)
                qml.RX(c + c ** 2 + tf.sin(a), wires=0)
                expval(qml.PauliZ(0))

            assert qtape.get_parameters() == [a * c, b, c + c ** 2 + tf.sin(a)]
            res = qtape.execute(dev)

        res = tape.jacobian(res, [a, b, c])
        assert isinstance(res[0], tf.Tensor)
        assert res[1] is None
        assert isinstance(res[2], tf.Tensor)

        spy.assert_not_called()

    def test_no_trainable_parameters(self, mocker, tol):
        """Test evaluation if there are no trainable parameters"""
        spy = mocker.spy(QuantumTape, "jacobian")
        dev = qml.device("default.qubit.tf", wires=2)

        with tf.GradientTape() as tape:
            with QuantumTape() as qtape:
                qml.RY(0.2, wires=0)
                qml.RX(tf.constant(0.1), wires=0)
                qml.CNOT(wires=[0, 1])
                expval(qml.PauliZ(0))
                expval(qml.PauliZ(1))

            res = qtape.execute(dev)

        assert res.shape == (2,)
        assert isinstance(res, tf.Tensor)
        spy.assert_not_called()

    @pytest.mark.parametrize("U", [tf.constant([[0, 1], [1, 0]]), np.array([[0, 1], [1, 0]])])
    def test_matrix_parameter(self, U, mocker, tol):
        """Test that the TF interface works correctly
        with a matrix parameter"""
        spy = mocker.spy(QuantumTape, "jacobian")
        a = tf.Variable(0.1, dtype=tf.float64)

        dev = qml.device("default.qubit.tf", wires=2)

        with tf.GradientTape() as tape:
            with QuantumTape() as qtape:
                qml.QubitUnitary(U, wires=0)
                qml.RY(a, wires=0)
                expval(qml.PauliZ(0))
            res = qtape.execute(dev)

        assert np.allclose(res, -tf.cos(a), atol=tol, rtol=0)

        res = tape.jacobian(res, a)
        assert np.allclose(res, tf.sin(a), atol=tol, rtol=0)
        spy.assert_not_called()

    def test_differentiable_expand(self, mocker, tol):
        """Test that operation and nested tapes expansion
        is differentiable"""
        spy = mocker.spy(QuantumTape, "jacobian")
        mock = mocker.patch.object(qml.operation.Operation, "do_check_domain", False)

        class U3(qml.U3):
            def expand(self):
                tape = QuantumTape()
                theta, phi, lam = self.data
                wires = self.wires
                tape._ops += [
                    qml.Rot(lam, theta, -lam, wires=wires),
                    qml.PhaseShift(phi + lam, wires=wires),
                ]
                return tape

        qtape = QuantumTape()

        dev = qml.device("default.qubit.tf", wires=1)
        a = np.array(0.1)
        p = tf.Variable([0.1, 0.2, 0.3], dtype=tf.float64)

        with tf.GradientTape() as tape:

            with qtape:
                qml.RX(a, wires=0)
                U3(p[0], p[1], p[2], wires=0)
                expval(qml.PauliX(0))

            qtape = qtape.expand()

            assert [i.name for i in qtape.operations] == ["RX", "Rot", "PhaseShift"]
            assert np.all(qtape.get_parameters() == [a, p[2], p[0], -p[2], p[1] + p[2]])

            res = qtape.execute(device=dev)

        expected = tf.cos(a) * tf.cos(p[1]) * tf.sin(p[0]) + tf.sin(a) * (
            tf.cos(p[2]) * tf.sin(p[1]) + tf.cos(p[0]) * tf.cos(p[1]) * tf.sin(p[2])
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = tape.jacobian(res, p)
        expected = np.array(
            [
                tf.cos(p[1]) * (tf.cos(a) * tf.cos(p[0]) - tf.sin(a) * tf.sin(p[0]) * tf.sin(p[2])),
                tf.cos(p[1]) * tf.cos(p[2]) * tf.sin(a)
                - tf.sin(p[1])
                * (tf.cos(a) * tf.sin(p[0]) + tf.cos(p[0]) * tf.sin(a) * tf.sin(p[2])),
                tf.sin(a)
                * (tf.cos(p[0]) * tf.cos(p[1]) * tf.cos(p[2]) - tf.sin(p[1]) * tf.sin(p[2])),
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)
        spy.assert_not_called()

    def test_probability_differentiation(self, tol):
        """Tests correct output shape and evaluation for a tape
        with multiple prob outputs"""

        dev = qml.device("default.qubit.tf", wires=2)
        x = tf.Variable(0.543, dtype=tf.float64)
        y = tf.Variable(-0.654, dtype=tf.float64)

        with tf.GradientTape() as tape:
            with QuantumTape() as qtape:
                qml.RX(x, wires=[0])
                qml.RY(y, wires=[1])
                qml.CNOT(wires=[0, 1])
                probs(wires=[0])
                probs(wires=[1])

            res = qtape.execute(dev)

        expected = np.array(
            [
                [tf.cos(x / 2) ** 2, tf.sin(x / 2) ** 2],
                [(1 + tf.cos(x) * tf.cos(y)) / 2, (1 - tf.cos(x) * tf.cos(y)) / 2],
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = tape.jacobian(res, [x, y])
        expected = np.array(
            [
                [
                    [-tf.sin(x) / 2, tf.sin(x) / 2],
                    [-tf.sin(x) * tf.cos(y) / 2, tf.cos(y) * tf.sin(x) / 2],
                ],
                [
                    [0, 0],
                    [-tf.cos(x) * tf.sin(y) / 2, tf.cos(x) * tf.sin(y) / 2],
                ],
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_ragged_differentiation(self, monkeypatch, tol):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""
        dev = qml.device("default.qubit.tf", wires=2)
        x = tf.Variable(0.543, dtype=tf.float64)
        y = tf.Variable(-0.654, dtype=tf.float64)

        def _asarray(args, dtype=tf.float64):
            res = [tf.reshape(i, [-1]) for i in args]
            res = tf.concat(res, axis=0)
            return tf.cast(res, dtype=dtype)

        # The current DefaultQubitTF device provides an _asarray method that does
        # not work correctly for ragged arrays. For ragged arrays, we would like _asarray to
        # flatten the array. Here, we patch the _asarray method on the device to achieve this
        # behaviour; once the tape has moved from the beta folder, we should implement
        # this change directly in the device.
        monkeypatch.setattr(dev, "_asarray", _asarray)

        with tf.GradientTape() as tape:
            with QuantumTape() as qtape:
                qml.RX(x, wires=[0])
                qml.RY(y, wires=[1])
                qml.CNOT(wires=[0, 1])
                expval(qml.PauliZ(0))
                probs(wires=[1])

            res = qtape.execute(dev)

        expected = np.array(
            [tf.cos(x), (1 + tf.cos(x) * tf.cos(y)) / 2, (1 - tf.cos(x) * tf.cos(y)) / 2]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = tape.jacobian(res, [x, y])
        expected = np.array(
            [
                [-tf.sin(x), -tf.sin(x) * tf.cos(y) / 2, tf.cos(y) * tf.sin(x) / 2],
                [0, -tf.cos(x) * tf.sin(y) / 2, tf.cos(x) * tf.sin(y) / 2],
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_sampling(self):
        """Test sampling works as expected"""
        dev = qml.device("default.qubit.tf", wires=2, shots=10)

        with tf.GradientTape() as tape:
            with QuantumTape() as qtape:
                qml.Hadamard(wires=[0])
                qml.CNOT(wires=[0, 1])
                sample(qml.PauliZ(0))
                sample(qml.PauliX(1))

            res = qtape.execute(dev)

        assert res.shape == (2, 10)
        assert isinstance(res, tf.Tensor)


@pytest.mark.parametrize(
    "dev_name,diff_method", [["default.qubit", "finite-diff"], ["default.qubit.tf", "backprop"]]
)
class TestQNode:
    """Same tests as above, but this time via the QNode interface!"""

    def test_execution_no_interface(self, dev_name, diff_method):
        """Test execution works without an interface"""
        if diff_method == "backprop":
            pytest.skip("Test does not support backprop")

        dev = qml.device(dev_name, wires=1)

        @qnode(dev)
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return expval(qml.PauliZ(0))

        a = tf.Variable(0.1)

        with tf.GradientTape() as tape:
            res = circuit(a)

        assert circuit.qtape.interface == "autograd"

        # without the interface, the tape simply returns an array of results
        assert isinstance(res, np.ndarray)
        assert res.shape == (1,)
        assert isinstance(res[0], float)

        # without the interface, the tape is unable to deduce
        # trainable parameters
        assert circuit.qtape.trainable_params == {0, 1}

        # gradients should cause an error
        with pytest.raises(AttributeError, match="has no attribute '_id'"):
            assert tape.gradient(res, a) is None

    def test_execution_with_interface(self, dev_name, diff_method):
        """Test execution works with the interface"""
        if diff_method == "backprop":
            pytest.skip("Test does not support backprop")

        dev = qml.device(dev_name, wires=1)

        @qnode(dev, interface="tf")
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return expval(qml.PauliZ(0))

        a = tf.Variable(0.1)
        circuit(a)

        # if executing outside a gradient tape, the number of trainable parameters
        # cannot be determined by TensorFlow
        assert circuit.qtape.trainable_params == set()

        with tf.GradientTape() as tape:
            res = circuit(a)

        assert circuit.qtape.interface == "tf"

        # with the interface, the tape returns tensorflow tensors
        assert isinstance(res, tf.Tensor)
        assert res.shape == (1,)

        # the tape is able to deduce trainable parameters
        assert circuit.qtape.trainable_params == {0}

        # gradients should work
        grad = tape.gradient(res, a)
        assert isinstance(grad, tf.Tensor)
        assert grad.shape == tuple()

    def test_interface_swap(self, dev_name, diff_method, tol):
        """Test that the TF interface can be applied to a QNode
        with a pre-existing interface"""
        if diff_method == "backprop":
            pytest.skip("Test does not support backprop")

        dev = qml.device(dev_name, wires=1)

        @qnode(dev, interface="autograd")
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return expval(qml.PauliZ(0))

        from pennylane import numpy as anp

        a = anp.array(0.1, requires_grad=True)

        res1 = circuit(a)
        grad_fn = qml.grad(circuit)
        grad1 = grad_fn(a)

        # switch to TF interface
        circuit.to_tf()

        a = tf.Variable(0.1, dtype=tf.float64)

        with tf.GradientTape() as tape:
            res2 = circuit(a)

        grad2 = tape.gradient(res2, a)
        assert np.allclose(res1, res2, atol=tol, rtol=0)
        assert np.allclose(grad1, grad2, atol=tol, rtol=0)

    def test_jacobian(self, dev_name, diff_method, mocker, tol):
        """Test jacobian calculation"""
        spy = mocker.spy(QuantumTape, "jacobian")
        a = tf.Variable(0.1, dtype=tf.float64)
        b = tf.Variable(0.2, dtype=tf.float64)

        dev = qml.device(dev_name, wires=2)

        @qnode(dev, diff_method=diff_method, interface="tf")
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return [expval(qml.PauliZ(0)), expval(qml.PauliY(1))]

        with tf.GradientTape() as tape:
            res = circuit(a, b)

        assert circuit.qtape.trainable_params == {0, 1}

        assert isinstance(res, tf.Tensor)
        assert res.shape == (2,)

        expected = [tf.cos(a), -tf.cos(a) * tf.sin(b)]
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = tape.jacobian(res, [a, b])
        expected = [[-tf.sin(a), tf.sin(a) * tf.sin(b)], [0, -tf.cos(a) * tf.cos(b)]]
        assert np.allclose(res, expected, atol=tol, rtol=0)

        if diff_method == "finite-diff":
            spy.assert_called()
        elif diff_method == "backprop":
            spy.assert_not_called()

    def test_jacobian_dtype(self, dev_name, diff_method, tol):
        """Test calculating the jacobian with a different datatype"""
        if diff_method == "backprop":
            pytest.skip("Test does not support backprop")

        a = tf.Variable(0.1, dtype=tf.float32)
        b = tf.Variable(0.2, dtype=tf.float32)

        dev = qml.device("default.qubit", wires=2)

        @qnode(dev)
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return [expval(qml.PauliZ(0)), expval(qml.PauliY(1))]

        circuit.to_tf(dtype=tf.float32)
        assert circuit.dtype is tf.float32

        with tf.GradientTape() as tape:
            res = circuit(a, b)

        assert circuit.qtape.interface == "tf"
        assert circuit.qtape.trainable_params == {0, 1}

        assert isinstance(res, tf.Tensor)
        assert res.shape == (2,)
        assert res.dtype is tf.float32

        res = tape.jacobian(res, [a, b])
        assert [r.dtype is tf.float32 for r in res]

    def test_jacobian_options(self, dev_name, diff_method, mocker, tol):
        """Test setting jacobian options"""
        if diff_method == "backprop":
            pytest.skip("Test does not support backprop")

        spy = mocker.spy(QuantumTape, "numeric_pd")

        a = tf.Variable([0.1, 0.2])

        dev = qml.device("default.qubit", wires=1)

        @qnode(dev, interface="tf", h=1e-8, order=2)
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return expval(qml.PauliZ(0))

        with tf.GradientTape() as tape:
            res = circuit(a)

        tape.jacobian(res, a)

        for args in spy.call_args_list:
            assert args[1]["order"] == 2
            assert args[1]["h"] == 1e-8

    def test_changing_trainability(self, dev_name, diff_method, mocker, tol):
        """Test changing the trainability of parameters changes the
        number of differentiation requests made"""
        if diff_method == "backprop":
            pytest.skip("Test does not support backprop")

        a = tf.Variable(0.1, dtype=tf.float64)
        b = tf.Variable(0.2, dtype=tf.float64)

        dev = qml.device("default.qubit", wires=2)

        @qnode(dev, interface="tf", diff_method="finite-diff")
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return expval(qml.PauliZ(0)), expval(qml.PauliY(1))

        with tf.GradientTape() as tape:
            res = circuit(a, b)

        # the tape has reported both gate arguments as trainable
        assert circuit.qtape.trainable_params == {0, 1}

        expected = [tf.cos(a), -tf.cos(a) * tf.sin(b)]
        assert np.allclose(res, expected, atol=tol, rtol=0)

        spy = mocker.spy(QuantumTape, "numeric_pd")

        jac = tape.jacobian(res, [a, b])
        expected = [
            [-tf.sin(a), tf.sin(a) * tf.sin(b)],
            [0, -tf.cos(a) * tf.cos(b)],
        ]
        assert np.allclose(jac, expected, atol=tol, rtol=0)

        # QuantumTape.numeric_pd has been called for each argument
        assert len(spy.call_args_list) == 2

        # make the second QNode argument a constant
        a = tf.Variable(0.54, dtype=tf.float64)
        b = tf.constant(0.8, dtype=tf.float64)

        with tf.GradientTape() as tape:
            res = circuit(a, b)

        # the tape has reported only the first argument as trainable
        assert circuit.qtape.trainable_params == {0}

        expected = [tf.cos(a), -tf.cos(a) * tf.sin(b)]
        assert np.allclose(res, expected, atol=tol, rtol=0)

        spy.call_args_list = []
        jac = tape.jacobian(res, a)
        expected = [-tf.sin(a), tf.sin(a) * tf.sin(b)]
        assert np.allclose(jac, expected, atol=tol, rtol=0)

        # QuantumTape.numeric_pd has been called only once
        assert len(spy.call_args_list) == 1

    def test_classical_processing(self, dev_name, diff_method, tol):
        """Test classical processing within the quantum tape"""
        a = tf.Variable(0.1, dtype=tf.float64)
        b = tf.constant(0.2, dtype=tf.float64)
        c = tf.Variable(0.3, dtype=tf.float64)

        dev = qml.device(dev_name, wires=1)

        @qnode(dev, diff_method=diff_method, interface="tf")
        def circuit(a, b, c):
            qml.RY(a * c, wires=0)
            qml.RZ(b, wires=0)
            qml.RX(c + c ** 2 + tf.sin(a), wires=0)
            return expval(qml.PauliZ(0))

        with tf.GradientTape() as tape:
            res = circuit(a, b, c)

        if diff_method == "finite-diff":
            assert circuit.qtape.trainable_params == {0, 2}
            assert circuit.qtape.get_parameters() == [a * c, c + c ** 2 + tf.sin(a)]

        res = tape.jacobian(res, [a, b, c])

        assert isinstance(res[0], tf.Tensor)
        assert res[1] is None
        assert isinstance(res[2], tf.Tensor)

    def test_no_trainable_parameters(self, dev_name, diff_method, tol):
        """Test evaluation and Jacobian if there are no trainable parameters"""
        dev = qml.device(dev_name, wires=2)

        @qnode(dev, diff_method=diff_method, interface="tf")
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            qml.CNOT(wires=[0, 1])
            return expval(qml.PauliZ(0)), expval(qml.PauliZ(1))

        a = 0.1
        b = tf.constant(0.2, dtype=tf.float64)

        with tf.GradientTape() as tape:
            res = circuit(a, b)

        if diff_method == "finite-diff":
            assert circuit.qtape.trainable_params == set()

        assert res.shape == (2,)
        assert isinstance(res, tf.Tensor)

    @pytest.mark.parametrize("U", [tf.constant([[0, 1], [1, 0]]), np.array([[0, 1], [1, 0]])])
    def test_matrix_parameter(self, dev_name, diff_method, U, tol):
        """Test that the TF interface works correctly
        with a matrix parameter"""
        a = tf.Variable(0.1, dtype=tf.float64)

        dev = qml.device(dev_name, wires=2)

        @qnode(dev, diff_method=diff_method, interface="tf")
        def circuit(U, a):
            qml.QubitUnitary(U, wires=0)
            qml.RY(a, wires=0)
            return expval(qml.PauliZ(0))

        with tf.GradientTape() as tape:
            res = circuit(U, a)

        if diff_method == "finite-diff":
            assert circuit.qtape.trainable_params == {1}

        assert np.allclose(res, -tf.cos(a), atol=tol, rtol=0)

        res = tape.jacobian(res, a)
        assert np.allclose(res, tf.sin(a), atol=tol, rtol=0)

    def test_differentiable_expand(self, dev_name, diff_method, mocker, tol):
        """Test that operation and nested tapes expansion
        is differentiable"""
        mock = mocker.patch.object(qml.operation.Operation, "do_check_domain", False)

        class U3(qml.U3):
            def expand(self):
                theta, phi, lam = self.data
                wires = self.wires

                with QuantumTape() as tape:
                    qml.Rot(lam, theta, -lam, wires=wires)
                    qml.PhaseShift(phi + lam, wires=wires)

                return tape

        dev = qml.device(dev_name, wires=1)
        a = np.array(0.1)
        p = tf.Variable([0.1, 0.2, 0.3], dtype=tf.float64)

        @qnode(dev, diff_method=diff_method, interface="tf")
        def circuit(a, p):
            qml.RX(a, wires=0)
            U3(p[0], p[1], p[2], wires=0)
            return expval(qml.PauliX(0))

        with tf.GradientTape() as tape:
            res = circuit(a, p)

        if diff_method == "finite-diff":
            assert circuit.qtape.trainable_params == {1, 2, 3, 4}
        elif diff_method == "backprop":
            assert circuit.qtape.trainable_params == {0, 1, 2, 3, 4}

        assert [i.name for i in circuit.qtape.operations] == ["RX", "Rot", "PhaseShift"]

        if diff_method == "finite-diff":
            assert np.all(circuit.qtape.get_parameters() == [p[2], p[0], -p[2], p[1] + p[2]])
        elif diff_method == "backprop":
            assert np.all(circuit.qtape.get_parameters() == [a, p[2], p[0], -p[2], p[1] + p[2]])

        expected = tf.cos(a) * tf.cos(p[1]) * tf.sin(p[0]) + tf.sin(a) * (
            tf.cos(p[2]) * tf.sin(p[1]) + tf.cos(p[0]) * tf.cos(p[1]) * tf.sin(p[2])
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = tape.jacobian(res, p)
        expected = np.array(
            [
                tf.cos(p[1]) * (tf.cos(a) * tf.cos(p[0]) - tf.sin(a) * tf.sin(p[0]) * tf.sin(p[2])),
                tf.cos(p[1]) * tf.cos(p[2]) * tf.sin(a)
                - tf.sin(p[1])
                * (tf.cos(a) * tf.sin(p[0]) + tf.cos(p[0]) * tf.sin(a) * tf.sin(p[2])),
                tf.sin(a)
                * (tf.cos(p[0]) * tf.cos(p[1]) * tf.cos(p[2]) - tf.sin(p[1]) * tf.sin(p[2])),
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_probability_differentiation(self, dev_name, diff_method, tol):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""

        dev = qml.device(dev_name, wires=2)
        x = tf.Variable(0.543, dtype=tf.float64)
        y = tf.Variable(-0.654, dtype=tf.float64)

        @qnode(dev, diff_method=diff_method, interface="tf")
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return probs(wires=[0]), probs(wires=[1])

        with tf.GradientTape() as tape:
            res = circuit(x, y)

        expected = np.array(
            [
                [tf.cos(x / 2) ** 2, tf.sin(x / 2) ** 2],
                [(1 + tf.cos(x) * tf.cos(y)) / 2, (1 - tf.cos(x) * tf.cos(y)) / 2],
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = tape.jacobian(res, [x, y])
        expected = np.array(
            [
                [
                    [-tf.sin(x) / 2, tf.sin(x) / 2],
                    [-tf.sin(x) * tf.cos(y) / 2, tf.cos(y) * tf.sin(x) / 2],
                ],
                [
                    [0, 0],
                    [-tf.cos(x) * tf.sin(y) / 2, tf.cos(x) * tf.sin(y) / 2],
                ],
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_ragged_differentiation(self, dev_name, diff_method, monkeypatch, tol):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""
        dev = qml.device(dev_name, wires=2)
        x = tf.Variable(0.543, dtype=tf.float64)
        y = tf.Variable(-0.654, dtype=tf.float64)

        def _asarray(args, dtype=tf.float64):
            res = [tf.reshape(i, [-1]) for i in args]
            res = tf.concat(res, axis=0)
            return tf.cast(res, dtype=dtype)

        if dev_name == "default.qubit.tf":
            # we need to patch the asarray method on the device
            monkeypatch.setattr(dev, "_asarray", _asarray)

        @qnode(dev, diff_method=diff_method, interface="tf")
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return [expval(qml.PauliZ(0)), probs(wires=[1])]

        with tf.GradientTape() as tape:
            res = circuit(x, y)

        expected = np.array(
            [tf.cos(x), (1 + tf.cos(x) * tf.cos(y)) / 2, (1 - tf.cos(x) * tf.cos(y)) / 2]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = tape.jacobian(res, [x, y])
        expected = np.array(
            [
                [-tf.sin(x), -tf.sin(x) * tf.cos(y) / 2, tf.cos(y) * tf.sin(x) / 2],
                [0, -tf.cos(x) * tf.sin(y) / 2, tf.cos(x) * tf.sin(y) / 2],
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_sampling(self, dev_name, diff_method):
        """Test sampling works as expected"""
        dev = qml.device(dev_name, wires=2, shots=10)

        @qnode(dev, diff_method=diff_method, interface="tf")
        def circuit():
            qml.Hadamard(wires=[0])
            qml.CNOT(wires=[0, 1])
            return [sample(qml.PauliZ(0)), sample(qml.PauliX(1))]

        with tf.GradientTape() as tape:
            res = circuit()

        assert res.shape == (2, 10)
        assert isinstance(res, tf.Tensor)


def qtransform(qnode, a, interface=tf):
    """transforms every RY(y) gate in a circuit to RX(-a*cos(y))"""

    def construct(self, args, kwargs):
        """New quantum tape construct method, that performs
        the transform on the tape in a define-by-run manner"""
        global t_op

        QNode.construct(self, args, kwargs)

        new_ops = []
        for o in self.qtape.operations:
            if isinstance(o, qml.RY):
                t_op = qml.RX(-a * interface.cos(o.data[0]), wires=o.wires)
                new_ops.append(t_op)
            else:
                new_ops.append(o)

        self.qtape._ops = new_ops
        self.qtape._update()

    import copy

    new_qnode = copy.deepcopy(qnode)
    new_qnode.construct = construct.__get__(new_qnode, QNode)
    return new_qnode


@pytest.mark.parametrize(
    "dev_name,diff_method", [("default.qubit", "finite-diff"), ("default.qubit.tf", "backprop")]
)
def test_transform(dev_name, diff_method, monkeypatch, tol):
    """Test an example transform"""
    monkeypatch.setattr(qml.operation.Operation, "do_check_domain", False)

    dev = qml.device(dev_name, wires=1)

    @qnode(dev, interface="tf", diff_method=diff_method)
    def circuit(weights):
        global op1, op2
        op1 = qml.RY(weights[0], wires=0)
        op2 = qml.RX(weights[1], wires=0)
        return expval(qml.PauliZ(wires=0))

    weights = tf.Variable([0.32, 0.543], dtype=tf.float64)
    a = tf.Variable(0.5, dtype=tf.float64)

    with tf.GradientTape(persistent=True) as tape:
        new_qnode = qtransform(circuit, a)
        res = new_qnode(weights)[0]
        res2 = circuit(tf.sin(weights))[0]
        loss = res + res2

    assert circuit.qtape.operations == [op1, op2]
    assert new_qnode.qtape.operations[0] == t_op
    assert new_qnode.qtape.operations[1].name == op2.name
    assert new_qnode.qtape.operations[1].wires == op2.wires

    assert np.all(circuit.qtape.get_parameters() == tf.sin(weights))
    assert np.all(new_qnode.qtape.get_parameters() == [-a * tf.cos(weights[0]), weights[1]])

    grad = tape.gradient(loss, [weights, a])
    assert len(grad) == 2
    assert grad[0].shape == weights.shape
    assert grad[1].shape == a.shape

    assert np.allclose(loss, 1.8244501889992706, atol=tol, rtol=0)
    assert np.allclose(grad[0], [-0.26610258, -0.47053553], atol=tol, rtol=0)
    assert np.allclose(grad[1], 0.06486032, atol=tol, rtol=0)

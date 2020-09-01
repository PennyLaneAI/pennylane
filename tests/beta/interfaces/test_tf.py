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
import numpy as np

import pennylane as qml
from pennylane.beta.tapes import QuantumTape
from pennylane.beta.queuing import expval, var, sample, probs
from pennylane.beta.interfaces.tf import TFInterface


tf = pytest.importorskip("tensorflow", minversion="2.1")


class TestTFQuantumTape:
    """Test the autograd interface applied to a tape"""

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

        TFInterface.apply(QuantumTape(), dtype=tf.float32)
        assert tape.interface == "tf"
        assert isinstance(tape, TFInterface)
        assert tape.__bare__ == QuantumTape
        assert tape.dtype is tf.float64

    def test_get_parameters(self):
        """Test that the get parameters function correctly sets and returns the
        trainable parameters"""
        a = tf.Variable(0.1)
        b = tf.constant(0.2)
        c = tf.Variable(0.3)
        d = tf.constant(0.4)

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
        """Test calculating the jacobian with a different datatype"""
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

        jac = tape.jacobian(res2, [a, b])
        expected = [
            [-2 * tf.sin(2 * a), 2 * tf.sin(2 * a) * tf.sin(b)],
            [0, -tf.cos(2 * a) * tf.cos(b)],
        ]
        assert np.allclose(jac, expected, atol=tol, rtol=0)

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
        """Test evaluation and Jacobian if there are no trainable parameters"""
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

    def test_differentiable_expand(self, tol):
        """Test that operation and nested tapes expansion
        is differentiable"""

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
        with prob and expval outputs"""

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
        """Test evaluation and Jacobian if there are no trainable parameters"""
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
        with prob and expval outputs"""

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

    def test_ragged_differentiation(self, tol):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""
        dev = qml.device("default.qubit.tf", wires=2)
        x = tf.Variable(0.543, dtype=tf.float64)
        y = tf.Variable(-0.654, dtype=tf.float64)

        def _asarray(args, dtype=tf.float64):
            res = [tf.reshape(i, [-1]) for i in args]
            res = tf.concat(res, axis=0)
            return tf.cast(res, dtype=dtype)

        dev._asarray = _asarray

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

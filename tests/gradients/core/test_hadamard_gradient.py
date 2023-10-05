# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
Tests for the gradients.hadamard_gradient module.
"""

import warnings
import pytest

import pennylane as qml
from pennylane import numpy as np


def grad_fn(tape, dev, fn=qml.gradients.hadamard_grad, **kwargs):
    """Utility function to automate execution and processing of gradient tapes"""
    tapes, fn = fn(tape, **kwargs)
    return fn(dev.execute(tapes)), tapes


def cost1(x):
    """Cost function."""
    qml.Rot(*x, wires=0)
    return qml.expval(qml.PauliZ(0))


def cost2(x):
    """Cost function."""
    qml.Rot(*x, wires=0)
    return [qml.expval(qml.PauliZ(0))]


def cost3(x):
    """Cost function."""
    qml.Rot(*x, wires=0)
    return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]


def cost4(x):
    """Cost function."""
    qml.Rot(*x, wires=0)
    return qml.probs([0, 1])


def cost5(x):
    """Cost function."""
    qml.Rot(*x, wires=0)
    return [qml.probs([0, 1])]


def cost6(x):
    """Cost function."""
    qml.Rot(*x, wires=0)
    return [qml.probs([0, 1]), qml.probs([2, 3])]


class TestHadamardGrad:
    """Unit tests for the hadamard_grad function"""

    def test_batched_tape_raises(self):
        """Test that an error is raised for a broadcasted/batched tape."""
        tape = qml.tape.QuantumScript([qml.RX([0.4, 0.2], 0)], [qml.expval(qml.PauliZ(0))])
        _match = "Computing the gradient of broadcasted tapes with the Hadamard test gradient"
        with pytest.raises(NotImplementedError, match=_match):
            qml.gradients.hadamard_grad(tape)

    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, 2 * np.pi, 7))
    @pytest.mark.parametrize("G", [qml.RX, qml.RY, qml.RZ, qml.PhaseShift, qml.U1])
    def test_pauli_rotation_gradient(self, G, theta, tol):
        """Tests that the automatic gradients of Pauli rotations are correct."""
        dev = qml.device("default.qubit", wires=2)

        with qml.queuing.AnnotatedQueue() as q:
            qml.StatePrep(np.array([1.0, -1.0], requires_grad=False) / np.sqrt(2), wires=0)
            G(theta, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {1}

        res_hadamard, tapes = grad_fn(tape, dev)

        assert len(tapes) == 1

        assert isinstance(res_hadamard, np.ndarray)
        assert res_hadamard.shape == ()

        res_param_shift, _ = grad_fn(tape, dev, qml.gradients.param_shift)

        assert np.allclose(res_hadamard, res_param_shift, atol=tol, rtol=0)

    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, 2 * np.pi, 7))
    def test_rot_gradient(self, theta, tol):
        """Tests that the automatic gradient of an arbitrary Euler-angle-parameterized gate is correct."""
        dev = qml.device("default.qubit", wires=2)
        params = np.array([theta, theta**3, np.sqrt(2) * theta])

        with qml.queuing.AnnotatedQueue() as q:
            qml.StatePrep(np.array([1.0, -1.0], requires_grad=False) / np.sqrt(2), wires=0)
            qml.Rot(*params, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {1, 2, 3}

        num_params = len(tape.trainable_params)
        res_hadamard, tapes = grad_fn(tape, dev)

        assert len(tapes) == num_params
        assert isinstance(res_hadamard, tuple)
        assert len(res_hadamard) == num_params

        res_param_shift, _ = grad_fn(tape, dev, qml.gradients.param_shift)

        assert np.allclose(res_hadamard, res_param_shift, atol=tol, rtol=0)

    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, 2 * np.pi, 7))
    @pytest.mark.parametrize("G", [qml.CRX, qml.CRY, qml.CRZ])
    def test_controlled_rotation_gradient(self, G, theta, tol):
        """Test gradient of controlled rotation gates"""
        dev = qml.device("default.qubit", wires=3)

        with qml.queuing.AnnotatedQueue() as q:
            qml.StatePrep(np.array([1.0, -1.0], requires_grad=False) / np.sqrt(2), wires=0)
            G(theta, wires=[0, 1])
            qml.expval(qml.PauliX(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {1}

        res_hadamard = dev.execute(tape)
        assert np.allclose(res_hadamard, -np.cos(theta / 2), atol=tol, rtol=0)

        res_hadamard, _ = grad_fn(tape, dev)

        res_param_shift, _ = grad_fn(tape, dev, qml.gradients.param_shift)
        assert np.allclose(res_hadamard, res_param_shift, atol=tol, rtol=0)

        expected = np.sin(theta / 2) / 2

        assert np.allclose(res_hadamard, expected, atol=tol, rtol=0)

    def test_control_rotations(self, tol):
        """Test RX, CRX and CRY."""
        dev = qml.device("default.qubit", wires=4)

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(0.1, wires=0)
            qml.CRY(0.2, wires=[2, 0])
            qml.CRX(0.3, wires=[0, 1])
            qml.expval(qml.PauliX(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {0, 1, 2}

        res_hadamard, _ = grad_fn(tape, dev)
        assert np.allclose(res_hadamard, np.zeros(3), atol=tol, rtol=0)

    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, 2 * np.pi, 7))
    @pytest.mark.parametrize("G", [qml.CRX, qml.CRY, qml.CRZ])
    def test_controlled_rotation_gradient_multi(self, G, theta, tol):
        """Test gradient of controlled rotation gates with multiple measurements"""
        dev = qml.device("default.qubit", wires=3)

        with qml.queuing.AnnotatedQueue() as q:
            qml.StatePrep(np.array([1.0, -1.0], requires_grad=False) / np.sqrt(2), wires=0)
            G(theta, wires=[0, 1])
            qml.expval(qml.PauliX(0))
            qml.probs(wires=[1])

        tape = qml.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {1}

        res_hadamard, _ = grad_fn(tape, dev)
        res_param_shift, _ = grad_fn(tape, dev, qml.gradients.param_shift)

        assert isinstance(res_hadamard, tuple)
        assert np.allclose(res_hadamard[0], res_param_shift[0], atol=tol, rtol=0)
        assert np.allclose(res_hadamard[1], res_param_shift[1], atol=tol, rtol=0)

    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, 2 * np.pi, 7))
    @pytest.mark.parametrize("G", [qml.IsingXX, qml.IsingYY, qml.IsingZZ])
    def test_ising_gradient(self, G, theta, tol):
        """Test gradient of Ising coupling gates"""
        dev = qml.device("default.qubit", wires=3)

        with qml.queuing.AnnotatedQueue() as q:
            qml.StatePrep(np.array([1.0, -1.0], requires_grad=False) / np.sqrt(2), wires=0)
            G(theta, wires=[0, 1])
            qml.expval(qml.PauliZ(1))

        tape = qml.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {1}

        res_hadamard, _ = grad_fn(tape, dev)

        res_param_shift, _ = grad_fn(tape, dev, qml.gradients.param_shift)

        assert np.allclose(res_hadamard, res_param_shift, atol=tol, rtol=0)

    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, np.pi, 7))
    def test_CRot_gradient_with_expansion(self, theta, tol):
        """Tests that the automatic gradient of an arbitrary controlled Euler-angle-parameterized
        gate is correct."""
        dev = qml.device("default.qubit", wires=3)
        a, b, c = np.array([theta, theta**3, np.sqrt(2) * theta])

        with qml.queuing.AnnotatedQueue() as q:
            qml.StatePrep(np.array([1.0, -1.0], requires_grad=False) / np.sqrt(2), wires=0)
            qml.CRot(a, b, c, wires=[0, 1])
            qml.expval(qml.PauliX(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {1, 2, 3}

        res_hadamard = dev.execute(tape)
        expected = -np.cos(b / 2) * np.cos(0.5 * (a + c))
        assert np.allclose(res_hadamard, expected, atol=tol, rtol=0)

        grad, _ = grad_fn(tape, dev)

        expected = np.array(
            [
                0.5 * np.cos(b / 2) * np.sin(0.5 * (a + c)),
                0.5 * np.sin(b / 2) * np.cos(0.5 * (a + c)),
                0.5 * np.cos(b / 2) * np.sin(0.5 * (a + c)),
            ]
        )

        assert isinstance(grad, tuple)
        assert len(grad) == 5
        assert np.allclose(-grad[1] / 2, expected[0], atol=tol, rtol=0)
        assert np.allclose(-grad[2] / 2, expected[1], atol=tol, rtol=0)
        assert np.allclose(-grad[1] / 2, expected[2], atol=tol, rtol=0)

    def test_gradients_agree_finite_differences(self, tol):
        """Tests that the Hadamard test gradient agrees with the first and second
        order finite differences"""
        params = np.array([0.1, -1.6, np.pi / 5])

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(params[0], wires=[0])
            qml.CNOT(wires=[0, 1])
            qml.RY(-1.6, wires=[0])
            qml.RY(params[1], wires=[1])
            qml.CNOT(wires=[1, 0])
            qml.RX(params[2], wires=[0])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        tape.trainable_params = {0, 2, 3}
        dev = qml.device("default.qubit", wires=3)

        grad_F1, _ = grad_fn(tape, dev, fn=qml.gradients.finite_diff, approx_order=1)
        grad_F2, _ = grad_fn(
            tape, dev, fn=qml.gradients.finite_diff, approx_order=2, strategy="center"
        )
        grad_A, _ = grad_fn(tape, dev)

        # gradients computed with different methods must agree
        assert np.allclose(grad_A, grad_F1, atol=tol, rtol=0)
        assert np.allclose(grad_A, grad_F2, atol=tol, rtol=0)

    @pytest.mark.parametrize("angle", np.linspace(0, 2 * np.pi, 7, requires_grad=True))
    @pytest.mark.parametrize("pauli_word", ["XX", "YY", "ZZ", "XY", "YX", "ZX", "ZY", "YZ"])
    def test_differentiability_paulirot(self, angle, pauli_word, tol):
        """Test that differentiation of PauliRot works."""

        dev = qml.device("default.qubit", wires=3)
        with qml.queuing.AnnotatedQueue() as q:
            qml.PauliRot(angle, pauli_word, wires=[0, 1])
            qml.expval(qml.PauliY(0) @ qml.PauliX(1))

        tape = qml.tape.QuantumScript.from_queue(q)

        res_hadamard, _ = grad_fn(tape, dev)
        res_param_shift, _ = grad_fn(tape, dev, fn=qml.gradients.param_shift)

        assert np.allclose(res_hadamard, res_param_shift, tol)

    def test_single_expectation_value(self, tol):
        """Tests correct output shape and evaluation for a tape
        with a single expval output"""
        dev = qml.device("default.qubit", wires=3)
        x = 0.543
        y = -0.654

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        tape = qml.tape.QuantumScript.from_queue(q)

        res_hadamard, tapes = grad_fn(tape, dev)

        assert len(tapes) == 2

        assert len(res_hadamard) == 2
        assert not isinstance(res_hadamard[0], tuple)
        assert not isinstance(res_hadamard[1], tuple)

        expected = np.array([-np.sin(y) * np.sin(x), np.cos(y) * np.cos(x)])
        assert np.allclose(res_hadamard[0], expected[0], atol=tol, rtol=0)
        assert np.allclose(res_hadamard[1], expected[1], atol=tol, rtol=0)

    def test_multiple_expectation_values(self, tol):
        """Tests correct output shape and evaluation for a tape
        with multiple expval outputs"""
        dev = qml.device("default.qubit", wires=3)
        x = 0.543
        y = -0.654

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.expval(qml.PauliX(1))

        tape = qml.tape.QuantumScript.from_queue(q)

        res_hadamard, tapes = grad_fn(tape, dev)

        assert len(tapes) == 2

        assert len(res_hadamard) == 2
        assert len(res_hadamard[0]) == 2
        assert len(res_hadamard[1]) == 2

        expected = np.array([[-np.sin(x), 0], [0, np.cos(y)]])
        assert np.allclose(res_hadamard[0], expected[0], atol=tol, rtol=0)
        assert np.allclose(res_hadamard[1], expected[1], atol=tol, rtol=0)

    def test_diff_single_probs(self, tol):
        """Test diff of single probabilities."""
        dev = qml.device("default.qubit", wires=3)
        x = 0.543
        y = -0.654

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.probs(wires=[1])

        tape = qml.tape.QuantumScript.from_queue(q)

        res_hadamard, tapes = grad_fn(tape, dev)

        assert len(tapes) == 2
        assert isinstance(res_hadamard, tuple)
        assert len(res_hadamard) == 2

        assert res_hadamard[0].shape == (2,)
        assert res_hadamard[1].shape == (2,)

        expected = np.array(
            [
                [-np.sin(x) * np.cos(y) / 2, -np.cos(x) * np.sin(y) / 2],
                [np.cos(y) * np.sin(x) / 2, np.cos(x) * np.sin(y) / 2],
            ]
        )

        assert np.allclose(res_hadamard[0], expected.T[0], atol=tol, rtol=0)
        assert np.allclose(res_hadamard[1], expected.T[1], atol=tol, rtol=0)

    def test_prob_expectation_values(self, tol):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""

        dev = qml.device("default.qubit", wires=3)
        x = 0.543
        y = -0.654

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.probs(wires=[0, 1])

        tape = qml.tape.QuantumScript.from_queue(q)

        res_hadamard, tapes = grad_fn(tape, dev)

        assert len(tapes) == 2

        assert isinstance(res_hadamard, tuple)
        assert len(res_hadamard) == 2

        assert isinstance(res_hadamard[0], tuple)
        assert len(res_hadamard[0]) == 2

        assert isinstance(res_hadamard[0][0], np.ndarray)
        assert res_hadamard[0][0].shape == ()
        assert isinstance(res_hadamard[0][1], np.ndarray)
        assert res_hadamard[0][1].shape == ()

        assert isinstance(res_hadamard[1], tuple)
        assert len(res_hadamard[1]) == 2

        assert isinstance(res_hadamard[1][0], np.ndarray)
        assert res_hadamard[1][0].shape == (4,)
        assert isinstance(res_hadamard[1][1], np.ndarray)
        assert res_hadamard[1][1].shape == (4,)

        expval_expected = [-2 * np.sin(x) / 2, 0]
        probs_expected = (
            np.array(
                [
                    [
                        -(np.cos(y / 2) ** 2 * np.sin(x)),
                        -(np.cos(x / 2) ** 2 * np.sin(y)),
                    ],
                    [
                        -(np.sin(x) * np.sin(y / 2) ** 2),
                        (np.cos(x / 2) ** 2 * np.sin(y)),
                    ],
                    [
                        (np.sin(x) * np.sin(y / 2) ** 2),
                        (np.sin(x / 2) ** 2 * np.sin(y)),
                    ],
                    [
                        (np.cos(y / 2) ** 2 * np.sin(x)),
                        -(np.sin(x / 2) ** 2 * np.sin(y)),
                    ],
                ]
            )
            / 2
        )
        # Expvals
        assert np.allclose(res_hadamard[0][0], expval_expected[0], tol)
        assert np.allclose(res_hadamard[0][1], expval_expected[1], tol)

        # Probs
        assert np.allclose(res_hadamard[1][0], probs_expected[:, 0], tol)
        assert np.allclose(res_hadamard[1][1], probs_expected[:, 1], tol)

    costs_and_expected_expval = [
        (cost1, [3]),
        (cost2, [3]),
        (cost3, [2, 3]),
    ]

    @pytest.mark.parametrize("cost, expected_shape", costs_and_expected_expval)
    def test_output_shape_matches_qnode_expval(self, cost, expected_shape):
        """Test that the transform output shape matches that of the QNode."""
        dev = qml.device("default.qubit", wires=4)

        x = np.random.rand(3)
        circuit = qml.QNode(cost, dev)

        res_hadamard = qml.gradients.hadamard_grad(circuit)(x)

        assert isinstance(res_hadamard, (tuple, list))
        if len(res_hadamard) == 1:
            res_hadamard = res_hadamard[0]
        assert len(res_hadamard) == expected_shape[0]

        if len(expected_shape) > 1:
            for r in res_hadamard:
                assert isinstance(r, tuple)
                assert len(r) == expected_shape[1]

    costs_and_expected_probs = [
        (cost4, [3, 4]),
        (cost5, [3, 4]),
        (cost6, [2, 3, 4]),
    ]

    @pytest.mark.parametrize("cost, expected_shape", costs_and_expected_probs)
    def test_output_shape_matches_qnode_probs(self, cost, expected_shape):
        """Test that the transform output shape matches that of the QNode."""
        dev = qml.device("default.qubit", wires=5)

        x = np.random.rand(3)
        circuit = qml.QNode(cost, dev)

        res_hadamard = qml.gradients.hadamard_grad(circuit)(x)
        assert isinstance(res_hadamard, (tuple, list))
        if len(res_hadamard) == 1:
            res_hadamard = res_hadamard[0]
        assert len(res_hadamard) == expected_shape[0]

        if len(expected_shape) > 2:
            for r in res_hadamard:
                assert isinstance(r, tuple)
                assert len(r) == expected_shape[1]

                for r_ in r:
                    assert isinstance(r_, qml.numpy.ndarray)
                    assert len(r_) == expected_shape[2]

        elif len(expected_shape) > 1:
            for r in res_hadamard:
                assert isinstance(r, qml.numpy.ndarray)
                assert len(r) == expected_shape[1]

    def test_multi_measure_no_warning(self):
        """Test computing the gradient of a tape that contains multiple
        measurements omits no warnings."""

        dev = qml.device("default.qubit", wires=4)

        par1 = qml.numpy.array(0.3)
        par2 = qml.numpy.array(0.1)

        with qml.queuing.AnnotatedQueue() as q:
            qml.RY(par1, wires=0)
            qml.RX(par2, wires=1)
            qml.probs(wires=[1, 2])
            qml.expval(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q)

        with warnings.catch_warnings(record=True) as record:
            grad_fn(tape, dev=dev)

        assert len(record) == 0

    @pytest.mark.parametrize("shots", [None, 100])
    def test_shots_attribute(self, shots):
        """Tests that the shots attribute is copied to the new tapes"""
        dev = qml.device("default.qubit", wires=3)
        x = 0.543
        y = -0.654

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        tape = qml.tape.QuantumScript.from_queue(q, shots=shots)
        _, tapes = grad_fn(tape, dev)

        assert all(new_tape.shots == tape.shots for new_tape in tapes)


class TestHadamardGradEdgeCases:
    """Test the Hadamard gradient transform and edge cases such as non diff parameters, auxiliary wires, etc..."""

    # pylint:disable=too-many-public-methods

    device_wires = [qml.wires.Wires([0, 1, "aux"])]
    device_wires_no_aux = [qml.wires.Wires([0, 1, 2])]

    working_wires = [None, qml.wires.Wires("aux"), "aux"]
    already_used_wires = [qml.wires.Wires(0), qml.wires.Wires(1)]

    @pytest.mark.parametrize("aux_wire", working_wires)
    @pytest.mark.parametrize("device_wires", device_wires)
    def test_aux_wire(self, aux_wire, device_wires):
        """Test the aux wire is available."""
        # One qubit is added to the device 2 + 1
        dev = qml.device("default.qubit", wires=device_wires)
        x = 0.543
        y = -0.654

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        tape = qml.tape.QuantumScript.from_queue(q)

        tapes, _ = qml.gradients.hadamard_grad(tape, aux_wire=aux_wire, device_wires=dev.wires)
        assert len(tapes) == 2
        tapes, _ = qml.gradients.hadamard_grad(tape, aux_wire=aux_wire)
        assert len(tapes) == 2

    @pytest.mark.parametrize("aux_wire", already_used_wires)
    @pytest.mark.parametrize("device_wires", device_wires)
    def test_aux_wire_already_used_wires(self, aux_wire, device_wires):
        """Test the aux wire is available."""
        dev = qml.device("default.qubit", wires=device_wires)
        x = 0.543
        y = -0.654

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        tape = qml.tape.QuantumScript.from_queue(q)

        _match = "The requested auxiliary wire is already in use by the circuit"
        with pytest.raises(qml.wires.WireError, match=_match):
            qml.gradients.hadamard_grad(tape, aux_wire=aux_wire, device_wires=dev.wires)

    @pytest.mark.parametrize("device_wires", device_wires_no_aux)
    def test_requested_wire_not_exist(self, device_wires):
        """Test if the aux wire is not on the device an error is raised."""
        aux_wire = qml.wires.Wires("aux")
        dev = qml.device("default.qubit", wires=device_wires)
        x = 0.543
        y = -0.654

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        tape = qml.tape.QuantumScript.from_queue(q)
        _match = "The requested auxiliary wire does not exist on the used device"
        with pytest.raises(qml.wires.WireError, match=_match):
            qml.gradients.hadamard_grad(tape, aux_wire=aux_wire, device_wires=dev.wires)

    @pytest.mark.parametrize("aux_wire", [None] + already_used_wires)
    def test_device_not_enough_wires(self, aux_wire):
        """Test that an error is raised when the device cannot accept an auxiliary wire
        because it is full."""
        dev = qml.device("default.qubit", wires=2)

        m = qml.expval(qml.PauliZ(0) @ qml.PauliX(1))
        tape = qml.tape.QuantumScript([qml.RX(0.543, wires=[0]), qml.RY(-0.654, wires=[1])], [m])

        if aux_wire is None:
            _match = "The device has no free wire for the auxiliary wire."
        else:
            _match = "The requested auxiliary wire is already in use by the circuit."
        with pytest.raises(qml.wires.WireError, match=_match):
            qml.gradients.hadamard_grad(tape, aux_wire=aux_wire, device_wires=dev.wires)

    def test_device_wire_does_not_exist(self):
        """Test that an error is raised when the device cannot accept an auxiliary wire
        because it does not exist on the device."""
        aux_wire = qml.wires.Wires("aux")
        dev = qml.device("default.qubit", wires=2)
        m = qml.expval(qml.PauliZ(0) @ qml.PauliX(1))
        tape = qml.tape.QuantumScript([qml.RX(0.543, wires=[0]), qml.RY(-0.654, wires=[1])], [m])

        _match = "The requested auxiliary wire does not exist on the used device."
        with pytest.raises(qml.wires.WireError, match=_match):
            qml.gradients.hadamard_grad(tape, aux_wire=aux_wire, device_wires=dev.wires)

    def test_empty_circuit(self):
        """Test that an empty circuit works correctly"""
        with qml.queuing.AnnotatedQueue() as q:
            qml.expval(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        with pytest.warns(UserWarning, match="gradient of a tape with no trainable parameters"):
            tapes, _ = qml.gradients.hadamard_grad(tape)
        assert not tapes

    def test_all_parameters_independent(self):
        """Test that a circuit where all parameters do not affect the output"""
        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(0.4, wires=0)
            qml.expval(qml.PauliZ(1))

        tape = qml.tape.QuantumScript.from_queue(q)
        tapes, _ = qml.gradients.hadamard_grad(tape)
        assert not tapes

    def test_state_non_differentiable_error(self):
        """Test error raised if attempting to differentiate with
        respect to a state"""

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[1])
            qml.state()

        tape = qml.tape.QuantumScript.from_queue(q)
        _match = r"return the state with the Hadamard test gradient transform"
        with pytest.raises(ValueError, match=_match):
            qml.gradients.hadamard_grad(tape)

    def test_variance_non_differentiable_error(self):
        """Test error raised if attempting to differentiate with respect to a variance"""

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[1])
            qml.var(qml.PauliZ(wires=0))

        tape = qml.tape.QuantumScript.from_queue(q)
        with pytest.raises(
            ValueError,
            match=(
                r"Computing the gradient of variances with the Hadamard test "
                "gradient transform is not supported."
            ),
        ):
            qml.gradients.hadamard_grad(tape)

    def test_independent_parameter(self, mocker):
        """Test that an independent parameter is skipped
        during the Jacobian computation."""
        spy = mocker.spy(qml.gradients.hadamard_gradient, "_expval_hadamard_grad")

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[1])  # does not have any impact on the expval
            qml.expval(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        dev = qml.device("default.qubit", wires=3)

        res_hadamard, tapes = grad_fn(tape, dev)

        assert len(tapes) == 1

        assert isinstance(res_hadamard, tuple)
        assert len(res_hadamard) == 2
        assert res_hadamard[0].shape == ()
        assert res_hadamard[1].shape == ()

        # only called for parameter 0
        assert spy.call_args[0][0:2] == (tape, [0])

    @pytest.mark.autograd
    def test_no_trainable_params_qnode_autograd(self, mocker):
        """Test that the correct ouput and warning is generated in the absence of any trainable
        parameters"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="autograd")
        def circuit(weights):
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        weights = [0.1, 0.2]
        with pytest.raises(qml.QuantumFunctionError, match="No trainable parameters."):
            qml.gradients.hadamard_grad(circuit)(weights)

    @pytest.mark.torch
    def test_no_trainable_params_qnode_torch(self, mocker):
        """Test that the correct ouput and warning is generated in the absence of any trainable
        parameters"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="torch")
        def circuit(weights):
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        weights = [0.1, 0.2]
        with pytest.raises(qml.QuantumFunctionError, match="No trainable parameters."):
            qml.gradients.hadamard_grad(circuit)(weights)

    @pytest.mark.tf
    def test_no_trainable_params_qnode_tf(self, mocker):
        """Test that the correct ouput and warning is generated in the absence of any trainable
        parameters"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="tf")
        def circuit(weights):
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        weights = [0.1, 0.2]
        with pytest.raises(qml.QuantumFunctionError, match="No trainable parameters."):
            qml.gradients.hadamard_grad(circuit)(weights)

    @pytest.mark.jax
    def test_no_trainable_params_qnode_jax(self, mocker):
        """Test that the correct ouput and warning is generated in the absence of any trainable
        parameters"""
        dev = qml.device("default.qubit", wires=2)
        spy = mocker.spy(qml.devices.qubit, "measure")

        @qml.qnode(dev, interface="jax")
        def circuit(weights):
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        weights = [0.1, 0.2]
        with pytest.raises(qml.QuantumFunctionError, match="No trainable parameters."):
            qml.gradients.hadamard_grad(circuit)(weights)

    @pytest.mark.autograd
    def test_no_trainable_params_qnode_autograd_legacy(self, mocker):
        """Test that the correct ouput and warning is generated in the absence of any trainable
        parameters"""
        dev = qml.device("default.qubit.autograd", wires=2)

        @qml.qnode(dev, interface="autograd")
        def circuit(weights):
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        weights = [0.1, 0.2]
        with pytest.raises(qml.QuantumFunctionError, match="No trainable parameters."):
            qml.gradients.hadamard_grad(circuit)(weights)

    @pytest.mark.torch
    def test_no_trainable_params_qnode_torch_legacy(self, mocker):
        """Test that the correct ouput and warning is generated in the absence of any trainable
        parameters"""
        dev = qml.device("default.qubit.torch", wires=2)

        @qml.qnode(dev, interface="torch")
        def circuit(weights):
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        weights = [0.1, 0.2]
        with pytest.raises(qml.QuantumFunctionError, match="No trainable parameters."):
            qml.gradients.hadamard_grad(circuit)(weights)

    @pytest.mark.tf
    def test_no_trainable_params_qnode_tf_legacy(self, mocker):
        """Test that the correct ouput and warning is generated in the absence of any trainable
        parameters"""
        dev = qml.device("default.qubit.tf", wires=2)
        spy = mocker.spy(dev, "expval")

        @qml.qnode(dev, interface="tf")
        def circuit(weights):
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        weights = [0.1, 0.2]
        with pytest.raises(qml.QuantumFunctionError, match="No trainable parameters."):
            qml.gradients.hadamard_grad(circuit)(weights)

    @pytest.mark.jax
    def test_no_trainable_params_qnode_jax_legacy(self, mocker):
        """Test that the correct ouput and warning is generated in the absence of any trainable
        parameters"""
        dev = qml.device("default.qubit.jax", wires=2)

        @qml.qnode(dev, interface="jax")
        def circuit(weights):
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        weights = [0.1, 0.2]
        with pytest.raises(qml.QuantumFunctionError, match="No trainable parameters."):
            res_hadamard = qml.gradients.hadamard_grad(circuit)(weights)

    def test_no_trainable_params_tape(self):
        """Test that the correct ouput and warning is generated in the absence of any trainable
        parameters"""
        dev = qml.device("default.qubit", wires=2)

        weights = [0.1, 0.2]
        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        tape = qml.tape.QuantumScript.from_queue(q)
        # TODO: remove once #2155 is resolved
        tape.trainable_params = []

        with pytest.warns(UserWarning, match="gradient of a tape with no trainable parameters"):
            g_tapes, post_processing = qml.gradients.hadamard_grad(tape)
        res_hadamard = post_processing(qml.execute(g_tapes, dev, None))

        assert g_tapes == []
        assert isinstance(res_hadamard, np.ndarray)
        assert res_hadamard.shape == (0,)

    def test_no_trainable_params_multiple_return_tape(self):
        """Test that the correct ouput and warning is generated in the absence of any trainable
        parameters with multiple returns."""
        dev = qml.device("default.qubit", wires=2)

        weights = [0.1, 0.2]
        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
            qml.probs(wires=[0, 1])

        tape = qml.tape.QuantumScript.from_queue(q)
        tape.trainable_params = []

        with pytest.warns(UserWarning, match="gradient of a tape with no trainable parameters"):
            res_hadamard, tapes = grad_fn(tape, dev)

        assert tapes == []
        assert isinstance(res_hadamard, tuple)
        for r in res_hadamard:
            assert isinstance(r, np.ndarray)
            assert r.shape == (0,)

    def test_all_zero_diff_methods_tape(self):
        """Test that the transform works correctly when the diff method for every parameter is
        identified to be 0, and that no tapes were generated."""
        dev = qml.device("default.qubit", wires=3)

        params = np.array([0.5, 0.5, 0.5], requires_grad=True)

        with qml.queuing.AnnotatedQueue() as q:
            qml.Rot(*params, wires=0)
            qml.probs([2, 3])

        tape = qml.tape.QuantumScript.from_queue(q)

        res_hadamard, tapes = grad_fn(tape, dev)

        assert tapes == []

        assert isinstance(res_hadamard, tuple)

        assert len(res_hadamard) == 3

        assert isinstance(res_hadamard[0], np.ndarray)
        assert res_hadamard[0].shape == (4,)
        assert np.allclose(res_hadamard[0], 0)

        assert isinstance(res_hadamard[1], np.ndarray)
        assert res_hadamard[1].shape == (4,)
        assert np.allclose(res_hadamard[1], 0)

        assert isinstance(res_hadamard[2], np.ndarray)
        assert res_hadamard[2].shape == (4,)
        assert np.allclose(res_hadamard[2], 0)

    def test_all_zero_diff_methods_multiple_returns_tape(self):
        """Test that the transform works correctly when the diff method for every parameter is
        identified to be 0, and that no tapes were generated."""

        dev = qml.device("default.qubit", wires=3)

        params = np.array([0.5, 0.5, 0.5], requires_grad=True)

        with qml.queuing.AnnotatedQueue() as q:
            qml.Rot(*params, wires=0)
            qml.expval(qml.PauliZ(wires=2))
            qml.probs([2, 3])

        tape = qml.tape.QuantumScript.from_queue(q)

        res_hadamard, tapes = grad_fn(tape, dev)

        assert tapes == []

        assert isinstance(res_hadamard, tuple)

        assert len(res_hadamard) == 2

        # First elem
        assert len(res_hadamard[0]) == 3

        assert isinstance(res_hadamard[0][0], np.ndarray)
        assert res_hadamard[0][0].shape == ()
        assert np.allclose(res_hadamard[0][0], 0)

        assert isinstance(res_hadamard[0][1], np.ndarray)
        assert res_hadamard[0][1].shape == ()
        assert np.allclose(res_hadamard[0][1], 0)

        assert isinstance(res_hadamard[0][2], np.ndarray)
        assert res_hadamard[0][2].shape == ()
        assert np.allclose(res_hadamard[0][2], 0)

        # Second elem
        assert len(res_hadamard[0]) == 3

        assert isinstance(res_hadamard[1][0], np.ndarray)
        assert res_hadamard[1][0].shape == (4,)
        assert np.allclose(res_hadamard[1][0], 0)

        assert isinstance(res_hadamard[1][1], np.ndarray)
        assert res_hadamard[1][1].shape == (4,)
        assert np.allclose(res_hadamard[1][1], 0)

        assert isinstance(res_hadamard[1][2], np.ndarray)
        assert res_hadamard[1][2].shape == (4,)
        assert np.allclose(res_hadamard[1][2], 0)

    def test_all_zero_diff_methods(self):
        """Test that the transform works correctly when the diff method for every parameter is
        identified to be 0, and that no tapes were generated."""
        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def circuit(params):
            qml.Rot(*params, wires=0)
            return qml.probs([2, 3])

        params = np.array([0.5, 0.5, 0.5], requires_grad=True)

        result = qml.gradients.hadamard_grad(circuit)(params)

        assert isinstance(result, tuple)

        assert len(result) == 3

        assert isinstance(result[0], np.ndarray)
        assert result[0].shape == (4,)
        assert np.allclose(result[0], 0)

        assert isinstance(result[1], np.ndarray)
        assert result[1].shape == (4,)
        assert np.allclose(result[1], 0)

        assert isinstance(result[2], np.ndarray)
        assert result[2].shape == (4,)
        assert np.allclose(result[2], 0)


class TestHadamardTestGradDiff:
    """Test that the transform is differentiable"""

    @pytest.mark.autograd
    @pytest.mark.parametrize("dev_name", ["default.qubit", "default.qubit.autograd"])
    def test_autograd(self, dev_name):
        """Tests that the output of the hadamard gradient transform
        can be differentiated using autograd, yielding second derivatives."""
        dev = qml.device(dev_name, wires=3)
        execute_fn = dev.execute if dev_name == "default.qubit" else dev.batch_execute
        params = np.array([0.543, -0.654], requires_grad=True)

        def cost_fn_hadamard(x):
            with qml.queuing.AnnotatedQueue() as q:
                qml.RX(x[0], wires=[0])
                qml.RY(x[1], wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

            tape = qml.tape.QuantumScript.from_queue(q)
            tape.trainable_params = {0, 1}
            tapes, fn = qml.gradients.hadamard_grad(tape)
            jac = fn(execute_fn(tapes))
            return qml.math.stack(jac)

        def cost_fn_param_shift(x):
            with qml.queuing.AnnotatedQueue() as q:
                qml.RX(x[0], wires=[0])
                qml.RY(x[1], wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

            tape = qml.tape.QuantumScript.from_queue(q)
            tape.trainable_params = {0, 1}
            tapes, fn = qml.gradients.param_shift(tape)
            jac = fn(execute_fn(tapes))
            return qml.math.stack(jac)

        res_hadamard = qml.jacobian(cost_fn_hadamard)(params)
        res_param_shift = qml.jacobian(cost_fn_param_shift)(params)
        assert np.allclose(res_hadamard, res_param_shift)

    @pytest.mark.tf
    @pytest.mark.parametrize("dev_name", ["default.qubit", "default.qubit.tf"])
    def test_tf(self, dev_name):
        """Tests that the output of the hadamard gradient transform
        can be differentiated using TF, yielding second derivatives."""
        import tensorflow as tf

        dev = qml.device(dev_name, wires=3)
        execute_fn = dev.execute if dev_name == "default.qubit" else dev.batch_execute
        params = tf.Variable([0.543, -0.654], dtype=tf.float64)

        with tf.GradientTape() as t_h:
            with qml.queuing.AnnotatedQueue() as q:
                qml.RX(params[0], wires=[0])
                qml.RY(params[1], wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

            tape = qml.tape.QuantumScript.from_queue(q)
            tape.trainable_params = {0, 1}
            tapes, fn = qml.gradients.hadamard_grad(tape)
            jac_h = fn(execute_fn(tapes))
            jac_h = qml.math.stack(jac_h)

        with tf.GradientTape() as t_p:
            with qml.queuing.AnnotatedQueue() as q:
                qml.RX(params[0], wires=[0])
                qml.RY(params[1], wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

            tape = qml.tape.QuantumScript.from_queue(q)
            tape.trainable_params = {0, 1}
            tapes, fn = qml.gradients.param_shift(tape)
            jac_p = fn(execute_fn(tapes))
            jac_p = qml.math.stack(jac_p)

        res_hadamard = t_h.jacobian(jac_h, params)
        res_param_shift = t_p.jacobian(jac_p, params)

        assert np.allclose(res_hadamard, res_param_shift)

    @pytest.mark.torch
    @pytest.mark.parametrize("dev_name", ["default.qubit", "default.qubit.torch"])
    def test_torch(self, dev_name):
        """Tests that the output of the hadamard gradient transform
        can be differentiated using Torch, yielding second derivatives."""
        import torch

        dev = qml.device(dev_name, wires=3)
        execute_fn = dev.execute if dev_name == "default.qubit" else dev.batch_execute
        params = torch.tensor([0.543, -0.654], dtype=torch.float64, requires_grad=True)

        def cost_h(x):
            with qml.queuing.AnnotatedQueue() as q:
                qml.RX(x[0], wires=[0])
                qml.RY(x[1], wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

            tape = qml.tape.QuantumScript.from_queue(q)
            tapes, fn = qml.gradients.hadamard_grad(tape)

            jac = fn(execute_fn(tapes))
            return jac

        def cost_p(x):
            with qml.queuing.AnnotatedQueue() as q:
                qml.RX(x[0], wires=[0])
                qml.RY(x[1], wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

            tape = qml.tape.QuantumScript.from_queue(q)
            tapes, fn = qml.gradients.param_shift(tape)

            jac = fn(execute_fn(tapes))
            return jac

        res_hadamard = torch.autograd.functional.jacobian(cost_h, params)
        res_param_shift = torch.autograd.functional.jacobian(cost_p, params)

        assert np.allclose(res_hadamard[0].detach(), res_param_shift[0].detach())
        assert np.allclose(res_hadamard[1].detach(), res_param_shift[1].detach())

    @pytest.mark.jax
    @pytest.mark.parametrize("dev_name", ["default.qubit", "default.qubit.jax"])
    def test_jax(self, dev_name):
        """Tests that the output of the hadamard gradient transform
        can be differentiated using JAX, yielding second derivatives."""
        import jax
        from jax import numpy as jnp
        from jax.config import config

        config.update("jax_enable_x64", True)

        dev = qml.device(dev_name, wires=3)
        execute_fn = dev.execute if dev_name == "default.qubit" else dev.batch_execute
        params = jnp.array([0.543, -0.654])

        def cost_h(x):
            with qml.queuing.AnnotatedQueue() as q:
                qml.RX(x[0], wires=[0])
                qml.RY(x[1], wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

            tape = qml.tape.QuantumScript.from_queue(q)
            tape.trainable_params = {0, 1}
            tapes, fn = qml.gradients.hadamard_grad(tape)

            jac = fn(execute_fn(tapes))
            return jac

        def cost_p(x):
            with qml.queuing.AnnotatedQueue() as q:
                qml.RX(x[0], wires=[0])
                qml.RY(x[1], wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

            tape = qml.tape.QuantumScript.from_queue(q)
            tape.trainable_params = {0, 1}
            tapes, fn = qml.gradients.hadamard_grad(tape)

            jac = fn(execute_fn(tapes))
            return jac

        res_hadamard = jax.jacobian(cost_h)(params)
        res_param_shift = jax.jacobian(cost_p)(params)
        assert np.allclose(res_hadamard, res_param_shift)


@pytest.mark.parametrize("argnums", [[0], [1], [0, 1]])
@pytest.mark.parametrize("interface", ["jax"])
@pytest.mark.jax
class TestJaxArgnums:
    """Class to test the integration of argnums (Jax) and the hadamard grad transform."""

    expected_jacs = []
    interfaces = ["auto", "jax"]

    def test_argnum_error(self, argnums, interface):
        """Test that giving argnum to Jax, raises an error."""
        import jax

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev, interface=interface)
        def circuit(x, y):
            qml.RX(x[0], wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        x = jax.numpy.array([0.543, 0.2])
        y = jax.numpy.array(-0.654)

        with pytest.raises(
            qml.QuantumFunctionError,
            match="argnum does not work with the Jax interface. You should use argnums instead.",
        ):
            qml.gradients.hadamard_grad(circuit, argnum=argnums)(x, y)

    def test_single_expectation_value(self, argnums, interface):
        """Test for single expectation value."""
        import jax

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev, interface=interface)
        def circuit(x, y):
            qml.RX(x[0], wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        x = jax.numpy.array([0.543, 0.2])
        y = jax.numpy.array(-0.654)

        res = qml.gradients.hadamard_grad(circuit, argnums=argnums)(x, y)

        expected_0 = np.array([-np.sin(y) * np.sin(x[0]), 0])
        expected_1 = np.array(np.cos(y) * np.cos(x[0]))

        if argnums == [0]:
            assert np.allclose(res, expected_0)
        if argnums == [1]:
            assert np.allclose(res, expected_1)
        if argnums == [0, 1]:
            assert np.allclose(res[0], expected_0)
            assert np.allclose(res[1], expected_1)

    def test_multi_expectation_values(self, argnums, interface):
        """Test for multiple expectation values."""
        import jax

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev, interface=interface)
        def circuit(x, y):
            qml.RX(x[0], wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(1))

        x = jax.numpy.array([0.543, 0.2])
        y = jax.numpy.array(-0.654)

        res = qml.gradients.hadamard_grad(circuit, argnums=argnums)(x, y)

        expected_0 = np.array([[-np.sin(x[0]), 0.0], [0.0, 0.0]])
        expected_1 = np.array([0, np.cos(y)])

        if argnums == [0]:
            assert np.allclose(res[0], expected_0[0])
            assert np.allclose(res[1], expected_0[1])
        if argnums == [1]:
            assert np.allclose(res, expected_1)
        if argnums == [0, 1]:
            assert np.allclose(res[0][0], expected_0[0])
            assert np.allclose(res[0][1], expected_0[1])
            assert np.allclose(res[1][0], expected_1[0])
            assert np.allclose(res[1][1], expected_1[1])

    def test_hessian(self, argnums, interface):
        """Test for hessian."""
        import jax

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev, interface=interface)
        def circuit(x, y):
            qml.RX(x[0], wires=[0])
            qml.RY(x[1], wires=[1])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        x = jax.numpy.array([0.543, -0.654])
        y = jax.numpy.array(-0.123)

        res = jax.jacobian(qml.gradients.hadamard_grad(circuit, argnums=argnums), argnums=argnums)(
            x, y
        )
        res_expected = jax.hessian(circuit, argnums=argnums)(x, y)

        if argnums == [0]:
            assert np.allclose(res[0][0], res_expected[0][0][0])
            assert np.allclose(res[1][0], res_expected[0][0][1])
        else:
            if len(argnums) != 1:
                res = res[0]

            for r, r_e in zip(res, res_expected[0]):
                assert np.allclose(r, r_e)

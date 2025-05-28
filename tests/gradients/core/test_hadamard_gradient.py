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
# pylint: disable=use-implicit-booleaness-not-comparison
import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.exceptions import QuantumFunctionError


def grad_fn(tape, dev, fn=qml.gradients.hadamard_grad, **kwargs):
    """Utility function to automate execution and processing of gradient tapes"""
    tapes, fn = fn(tape, **kwargs)
    return fn(dev.execute(tapes)), tapes


def cost1(x):
    """Cost function."""
    qml.Rot(x[0], 0.3 * x[1], x[2], wires=0)
    return qml.expval(qml.PauliZ(0))


def cost2(x):
    """Cost function."""
    qml.Rot(*x, wires=0)
    return [qml.expval(qml.PauliZ(0))]


def cost3(x):
    """Cost function."""
    qml.Rot(*x, wires=0)
    return (qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)))


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
    return (qml.probs([0, 1]), qml.probs([2, 3]))


def cost7(x):
    """Cost function."""
    qml.RX(x, 0)
    return qml.expval(qml.PauliZ(0))


def cost8(x):
    """Cost function."""
    qml.RX(x, 0)
    return [qml.expval(qml.PauliZ(0))]


def cost9(x):
    """Cost function."""
    qml.RX(x, 0)
    return (qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)))


def cost10(x):
    """Cost function."""
    qml.Rot(*x, wires=0)
    qml.Hadamard(1)
    qml.X(2)
    return qml.probs()


def cost11(x):
    """Cost function."""
    qml.Rot(*x, wires=0)
    qml.Hadamard(1)
    qml.X(2)
    return qml.probs(), qml.probs([0, 2, 1, 4, 3])


def cost12(x):
    """Cost function."""
    qml.Rot(*x, wires=0)
    qml.Hadamard(1)
    qml.X(2)
    return qml.probs(op=qml.Hadamard(0) @ qml.Y(1) @ qml.Y(2))


class TestHadamardValidation:
    """Test validation of edge cases with the hadamard gradient."""

    @pytest.mark.parametrize("mode", ["invalid-mode"])
    def test_invalid_mode(self, mode):
        """Test that a ValueError is raised if an invalid mode is provided."""
        tape = qml.tape.QuantumScript([qml.RX(0.543, 0), qml.RY(-0.654, 0)], [qml.expval(qml.Z(0))])
        _match = r"Invalid mode"
        with pytest.raises(ValueError, match=_match):
            qml.gradients.hadamard_grad(tape, mode=mode)

    @pytest.mark.parametrize("mode", ["standard", "reversed", "direct", "reversed-direct"])
    def test_trainable_batched_tape_raises(self, mode):
        """Test that an error is raised for a broadcasted/batched tape if the broadcasted
        parameter is differentiated."""
        tape = qml.tape.QuantumScript([qml.RX([0.4, 0.2], 0)], [qml.expval(qml.PauliZ(0))])
        _match = r"Computing the gradient of broadcasted tapes"
        with pytest.raises(NotImplementedError, match=_match):
            qml.gradients.hadamard_grad(tape, mode=mode)

    @pytest.mark.parametrize("mode", ["standard", "reversed", "direct", "reversed-direct"])
    def test_error_trainable_obs_probs(self, mode):
        """Test that there's an error if the observable is trainable with a probs."""

        H = qml.numpy.array(2.0) * (qml.X(0) + qml.Y(0))
        tape = qml.tape.QuantumScript([], [qml.probs(op=H)])
        with pytest.raises(ValueError, match="Can only differentiate Hamiltonian coefficients"):
            qml.gradients.hadamard_grad(tape, mode=mode)

    @pytest.mark.parametrize("mode", ["standard", "reversed", "direct", "reversed-direct"])
    def test_tape_with_partitioned_shots_multiple_measurements_raises(self, mode):
        """Test that an error is raised with multiple measurements and partitioned shots."""
        tape = qml.tape.QuantumScript(
            [qml.RX(0.1, wires=0)],
            [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(0))],
            shots=(1000, 10000),
        )
        with pytest.raises(NotImplementedError):
            qml.gradients.hadamard_grad(tape, mode=mode)

    @pytest.mark.parametrize("mode", ["standard", "reversed"])
    @pytest.mark.parametrize("aux_wire", [qml.wires.Wires(0), qml.wires.Wires(1)])
    @pytest.mark.parametrize("device_wires", [qml.wires.Wires([0, 1, "aux"])])
    def test_aux_wire_already_used_wires(self, aux_wire, device_wires, mode):
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
            qml.gradients.hadamard_grad(tape, aux_wire=aux_wire, device_wires=dev.wires, mode=mode)

    @pytest.mark.parametrize("mode", ["standard", "reversed"])
    @pytest.mark.parametrize("device_wires", [qml.wires.Wires([0, 1, 2])])
    def test_requested_wire_not_exist(self, device_wires, mode):
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
            qml.gradients.hadamard_grad(tape, aux_wire=aux_wire, device_wires=dev.wires, mode=mode)

    @pytest.mark.parametrize("mode", ["standard", "reversed"])
    @pytest.mark.parametrize("aux_wire", [None, qml.wires.Wires(0), qml.wires.Wires(1)])
    def test_device_not_enough_wires(self, aux_wire, mode):
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
            qml.gradients.hadamard_grad(tape, aux_wire=aux_wire, device_wires=dev.wires, mode=mode)

    @pytest.mark.parametrize("mode", ["standard", "reversed"])
    def test_device_wire_does_not_exist(self, mode):
        """Test that an error is raised when the device cannot accept an auxiliary wire
        because it does not exist on the device."""
        aux_wire = qml.wires.Wires("aux")
        dev = qml.device("default.qubit", wires=2)
        m = qml.expval(qml.PauliZ(0) @ qml.PauliX(1))
        tape = qml.tape.QuantumScript([qml.RX(0.543, wires=[0]), qml.RY(-0.654, wires=[1])], [m])

        _match = "The requested auxiliary wire does not exist on the used device."
        with pytest.raises(qml.wires.WireError, match=_match):
            qml.gradients.hadamard_grad(tape, aux_wire=aux_wire, device_wires=dev.wires, mode=mode)

    @pytest.mark.parametrize("measurement", [qml.state(), qml.var(qml.Z(0))])
    @pytest.mark.parametrize("mode", ["standard", "reversed", "direct", "reversed-direct"])
    def test_state_non_differentiable_error(self, mode, measurement):
        """Test error raised if attempting to differentiate with
        respect to a state"""

        tape = qml.tape.QuantumScript([qml.RX(0.543, 0), qml.RY(-0.654, 0)], [measurement])
        _match = r"is not supported"
        with pytest.raises(ValueError, match=_match):
            qml.gradients.hadamard_grad(tape, mode=mode)

    @pytest.mark.parametrize("measurement", [qml.probs(wires=[0])])
    @pytest.mark.parametrize("mode", ["reversed", "direct", "reversed-direct"])
    def test_no_probs_with_new_modes(self, mode, measurement):
        """Test that a ValueError is raised if qml.probs is used with reversed, direct, or reversed direct"""

        tape = qml.tape.QuantumScript([qml.RX(0.543, 0), qml.RY(-0.654, 0)], [measurement])
        _match = r"is not supported"
        with pytest.raises(ValueError, match=_match):
            qml.gradients.hadamard_grad(tape, mode=mode)

    @pytest.mark.parametrize("mode", ("reversed", "reversed-direct"))
    def test_at_most_one_measurement_with_reversed(self, mode):
        """Test that a ValueError is raised if more than one expectation value is used
        with reversed and reversed-direct methods."""

        tape = qml.tape.QuantumScript(
            [qml.RX(0.543, 0), qml.RY(-0.654, 0)], [qml.expval(qml.Z(0)), qml.expval(qml.X(0))]
        )
        with pytest.raises(NotImplementedError):
            qml.gradients.hadamard_grad(tape, mode=mode)


class TestDifferentModes:
    """Unit test the different modes of gradient."""

    def test_reversed_mode(self):
        """Test that reversed mode hadamard gradient constructs the correct batch of tapes."""

        tape = qml.tape.QuantumScript(
            [qml.IsingXY(0.5, wires=(0, 1))], [qml.expval(qml.X(0) + qml.Y(0))], shots=50
        )
        batch, fn = qml.gradients.hadamard_grad(tape, mode="reversed")
        assert len(batch) == 2

        expected_H = qml.IsingXY(0.5, wires=(0, 1)).generator() @ qml.Y(2)
        expected0 = qml.tape.QuantumScript(
            [qml.IsingXY(0.5, wires=(0, 1)), qml.H(2), qml.CNOT((2, 0)), qml.H(2)],
            [qml.expval(expected_H)],
            shots=50,
        )
        qml.assert_equal(batch[0], expected0)
        expected1 = qml.tape.QuantumScript(
            [qml.IsingXY(0.5, wires=(0, 1)), qml.H(2), qml.CY((2, 0)), qml.H(2)],
            [qml.expval(expected_H)],
            shots=50,
        )
        qml.assert_equal(batch[1], expected1)

        assert qml.math.allclose(fn((1.0, 2.0)), -6.0)

    def test_direct_mode(self):
        """Directly test the batch output of applying the direct mode hadamard gradient."""
        tape = qml.tape.QuantumScript(
            [qml.IsingXY(0.5, wires=(0, 1))], [qml.expval(qml.X(0) + qml.Y(0))], shots=50
        )
        batch, fn = qml.gradients.hadamard_grad(tape, mode="direct")
        assert len(batch) == 4

        measurement = tape.measurements

        expected0 = qml.tape.QuantumScript(
            [qml.IsingXY(0.5, wires=(0, 1)), qml.evolve(qml.X(0) @ qml.X(1), np.pi / 4)],
            measurement,
            shots=50,
        )
        qml.assert_equal(batch[0], expected0)
        expected1 = qml.tape.QuantumScript(
            [qml.IsingXY(0.5, wires=(0, 1)), qml.evolve(qml.X(0) @ qml.X(1), -np.pi / 4)],
            measurement,
            shots=50,
        )
        qml.assert_equal(batch[1], expected1)
        expected2 = qml.tape.QuantumScript(
            [qml.IsingXY(0.5, wires=(0, 1)), qml.evolve(qml.Y(0) @ qml.Y(1), np.pi / 4)],
            measurement,
            shots=50,
        )
        qml.assert_equal(batch[2], expected2)
        expected3 = qml.tape.QuantumScript(
            [qml.IsingXY(0.5, wires=(0, 1)), qml.evolve(qml.Y(0) @ qml.Y(1), -np.pi / 4)],
            measurement,
            shots=50,
        )
        qml.assert_equal(batch[3], expected3)

        assert qml.math.allclose(fn((1.0, 2.0, 3.0, 4.0)), 0.5)

    def test_reversed_direct_mode(self):
        """Directly test tht batch output of applying the reversed direct mode of hadamard gradient."""
        tape = qml.tape.QuantumScript(
            [qml.IsingXY(0.5, wires=(0, 1))], [qml.expval(qml.X(0) + qml.Y(0))], shots=50
        )
        batch, fn = qml.gradients.hadamard_grad(tape, mode="reversed-direct")
        assert len(batch) == 4

        expected_H = qml.IsingXY(0.5, wires=(0, 1)).generator()
        expected0 = qml.tape.QuantumScript(
            [qml.IsingXY(0.5, wires=(0, 1)), qml.evolve(qml.X(0), np.pi / 4)],
            [qml.expval(expected_H)],
            shots=50,
        )
        qml.assert_equal(batch[0], expected0)

        expected1 = qml.tape.QuantumScript(
            [qml.IsingXY(0.5, wires=(0, 1)), qml.evolve(qml.X(0), -np.pi / 4)],
            [qml.expval(expected_H)],
            shots=50,
        )
        qml.assert_equal(batch[1], expected1)

        expected2 = qml.tape.QuantumScript(
            [qml.IsingXY(0.5, wires=(0, 1)), qml.evolve(qml.Y(0), np.pi / 4)],
            [qml.expval(expected_H)],
            shots=50,
        )
        qml.assert_equal(batch[2], expected2)

        expected3 = qml.tape.QuantumScript(
            [qml.IsingXY(0.5, wires=(0, 1)), qml.evolve(qml.Y(0), -np.pi / 4)],
            [qml.expval(expected_H)],
            shots=50,
        )
        qml.assert_equal(batch[3], expected3)

        assert qml.math.allclose(fn((1.0, 2.0, 3.0, 4.0)), -2.0)

    def test_reversed_mode_hadamard_obs(self):
        """Test that reversed mode can handle measuring an observable that does not have a pauli rep."""

        tape = qml.tape.QuantumScript([qml.RX(0.5, 0)], [qml.expval(qml.H(0))])
        batch, fn = qml.gradients.hadamard_grad(tape, mode="reversed")

        assert len(batch) == 2

        obs_terms = [qml.X(0), qml.Z(0)]
        H = qml.RX(0.5, 0).generator() @ qml.Y(1)
        for i, ob in enumerate(obs_terms):
            expected = qml.tape.QuantumScript(
                [qml.RX(0.5, 0), qml.H(1), qml.ctrl(ob, 1), qml.H(1)], [qml.expval(H)]
            )
            qml.assert_equal(expected.measurements[0].obs, batch[i].measurements[0].obs)
            qml.assert_equal(batch[i], expected)

        out = fn((1.0, 2.0))
        expected = -2 / np.sqrt(2) * (1.0 + 2.0)
        assert qml.math.allclose(out, expected)

    def test_reversed_direct_mode_hadamard_obs(self):
        """Test that reversed-direct mode can measure an observable that does not have a pauli rep."""
        # similar to the above test, we just want to directly check the four outputted tapes for
        # using reversed direct when taking the expectation value of a hadamard
        tape = qml.tape.QuantumScript([qml.RX(0.5, 0)], [qml.expval(qml.H(0))])
        batch, fn = qml.gradients.hadamard_grad(tape, mode="reversed-direct")

        assert len(batch) == 4

        obs_terms = [qml.X(0), qml.Z(0)]
        H = qml.RX(0.5, 0).generator()
        for i, ob in enumerate(obs_terms):
            expected0 = qml.tape.QuantumScript(
                [qml.RX(0.5, 0), qml.evolve(ob, np.pi / 4)], [qml.expval(H)]
            )
            qml.assert_equal(expected0.measurements[0].obs, batch[2 * i].measurements[0].obs)
            qml.assert_equal(batch[2 * i], expected0)
            expected1 = qml.tape.QuantumScript(
                [qml.RX(0.5, 0), qml.evolve(ob, -np.pi / 4)], [qml.expval(H)]
            )
            qml.assert_equal(expected1.measurements[0].obs, batch[2 * i + 1].measurements[0].obs)
            qml.assert_equal(batch[2 * i + 1], expected1)

            out = fn((1.0, 2.0, 3.0, 4.0))
            expected = 1 / np.sqrt(2) * (1.0 - 2.0) + 1 / np.sqrt(2) * (3.0 - 4.0)
            assert qml.math.allclose(out, expected)

    @pytest.mark.parametrize("mode", ["direct", "reversed-direct"])
    def test_no_available_work_wire_direct_methods(self, mode):
        """Test that direct and reversed direct work with no available work wires."""
        tape = qml.tape.QuantumScript([qml.RX(0.5, 0)], [qml.expval(qml.Z(0))])
        _ = qml.gradients.hadamard_grad(tape, mode=mode, device_wires=qml.wires.Wires(0))


# pylint: disable=too-many-public-methods
class TestHadamardGrad:
    """Unit tests for the hadamard_grad function"""

    @pytest.mark.parametrize("mode", ["standard", "reversed", "direct", "reversed-direct"])
    def test_nontrainable_batched_tape(self, mode):
        """Test that no error is raised for a broadcasted/batched tape if the broadcasted
        parameter is not differentiated, and that the results correspond to the stacked
        results of the single-tape derivatives."""
        dev = qml.device("default.qubit")
        x = [0.4, 0.2]
        tape = qml.tape.QuantumScript(
            [qml.RY(0.6, 0), qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))], trainable_params=[0]
        )
        batched_tapes, batched_fn = qml.gradients.hadamard_grad(tape, mode=mode)
        batched_grad = batched_fn(dev.execute(batched_tapes))
        separate_tapes = [
            qml.tape.QuantumScript(
                [qml.RY(0.6, 0), qml.RX(_x, 0)], [qml.expval(qml.PauliZ(0))], trainable_params=[0]
            )
            for _x in x
        ]
        separate_tapes_and_fns = [qml.gradients.hadamard_grad(t, mode=mode) for t in separate_tapes]
        separate_grad = [_fn(dev.execute(_tapes)) for _tapes, _fn in separate_tapes_and_fns]
        assert np.allclose(batched_grad, separate_grad)

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

        assert res_hadamard.shape == ()

        res_param_shift, _ = grad_fn(tape, dev, qml.gradients.param_shift)

        assert np.allclose(res_hadamard, res_param_shift, atol=tol, rtol=0)

    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, 2 * np.pi, 7))
    def test_rot_gradient(self, theta, tol):
        """Tests that the automatic gradient of an arbitrary Euler-angle-parametrized gate is correct."""
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
        # assert isinstance(res_hadamard, tuple)
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

        assert isinstance(res_hadamard, (list, tuple))
        assert np.allclose(res_hadamard[0], res_param_shift[0], atol=tol, rtol=0)
        assert np.allclose(res_hadamard[1], res_param_shift[1], atol=tol, rtol=0)

    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, 2 * np.pi, 7))
    @pytest.mark.parametrize("G", [qml.IsingXX, qml.IsingYY, qml.IsingZZ, qml.IsingXY])
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
        """Tests that the automatic gradient of an arbitrary controlled Euler-angle-parametrized
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

        assert isinstance(grad, (tuple, list))
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
            qml.expval(qml.X(0))

        tape = qml.tape.QuantumScript.from_queue(q)

        res_hadamard, tapes = grad_fn(tape, dev)

        assert len(tapes) == 2

        assert len(res_hadamard) == 3
        assert len(res_hadamard[0]) == 2
        assert len(res_hadamard[1]) == 2

        expected = np.array([[-np.sin(x), 0], [0, np.cos(y)], [0, 0]])
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
        assert isinstance(res_hadamard, (tuple, list))
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
            qml.probs()

        tape = qml.tape.QuantumScript.from_queue(q)

        res_hadamard, tapes = grad_fn(tape, dev)

        assert len(tapes) == 2

        assert isinstance(res_hadamard, (tuple, list))
        assert len(res_hadamard) == 3

        assert isinstance(res_hadamard[0], (tuple, list))
        assert len(res_hadamard[0]) == 2

        # assert isinstance(res_hadamard[0][0], np.ndarray)
        assert res_hadamard[0][0].shape == ()
        # assert isinstance(res_hadamard[0][1], np.ndarray)
        assert res_hadamard[0][1].shape == ()

        for res in res_hadamard[1:]:
            assert isinstance(res, (tuple, list))
            assert len(res) == 2

            assert isinstance(res[0], np.ndarray)
            assert res[0].shape == (4,)
            assert isinstance(res[1], np.ndarray)
            assert res[1].shape == (4,)

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
        for res in res_hadamard[1:]:
            assert np.allclose(res[0], probs_expected[:, 0], tol)
            assert np.allclose(res[1], probs_expected[:, 1], tol)

    costs_and_expected_expval_scalar = [
        (cost7, (), np.ndarray),
        (cost8, (1,), list),
        (cost9, (2,), tuple),
    ]

    @pytest.mark.parametrize("mode", ["standard", "reversed", "direct", "reversed-direct"])
    @pytest.mark.parametrize("cost, exp_shape, exp_type", costs_and_expected_expval_scalar)
    def test_output_shape_matches_qnode_expval_scalar(self, cost, exp_shape, exp_type, mode):
        """Test that the transform output shape matches that of the QNode for
        expectation values and a scalar parameter."""
        if mode in {"reversed", "reversed-direct"} and exp_shape == (2,):
            pytest.xfail("reversed modes do not work with more than one output value.")
        dev = qml.device("default.qubit", wires=4)

        x = np.array(0.419)
        circuit = qml.QNode(cost, dev)

        res_hadamard = qml.gradients.hadamard_grad(circuit, mode=mode)(x)

        assert isinstance(res_hadamard, exp_type)
        assert np.array(res_hadamard).shape == exp_shape

    costs_and_expected_expval_array = [
        (cost1, [3], np.ndarray),
        (cost2, [3], list),
        (cost3, [2, 3], tuple),
    ]

    @pytest.mark.parametrize("mode", ["standard", "reversed", "direct", "reversed-direct"])
    @pytest.mark.parametrize("cost, exp_shape, exp_type", costs_and_expected_expval_array)
    def test_output_shape_matches_qnode_expval_array(self, cost, exp_shape, exp_type, mode):
        """Test that the transform output shape matches that of the QNode for
        expectation values and an array-valued parameter."""
        if mode in {"reversed", "reversed-direct"} and exp_shape == [2, 3]:
            pytest.xfail("reversed modes do not work with more than one output value.")

        dev = qml.device("default.qubit", wires=4)

        x = np.random.rand(3)
        circuit = qml.QNode(cost, dev)

        res_hadamard = qml.gradients.hadamard_grad(circuit, mode=mode)(x)

        assert isinstance(res_hadamard, exp_type)
        if len(res_hadamard) == 1:
            res_hadamard = res_hadamard[0]
        assert len(res_hadamard) == exp_shape[0]

        if len(exp_shape) > 1:
            for r in res_hadamard:
                assert isinstance(r, np.ndarray)
                assert len(r) == exp_shape[1]

    costs_and_expected_probs = [
        (cost4, [4, 3], np.ndarray),
        (cost5, [4, 3], list),
        (cost6, [2, 4, 3], tuple),
        (cost10, [8, 3], np.ndarray),  # Note that the shape here depends on the device
        (cost11, [2, 32, 3], tuple),
        (cost12, [8, 3], np.ndarray),
    ]

    @pytest.mark.parametrize(
        "mode", ["standard"]
    )  # other modes should not work with probs at the moment
    @pytest.mark.parametrize("cost, exp_shape, exp_type", costs_and_expected_probs)
    def test_output_shape_matches_qnode_probs(self, cost, exp_shape, exp_type, mode):
        """Test that the transform output shape matches that of the QNode."""
        dev = qml.device("default.qubit", wires=6)

        x = np.random.rand(3)
        circuit = qml.QNode(cost, dev)

        res_hadamard = qml.gradients.hadamard_grad(circuit, mode=mode)(x)
        assert isinstance(res_hadamard, exp_type)
        if len(res_hadamard) == 1:
            res_hadamard = res_hadamard[0]
        assert len(res_hadamard) == exp_shape[0]

        # Also check on the tape level
        tape = qml.workflow.construct_tape(circuit)(x)
        tapes, fn = qml.gradients.hadamard_grad(tape, mode=mode)
        res_hadamard_tape = qml.math.moveaxis(qml.math.stack(fn(dev.execute(tapes))), -2, -1)

        for res in [res_hadamard, res_hadamard_tape]:
            if len(exp_shape) > 2:
                for r in res:
                    assert isinstance(r, np.ndarray)
                    assert len(r) == exp_shape[1]

                    for r_ in r:
                        assert isinstance(r_, np.ndarray)
                        assert len(r_) == exp_shape[2]

            elif len(exp_shape) > 1:
                for r in res:
                    assert isinstance(r, np.ndarray)
                    assert len(r) == exp_shape[1]

    @pytest.mark.parametrize("shots", [None, 100, (1000, 1000)])
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

    @pytest.mark.parametrize("mode", ["standard", "reversed", "direct", "reversed-direct"])
    @pytest.mark.autograd
    def test_no_trainable_params_qnode_autograd(self, mode):
        """Test that the correct ouput and warning is generated in the absence of any trainable
        parameters"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="autograd")
        def circuit(weights):
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        weights = [0.1, 0.2]
        with pytest.raises(QuantumFunctionError, match="No trainable parameters."):
            qml.gradients.hadamard_grad(circuit, mode=mode)(weights)

    @pytest.mark.parametrize("mode", ["standard", "reversed", "direct", "reversed-direct"])
    @pytest.mark.torch
    def test_no_trainable_params_qnode_torch(self, mode):
        """Test that the correct output and warning is generated in the absence of any trainable
        parameters"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="torch")
        def circuit(weights):
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        weights = [0.1, 0.2]
        with pytest.raises(QuantumFunctionError, match="No trainable parameters."):
            qml.gradients.hadamard_grad(circuit, mode=mode)(weights)

    @pytest.mark.parametrize("mode", ["standard", "reversed", "direct", "reversed-direct"])
    @pytest.mark.tf
    def test_no_trainable_params_qnode_tf(self, mode):
        """Test that the correct ouput and warning is generated in the absence of any trainable
        parameters"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="tf")
        def circuit(weights):
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        weights = [0.1, 0.2]
        with pytest.raises(QuantumFunctionError, match="No trainable parameters."):
            qml.gradients.hadamard_grad(circuit, mode=mode)(weights)

    @pytest.mark.parametrize("mode", ["standard", "reversed", "direct", "reversed-direct"])
    @pytest.mark.jax
    def test_no_trainable_params_qnode_jax(self, mode):
        """Test that the correct ouput and warning is generated in the absence of any trainable
        parameters"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="jax")
        def circuit(weights):
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        weights = [0.1, 0.2]
        with pytest.raises(QuantumFunctionError, match="No trainable parameters."):
            qml.gradients.hadamard_grad(circuit, mode=mode)(weights)

    @pytest.mark.parametrize("mode", ["standard", "reversed", "direct", "reversed-direct"])
    def test_no_trainable_params_tape(self, mode):
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
            g_tapes, post_processing = qml.gradients.hadamard_grad(tape, mode=mode)
        res_hadamard = post_processing(qml.execute(g_tapes, dev, None))

        assert g_tapes == []
        assert isinstance(res_hadamard, np.ndarray)
        assert res_hadamard.shape == (0,)


class TestHadamardGradEdgeCases:
    """Test the Hadamard gradient transform and edge cases such as non diff parameters, auxiliary wires, etc..."""

    # pylint:disable=too-many-public-methods

    device_wires = [qml.wires.Wires([0, 1, "aux"])]
    device_wires_no_aux = [qml.wires.Wires([0, 1, 2])]

    working_wires = [None, qml.wires.Wires("aux"), "aux"]
    already_used_wires = [qml.wires.Wires(0), qml.wires.Wires(1)]

    @pytest.mark.parametrize("mode", ["standard"])  # test is fairly standard specific
    @pytest.mark.parametrize("aux_wire", working_wires)
    @pytest.mark.parametrize("device_wires", device_wires)
    def test_aux_wire(self, aux_wire, device_wires, mode):
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

        tapes, _ = qml.gradients.hadamard_grad(
            tape, aux_wire=aux_wire, device_wires=dev.wires, mode=mode
        )
        assert len(tapes) == 2
        tapes, _ = qml.gradients.hadamard_grad(tape, aux_wire=aux_wire, mode=mode)
        assert len(tapes) == 2

    @pytest.mark.parametrize("mode", ["standard", "reversed", "direct", "reversed-direct"])
    def test_empty_circuit(self, mode):
        """Test that an empty circuit works correctly"""
        with qml.queuing.AnnotatedQueue() as q:
            qml.expval(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        with pytest.warns(UserWarning, match="gradient of a tape with no trainable parameters"):
            tapes, _ = qml.gradients.hadamard_grad(tape, mode=mode)
        assert not tapes

    @pytest.mark.parametrize("mode", ["standard", "reversed", "direct", "reversed-direct"])
    def test_all_parameters_independent(self, mode):
        """Test that a circuit where all parameters do not affect the output"""
        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(0.4, wires=0)
            qml.expval(qml.PauliZ(1))

        tape = qml.tape.QuantumScript.from_queue(q)
        tapes, _ = qml.gradients.hadamard_grad(tape, mode=mode)
        assert not tapes

    def test_independent_parameter(self):
        """Test that an independent parameter is skipped
        during the Jacobian computation."""

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[1])  # does not have any impact on the expval
            qml.expval(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        dev = qml.device("default.qubit", wires=3)

        res_hadamard, tapes = grad_fn(tape, dev)

        assert len(tapes) == 1

        assert isinstance(res_hadamard, (tuple, list))
        assert len(res_hadamard) == 2
        assert res_hadamard[0].shape == ()
        assert res_hadamard[1].shape == ()

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

        assert isinstance(res_hadamard, (list, tuple))

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

    @pytest.mark.parametrize("mode", ["standard"])
    @pytest.mark.parametrize("prefactor", [1.0, 2.0])
    def test_all_zero_diff_methods(self, prefactor, mode):
        """Test that the transform works correctly when the diff method for every parameter is
        identified to be 0, and that no tapes were generated."""
        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def circuit(params):
            qml.Rot(*(prefactor * params), wires=0)
            return qml.probs([2, 3])

        params = np.array([0.5, 0.5, 0.5], requires_grad=True)

        result = qml.gradients.hadamard_grad(circuit, mode=mode)(params)

        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 3)
        assert np.allclose(result, 0)


@pytest.mark.parametrize("mode", ["standard", "reversed", "direct", "reversed-direct"])
class TestHadamardTestGradDiff:
    """Test that the transform is differentiable"""

    @pytest.mark.autograd
    def test_autograd(self, mode):
        """Tests that the output of the hadamard gradient transform
        can be differentiated using autograd, yielding second derivatives."""
        dev = qml.device("default.qubit", wires=3)
        params = np.array([0.543, -0.654], requires_grad=True)

        def cost_fn_hadamard(x):
            with qml.queuing.AnnotatedQueue() as q:
                qml.RX(x[0], wires=[0])
                qml.RY(x[1], wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

            tape = qml.tape.QuantumScript.from_queue(q)
            tape.trainable_params = {0, 1}
            tapes, fn = qml.gradients.hadamard_grad(tape, mode=mode)
            jac = fn(dev.execute(tapes))
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
            jac = fn(dev.execute(tapes))
            return qml.math.stack(jac)

        res_hadamard = qml.jacobian(cost_fn_hadamard)(params)
        res_param_shift = qml.jacobian(cost_fn_param_shift)(params)
        assert np.allclose(res_hadamard, res_param_shift)

    @pytest.mark.tf
    def test_tf(self, mode):
        """Tests that the output of the hadamard gradient transform
        can be differentiated using TF, yielding second derivatives."""
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=3)
        params = tf.Variable([0.543, -0.654], dtype=tf.float64)

        with tf.GradientTape() as t_h:
            with qml.queuing.AnnotatedQueue() as q:
                qml.RX(params[0], wires=[0])
                qml.RY(params[1], wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

            tape = qml.tape.QuantumScript.from_queue(q)
            tape.trainable_params = {0, 1}
            tapes, fn = qml.gradients.hadamard_grad(tape, mode=mode)
            jac_h = fn(dev.execute(tapes))
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
            jac_p = fn(dev.execute(tapes))
            jac_p = qml.math.stack(jac_p)

        res_hadamard = t_h.jacobian(jac_h, params)
        res_param_shift = t_p.jacobian(jac_p, params)

        assert np.allclose(res_hadamard, res_param_shift)

    @pytest.mark.torch
    def test_torch(self, mode):
        """Tests that the output of the hadamard gradient transform
        can be differentiated using Torch, yielding second derivatives."""
        import torch

        dev = qml.device("default.qubit", wires=3)
        params = torch.tensor([0.543, -0.654], dtype=torch.float64, requires_grad=True)

        def cost_h(x):
            with qml.queuing.AnnotatedQueue() as q:
                qml.RX(x[0], wires=[0])
                qml.RY(x[1], wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

            tape = qml.tape.QuantumScript.from_queue(q)
            tapes, fn = qml.gradients.hadamard_grad(tape, mode=mode)

            return tuple(fn(dev.execute(tapes)))

        def cost_p(x):
            with qml.queuing.AnnotatedQueue() as q:
                qml.RX(x[0], wires=[0])
                qml.RY(x[1], wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

            tape = qml.tape.QuantumScript.from_queue(q)
            tapes, fn = qml.gradients.param_shift(tape)

            return tuple(fn(dev.execute(tapes)))

        res_hadamard = torch.autograd.functional.jacobian(cost_h, params)
        res_param_shift = torch.autograd.functional.jacobian(cost_p, params)

        assert np.allclose(res_hadamard[0].detach(), res_param_shift[0].detach())
        assert np.allclose(res_hadamard[1].detach(), res_param_shift[1].detach())

    @pytest.mark.jax
    def test_jax(self, mode):
        """Tests that the output of the hadamard gradient transform
        can be differentiated using JAX, yielding second derivatives."""
        import jax
        from jax import numpy as jnp

        dev = qml.device("default.qubit", wires=3)
        params = jnp.array([0.543, -0.654])

        def cost_h(x):
            with qml.queuing.AnnotatedQueue() as q:
                qml.RX(x[0], wires=[0])
                qml.RY(x[1], wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

            tape = qml.tape.QuantumScript.from_queue(q)
            tape.trainable_params = {0, 1}
            tapes, fn = qml.gradients.hadamard_grad(tape, mode=mode)

            return fn(dev.execute(tapes))

        def cost_p(x):
            with qml.queuing.AnnotatedQueue() as q:
                qml.RX(x[0], wires=[0])
                qml.RY(x[1], wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

            tape = qml.tape.QuantumScript.from_queue(q)
            tape.trainable_params = {0, 1}
            tapes, fn = qml.gradients.hadamard_grad(tape, mode=mode)

            return fn(dev.execute(tapes))

        res_hadamard = jax.jacobian(cost_h)(params)
        res_param_shift = jax.jacobian(cost_p)(params)
        assert np.allclose(res_hadamard, res_param_shift)


@pytest.mark.parametrize("argnums", [[0], [1], [0, 1]])
@pytest.mark.parametrize("interface", ["jax"])
@pytest.mark.parametrize("mode", ["standard", "reversed", "direct", "reversed-direct"])
@pytest.mark.jax
class TestJaxArgnums:
    """Class to test the integration of argnums (Jax) and the hadamard grad transform."""

    expected_jacs = []
    interfaces = ["auto", "jax"]

    def test_argnum_error(self, argnums, interface, mode):
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
            QuantumFunctionError,
            match="argnum does not work with the Jax interface. You should use argnums instead.",
        ):
            qml.gradients.hadamard_grad(circuit, argnum=argnums, mode=mode)(x, y)

    def test_single_expectation_value(self, argnums, interface, mode):
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

        res = qml.gradients.hadamard_grad(circuit, argnums=argnums, mode=mode)(x, y)

        expected_0 = np.array([-np.sin(y) * np.sin(x[0]), 0])
        expected_1 = np.array(np.cos(y) * np.cos(x[0]))

        if argnums == [0]:
            assert np.allclose(res, expected_0)
        if argnums == [1]:
            assert np.allclose(res, expected_1)
        if argnums == [0, 1]:
            assert np.allclose(res[0], expected_0)
            assert np.allclose(res[1], expected_1)

    def test_multi_expectation_values(self, argnums, interface, mode):
        """Test for multiple expectation values."""
        if mode in {"reversed", "reversed-direct"}:
            pytest.skip("these methods do not work with more than one observable.")
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

        res = qml.gradients.hadamard_grad(circuit, argnums=argnums, mode=mode)(x, y)

        expected_0 = np.array([[-np.sin(x[0]), 0.0], [0.0, 0.0]])
        expected_1 = np.array([0, np.cos(y)])

        if argnums == [0]:
            assert np.allclose(res[0], expected_0[0])
            assert np.allclose(res[1], expected_0[1])
        if argnums == [1]:
            assert np.allclose(res[0][0], expected_1[0])
            assert np.allclose(res[1][0], expected_1[1])
        if argnums == [0, 1]:
            assert np.allclose(res[0][0], expected_0[0])
            assert np.allclose(res[0][1], expected_0[1])
            assert np.allclose(res[1][0], expected_1[0])
            assert np.allclose(res[1][1], expected_1[1])

    def test_hessian(self, argnums, interface, mode):
        """Test for hessian."""
        import jax

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev, interface=interface)
        def circuit(x, y):
            qml.RX(x[0], wires=[0])
            qml.RY(x[1], wires=[1])
            qml.RY(y, wires=[1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        x = jax.numpy.array([0.543, -0.654])
        y = jax.numpy.array(-0.123)

        res = jax.jacobian(
            qml.gradients.hadamard_grad(circuit, argnums=argnums, mode=mode), argnums=argnums
        )(x, y)
        res_expected = jax.hessian(circuit, argnums=argnums)(x, y)

        if len(argnums) == 1:
            # jax.hessian produces an additional tuple axis, which we have to index away here
            assert np.allclose(res, res_expected[0])
        else:
            # The Hessian is a 2x2 nested tuple "matrix" for argnums=[0, 1]
            for r, r_e in zip(res, res_expected):
                for r_, r_e_ in zip(r, r_e):
                    assert np.allclose(r_, r_e_)

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
"""Unit tests for the probs module"""
import numpy as np
import pytest

import pennylane as qml
from pennylane import numpy as pnp
from pennylane.devices.qubit.measure import flatten_state
from pennylane.measurements import ProbabilityMP, Shots


# TODO: Remove this when new CustomMP are the default
def custom_measurement_process(device, spy):
    assert len(spy.call_args_list) > 0  # make sure method is mocked properly

    samples = device._samples  # pylint: disable=protected-access
    state = flatten_state(device._state, device.num_wires)  # pylint: disable=protected-access
    call_args_list = list(spy.call_args_list)
    for call_args in call_args_list:
        wires, shot_range, bin_size = (
            call_args.kwargs["wires"],
            call_args.kwargs["shot_range"],
            call_args.kwargs["bin_size"],
        )
        # no need to use op, because the observable has already been applied to ``dev._state``
        meas = qml.probs(wires=wires)
        old_res = device.probability(wires=wires, shot_range=shot_range, bin_size=bin_size)
        if device.shots is None:
            new_res = meas.process_state(state=state, wire_order=device.wires)
        else:
            new_res = meas.process_samples(
                samples=samples,
                wire_order=device.wires,
                shot_range=shot_range,
                bin_size=bin_size,
            )
        assert qml.math.allequal(old_res, new_res)


# make the test deterministic
np.random.seed(42)


@pytest.fixture(name="init_state")
def fixture_init_state():
    """Fixture that creates an initial state"""

    def _init_state(n):
        """An initial state over n wires"""
        state = np.random.random([2**n]) + np.random.random([2**n]) * 1j
        state /= np.linalg.norm(state)
        return state

    return _init_state


class TestProbs:
    """Tests for the probs function"""

    # pylint:disable=too-many-public-methods

    def test_queue(self):
        """Test that the right measurement class is queued."""
        dev = qml.device("default.qubit.legacy", wires=2)

        @qml.qnode(dev)
        def circuit():
            return qml.probs(wires=0)

        circuit()

        assert isinstance(circuit.tape[0], ProbabilityMP)

    @pytest.mark.parametrize("wires", [[0], [2, 1], ["a", "c", 3]])
    @pytest.mark.parametrize("shots", [None, 10])
    def test_shape(self, wires, shots):
        """Test that the shape is correct."""
        dev = qml.device("default.qubit.legacy", wires=3, shots=shots)
        res = qml.probs(wires=wires)
        assert res.shape(dev, Shots(shots)) == (2 ** len(wires),)

    @pytest.mark.parametrize("wires", [[0], [2, 1], ["a", "c", 3]])
    def test_shape_shot_vector(self, wires):
        """Test that the shape is correct with the shot vector too."""
        res = qml.probs(wires=wires)
        shot_vector = (1, 2, 3)
        dev = qml.device("default.qubit.legacy", wires=3, shots=shot_vector)
        assert res.shape(dev, Shots(shot_vector)) == (
            (2 ** len(wires),),
            (2 ** len(wires),),
            (2 ** len(wires),),
        )

    @pytest.mark.parametrize("shots", [None, 100])
    def test_probs_no_arguments(self, shots):
        """Test that using ``qml.probs`` with no arguments returns the probabilities of all wires."""
        dev = qml.device("default.qubit.legacy", wires=3, shots=shots)

        @qml.qnode(dev)
        def circuit():
            return qml.probs()

        res = circuit()

        assert qml.math.allequal(res, [1, 0, 0, 0, 0, 0, 0, 0])

    def test_full_prob(self, init_state, tol, mocker):
        """Test that the correct probability is returned."""
        dev = qml.device("default.qubit.legacy", wires=4)
        spy = mocker.spy(qml.QubitDevice, "probability")

        state = init_state(4)

        @qml.qnode(dev)
        def circuit():
            qml.StatePrep(state, wires=list(range(4)))
            return qml.probs(wires=range(4))

        res = circuit()
        expected = np.abs(state) ** 2
        assert np.allclose(res, expected, atol=tol, rtol=0)

        custom_measurement_process(dev, spy)

    def test_marginal_prob(self, init_state, tol, mocker):
        """Test that the correct marginal probability is returned."""
        dev = qml.device("default.qubit.legacy", wires=4)
        spy = mocker.spy(qml.QubitDevice, "probability")

        state = init_state(4)

        @qml.qnode(dev)
        def circuit():
            qml.StatePrep(state, wires=list(range(4)))
            return qml.probs(wires=[1, 3])

        res = circuit()
        expected = np.reshape(np.abs(state) ** 2, [2] * 4)
        expected = np.einsum("ijkl->jl", expected).flatten()
        assert np.allclose(res, expected, atol=tol, rtol=0)

        custom_measurement_process(dev, spy)

    def test_marginal_prob_more_wires(self, init_state, mocker, tol):
        """Test that the correct marginal probability is returned, when the
        states_to_binary method is used for probability computations."""
        dev = qml.device("default.qubit.legacy", wires=4)
        spy_probs = mocker.spy(qml.QubitDevice, "probability")
        state = init_state(4)

        @qml.qnode(dev)
        def circuit():
            qml.StatePrep(state, wires=list(range(4)))
            return qml.probs(wires=[1, 0, 3])

        res = circuit()

        expected = np.reshape(np.abs(state) ** 2, [2] * 4)
        expected = np.einsum("ijkl->jil", expected).flatten()
        assert np.allclose(res, expected, atol=tol, rtol=0)

        custom_measurement_process(dev, spy_probs)

    @pytest.mark.parametrize("shots", [None, 10000, [10000, 10000]])
    @pytest.mark.parametrize("phi", np.arange(0, 2 * np.pi, np.pi / 3))
    @pytest.mark.parametrize("device_name", ["default.qubit.legacy", "default.mixed"])
    def test_observable_is_measurement_value(
        self, shots, phi, mocker, tol, tol_stochastic, device_name
    ):  # pylint: disable=too-many-arguments
        """Test that probs for mid-circuit measurement values
        are correct for a single measurement value."""
        np.random.seed(42)
        dev = qml.device(device_name, wires=2, shots=shots)

        @qml.qnode(dev)
        def circuit(phi):
            qml.RX(phi, 0)
            m0 = qml.measure(0)
            return qml.probs(op=m0)

        new_dev = circuit.device
        spy = mocker.spy(qml.QubitDevice, "probability")

        res = circuit(phi)

        atol = tol if shots is None else tol_stochastic
        expected = np.array([np.cos(phi / 2) ** 2, np.sin(phi / 2) ** 2])

        if not isinstance(shots, list):
            assert np.allclose(np.array(res), expected, atol=atol, rtol=0)
        else:
            for r in res:  # pylint: disable=not-an-iterable
                assert np.allclose(r, expected, atol=atol, rtol=0)

        if device_name != "default.mixed":
            custom_measurement_process(new_dev, spy)

    @pytest.mark.parametrize("shots", [None, 10000, [10000, 10000]])
    @pytest.mark.parametrize("phi", np.arange(0, 2 * np.pi, np.pi / 3))
    @pytest.mark.parametrize("device_name", ["default.qubit.legacy", "default.mixed"])
    def test_observable_is_measurement_value_list(
        self, shots, phi, mocker, tol, tol_stochastic, device_name
    ):  # pylint: disable=too-many-arguments
        """Test that probs for mid-circuit measurement values
        are correct for a list of measurement value."""
        np.random.seed(42)
        dev = qml.device(device_name, wires=6, shots=shots)

        @qml.qnode(dev)
        def circuit(phi):
            qml.RX(phi, 0)
            m0 = qml.measure(0)
            qml.RX(0.5 * phi, 1)
            m1 = qml.measure(1)
            qml.RX(2.0 * phi, 2)
            m2 = qml.measure(2)
            return qml.probs(op=[m0, m1, m2])

        new_dev = circuit.device
        spy = mocker.spy(qml.QubitDevice, "probability")

        res = circuit(phi, shots=shots)

        @qml.qnode(dev)
        def expected_circuit(phi):
            qml.RX(phi, 0)
            qml.RX(0.5 * phi, 1)
            qml.RX(2.0 * phi, 2)
            return qml.probs(wires=[0, 1, 2])

        expected = expected_circuit(phi)

        atol = tol if shots is None else tol_stochastic

        if not isinstance(shots, list):
            assert np.allclose(np.array(res), expected, atol=atol, rtol=0)
        else:
            for r in res:  # pylint: disable=not-an-iterable
                assert np.allclose(r, expected, atol=atol, rtol=0)

        if device_name != "default.mixed":
            custom_measurement_process(new_dev, spy)

    def test_integration(self, tol, mocker):
        """Test the probability is correct for a known state preparation."""
        dev = qml.device("default.qubit.legacy", wires=2)
        spy = mocker.spy(qml.QubitDevice, "probability")

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0, 1])

        # expected probability, using [00, 01, 10, 11]
        # ordering, is [0.5, 0.5, 0, 0]

        res = circuit()
        expected = np.array([0.5, 0.5, 0, 0])
        assert np.allclose(res, expected, atol=tol, rtol=0)

        custom_measurement_process(dev, spy)

    @pytest.mark.parametrize("shots", [100, [1, 10, 100]])
    def test_integration_analytic_false(self, tol, mocker, shots):
        """Test the probability is correct for a known state preparation when the
        analytic attribute is set to False."""
        dev = qml.device("default.qubit.legacy", wires=3, shots=shots)
        spy = mocker.spy(qml.QubitDevice, "probability")

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(0)
            return qml.probs(wires=dev.wires)

        res = circuit()
        expected = np.array([0, 0, 0, 0, 1, 0, 0, 0])
        assert np.allclose(res, expected, atol=tol, rtol=0)

        custom_measurement_process(dev, spy)

    @pytest.mark.parametrize("shots", [None, 100])
    def test_batch_size(self, mocker, shots):
        """Test the probability is correct for a batched input."""
        dev = qml.device("default.qubit.legacy", wires=1, shots=shots)
        spy = mocker.spy(qml.QubitDevice, "probability")

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, 0)
            return qml.probs(wires=dev.wires)  # TODO: Use ``qml.probs()`` when supported

        x = np.array([0, np.pi / 2])
        res = circuit(x)
        expected = [[1.0, 0.0], [0.5, 0.5]]
        assert np.allclose(res, expected, atol=0.1, rtol=0.1)

        custom_measurement_process(dev, spy)

    @pytest.mark.autograd
    def test_numerical_analytic_diff_agree(self, tol, mocker):
        """Test that the finite difference and parameter shift rule
        provide the same Jacobian."""
        w = 4
        dev = qml.device("default.qubit.legacy", wires=w)
        spy = mocker.spy(qml.QubitDevice, "probability")

        def circuit(x, y, z):
            for i in range(w):
                qml.RX(x, wires=i)
                qml.PhaseShift(z, wires=i)
                qml.RY(y, wires=i)

            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[2, 3])

            return qml.probs(wires=[1, 3])

        params = pnp.array([0.543, -0.765, -0.3], requires_grad=True)

        circuit_F = qml.QNode(circuit, dev, diff_method="finite-diff")
        circuit_A = qml.QNode(circuit, dev, diff_method="parameter-shift")
        res_F = qml.jacobian(circuit_F)(*params)
        res_A = qml.jacobian(circuit_A)(*params)

        # Both jacobians should be of shape (2**prob.wires, num_params)
        assert isinstance(res_F, tuple) and len(res_F) == 3
        assert all(_r.shape == (2**2,) for _r in res_F)
        assert isinstance(res_A, tuple) and len(res_A) == 3
        assert all(_r.shape == (2**2,) for _r in res_A)

        # Check that they agree up to numeric tolerance
        assert all(np.allclose(_rF, _rA, atol=tol, rtol=0) for _rF, _rA in zip(res_F, res_A))

        custom_measurement_process(dev, spy)

    @pytest.mark.parametrize("hermitian", [1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])])
    def test_prob_generalize_param_one_qubit(self, hermitian, tol, mocker):
        """Test that the correct probability is returned."""
        dev = qml.device("default.qubit.legacy", wires=1)
        spy = mocker.spy(qml.QubitDevice, "probability")

        @qml.qnode(dev)
        def circuit(x):
            qml.RZ(x, wires=0)
            return qml.probs(op=qml.Hermitian(hermitian, wires=0))

        res = circuit(0.56)

        def circuit_rotated(x):
            qml.RZ(x, wires=0)
            qml.Hermitian(hermitian, wires=0).diagonalizing_gates()

        state = np.array([1, 0])
        matrix = qml.matrix(circuit_rotated, wire_order=[0])(0.56)
        state = np.dot(matrix, state)
        expected = np.reshape(np.abs(state) ** 2, [2] * 1)
        expected = expected.flatten()

        assert np.allclose(res, expected, atol=tol, rtol=0)

        custom_measurement_process(dev, spy)

    @pytest.mark.parametrize("hermitian", [1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])])
    def test_prob_generalize_param(self, hermitian, tol, mocker):
        """Test that the correct probability is returned."""
        dev = qml.device("default.qubit.legacy", wires=3)
        spy = mocker.spy(qml.QubitDevice, "probability")

        @qml.qnode(dev)
        def circuit(x, y):
            qml.RZ(x, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 2])
            return qml.probs(op=qml.Hermitian(hermitian, wires=0))

        res = circuit(0.56, 0.1)

        def circuit_rotated(x, y):
            qml.RZ(x, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 2])
            qml.Hermitian(hermitian, wires=0).diagonalizing_gates()

        state = np.array([1, 0, 0, 0, 0, 0, 0, 0])
        matrix = qml.matrix(circuit_rotated, wire_order=[0, 1, 2])(0.56, 0.1)
        state = np.dot(matrix, state)
        expected = np.reshape(np.abs(state) ** 2, [2] * 3)
        expected = np.einsum("ijk->i", expected).flatten()
        assert np.allclose(res, expected, atol=tol, rtol=0)

        custom_measurement_process(dev, spy)

    @pytest.mark.parametrize("hermitian", [1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])])
    def test_prob_generalize_param_multiple(self, hermitian, tol, mocker):
        """Test that the correct probability is returned."""
        dev = qml.device("default.qubit.legacy", wires=3)
        spy = mocker.spy(qml.QubitDevice, "probability")

        @qml.qnode(dev)
        def circuit(x, y):
            qml.RZ(x, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 2])
            return (
                qml.probs(op=qml.Hermitian(hermitian, wires=0)),
                qml.probs(wires=[1]),
                qml.probs(wires=[2]),
            )

        res = circuit(0.56, 0.1)
        res = np.reshape(res, (3, 2))

        def circuit_rotated(x, y):
            qml.RZ(x, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 2])
            qml.Hermitian(hermitian, wires=0).diagonalizing_gates()

        state = np.array([1, 0, 0, 0, 0, 0, 0, 0])
        matrix = qml.matrix(circuit_rotated, wire_order=[0, 1, 2])(0.56, 0.1)
        state = np.dot(matrix, state)

        expected = np.reshape(np.abs(state) ** 2, [2] * 3)
        expected_0 = np.einsum("ijk->i", expected).flatten()
        expected_1 = np.einsum("ijk->j", expected).flatten()
        expected_2 = np.einsum("ijk->k", expected).flatten()

        assert np.allclose(res[0], expected_0, atol=tol, rtol=0)
        assert np.allclose(res[1], expected_1, atol=tol, rtol=0)
        assert np.allclose(res[2], expected_2, atol=tol, rtol=0)

        custom_measurement_process(dev, spy)

    @pytest.mark.parametrize("hermitian", [1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])])
    @pytest.mark.parametrize("wire", [0, 1, 2, 3])
    def test_prob_generalize_initial_state(self, hermitian, wire, init_state, tol, mocker):
        """Test that the correct probability is returned."""
        # pylint:disable=too-many-arguments
        dev = qml.device("default.qubit.legacy", wires=4)
        spy = mocker.spy(qml.QubitDevice, "probability")
        state = init_state(4)

        @qml.qnode(dev)
        def circuit():
            qml.StatePrep(state, wires=list(range(4)))
            qml.PauliX(wires=0)
            qml.PauliX(wires=1)
            qml.PauliX(wires=2)
            qml.PauliX(wires=3)
            return qml.probs(op=qml.Hermitian(hermitian, wires=wire))

        res = circuit()

        def circuit_rotated():
            qml.PauliX(wires=0)
            qml.PauliX(wires=1)
            qml.PauliX(wires=2)
            qml.PauliX(wires=3)
            qml.Hermitian(hermitian, wires=wire).diagonalizing_gates()

        matrix = qml.matrix(circuit_rotated, wire_order=[0, 1, 2, 3])()
        state = np.dot(matrix, state)
        expected = np.reshape(np.abs(state) ** 2, [2] * 4)

        if wire == 0:
            expected = np.einsum("ijkl->i", expected).flatten()
        elif wire == 1:
            expected = np.einsum("ijkl->j", expected).flatten()
        elif wire == 2:
            expected = np.einsum("ijkl->k", expected).flatten()
        elif wire == 3:
            expected = np.einsum("ijkl->l", expected).flatten()

        assert np.allclose(res, expected, atol=tol, rtol=0)

        custom_measurement_process(dev, spy)

    @pytest.mark.parametrize("operation", [qml.PauliX, qml.PauliY, qml.Hadamard])
    @pytest.mark.parametrize("wire", [0, 1, 2, 3])
    def test_operation_prob(self, operation, wire, init_state, tol, mocker):
        "Test the rotated probability with different wires and rotating operations."
        # pylint:disable=too-many-arguments
        dev = qml.device("default.qubit.legacy", wires=4)
        spy = mocker.spy(qml.QubitDevice, "probability")
        state = init_state(4)

        @qml.qnode(dev)
        def circuit():
            qml.StatePrep(state, wires=list(range(4)))
            qml.PauliX(wires=0)
            qml.PauliZ(wires=1)
            qml.PauliY(wires=2)
            qml.PauliZ(wires=3)
            return qml.probs(op=operation(wires=wire))

        res = circuit()

        def circuit_rotated():
            qml.PauliX(wires=0)
            qml.PauliZ(wires=1)
            qml.PauliY(wires=2)
            qml.PauliZ(wires=3)
            operation(wires=wire).diagonalizing_gates()

        matrix = qml.matrix(circuit_rotated, wire_order=[0, 1, 2, 3])()
        state = np.dot(matrix, state)
        expected = np.reshape(np.abs(state) ** 2, [2] * 4)

        if wire == 0:
            expected = np.einsum("ijkl->i", expected).flatten()
        elif wire == 1:
            expected = np.einsum("ijkl->j", expected).flatten()
        elif wire == 2:
            expected = np.einsum("ijkl->k", expected).flatten()
        elif wire == 3:
            expected = np.einsum("ijkl->l", expected).flatten()

        assert np.allclose(res, expected, atol=tol, rtol=0)

        custom_measurement_process(dev, spy)

    @pytest.mark.parametrize("observable", [(qml.PauliX, qml.PauliY)])
    def test_observable_tensor_prob(self, observable, init_state, tol, mocker):
        "Test the rotated probability with a tensor observable."
        dev = qml.device("default.qubit.legacy", wires=4)
        spy = mocker.spy(qml.QubitDevice, "probability")
        state = init_state(4)

        @qml.qnode(dev)
        def circuit():
            qml.StatePrep(state, wires=list(range(4)))
            qml.PauliX(wires=0)
            qml.PauliZ(wires=1)
            qml.PauliY(wires=2)
            qml.PauliZ(wires=3)
            return qml.probs(op=observable[0](wires=0) @ observable[1](wires=1))

        res = circuit()

        def circuit_rotated():
            qml.PauliX(wires=0)
            qml.PauliZ(wires=1)
            qml.PauliY(wires=2)
            qml.PauliZ(wires=3)
            observable[0](wires=0).diagonalizing_gates()
            observable[1](wires=1).diagonalizing_gates()

        matrix = qml.matrix(circuit_rotated, wire_order=[0, 1, 2, 3])()
        state = np.dot(matrix, state)
        expected = np.reshape(np.abs(state) ** 2, [2] * 4)

        expected = np.einsum("ijkl->ij", expected).flatten()

        assert np.allclose(res, expected, atol=tol, rtol=0)

        custom_measurement_process(dev, spy)

    @pytest.mark.parametrize("coeffs, obs", [([1, 1], [qml.PauliX(wires=0), qml.PauliX(wires=1)])])
    def test_hamiltonian_error(self, coeffs, obs, init_state):
        "Test that an error is returned for hamiltonians."
        H = qml.Hamiltonian(coeffs, obs)

        dev = qml.device("default.qubit.legacy", wires=4)
        state = init_state(4)

        @qml.qnode(dev)
        def circuit():
            qml.StatePrep(state, wires=list(range(4)))
            qml.PauliX(wires=0)
            qml.PauliZ(wires=1)
            qml.PauliY(wires=2)
            qml.PauliZ(wires=3)
            return qml.probs(op=H)

        with pytest.raises(
            qml.QuantumFunctionError,
            match="Hamiltonians are not supported for rotating probabilities.",
        ):
            circuit()

    @pytest.mark.parametrize(
        "operation", [qml.SingleExcitation, qml.SingleExcitationPlus, qml.SingleExcitationMinus]
    )
    def test_generalize_prob_not_hermitian(self, operation):
        """Test that Operators that do not have a diagonalizing_gates representation cannot
        be used in probability measurements."""

        dev = qml.device("default.qubit.legacy", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(wires=0)
            qml.PauliZ(wires=1)
            return qml.probs(op=operation(0.56, wires=[0, 1]))

        with pytest.raises(
            qml.QuantumFunctionError,
            match="does not define diagonalizing gates : cannot be used to rotate the probability",
        ):
            circuit()

    @pytest.mark.parametrize("hermitian", [1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])])
    def test_prob_wires_and_hermitian(self, hermitian):
        """Test that we can cannot give simultaneously wires and a hermitian."""

        dev = qml.device("default.qubit.legacy", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(wires=0)
            return qml.probs(op=qml.Hermitian(hermitian, wires=0), wires=1)

        with pytest.raises(
            qml.QuantumFunctionError,
            match="Cannot specify the wires to probs if an observable is "
            "provided. The wires for probs will be determined directly from the observable.",
        ):
            circuit()

    def test_non_commuting_probs_raises_error(self):
        dev = qml.device("default.qubit.legacy", wires=5)

        @qml.qnode(dev)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliX(0)), qml.probs(wires=[0, 1])

        with pytest.raises(
            qml.QuantumFunctionError, match="Only observables that are qubit-wise commuting"
        ):
            circuit(1, 2)

    def test_commuting_probs_in_computational_basis(self):
        """Test that `qml.probs` can be used in the computational basis with other commuting observables."""
        dev = qml.device("default.qubit.legacy", wires=5)

        @qml.qnode(dev)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        res = circuit(1, 2)

        @qml.qnode(dev)
        def circuit2(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(dev)
        def circuit3(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0, 1])

        res2 = circuit2(1, 2)
        res3 = circuit3(1, 2)

        assert res[0] == res2
        assert qml.math.allequal(res[1:], res3)

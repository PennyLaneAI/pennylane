# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
Unit tests for the :mod:`pennylane.plugin.DefaultQutrit` device.
"""
import math

import pytest
import pennylane as qml
from pennylane import numpy as np, DeviceError
from pennylane.devices.default_qutrit import DefaultQutrit
from pennylane.wires import Wires, WireError

OMEGA = np.exp(2 * np.pi * 1j / 3)


U_thadamard_01 = np.multiply(
    1 / np.sqrt(2),
    np.array(
        [[1, 1, 0], [1, -1, 0], [0, 0, np.sqrt(2)]],
    ),
)

U_x_02 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.complex128)

U_z_12 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.complex128)

U_shift = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.complex128)

U_clock = np.array([[1, 0, 0], [0, OMEGA, 0], [0, 0, OMEGA**2]])

U_tswap = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
    ],
    dtype=np.complex128,
)

U_tadd = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
    ],
    dtype=np.complex128,
)


def test_analytic_deprecation():
    """Tests if the kwarg `analytic` is used and displays error message."""
    msg = "The analytic argument has been replaced by shots=None. "
    msg += "Please use shots=None instead of analytic=True."

    with pytest.raises(
        DeviceError,
        match=msg,
    ):
        qml.device("default.qutrit", wires=1, shots=1, analytic=True)


def test_dtype_errors():
    """Test that if an incorrect dtype is provided to the device then an error is raised."""
    with pytest.raises(DeviceError, match="Real datatype must be a floating point type."):
        qml.device("default.qutrit", wires=1, r_dtype=np.complex128)
    with pytest.raises(
        DeviceError, match="Complex datatype must be a complex floating point type."
    ):
        qml.device("default.qutrit", wires=1, c_dtype=np.float64)


# TODO: Add tests for expval, var, sample and tensor observables after addition of observables
# TODO: Add tests to check for dtype preservation after more ops and observables have been added
# TODO: Add tests for operations that will have custom internal implementations for default.qutrit once added
# TODO: Add tests for inverse decomposition once decomposible operations are added
# TODO: Add tests for verifying correct behaviour of different observables on default qutrit device.
# TODO: Add tests for devices with shots for testing correct behaviour of sample


class TestApply:
    """Tests that operations and inverses of certain operations are applied correctly or that the proper
    errors are raised.
    """

    # TODO: Add tests for non-parametric ops after they're implemented

    # TODO: Add more data as parametric ops get added
    test_data_single_wire_with_parameters = [
        (qml.QutritUnitary, [1, 0, 0], [1, 1, 0] / np.sqrt(2), U_thadamard_01),
        (qml.QutritUnitary, [1, 0, 0], [0, 0, 1], U_x_02),
        (qml.QutritUnitary, [1, 0, 0], [1, 0, 0], U_z_12),
        (qml.QutritUnitary, [0, 1, 0], [0, 1, 0], U_x_02),
        (qml.QutritUnitary, [0, 0, 1], [0, 0, -1], U_z_12),
        (qml.QutritUnitary, [0, 1, 0], [0, 0, 1], U_shift),
        (qml.QutritUnitary, [0, 1, 0], [0, OMEGA, 0], U_clock),
    ]

    @pytest.mark.parametrize(
        "operation, input, expected_output, par", test_data_single_wire_with_parameters
    )
    def test_apply_operation_single_wire_with_parameters(
        self, qutrit_device_1_wire, tol, operation, input, expected_output, par
    ):
        """Tests that applying an operation yields the expected output state for single wire
        operations that have parameters."""

        qutrit_device_1_wire._state = np.array(input, dtype=qutrit_device_1_wire.C_DTYPE)

        qutrit_device_1_wire.apply([operation(par, wires=[0])])

        assert np.allclose(qutrit_device_1_wire._state, np.array(expected_output), atol=tol, rtol=0)
        assert qutrit_device_1_wire._state.dtype == qutrit_device_1_wire.C_DTYPE

    # TODO: Add more data as parametric ops get added
    test_data_single_wire_with_parameters_inverse = [
        (qml.QutritUnitary, [1, 0, 0], [0, 0, 1], U_shift),
        (qml.QutritUnitary, [0, 0, 1], [0, 1, 0], U_shift),
        (qml.QutritUnitary, [0, OMEGA, 0], [0, 1, 0], U_clock),
        (qml.QutritUnitary, [0, 0, OMEGA**2], [0, 0, 1], U_clock),
    ]

    @pytest.mark.parametrize(
        "operation, input, expected_output, par", test_data_single_wire_with_parameters_inverse
    )
    def test_apply_operation_single_wire_with_parameters_inverse(
        self, qutrit_device_1_wire, tol, operation, input, expected_output, par
    ):
        """Tests that applying an operation yields the expected output state for single wire
        operations that have parameters."""

        qutrit_device_1_wire._state = np.array(input, dtype=qutrit_device_1_wire.C_DTYPE)

        qutrit_device_1_wire.apply([operation(par, wires=[0]).inv()])

        assert np.allclose(qutrit_device_1_wire._state, np.array(expected_output), atol=tol, rtol=0)
        assert qutrit_device_1_wire._state.dtype == qutrit_device_1_wire.C_DTYPE

    # TODO: Add more ops as parametric operations get added
    test_data_two_wires_with_parameters = [
        (qml.QutritUnitary, [0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0], U_tswap),
        (qml.QutritUnitary, [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], U_tswap),
        (
            qml.QutritUnitary,
            [0, 0, 1, 0, 0, 0, 0, 1, 0] / np.sqrt(2),
            [0, 0, 0, 0, 0, 1, 1, 0, 0] / np.sqrt(2),
            U_tswap,
        ),
        (
            qml.QutritUnitary,
            np.multiply(0.5, [0, 1, 1, 0, 0, 0, 0, 1, 1]),
            np.multiply(0.5, [0, 0, 0, 1, 0, 1, 1, 0, 1]),
            U_tswap,
        ),
        (qml.QutritUnitary, [0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0], U_tadd),
        (qml.QutritUnitary, [0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0], U_tadd),
    ]

    # TODO: Add more ops as parametric operations get added
    test_data_two_wires_with_parameters_inverse = [
        (qml.QutritUnitary, [0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0], U_tswap),
        (qml.QutritUnitary, [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], U_tswap),
        (
            qml.QutritUnitary,
            [0, 0, 1, 0, 0, 0, 0, 1, 0] / np.sqrt(2),
            [0, 0, 0, 0, 0, 1, 1, 0, 0] / np.sqrt(2),
            U_tswap,
        ),
        (
            qml.QutritUnitary,
            np.multiply([0, 1, 1, 0, 0, 0, 0, 1, 1], 0.5),
            np.multiply([0, 0, 0, 1, 0, 1, 1, 0, 1], 0.5),
            U_tswap,
        ),
        (qml.QutritUnitary, [0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0], U_tadd),
        (qml.QutritUnitary, [0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0], U_tadd),
        (qml.QutritUnitary, [0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0], U_tadd),
    ]

    @pytest.mark.parametrize(
        "operation,input,expected_output,par", test_data_two_wires_with_parameters
    )
    def test_apply_operation_two_wires_with_parameters(
        self, qutrit_device_2_wires, tol, operation, input, expected_output, par
    ):
        """Tests that applying an operation yields the expected output state for two wire
        operations that have parameters."""

        qutrit_device_2_wires._state = np.array(input, dtype=qutrit_device_2_wires.C_DTYPE).reshape(
            (3, 3)
        )
        qutrit_device_2_wires.apply([operation(par, wires=[0, 1])])

        assert np.allclose(
            qutrit_device_2_wires._state.flatten(), np.array(expected_output), atol=tol, rtol=0
        )
        assert qutrit_device_2_wires._state.dtype == qutrit_device_2_wires.C_DTYPE

    @pytest.mark.parametrize(
        "operation,input,expected_output,par", test_data_two_wires_with_parameters_inverse
    )
    def test_apply_operation_two_wires_with_parameters_inverse(
        self, qutrit_device_2_wires, tol, operation, input, expected_output, par
    ):
        """Tests that applying an operation yields the expected output state for two wire
        operations that have parameters."""

        qutrit_device_2_wires._state = np.array(input, dtype=qutrit_device_2_wires.C_DTYPE).reshape(
            (3, 3)
        )
        qutrit_device_2_wires.apply([operation(par, wires=[0, 1]).inv()])

        assert np.allclose(
            qutrit_device_2_wires._state.flatten(), np.array(expected_output), atol=tol, rtol=0
        )
        assert qutrit_device_2_wires._state.dtype == qutrit_device_2_wires.C_DTYPE

    def test_apply_rotations_one_wire(self, qutrit_device_1_wire, tol):
        """Tests that rotations are applied in correct order after operations"""

        state = [1, 0, 0]
        qutrit_device_1_wire._state = np.array(state, dtype=qutrit_device_1_wire.C_DTYPE)

        ops = [
            qml.QutritUnitary(U_shift, wires=0).inv(),
            qml.QutritUnitary(U_thadamard_01, wires=0),
        ]
        rotations = [
            qml.QutritUnitary(U_thadamard_01, wires=0),
            qml.QutritUnitary(U_shift, wires=0),
        ]

        qutrit_device_1_wire.apply(ops, rotations)

        assert np.allclose(qutrit_device_1_wire._state.flatten(), state)

    # TODO: Add tests for state preperation ops after they're implemented


class TestDefaultQutritIntegration:
    """Integration tests for default.qutrit. This test ensures it integrates
    properly with the PennyLane interface, in particular QNode."""

    def test_defines_correct_capabilities(self):
        """Test that the device defines the right capabilities"""

        dev = qml.device("default.qutrit", wires=1)
        cap = dev.capabilities()
        capabilities = {
            "model": "qutrit",
            "supports_finite_shots": True,
            "supports_tensor_observables": True,
            "returns_probs": True,
            "returns_state": True,
            "supports_reversible_diff": True,
            "supports_inverse_operations": True,
            "supports_analytic_computation": True,
            "supports_broadcasting": False,
        }
        assert cap == capabilities

    three_wire_final_state = np.zeros(27)
    three_wire_final_state[0] = 1
    three_wire_final_state[4] = 1

    four_wire_final_state = np.zeros(81)
    four_wire_final_state[0] = 1
    four_wire_final_state[1] = 1
    four_wire_final_state[36] = 1
    four_wire_final_state[37] = 1
    state_measurement_data = [
        (1, U_thadamard_01, np.array([1, 1, 0]) / np.sqrt(2)),
        (2, U_tadd @ np.kron(U_shift, np.eye(3)), [0, 0, 0, 0, 1, 0, 0, 0, 0]),
        (1, U_shift, [0, 1, 0]),
        (
            2,
            U_tswap @ np.kron(U_thadamard_01, np.eye(3)),
            np.array([1, 1, 0, 0, 0, 0, 0, 0, 0]) / np.sqrt(2),
        ),
        (1, U_clock @ U_shift @ U_thadamard_01, np.array([0, OMEGA, OMEGA**2]) / np.sqrt(2)),
        (
            3,
            np.kron(np.eye(3), U_tadd) @ np.kron(np.eye(3), np.kron(U_thadamard_01, np.eye(3))),
            three_wire_final_state / np.sqrt(2),
        ),
        (
            4,
            np.kron(U_tadd, U_tswap)
            @ np.kron(U_thadamard_01, np.eye(27))
            @ np.kron(np.eye(9), np.kron(U_thadamard_01, np.eye(3))),
            four_wire_final_state / 2.0,
        ),
    ]

    @pytest.mark.parametrize("num_wires, mat, expected_out", state_measurement_data)
    def test_qutrit_circuit_state_measurement(self, num_wires, mat, expected_out, tol):
        """Tests if state returned by state function is correct"""
        dev = qml.device("default.qutrit", wires=num_wires)

        @qml.qnode(dev)
        def circuit(mat):
            qml.QutritUnitary(mat, wires=list(range(num_wires)))
            return qml.state()

        state = circuit(mat)
        assert np.allclose(state, expected_out, atol=tol)


class TestProbabilityIntegration:
    """Test probability method for when computation is/is not analytic"""

    def mock_analytic_counter(self, wires=None):
        self.analytic_counter += 1
        return np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)

    @pytest.mark.parametrize(
        "x", [[U_thadamard_01, U_clock], [U_shift, U_thadamard_01], [U_z_12, U_thadamard_01]]
    )
    def test_probability(self, x, tol):
        """Test that the probability function works for finite and infinite shots"""
        dev = qml.device("default.qutrit", wires=2, shots=1000)
        dev_analytic = qml.device("default.qutrit", wires=2, shots=None)

        def circuit(x):
            qml.QutritUnitary(x[0], wires=0)
            qml.QutritUnitary(x[1], wires=0)
            qml.QutritUnitary(U_tadd, wires=[0, 1])
            return qml.probs(wires=[0, 1])

        prob = qml.QNode(circuit, dev)
        prob_analytic = qml.QNode(circuit, dev_analytic)

        assert np.isclose(prob(x).sum(), 1, atol=tol, rtol=0)
        assert np.allclose(prob_analytic(x), prob(x), atol=0.1, rtol=0)
        assert not np.array_equal(prob_analytic(x), prob(x))

    def test_call_generate_samples(self, monkeypatch):
        """Test analytic_probability call when generating samples"""
        self.analytic_counter = False

        dev = qml.device("default.qutrit", wires=2, shots=1000)
        monkeypatch.setattr(dev, "analytic_probability", self.mock_analytic_counter)

        # generate samples through `generate_samples` (using 'analytic_probability')
        dev.generate_samples()

        # should call `analytic_probability` once through `generate_samples`
        assert self.analytic_counter == 1

    def test_stateless_analytic_return(self):
        """Test that analytic_probability returns None if device is stateless"""
        dev = qml.device("default.qutrit", wires=2)
        dev._state = None

        assert dev.analytic_probability() is None


class TestWiresIntegration:
    """Test that the device integrates with PennyLane's wire management."""

    def make_circuit_probs(self, wires):
        """Factory for a qnode returning probabilities using arbitrary wire labels."""
        dev = qml.device("default.qutrit", wires=wires)
        n_wires = len(wires)

        @qml.qnode(dev)
        def circuit():
            qml.QutritUnitary(U_shift, wires=wires[0 % n_wires])
            qml.QutritUnitary(U_clock, wires=wires[1 % n_wires])
            if n_wires > 1:
                qml.QutritUnitary(U_tswap, wires=[wires[0], wires[1]])
            return qml.probs(wires=wires)

        return circuit

    @pytest.mark.parametrize(
        "wires1, wires2",
        [
            (["a", "c", "d"], [2, 3, 0]),
            ([-1, -2, -3], ["q1", "ancilla", 2]),
            (["a", "c"], [3, 0]),
            ([-1, -2], ["ancilla", 2]),
            (["a"], ["nothing"]),
        ],
    )
    def test_wires_probs(self, wires1, wires2, tol):
        """Test that the probability vector of a circuit is independent from the wire labels used."""

        circuit1 = self.make_circuit_probs(wires1)
        circuit2 = self.make_circuit_probs(wires2)

        assert np.allclose(circuit1(), circuit2(), tol)

    def test_wires_not_found_exception(self):
        """Tests that an exception is raised when wires not present on the device are adressed."""
        dev = qml.device("default.qutrit", wires=["a", "b"])

        with qml.tape.QuantumTape() as tape:
            qml.QutritUnitary(np.eye(3), wires="c")

        with pytest.raises(WireError, match="Did not find some of the wires"):
            dev.execute(tape)

    wires_to_try = [
        (1, Wires([0]), Wires([0])),
        (4, Wires([1, 3]), Wires([1, 3])),
        (["a", 2], Wires([2]), Wires([1])),
        (["a", 2], Wires([2, "a"]), Wires([1, 0])),
    ]

    @pytest.mark.parametrize("dev_wires, wires_to_map, res", wires_to_try)
    def test_map_wires_caches(self, dev_wires, wires_to_map, res, mock_device):
        """Test that multiple calls to map_wires will use caching."""
        dev = qml.device("default.qutrit", wires=dev_wires)

        original_hits = dev.map_wires.cache_info().hits
        original_misses = dev.map_wires.cache_info().misses

        # The first call is computed: it's a miss as it didn't come from the cache
        dev.map_wires(wires_to_map)

        # The number of misses increased
        assert dev.map_wires.cache_info().misses > original_misses

        # The second call comes from the cache: it's a hit
        dev.map_wires(wires_to_map)

        # The number of hits increased
        assert dev.map_wires.cache_info().hits > original_hits


class TestApplyOperationUnit:
    """Unit tests for the internal _apply_operation method."""

    # TODO: Add test for testing operations with internal implementations

    @pytest.mark.parametrize("inverse", [True, False])
    def test_apply_tensordot_case(self, inverse, monkeypatch):
        """Tests the case when np.tensordot is used to apply an operation in
        default.qutrit."""
        dev = qml.device("default.qutrit", wires=2)

        test_state = np.array([1, 0, 0])
        wires = [0, 1]

        class TestSwap(qml.operation.Operation):
            num_wires = 2

            @staticmethod
            def compute_matrix(*params, **hyperparams):
                return U_tswap

        dev.operations.add("TestSwap")
        op = TestSwap(wires=wires)

        assert op.name in dev.operations
        assert op.name not in dev._apply_ops

        if inverse:
            op = op.inv()

        # Set the internal _apply_unitary_tensordot
        history = []
        mock_apply_tensordot = lambda state, matrix, wires: history.append((state, matrix, wires))

        with monkeypatch.context() as m:
            m.setattr(dev, "_apply_unitary", mock_apply_tensordot)

            dev._apply_operation(test_state, op)

            res_state, res_mat, res_wires = history[0]

            assert np.allclose(res_state, test_state)
            assert np.allclose(res_mat, op.matrix())
            assert np.allclose(res_wires, wires)

    def test_identity_skipped(self, mocker):
        """Test that applying the identity operation does not perform any additional computations"""
        dev = qml.device("default.qutrit", wires=1)

        starting_state = np.array([1, 0, 0])
        op = qml.Identity(0)

        spy_unitary = mocker.spy(dev, "_apply_unitary")

        res = dev._apply_operation(starting_state, op)

        assert res is starting_state
        spy_unitary.assert_not_called()

    @pytest.mark.parametrize("inverse", [True, False])
    def test_internal_apply_ops_case(self, inverse, monkeypatch, mocker):
        """Tests that if we provide an operation that has an internal
        implementation, then we use that specific implementation.

        This test provides a new internal function that `default.qutrit` uses to
        apply `QutritUnitary` (rather than redefining the gate itself).
        Note: `QutritUnitary` is not in `DefaultQutrit._apply_ops`, and will be
        temporarily added for this test.
        """
        dev = qml.device("default.qutrit", wires=1)

        # Create a dummy operation
        expected_test_output = np.ones(1)
        supported_gate_application = lambda *args, **kwargs: expected_test_output

        with monkeypatch.context() as m:
            # Set the internal ops implementations dict
            m.setattr(dev, "_apply_ops", {"QutritUnitary": supported_gate_application})

            test_state = np.array([1, 0, 0])
            op = (
                qml.QutritUnitary(U_shift, wires=0)
                if not inverse
                else qml.QutritUnitary(U_shift, wires=0).inv()
            )
            spy_unitary = mocker.spy(dev, "_apply_unitary")

            res = dev._apply_operation(test_state, op)
            assert np.allclose(res, expected_test_output)
            spy_unitary.assert_not_called()


class TestDensityMatrix:
    """Unit tests for the internal density_matrix method"""

    output = np.array([[0, 1, 0, 0, 1, 0, 0, 0, 0]]) / np.sqrt(2)
    output_0 = np.array([[1, 1, 0]]) / np.sqrt(2)
    output_1 = np.array([[0, 1, 0]])
    density_matrix_data = [
        (Wires([0, 1]), output.T @ output),
        (Wires([1, 0]), output.T @ output),
        (Wires([0]), output_0.T @ output_0),
        (Wires([1]), output_1.T @ output_1),
    ]

    @pytest.mark.parametrize("wires, expected", density_matrix_data)
    def test_density_matrix_all_wires(self, qutrit_device_2_wires, wires, expected):
        """Test that the density matrix is correct for the requested wires"""

        ops = [
            qml.QutritUnitary(U_thadamard_01, wires=0),
            qml.QutritUnitary(U_shift, wires=1),
        ]
        qutrit_device_2_wires.apply(ops)

        assert np.allclose(qutrit_device_2_wires.density_matrix(wires), expected)

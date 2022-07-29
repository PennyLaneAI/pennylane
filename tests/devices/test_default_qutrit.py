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

from gate_data import OMEGA, TSHIFT, TCLOCK, TSWAP, TADD, GELL_MANN


U_thadamard_01 = np.multiply(
    1 / np.sqrt(2),
    np.array(
        [[1, 1, 0], [1, -1, 0], [0, 0, np.sqrt(2)]],
    ),
)

U_x_02 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.complex128)

U_z_12 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.complex128)


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
        (qml.QutritUnitary, [0, 1, 0], [0, 0, 1], TSHIFT),
        (qml.QutritUnitary, [0, 1, 0], [0, OMEGA, 0], TCLOCK),
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
        (qml.QutritUnitary, [1, 0, 0], [0, 0, 1], TSHIFT),
        (qml.QutritUnitary, [0, 0, 1], [0, 1, 0], TSHIFT),
        (qml.QutritUnitary, [0, OMEGA, 0], [0, 1, 0], TCLOCK),
        (qml.QutritUnitary, [0, 0, OMEGA**2], [0, 0, 1], TCLOCK),
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
        (qml.QutritUnitary, [0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0], TSWAP),
        (qml.QutritUnitary, [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], TSWAP),
        (
            qml.QutritUnitary,
            [0, 0, 1, 0, 0, 0, 0, 1, 0] / np.sqrt(2),
            [0, 0, 0, 0, 0, 1, 1, 0, 0] / np.sqrt(2),
            TSWAP,
        ),
        (
            qml.QutritUnitary,
            np.multiply(0.5, [0, 1, 1, 0, 0, 0, 0, 1, 1]),
            np.multiply(0.5, [0, 0, 0, 1, 0, 1, 1, 0, 1]),
            TSWAP,
        ),
        (qml.QutritUnitary, [0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0], TADD),
        (qml.QutritUnitary, [0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0], TADD),
    ]

    # TODO: Add more ops as parametric operations get added
    test_data_two_wires_with_parameters_inverse = [
        (qml.QutritUnitary, [0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0], TSWAP),
        (qml.QutritUnitary, [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], TSWAP),
        (
            qml.QutritUnitary,
            [0, 0, 1, 0, 0, 0, 0, 1, 0] / np.sqrt(2),
            [0, 0, 0, 0, 0, 1, 1, 0, 0] / np.sqrt(2),
            TSWAP,
        ),
        (
            qml.QutritUnitary,
            np.multiply([0, 1, 1, 0, 0, 0, 0, 1, 1], 0.5),
            np.multiply([0, 0, 0, 1, 0, 1, 1, 0, 1], 0.5),
            TSWAP,
        ),
        (qml.QutritUnitary, [0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0], TADD),
        (qml.QutritUnitary, [0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0], TADD),
        (qml.QutritUnitary, [0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0], TADD),
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
        """Tests that applying the inverse of an operation yields the expected output state for two wire
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
            qml.QutritUnitary(TSHIFT, wires=0).inv(),
            qml.QutritUnitary(U_thadamard_01, wires=0),
        ]
        rotations = [
            qml.QutritUnitary(U_thadamard_01, wires=0),
            qml.QutritUnitary(TSHIFT, wires=0),
        ]

        qutrit_device_1_wire.apply(ops, rotations)

        assert np.allclose(qutrit_device_1_wire._state.flatten(), state)

    # TODO: Add tests for state preperation ops after they're implemented


class TestExpval:
    """Tests that expectation values are properly calculated or that the proper errors are raised."""

    @pytest.mark.parametrize(
        "observable,state,expected_output,mat",
        [
            (qml.THermitian, [1, 0, 0], 1, [[1, 1j, 0], [-1j, 1, 0], [0, 0, 1]]),
            (qml.THermitian, [0, 1, 0], -1, [[1, 0, 0], [0, -1, 0], [0, 0, 0]]),
            (
                qml.THermitian,
                [1 / math.sqrt(3), -1 / math.sqrt(3), 1j / math.sqrt(3)],
                0,
                [[0, -1j, 0], [1j, 0, 0], [0, 0, 0]],
            ),
            (qml.GellMannObs, [1, 0, 0], 0, 1),
            (qml.GellMannObs, [0, 0, 1], 0, 1),
            (qml.GellMannObs, [1 / math.sqrt(2), -1j / math.sqrt(2), 0], -1, 2),
            (qml.GellMannObs, [1, 0, 0], 0, 2),
            (qml.GellMannObs, [1, 0, 0], 1, 3),
            (qml.GellMannObs, [0, 1 / math.sqrt(2), 1 / math.sqrt(2)], -0.5, 3),
            (qml.GellMannObs, [1 / math.sqrt(2), 0, -1 / math.sqrt(2)], -1, 4),
            (qml.GellMannObs, [1 / math.sqrt(3), 1 / math.sqrt(3), 1 / math.sqrt(3)], 2 / 3, 4),
            (qml.GellMannObs, [1 / math.sqrt(2), 0, 1j / math.sqrt(2)], 1, 5),
            (qml.GellMannObs, [0, 1, 0], 0, 5),
            (qml.GellMannObs, [0, 0, 1], 0, 6),
            (qml.GellMannObs, [1 / math.sqrt(2), 1 / math.sqrt(2), 0], 0, 6),
            (qml.GellMannObs, [0, 1 / math.sqrt(2), 1j / math.sqrt(2)], 1, 7),
            (qml.GellMannObs, [0, 1 / math.sqrt(2), -1j / math.sqrt(2)], -1, 7),
            (qml.GellMannObs, [0, 0, 1], -2 / math.sqrt(3), 8),
            (qml.GellMannObs, [1 / math.sqrt(3), 1 / math.sqrt(3), 1 / math.sqrt(3)], 0, 8),
        ],
    )
    def test_expval_single_wire_with_parameters(
        self, qutrit_device_1_wire, tol, observable, state, expected_output, mat
    ):
        """Tests that expectation values are properly calculated for single-wire observables with parameters."""

        obs = (
            operation(par, wires=[0])
            if isinstance(par, int)
            else operation(np.array(par), wires=[0])
        )

        qutrit_device_1_wire.reset()
        qutrit_device_1_wire._state = np.array(state).reshape([3])
        qutrit_device_1_wire.apply([], obs.diagonalizing_gates())
        res = qutrit_device_1_wire.expval(obs)

        assert np.isclose(res, expected_output, atol=tol, rtol=0)

    A = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 2]])
    B = np.array([[4, 0, 0], [0, -2, 0], [0, 0, 1]])
    obs_1 = np.kron(A, B)

    C = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]])
    D = np.array([[1, 2, 3], [2, 1, 3], [3, 3, 2]])
    obs_2 = np.kron(C, D)

    @pytest.mark.parametrize(
        "observable,state,expected_output,mat",
        [
            (
                qml.THermitian,
                [1 / math.sqrt(3), 0, 1 / math.sqrt(3), 1 / math.sqrt(3), 0, 0, 0, 0, 0],
                1 / 3,
                obs_1,
            ),
            (
                qml.THermitian,
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
                2,
                obs_1,
            ),
            (
                qml.THermitian,
                [0.5, 0, 0, 0.5, 0, 0, 0, 0, 1 / math.sqrt(2)],
                1,
                obs_1,
            ),
            (
                qml.THermitian,
                [
                    3.73671170e-01 - 0.00000000e00j,
                    3.73671170e-01 - 8.75889651e-19j,
                    4.69829451e-01 + 7.59104364e-19j,
                    -2.74458036e-17 - 3.73671170e-01j,
                    -1.98254112e-18 - 3.73671170e-01j,
                    1.04702953e-17 - 4.69829451e-01j,
                    0.00000000e00 + 0.00000000e00j,
                    0.00000000e00 + 0.00000000e00j,
                    0.00000000e00 + 0.00000000e00j,
                ],
                -6.772,
                obs_2,
            ),
            (
                qml.THermitian,
                [1 / 3] * 9,
                0,
                obs_2,
            ),
            (
                qml.THermitian,
                [0, 0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0],
                0,
                obs_2,
            ),
        ],
    )
    def test_expval_two_wires_with_parameters(
        self, qutrit_device_2_wires, tol, observable, state, expected_output, mat
    ):
        """Tests that expectation values are properly calculated for two-wire observables with parameters."""

        obs = observable(np.array(mat), wires=[0, 1])

        qutrit_device_2_wires.reset()
        qutrit_device_2_wires._state = np.array(state).reshape([3] * 2)
        qutrit_device_2_wires.apply([], obs.diagonalizing_gates())
        res = qutrit_device_2_wires.expval(obs)
        assert np.isclose(res, expected_output, atol=tol, rtol=0)

    def test_expval_estimate(self):
        """Test that the expectation value is not analytically calculated"""

        dev = qml.device("default.qutrit", wires=1, shots=3)

        @qml.qnode(dev)
        def circuit():
            return qml.expval(qml.THermitian(np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]), wires=0))

        expval = circuit()

        # With 3 samples we are guaranteed to see a difference between
        # an estimated expectation value and an analytically calculated one
        assert not np.isclose(expval, 0.0)


class TestVar:
    """Tests that variances are properly calculated."""

    @pytest.mark.parametrize(
        "observable,state,expected_output,mat",
        [
            (qml.THermitian, [1, 0, 0], 1, [[1, 1j, 0], [-1j, 1, 0], [0, 0, 1]]),
            (qml.THermitian, [0, 1, 0], 1, [[1, 1j, 0], [-1j, 1, 0], [0, 0, 1]]),
            (
                qml.THermitian,
                [1 / math.sqrt(3), -1 / math.sqrt(3), 1j / math.sqrt(3)],
                2 / 3,
                [[1, 1j, 0], [-1j, 1, 0], [0, 0, 1]],
            ),
            (qml.GellMannObs, [1, 0, 0], 1, 1),
            (qml.GellMannObs, [0, 0, 1], 0, 1),
            (qml.GellMannObs, [1 / math.sqrt(2), -1j / math.sqrt(2), 0], 0, 2),
            (qml.GellMannObs, [1, 0, 0], 1, 2),
            (qml.GellMannObs, [1, 0, 0], 0, 3),
            (qml.GellMannObs, [0, 1 / math.sqrt(2), 1 / math.sqrt(2)], 0.25, 3),
            (qml.GellMannObs, [1 / math.sqrt(2), 0, -1 / math.sqrt(2)], 0, 4),
            (qml.GellMannObs, [1 / math.sqrt(3), 1 / math.sqrt(3), 1 / math.sqrt(3)], 2 / 9, 4),
            (qml.GellMannObs, [1 / math.sqrt(2), 0, 1j / math.sqrt(2)], 0, 5),
            (qml.GellMannObs, [0, 1, 0], 0, 5),
            (qml.GellMannObs, [0, 0, 1], 1, 6),
            (qml.GellMannObs, [1 / math.sqrt(2), 1 / math.sqrt(2), 0], 0.5, 6),
            (qml.GellMannObs, [0, 1 / math.sqrt(2), 1j / math.sqrt(2)], 0, 7),
            (qml.GellMannObs, [0, 1 / math.sqrt(2), -1j / math.sqrt(2)], 0, 7),
            (qml.GellMannObs, [0, 0, 1], 0, 8),
            (qml.GellMannObs, [1 / math.sqrt(3), 1 / math.sqrt(3), 1 / math.sqrt(3)], 2 / 3, 8),
        ],
    )
    def test_var_single_wire_with_parameters(
        self, qutrit_device_1_wire, tol, observable, state, expected_output, mat
    ):
        """Tests that variances are properly calculated for single-wire observables with parameters."""

        obs = (
            operation(par, wires=[0])
            if isinstance(par, int)
            else operation(np.array(par), wires=[0])
        )

        qutrit_device_1_wire.reset()
        qutrit_device_1_wire._state = np.array(state).reshape([3])
        qutrit_device_1_wire.apply([], obs.diagonalizing_gates())
        res = qutrit_device_1_wire.var(obs)

        assert np.isclose(res, expected_output, atol=tol, rtol=0)

    A = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 2]])
    B = np.array([[4, 0, 0], [0, -2, 0], [0, 0, 1]])
    obs_1 = np.kron(A, B)

    C = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]])
    D = np.array([[1, 2, 3], [2, 1, 3], [3, 3, 2]])
    obs_2 = np.kron(C, D)

    @pytest.mark.parametrize(
        "observable,state,expected_output,mat",
        [
            (
                qml.THermitian,
                [1 / math.sqrt(3), 0, 1 / math.sqrt(3), 1 / math.sqrt(3), 0, 0, 0, 0, 0],
                10.88888889,
                obs_1,
            ),
            (
                qml.THermitian,
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
                0,
                obs_1,
            ),
            (
                qml.THermitian,
                [0.5, 0, 0, 0.5, 0, 0, 0, 0, 1 / math.sqrt(2)],
                9,
                obs_1,
            ),
            (
                qml.THermitian,
                [0, 0, 1 / math.sqrt(2), 1 / math.sqrt(2), 0, 0, 0, 0, 0],
                18,
                obs_2,
            ),
            (
                qml.THermitian,
                [1 / 3] * 9,
                30.22222,
                obs_2,
            ),
            (
                qml.THermitian,
                [0, 1 / 2, 0, 1 / 2, 0, 1 / 2, 0, 1 / 2, 0],
                20,
                obs_2,
            ),
        ],
    )
    def test_var_two_wires_with_parameters(
        self, qutrit_device_2_wires, tol, observable, state, expected_output, mat
    ):
        """Tests that variances are properly calculated for two-wire observables with parameters."""

        obs = observable(np.array(mat), wires=[0, 1])

        qutrit_device_2_wires.reset()
        qutrit_device_2_wires._state = np.array(state).reshape([3] * 2)
        qutrit_device_2_wires.apply([], obs.diagonalizing_gates())
        res = qutrit_device_2_wires.var(obs)

        assert np.isclose(res, expected_output, atol=tol, rtol=0)

    def test_var_estimate(self):
        """Test that the var is not analytically calculated"""

        dev = qml.device("default.qutrit", wires=1, shots=3)

        @qml.qnode(dev)
        def circuit():
            return qml.var(qml.THermitian(np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]), wires=0))

        var = circuit()

        # With 3 samples we are guaranteed to see a difference between
        # an estimated variance and an analytically calculated one
        assert not np.isclose(var, 1.0)


class TestSample:
    """Tests that samples are properly calculated."""

    def test_sample_dimensions(self):
        """Tests if the samples returned by the sample function have
        the correct dimensions
        """

        # Explicitly resetting is necessary as the internal
        # state is set to None in __init__ and only properly
        # initialized during reset
        dev = qml.device("default.qutrit", wires=2, shots=1000)

        dev.apply([qml.QutritUnitary(TSHIFT, wires=0)])

        dev.shots = 10
        dev._wires_measured = {0}
        dev._samples = dev.generate_samples()
        s1 = dev.sample(qml.THermitian(np.eye(3), wires=0))
        assert np.array_equal(s1.shape, (10,))

        dev.reset()
        dev.shots = 12
        dev._wires_measured = {1}
        dev._samples = dev.generate_samples()
        s2 = dev.sample(qml.THermitian(np.eye(3), wires=1))
        assert np.array_equal(s2.shape, (12,))

        dev.reset()
        dev.shots = 17
        dev._wires_measured = {0, 1}
        dev._samples = dev.generate_samples()
        s3 = dev.sample(qml.THermitian(np.eye(3), wires=0) @ qml.THermitian(np.eye(3), wires=1))
        assert np.array_equal(s3.shape, (17,))

    def test_sample_values(self, tol):
        """Tests if the samples returned by sample have
        the correct values
        """

        # Explicitly resetting is necessary as the internal
        # state is set to None in __init__ and only properly
        # initialized during reset
        dev = qml.device("default.qutrit", wires=2, shots=1000)

        dev.apply([qml.QutritUnitary(TSHIFT, wires=0)])
        dev._wires_measured = {0}
        dev._samples = dev.generate_samples()

        s1 = dev.sample(qml.THermitian(np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]), wires=0))

        # s1 should only contain 1 and -1, which is guaranteed if
        # they square to 1
        assert np.allclose(s1**2, 1, atol=tol, rtol=0)


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
        (2, TADD @ np.kron(TSHIFT, np.eye(3)), [0, 0, 0, 0, 1, 0, 0, 0, 0]),
        (1, TSHIFT, [0, 1, 0]),
        (
            2,
            TSWAP @ np.kron(U_thadamard_01, np.eye(3)),
            np.array([1, 1, 0, 0, 0, 0, 0, 0, 0]) / np.sqrt(2),
        ),
        (1, TCLOCK @ TSHIFT @ U_thadamard_01, np.array([0, OMEGA, OMEGA**2]) / np.sqrt(2)),
        (
            3,
            np.kron(np.eye(3), TADD) @ np.kron(np.eye(3), np.kron(U_thadamard_01, np.eye(3))),
            three_wire_final_state / np.sqrt(2),
        ),
        (
            4,
            np.kron(TADD, TSWAP)
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


class TestTensorExpval:
    """Test tensor expectation values"""

    def test_hermitian_hermitian(self, tol):
        """Test that a tensor product involving two Hermitian matrices works correctly"""
        dev = qml.device("default.qutrit", wires=3)

        A1 = np.array([[1, 2, 3], [2, 1, 3], [3, 3, 2]])

        A = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]])
        B = np.array([[4, 0, 0], [0, -2, 0], [0, 0, 1]])
        A2 = np.kron(A, B)

        obs = qml.THermitian(A1, wires=[0]) @ qml.THermitian(A2, wires=[1, 2])

        dev.apply(
            [
                qml.QutritUnitary(TSHIFT, wires=0),
                qml.QutritUnitary(TADD, wires=[0, 1]),
                qml.QutritUnitary(TSHIFT, wires=0),
            ],
            obs.diagonalizing_gates(),
        )

        res = dev.expval(obs)

        expected = 0.0
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_hermitian_two_wires_identity_expectation(self, tol):
        """Test that a tensor product involving a Hermitian matrix for two wires and the identity works correctly"""
        dev = qml.device("default.qutrit", wires=3)

        A = np.array([[-2, 0, 0], [0, 8, 0], [0, 0, -1]])
        Identity = np.eye(3)
        H = np.kron(np.kron(Identity, Identity), A)
        obs = qml.THermitian(H, wires=[2, 1, 0])

        dev.apply(
            [
                qml.QutritUnitary(U_thadamard_01, wires=0),
                qml.QutritUnitary(TSHIFT, wires=0),
                qml.QutritUnitary(TSHIFT, wires=1),
                qml.QutritUnitary(TADD, wires=[0, 1]),
            ],
            obs.diagonalizing_gates(),
        )
        res = dev.expval(obs)

        expected = 3.5 * 1 * 1
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "index_1, eval_1", list(zip(list(range(1, 9)), [0, 0, -1, 0, 0, 0, 0, 1 / math.sqrt(3)]))
    )
    @pytest.mark.parametrize(
        "index_2, eval_2", list(zip(list(range(1, 9)), [0, 0, 0, 0, 0, 0, 0, -2 / math.sqrt(3)]))
    )
    def test_gell_mann_tensor(self, index_1, index_2, eval_1, eval_2, tol):
        """Test that the expectation value of the tensor product of two Gell-Mann observables is
        correct"""
        dev = qml.device("default.qutrit", wires=2)
        obs = qml.GellMannObs(index_1, wires=0) @ qml.GellMannObs(index_2, wires=1)

        dev.apply(
            [
                qml.QutritUnitary(TSHIFT, wires=0),
                qml.QutritUnitary(TSHIFT, wires=1),
                qml.QutritUnitary(TADD, wires=[0, 1]),
            ],
            obs.diagonalizing_gates(),
        )
        res = dev.expval(obs)

        expected = eval_1 * eval_2
        assert np.isclose(res, expected, atol=tol, rtol=0)


class TestTensorVar:
    """Tests for variance of tensor observables"""

    @pytest.mark.parametrize("index_1", list(range(1, 9)))
    @pytest.mark.parametrize("index_2", list(range(1, 9)))
    def test_gell_mann_tensor(self, index_1, index_2, tol):
        """Test that the variance of tensor Gell-Mann observables is correct"""
        dev = qml.device("default.qutrit", wires=2)
        obs = qml.GellMannObs(index_1, wires=0) @ qml.GellMannObs(index_2, wires=1)

        dev.apply(
            [
                qml.QutritUnitary(U_thadamard_01, wires=0),
                qml.QutritUnitary(TSHIFT, wires=0),
                qml.QutritUnitary(TSHIFT, wires=1),
                qml.QutritUnitary(TADD, wires=[0, 1]),  # (|12> + |20>) / sqrt(2)
            ],
            obs.diagonalizing_gates(),
        )
        res = dev.var(obs)

        state = np.array([[0, 0, 0, 0, 0, 1, 1, 0, 0]]) / math.sqrt(2)
        obs_mat = np.kron(GELL_MANN[index_1 - 1], GELL_MANN[index_2 - 1])

        expected = (
            state.conj() @ obs_mat @ obs_mat @ state.T - (state.conj() @ obs_mat @ state.T) ** 2
        )
        assert np.isclose(res, expected[0], atol=tol, rtol=0)

    def test_hermitian(self, tol):
        dev = qml.device("default.qutrit", wires=3)

        A1 = np.array([[3, 0, 0], [0, -3, 0], [0, 0, -2]])

        A = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 2]])
        B = np.array([[4, 0, 0], [0, -2, 0], [0, 0, 1]])
        A2 = np.kron(A, B)

        obs = qml.THermitian(A1, wires=[0]) @ qml.THermitian(A2, wires=[1, 2])

        dev.apply(
            [
                qml.QutritUnitary(TSHIFT, wires=0),
                qml.QutritUnitary(TADD, wires=[0, 1]),
                qml.QutritUnitary(TSHIFT, wires=0),
                qml.QutritUnitary(U_thadamard_01, wires=2),
            ],
            obs.diagonalizing_gates(),
        )

        res = dev.var(obs)
        expected = 36
        assert np.isclose(res, expected, atol=tol, rtol=0)


# TODO: Add tests for tensor non-parametrized observables
class TestTensorSample:
    pass


class TestProbabilityIntegration:
    """Test probability method for when computation is/is not analytic"""

    def mock_analytic_counter(self, wires=None):
        self.analytic_counter += 1
        return np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)

    @pytest.mark.parametrize(
        "x", [[U_thadamard_01, TCLOCK], [TSHIFT, U_thadamard_01], [U_z_12, U_thadamard_01]]
    )
    def test_probability(self, x, tol):
        """Test that the probability function works for finite and infinite shots"""
        dev = qml.device("default.qutrit", wires=2, shots=1000)
        dev_analytic = qml.device("default.qutrit", wires=2, shots=None)

        def circuit(x):
            qml.QutritUnitary(x[0], wires=0)
            qml.QutritUnitary(x[1], wires=0)
            qml.QutritUnitary(TADD, wires=[0, 1])
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
            qml.QutritUnitary(TSHIFT, wires=wires[0 % n_wires])
            qml.QutritUnitary(TCLOCK, wires=wires[1 % n_wires])
            if n_wires > 1:
                qml.QutritUnitary(TSWAP, wires=[wires[0], wires[1]])
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
        """Test that the probability vector of a circuit is independent of the wire labels used."""

        circuit1 = self.make_circuit_probs(wires1)
        circuit2 = self.make_circuit_probs(wires2)

        assert np.allclose(circuit1(), circuit2(), tol)

    def test_wires_not_found_exception(self):
        """Tests that an exception is raised when wires not present on the device are addressed."""
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


@pytest.mark.parametrize("inverse", [True, False])
class TestApplyOps:
    """Tests for special methods listed in _apply_ops that use array manipulation tricks to apply
    gates in DefaultQutrit."""

    state = np.arange(3**4, dtype=np.complex128).reshape((3, 3, 3, 3))
    dev = qml.device("default.qutrit", wires=4)

    single_qutrit_ops = [
        (qml.TShift, dev._apply_tshift),
        (qml.TClock, dev._apply_tclock),
    ]

    # TODO: Add tests for two-qutrit ops once they are added
    two_qutrit_ops = []

    @pytest.mark.parametrize("op, method", single_qutrit_ops)
    def test_apply_single_qutrit_op(self, op, method, inverse):
        """Test if the application of single qutrit operations is correct"""
        state_out = method(self.state, axes=[1], inverse=inverse)
        op = op(wires=[1])
        matrix = op.inv().matrix() if inverse else op.matrix()
        state_out_einsum = np.einsum("ab,ibjk->iajk", matrix, self.state)
        assert np.allclose(state_out, state_out_einsum)

    # TODO: Add tests for two-qutrit operations


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
                return TSWAP

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
                qml.QutritUnitary(TSHIFT, wires=0)
                if not inverse
                else qml.QutritUnitary(TSHIFT, wires=0).inv()
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
            qml.QutritUnitary(TSHIFT, wires=1),
        ]
        qutrit_device_2_wires.apply(ops)

        assert np.allclose(qutrit_device_2_wires.density_matrix(wires), expected)

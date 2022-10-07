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
Unit tests for the :mod:`pennylane.plugin.DefaultQubit` device.
"""
import cmath

# pylint: disable=protected-access,cell-var-from-loop
import math

import pytest
import pennylane as qml
from pennylane import numpy as np, DeviceError
from pennylane.devices.default_qubit import _get_slice, DefaultQubit
from pennylane.wires import Wires, WireError

U = np.array(
    [
        [0.83645892 - 0.40533293j, -0.20215326 + 0.30850569j],
        [-0.23889780 - 0.28101519j, -0.88031770 - 0.29832709j],
    ]
)

U2 = np.array(
    [
        [
            -0.07843244 - 3.57825948e-01j,
            0.71447295 - 5.38069384e-02j,
            0.20949966 + 6.59100734e-05j,
            -0.50297381 + 2.35731613e-01j,
        ],
        [
            -0.26626692 + 4.53837083e-01j,
            0.27771991 - 2.40717436e-01j,
            0.41228017 - 1.30198687e-01j,
            0.01384490 - 6.33200028e-01j,
        ],
        [
            -0.69254712 - 2.56963068e-02j,
            -0.15484858 + 6.57298384e-02j,
            -0.53082141 + 7.18073414e-02j,
            -0.41060450 - 1.89462315e-01j,
        ],
        [
            -0.09686189 - 3.15085273e-01j,
            -0.53241387 - 1.99491763e-01j,
            0.56928622 + 3.97704398e-01j,
            -0.28671074 - 6.01574497e-02j,
        ],
    ]
)

U_toffoli = np.diag([1 for i in range(8)])
U_toffoli[6:8, 6:8] = np.array([[0, 1], [1, 0]])

U_swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

U_cswap = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ]
)

H = np.array([[1.02789352, 1.61296440 - 0.3498192j], [1.61296440 + 0.3498192j, 1.23920938 + 0j]])

THETA = np.linspace(0.11, 1, 3)
PHI = np.linspace(0.32, 1, 3)
VARPHI = np.linspace(0.02, 1, 3)


def test_analytic_deprecation():
    """Tests if the kwarg `analytic` is used and displays error message."""
    msg = "The analytic argument has been replaced by shots=None. "
    msg += "Please use shots=None instead of analytic=True."

    with pytest.raises(
        DeviceError,
        match=msg,
    ):
        qml.device("default.qubit", wires=1, shots=1, analytic=True)


def test_dtype_errors():
    """Test that if an incorrect dtype is provided to the device then an error is raised."""
    with pytest.raises(DeviceError, match="Real datatype must be a floating point type."):
        qml.device("default.qubit", wires=1, r_dtype=np.complex128)
    with pytest.raises(
        DeviceError, match="Complex datatype must be a complex floating point type."
    ):
        qml.device("default.qubit", wires=1, c_dtype=np.float64)


def test_custom_op_with_matrix():
    """Test that a dummy op with a matrix is supported."""

    class DummyOp(qml.operation.Operation):
        num_wires = 1

        def compute_matrix(self):
            return np.eye(2)

    with qml.tape.QuantumTape() as tape:
        DummyOp(0)
        qml.state()

    dev = qml.device("default.qubit", wires=1)
    assert qml.math.allclose(dev.execute(tape), np.array([1, 0]))


class TestApply:
    """Tests that operations and inverses of certain operations are applied correctly or that the proper
    errors are raised.
    """

    test_data_no_parameters = [
        (qml.PauliX, [1, 0], np.array([0, 1])),
        (qml.PauliX, [1 / math.sqrt(2), 1 / math.sqrt(2)], [1 / math.sqrt(2), 1 / math.sqrt(2)]),
        (qml.PauliY, [1, 0], [0, 1j]),
        (qml.PauliY, [1 / math.sqrt(2), 1 / math.sqrt(2)], [-1j / math.sqrt(2), 1j / math.sqrt(2)]),
        (qml.PauliZ, [1, 0], [1, 0]),
        (qml.PauliZ, [1 / math.sqrt(2), 1 / math.sqrt(2)], [1 / math.sqrt(2), -1 / math.sqrt(2)]),
        (qml.S, [1, 0], [1, 0]),
        (qml.S, [1 / math.sqrt(2), 1 / math.sqrt(2)], [1 / math.sqrt(2), 1j / math.sqrt(2)]),
        (qml.T, [1, 0], [1, 0]),
        (
            qml.T,
            [1 / math.sqrt(2), 1 / math.sqrt(2)],
            [1 / math.sqrt(2), np.exp(1j * np.pi / 4) / math.sqrt(2)],
        ),
        (qml.Hadamard, [1, 0], [1 / math.sqrt(2), 1 / math.sqrt(2)]),
        (qml.Hadamard, [1 / math.sqrt(2), -1 / math.sqrt(2)], [0, 1]),
        (qml.Identity, [1, 0], [1, 0]),
        (qml.Identity, [1 / math.sqrt(2), 1 / math.sqrt(2)], [1 / math.sqrt(2), 1 / math.sqrt(2)]),
    ]

    test_data_no_parameters_inverses = [
        (qml.PauliX, [1, 0], np.array([0, 1])),
        (qml.PauliX, [1 / math.sqrt(2), 1 / math.sqrt(2)], [1 / math.sqrt(2), 1 / math.sqrt(2)]),
        (qml.PauliY, [1, 0], [0, 1j]),
        (qml.PauliY, [1 / math.sqrt(2), 1 / math.sqrt(2)], [-1j / math.sqrt(2), 1j / math.sqrt(2)]),
        (qml.PauliZ, [1, 0], [1, 0]),
        (qml.PauliZ, [1 / math.sqrt(2), 1 / math.sqrt(2)], [1 / math.sqrt(2), -1 / math.sqrt(2)]),
        (qml.S, [1, 0], [1, 0]),
        (qml.S, [1 / math.sqrt(2), 1 / math.sqrt(2)], [1 / math.sqrt(2), -1j / math.sqrt(2)]),
        (qml.T, [1, 0], [1, 0]),
        (
            qml.T,
            [1 / math.sqrt(2), 1 / math.sqrt(2)],
            [1 / math.sqrt(2), np.exp(-1j * np.pi / 4) / math.sqrt(2)],
        ),
        (qml.Hadamard, [1, 0], [1 / math.sqrt(2), 1 / math.sqrt(2)]),
        (qml.Hadamard, [1 / math.sqrt(2), -1 / math.sqrt(2)], [0, 1]),
        (qml.Identity, [1, 0], [1, 0]),
        (qml.Identity, [1 / math.sqrt(2), 1 / math.sqrt(2)], [1 / math.sqrt(2), 1 / math.sqrt(2)]),
    ]

    @pytest.mark.parametrize("operation,input,expected_output", test_data_no_parameters)
    def test_apply_operation_single_wire_no_parameters(
        self, qubit_device_1_wire, tol, operation, input, expected_output
    ):
        """Tests that applying an operation yields the expected output state for single wire
        operations that have no parameters."""

        qubit_device_1_wire._state = np.array(input, dtype=qubit_device_1_wire.C_DTYPE)
        qubit_device_1_wire.apply([operation(wires=[0])])

        assert np.allclose(qubit_device_1_wire._state, np.array(expected_output), atol=tol, rtol=0)
        assert qubit_device_1_wire._state.dtype == qubit_device_1_wire.C_DTYPE

    @pytest.mark.parametrize("operation,input,expected_output", test_data_no_parameters_inverses)
    def test_apply_operation_single_wire_no_parameters_inverse(
        self, qubit_device_1_wire, tol, operation, input, expected_output
    ):
        """Tests that applying an operation yields the expected output state for single wire
        operations that have no parameters."""

        qubit_device_1_wire._state = np.array(input, dtype=qubit_device_1_wire.C_DTYPE)
        qubit_device_1_wire.apply([operation(wires=[0]).inv()])

        assert np.allclose(qubit_device_1_wire._state, np.array(expected_output), atol=tol, rtol=0)
        assert qubit_device_1_wire._state.dtype == qubit_device_1_wire.C_DTYPE

    test_data_two_wires_no_parameters = [
        (qml.CNOT, [1, 0, 0, 0], [1, 0, 0, 0]),
        (qml.CNOT, [0, 0, 1, 0], [0, 0, 0, 1]),
        (
            qml.CNOT,
            [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)],
            [1 / math.sqrt(2), 0, 1 / math.sqrt(2), 0],
        ),
        (qml.SWAP, [1, 0, 0, 0], [1, 0, 0, 0]),
        (qml.SWAP, [0, 0, 1, 0], [0, 1, 0, 0]),
        (
            qml.SWAP,
            [1 / math.sqrt(2), 0, -1 / math.sqrt(2), 0],
            [1 / math.sqrt(2), -1 / math.sqrt(2), 0, 0],
        ),
        (qml.CZ, [1, 0, 0, 0], [1, 0, 0, 0]),
        (qml.CZ, [0, 0, 0, 1], [0, 0, 0, -1]),
        (
            qml.CZ,
            [1 / math.sqrt(2), 0, 0, -1 / math.sqrt(2)],
            [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)],
        ),
    ]

    test_data_iswap = [
        (qml.ISWAP, [1, 0, 0, 0], [1, 0, 0, 0]),
        (qml.ISWAP, [0, 0, 1, 0], [0, 1j, 0, 0]),
        (
            qml.ISWAP,
            [1 / math.sqrt(2), 0, -1 / math.sqrt(2), 0],
            [1 / math.sqrt(2), -1j / math.sqrt(2), 0, 0],
        ),
    ]

    test_data_iswap_inv = [
        (qml.ISWAP, [1, 0, 0, 0], [1, 0, 0, 0]),
        (qml.ISWAP, [0, 0, 1, 0], [0, -1j, 0, 0]),
        (
            qml.ISWAP,
            [1 / math.sqrt(2), 0, -1 / math.sqrt(2), 0],
            [1 / math.sqrt(2), 1j / math.sqrt(2), 0, 0],
        ),
    ]

    test_data_siswap = [
        (qml.SISWAP, [1, 0, 0, 0], [1, 0, 0, 0]),
        (qml.SISWAP, [0, 1, 0, 0], [0, 1 / math.sqrt(2), 1 / math.sqrt(2) * 1j, 0]),
        (
            qml.SISWAP,
            [1 / math.sqrt(2), 1 / math.sqrt(2), 0, 0],
            [1 / math.sqrt(2), 0.5, 0.5 * 1j, 0],
        ),
    ]

    test_data_sqisw = [
        (qml.SQISW, [1, 0, 0, 0], [1, 0, 0, 0]),
        (qml.SQISW, [0, 1, 0, 0], [0, 1 / math.sqrt(2), 1 / math.sqrt(2) * 1j, 0]),
        (
            qml.SQISW,
            [1 / math.sqrt(2), 1 / math.sqrt(2), 0, 0],
            [1 / math.sqrt(2), 0.5, 0.5 * 1j, 0],
        ),
    ]

    test_data_siswap_inv = [
        (
            qml.SISWAP,
            [1 / math.sqrt(2), 0, 1 / math.sqrt(2), 0],
            [1 / math.sqrt(2), -0.5 * 1j, 0.5, 0],
        ),
        (qml.SISWAP, [0, 0, 1, 0], [0, -1 / math.sqrt(2) * 1j, 1 / math.sqrt(2), 0]),
        (
            qml.SISWAP,
            [1 / math.sqrt(2), 0, -1 / math.sqrt(2), 0],
            [1 / math.sqrt(2), 0.5 * 1j, -0.5, 0],
        ),
    ]

    test_data_sqisw_inv = [
        (
            qml.SQISW,
            [1 / math.sqrt(2), 0, 1 / math.sqrt(2), 0],
            [1 / math.sqrt(2), -0.5 * 1j, 0.5, 0],
        ),
        (qml.SQISW, [0, 0, 1, 0], [0, -1 / math.sqrt(2) * 1j, 1 / math.sqrt(2), 0]),
        (
            qml.SQISW,
            [1 / math.sqrt(2), 0, -1 / math.sqrt(2), 0],
            [1 / math.sqrt(2), 0.5 * 1j, -0.5, 0],
        ),
    ]

    all_two_wires_no_parameters = (
        test_data_two_wires_no_parameters + test_data_iswap + test_data_siswap + test_data_sqisw
    )

    @pytest.mark.parametrize("operation,input,expected_output", all_two_wires_no_parameters)
    def test_apply_operation_two_wires_no_parameters(
        self, qubit_device_2_wires, tol, operation, input, expected_output
    ):
        """Tests that applying an operation yields the expected output state for two wire
        operations that have no parameters."""

        qubit_device_2_wires._state = np.array(input, dtype=qubit_device_2_wires.C_DTYPE).reshape(
            (2, 2)
        )
        qubit_device_2_wires.apply([operation(wires=[0, 1])])

        assert np.allclose(
            qubit_device_2_wires._state.flatten(), np.array(expected_output), atol=tol, rtol=0
        )
        assert qubit_device_2_wires._state.dtype == qubit_device_2_wires.C_DTYPE

    all_two_wires_no_parameters_inv = (
        test_data_two_wires_no_parameters
        + test_data_iswap_inv
        + test_data_siswap_inv
        + test_data_sqisw_inv
    )

    @pytest.mark.parametrize("operation,input,expected_output", all_two_wires_no_parameters_inv)
    def test_apply_operation_two_wires_no_parameters_inverse(
        self, qubit_device_2_wires, tol, operation, input, expected_output
    ):
        """Tests that applying an operation yields the expected output state for two wire
        operations that have no parameters."""

        qubit_device_2_wires._state = np.array(input, dtype=qubit_device_2_wires.C_DTYPE).reshape(
            (2, 2)
        )
        qubit_device_2_wires.apply([operation(wires=[0, 1]).inv()])

        assert np.allclose(
            qubit_device_2_wires._state.flatten(),
            np.array(expected_output),
            atol=tol,
            rtol=0,
        )
        assert qubit_device_2_wires._state.dtype == qubit_device_2_wires.C_DTYPE

    test_data_three_wires_no_parameters = [
        (qml.CSWAP, [1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0]),
        (qml.CSWAP, [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0]),
        (qml.CSWAP, [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0, 0]),
    ]

    @pytest.mark.parametrize("operation,input,expected_output", test_data_three_wires_no_parameters)
    def test_apply_operation_three_wires_no_parameters(
        self, qubit_device_3_wires, tol, operation, input, expected_output
    ):
        """Tests that applying an operation yields the expected output state for three wire
        operations that have no parameters."""

        qubit_device_3_wires._state = np.array(input, dtype=qubit_device_3_wires.C_DTYPE).reshape(
            (2, 2, 2)
        )
        qubit_device_3_wires.apply([operation(wires=[0, 1, 2])])

        assert np.allclose(
            qubit_device_3_wires._state.flatten(), np.array(expected_output), atol=tol, rtol=0
        )
        assert qubit_device_3_wires._state.dtype == qubit_device_3_wires.C_DTYPE

    @pytest.mark.parametrize("operation,input,expected_output", test_data_three_wires_no_parameters)
    def test_apply_operation_three_wires_no_parameters_inverse(
        self, qubit_device_3_wires, tol, operation, input, expected_output
    ):
        """Tests that applying the inverse of an operation yields the expected output state for three wire
        operations that have no parameters."""

        qubit_device_3_wires._state = np.array(input, dtype=qubit_device_3_wires.C_DTYPE).reshape(
            (2, 2, 2)
        )
        qubit_device_3_wires.apply([operation(wires=[0, 1, 2]).inv()])

        assert np.allclose(
            qubit_device_3_wires._state.flatten(), np.array(expected_output), atol=tol, rtol=0
        )
        assert qubit_device_3_wires._state.dtype == qubit_device_3_wires.C_DTYPE

    @pytest.mark.parametrize(
        "operation,expected_output,par",
        [
            (qml.BasisState, [0, 0, 1, 0], [1, 0]),
            (qml.BasisState, [0, 0, 1, 0], [1, 0]),
            (qml.BasisState, [0, 0, 0, 1], [1, 1]),
            (qml.QubitStateVector, [0, 0, 1, 0], [0, 0, 1, 0]),
            (qml.QubitStateVector, [0, 0, 1, 0], [0, 0, 1, 0]),
            (qml.QubitStateVector, [0, 0, 0, 1], [0, 0, 0, 1]),
            (
                qml.QubitStateVector,
                [1 / math.sqrt(3), 0, 1 / math.sqrt(3), 1 / math.sqrt(3)],
                [1 / math.sqrt(3), 0, 1 / math.sqrt(3), 1 / math.sqrt(3)],
            ),
            (
                qml.QubitStateVector,
                [1 / math.sqrt(3), 0, -1 / math.sqrt(3), 1 / math.sqrt(3)],
                [1 / math.sqrt(3), 0, -1 / math.sqrt(3), 1 / math.sqrt(3)],
            ),
        ],
    )
    def test_apply_operation_state_preparation(
        self, qubit_device_2_wires, tol, operation, expected_output, par
    ):
        """Tests that applying an operation yields the expected output state for single wire
        operations that have no parameters."""

        par = np.array(par)
        qubit_device_2_wires.reset()
        qubit_device_2_wires.apply([operation(par, wires=[0, 1])])

        assert np.allclose(
            qubit_device_2_wires._state.flatten(), np.array(expected_output), atol=tol, rtol=0
        )

    test_data_single_wire_with_parameters = [
        (qml.PhaseShift, [1, 0], [1, 0], [math.pi / 2]),
        (qml.PhaseShift, [0, 1], [0, 1j], [math.pi / 2]),
        (
            qml.PhaseShift,
            [1 / math.sqrt(2), 1 / math.sqrt(2)],
            [1 / math.sqrt(2), 1 / 2 + 1j / 2],
            [math.pi / 4],
        ),
        (qml.RX, [1, 0], [1 / math.sqrt(2), -1j * 1 / math.sqrt(2)], [math.pi / 2]),
        (qml.RX, [1, 0], [0, -1j], [math.pi]),
        (
            qml.RX,
            [1 / math.sqrt(2), 1 / math.sqrt(2)],
            [1 / 2 - 1j / 2, 1 / 2 - 1j / 2],
            [math.pi / 2],
        ),
        (qml.RY, [1, 0], [1 / math.sqrt(2), 1 / math.sqrt(2)], [math.pi / 2]),
        (qml.RY, [1, 0], [0, 1], [math.pi]),
        (qml.RY, [1 / math.sqrt(2), 1 / math.sqrt(2)], [0, 1], [math.pi / 2]),
        (qml.RZ, [1, 0], [1 / math.sqrt(2) - 1j / math.sqrt(2), 0], [math.pi / 2]),
        (qml.RZ, [0, 1], [0, 1j], [math.pi]),
        (
            qml.RZ,
            [1 / math.sqrt(2), 1 / math.sqrt(2)],
            [1 / 2 - 1j / 2, 1 / 2 + 1j / 2],
            [math.pi / 2],
        ),
        (qml.MultiRZ, [1, 0], [1 / math.sqrt(2) - 1j / math.sqrt(2), 0], [math.pi / 2]),
        (qml.MultiRZ, [0, 1], [0, 1j], [math.pi]),
        (
            qml.MultiRZ,
            [1 / math.sqrt(2), 1 / math.sqrt(2)],
            [1 / 2 - 1j / 2, 1 / 2 + 1j / 2],
            [math.pi / 2],
        ),
        (qml.Rot, [1, 0], [1 / math.sqrt(2) - 1j / math.sqrt(2), 0], [math.pi / 2, 0, 0]),
        (qml.Rot, [1, 0], [1 / math.sqrt(2), 1 / math.sqrt(2)], [0, math.pi / 2, 0]),
        (
            qml.Rot,
            [1 / math.sqrt(2), 1 / math.sqrt(2)],
            [1 / 2 - 1j / 2, 1 / 2 + 1j / 2],
            [0, 0, math.pi / 2],
        ),
        (
            qml.Rot,
            [1, 0],
            [-1j / math.sqrt(2), -1 / math.sqrt(2)],
            [math.pi / 2, -math.pi / 2, math.pi / 2],
        ),
        (
            qml.Rot,
            [1 / math.sqrt(2), 1 / math.sqrt(2)],
            [1 / 2 + 1j / 2, -1 / 2 + 1j / 2],
            [-math.pi / 2, math.pi, math.pi],
        ),
        (
            qml.QubitUnitary,
            [1, 0],
            [1j / math.sqrt(2), 1j / math.sqrt(2)],
            [
                np.array(
                    [
                        [1j / math.sqrt(2), 1j / math.sqrt(2)],
                        [1j / math.sqrt(2), -1j / math.sqrt(2)],
                    ]
                )
            ],
        ),
        (
            qml.QubitUnitary,
            [0, 1],
            [1j / math.sqrt(2), -1j / math.sqrt(2)],
            [
                np.array(
                    [
                        [1j / math.sqrt(2), 1j / math.sqrt(2)],
                        [1j / math.sqrt(2), -1j / math.sqrt(2)],
                    ]
                )
            ],
        ),
        (
            qml.QubitUnitary,
            [1 / math.sqrt(2), -1 / math.sqrt(2)],
            [0, 1j],
            [
                np.array(
                    [
                        [1j / math.sqrt(2), 1j / math.sqrt(2)],
                        [1j / math.sqrt(2), -1j / math.sqrt(2)],
                    ]
                )
            ],
        ),
        (qml.DiagonalQubitUnitary, [1, 0], [-1, 0], [np.array([-1, 1])]),
        (
            qml.DiagonalQubitUnitary,
            [1 / math.sqrt(2), 1 / math.sqrt(2)],
            [1 / math.sqrt(2), 1j / math.sqrt(2)],
            [np.array([1, 1j])],
        ),
        (
            qml.DiagonalQubitUnitary,
            [1 / 2, math.sqrt(3) / 4],
            [cmath.exp(1j * 0.4) / 2, cmath.exp(1j * -0.4) * math.sqrt(3) / 4],
            [np.array([cmath.exp(1j * 0.4), cmath.exp(1j * -0.4)])],
        ),
    ]

    test_data_single_wire_with_parameters_inverses = [
        (qml.PhaseShift, [1, 0], [1, 0], [math.pi / 2]),
        (qml.PhaseShift, [0, 1], [0, -1j], [math.pi / 2]),
        (
            qml.PhaseShift,
            [1 / math.sqrt(2), 1 / math.sqrt(2)],
            [1 / math.sqrt(2), 1 / 2 - 1j / 2],
            [math.pi / 4],
        ),
        (qml.RX, [1, 0], [1 / math.sqrt(2), 1j * 1 / math.sqrt(2)], [math.pi / 2]),
        (qml.RX, [1, 0], [0, 1j], [math.pi]),
        (
            qml.RX,
            [1 / math.sqrt(2), 1 / math.sqrt(2)],
            [1 / 2 + 1j / 2, 1 / 2 + 1j / 2],
            [math.pi / 2],
        ),
        (qml.RY, [1, 0], [1 / math.sqrt(2), -1 / math.sqrt(2)], [math.pi / 2]),
        (qml.RY, [1, 0], [0, -1], [math.pi]),
        (qml.RY, [1 / math.sqrt(2), 1 / math.sqrt(2)], [1, 0], [math.pi / 2]),
        (qml.RZ, [1, 0], [1 / math.sqrt(2) + 1j / math.sqrt(2), 0], [math.pi / 2]),
        (qml.RZ, [0, 1], [0, -1j], [math.pi]),
        (
            qml.RZ,
            [1 / math.sqrt(2), 1 / math.sqrt(2)],
            [1 / 2 + 1 / 2 * 1j, 1 / 2 - 1 / 2 * 1j],
            [math.pi / 2],
        ),
        (qml.MultiRZ, [1, 0], [1 / math.sqrt(2) + 1j / math.sqrt(2), 0], [math.pi / 2]),
        (qml.MultiRZ, [0, 1], [0, -1j], [math.pi]),
        (
            qml.MultiRZ,
            [1 / math.sqrt(2), 1 / math.sqrt(2)],
            [1 / 2 + 1 / 2 * 1j, 1 / 2 - 1 / 2 * 1j],
            [math.pi / 2],
        ),
        (qml.DiagonalQubitUnitary, [1, 0], [-1, 0], [np.array([-1, 1])]),
        (
            qml.DiagonalQubitUnitary,
            [1 / math.sqrt(2), 1 / math.sqrt(2)],
            [1 / math.sqrt(2), -1j / math.sqrt(2)],
            [np.array([1, 1j])],
        ),
        (
            qml.DiagonalQubitUnitary,
            [1 / 2, math.sqrt(3) / 4],
            [cmath.exp(-1j * 0.4) / 2, cmath.exp(1j * 0.4) * math.sqrt(3) / 4],
            [np.array([cmath.exp(1j * 0.4), cmath.exp(1j * -0.4)])],
        ),
    ]

    @pytest.mark.parametrize(
        "operation,input,expected_output,par", test_data_single_wire_with_parameters
    )
    def test_apply_operation_single_wire_with_parameters(
        self, qubit_device_1_wire, tol, operation, input, expected_output, par
    ):
        """Tests that applying an operation yields the expected output state for single wire
        operations that have parameters."""

        qubit_device_1_wire._state = np.array(input, dtype=qubit_device_1_wire.C_DTYPE)

        qubit_device_1_wire.apply([operation(*par, wires=[0])])

        assert np.allclose(qubit_device_1_wire._state, np.array(expected_output), atol=tol, rtol=0)
        assert qubit_device_1_wire._state.dtype == qubit_device_1_wire.C_DTYPE

    @pytest.mark.parametrize(
        "operation,input,expected_output,par", test_data_single_wire_with_parameters_inverses
    )
    def test_apply_operation_single_wire_with_parameters_inverse(
        self, qubit_device_1_wire, tol, operation, input, expected_output, par
    ):
        """Tests that applying the inverse of an operation yields the expected output state for single wire
        operations that have parameters."""

        qubit_device_1_wire._state = np.array(input, dtype=qubit_device_1_wire.C_DTYPE)
        qubit_device_1_wire.apply([operation(*par, wires=[0]).inv()])

        assert np.allclose(qubit_device_1_wire._state, np.array(expected_output), atol=tol, rtol=0)
        assert qubit_device_1_wire._state.dtype == qubit_device_1_wire.C_DTYPE

    test_data_two_wires_with_parameters = [
        (qml.CRX, [0, 1, 0, 0], [0, 1, 0, 0], [math.pi / 2]),
        (qml.CRX, [0, 0, 0, 1], [0, 0, -1j, 0], [math.pi]),
        (
            qml.CRX,
            [0, 1 / math.sqrt(2), 1 / math.sqrt(2), 0],
            [0, 1 / math.sqrt(2), 1 / 2, -1j / 2],
            [math.pi / 2],
        ),
        (qml.CRY, [0, 0, 0, 1], [0, 0, -1 / math.sqrt(2), 1 / math.sqrt(2)], [math.pi / 2]),
        (qml.CRY, [0, 0, 0, 1], [0, 0, -1, 0], [math.pi]),
        (
            qml.CRY,
            [1 / math.sqrt(2), 1 / math.sqrt(2), 0, 0],
            [1 / math.sqrt(2), 1 / math.sqrt(2), 0, 0],
            [math.pi / 2],
        ),
        (qml.CRZ, [0, 0, 0, 1], [0, 0, 0, 1 / math.sqrt(2) + 1j / math.sqrt(2)], [math.pi / 2]),
        (qml.CRZ, [0, 0, 0, 1], [0, 0, 0, 1j], [math.pi]),
        (
            qml.CRZ,
            [1 / math.sqrt(2), 1 / math.sqrt(2), 0, 0],
            [1 / math.sqrt(2), 1 / math.sqrt(2), 0, 0],
            [math.pi / 2],
        ),
        (qml.MultiRZ, [0, 0, 0, 1], [0, 0, 0, 1 / math.sqrt(2) - 1j / math.sqrt(2)], [math.pi / 2]),
        (qml.MultiRZ, [0, 0, 1, 0], [0, 0, 1j, 0], [math.pi]),
        (
            qml.MultiRZ,
            [1 / math.sqrt(2), 1 / math.sqrt(2), 0, 0],
            [1 / 2 - 1j / 2, 1 / 2 + 1j / 2, 0, 0],
            [math.pi / 2],
        ),
        (
            qml.CRot,
            [0, 0, 0, 1],
            [0, 0, 0, 1 / math.sqrt(2) + 1j / math.sqrt(2)],
            [math.pi / 2, 0, 0],
        ),
        (qml.CRot, [0, 0, 0, 1], [0, 0, -1 / math.sqrt(2), 1 / math.sqrt(2)], [0, math.pi / 2, 0]),
        (
            qml.CRot,
            [0, 0, 1 / math.sqrt(2), 1 / math.sqrt(2)],
            [0, 0, 1 / 2 - 1j / 2, 1 / 2 + 1j / 2],
            [0, 0, math.pi / 2],
        ),
        (
            qml.CRot,
            [0, 0, 0, 1],
            [0, 0, 1 / math.sqrt(2), 1j / math.sqrt(2)],
            [math.pi / 2, -math.pi / 2, math.pi / 2],
        ),
        (
            qml.CRot,
            [0, 1 / math.sqrt(2), 1 / math.sqrt(2), 0],
            [0, 1 / math.sqrt(2), 0, -1 / 2 + 1j / 2],
            [-math.pi / 2, math.pi, math.pi],
        ),
        (
            qml.QubitUnitary,
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [
                np.array(
                    [
                        [1, 0, 0, 0],
                        [0, 1 / math.sqrt(2), 1 / math.sqrt(2), 0],
                        [0, 1 / math.sqrt(2), -1 / math.sqrt(2), 0],
                        [0, 0, 0, 1],
                    ]
                )
            ],
        ),
        (
            qml.QubitUnitary,
            [0, 1, 0, 0],
            [0, 1 / math.sqrt(2), 1 / math.sqrt(2), 0],
            [
                np.array(
                    [
                        [1, 0, 0, 0],
                        [0, 1 / math.sqrt(2), 1 / math.sqrt(2), 0],
                        [0, 1 / math.sqrt(2), -1 / math.sqrt(2), 0],
                        [0, 0, 0, 1],
                    ]
                )
            ],
        ),
        (
            qml.QubitUnitary,
            [1 / 2, 1 / 2, -1 / 2, 1 / 2],
            [1 / 2, 0, 1 / math.sqrt(2), 1 / 2],
            [
                np.array(
                    [
                        [1, 0, 0, 0],
                        [0, 1 / math.sqrt(2), 1 / math.sqrt(2), 0],
                        [0, 1 / math.sqrt(2), -1 / math.sqrt(2), 0],
                        [0, 0, 0, 1],
                    ]
                )
            ],
        ),
        (qml.DiagonalQubitUnitary, [1, 0, 0, 0], [-1, 0, 0, 0], [np.array([-1, 1, 1, -1])]),
        (
            qml.DiagonalQubitUnitary,
            [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)],
            [1 / math.sqrt(2), 0, 0, -1 / math.sqrt(2)],
            [np.array([1, 1, 1, -1])],
        ),
        (qml.DiagonalQubitUnitary, [0, 0, 1, 0], [0, 0, 1j, 0], [np.array([-1, 1j, 1j, -1])]),
        (qml.IsingXX, [0, 0, 1, 0], [0, -1j / math.sqrt(2), 1 / math.sqrt(2), 0], [math.pi / 2]),
        (qml.IsingXX, [0, 0, 0, 1], [-1j / math.sqrt(2), 0, 0, 1 / math.sqrt(2)], [math.pi / 2]),
        (qml.IsingXX, [1, 0, 0, 0], [1 / math.sqrt(2), 0, 0, -1j / math.sqrt(2)], [math.pi / 2]),
        (qml.IsingYY, [0, 0, 1, 0], [0, -1j / math.sqrt(2), 1 / math.sqrt(2), 0], [math.pi / 2]),
        (qml.IsingYY, [0, 0, 0, 1], [1j / math.sqrt(2), 0, 0, 1 / math.sqrt(2)], [math.pi / 2]),
        (qml.IsingYY, [1, 0, 0, 0], [1 / math.sqrt(2), 0, 0, 1j / math.sqrt(2)], [math.pi / 2]),
        (qml.IsingZZ, [0, 0, 1, 0], [0, 0, 1 / math.sqrt(2) + 1j / math.sqrt(2), 0], [math.pi / 2]),
        (qml.IsingZZ, [0, 0, 0, 1], [0, 0, 0, 1 / math.sqrt(2) - 1j / math.sqrt(2)], [math.pi / 2]),
        (qml.IsingZZ, [1, 0, 0, 0], [1 / math.sqrt(2) - 1j / math.sqrt(2), 0, 0, 0], [math.pi / 2]),
    ]

    test_data_two_wires_with_parameters_inverses = [
        (qml.CRX, [0, 1, 0, 0], [0, 1, 0, 0], [math.pi / 2]),
        (qml.CRX, [0, 0, 0, 1], [0, 0, 1j, 0], [math.pi]),
        (
            qml.CRX,
            [0, 1 / math.sqrt(2), 1 / math.sqrt(2), 0],
            [0, 1 / math.sqrt(2), 1 / 2, 1j / 2],
            [math.pi / 2],
        ),
        (qml.MultiRZ, [0, 0, 0, 1], [0, 0, 0, 1 / math.sqrt(2) + 1j / math.sqrt(2)], [math.pi / 2]),
        (qml.MultiRZ, [0, 0, 1, 0], [0, 0, -1j, 0], [math.pi]),
        (
            qml.MultiRZ,
            [1 / math.sqrt(2), 1 / math.sqrt(2), 0, 0],
            [1 / 2 + 1j / 2, 1 / 2 - 1j / 2, 0, 0],
            [math.pi / 2],
        ),
        (qml.DiagonalQubitUnitary, [1, 0, 0, 0], [-1, 0, 0, 0], [np.array([-1, 1, 1, -1])]),
        (
            qml.DiagonalQubitUnitary,
            [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)],
            [1 / math.sqrt(2), 0, 0, -1 / math.sqrt(2)],
            [np.array([1, 1, 1, -1])],
        ),
        (qml.DiagonalQubitUnitary, [0, 0, 1, 0], [0, 0, -1j, 0], [np.array([-1, 1j, 1j, -1])]),
    ]

    @pytest.mark.parametrize(
        "operation,input,expected_output,par", test_data_two_wires_with_parameters
    )
    def test_apply_operation_two_wires_with_parameters(
        self, qubit_device_2_wires, tol, operation, input, expected_output, par
    ):
        """Tests that applying an operation yields the expected output state for two wire
        operations that have parameters."""

        qubit_device_2_wires._state = np.array(input, dtype=qubit_device_2_wires.C_DTYPE).reshape(
            (2, 2)
        )
        qubit_device_2_wires.apply([operation(*par, wires=[0, 1])])

        assert np.allclose(
            qubit_device_2_wires._state.flatten(), np.array(expected_output), atol=tol, rtol=0
        )
        assert qubit_device_2_wires._state.dtype == qubit_device_2_wires.C_DTYPE

    @pytest.mark.parametrize(
        "operation,input,expected_output,par", test_data_two_wires_with_parameters_inverses
    )
    def test_apply_operation_two_wires_with_parameters_inverse(
        self, qubit_device_2_wires, tol, operation, input, expected_output, par
    ):
        """Tests that applying the inverse of an operation yields the expected output state for two wire
        operations that have parameters."""

        qubit_device_2_wires._state = np.array(input, dtype=qubit_device_2_wires.C_DTYPE).reshape(
            (2, 2)
        )
        qubit_device_2_wires.apply([operation(*par, wires=[0, 1]).inv()])

        assert np.allclose(
            qubit_device_2_wires._state.flatten(), np.array(expected_output), atol=tol, rtol=0
        )
        assert qubit_device_2_wires._state.dtype == qubit_device_2_wires.C_DTYPE

    def test_apply_errors_qubit_state_vector(self, qubit_device_2_wires):
        """Test that apply fails for incorrect state preparation, and > 2 qubit gates"""
        with pytest.raises(ValueError, match="Sum of amplitudes-squared does not equal one."):
            qubit_device_2_wires.apply([qml.QubitStateVector(np.array([1, -1]), wires=[0])])

        with pytest.raises(ValueError, match=r"State vector must have shape \(2\*\*wires,\)."):
            p = np.array([1, 0, 1, 1, 0]) / np.sqrt(3)
            qubit_device_2_wires.apply([qml.QubitStateVector(p, wires=[0, 1])])

        with pytest.raises(
            DeviceError,
            match="Operation QubitStateVector cannot be used after other Operations have already been applied "
            "on a default.qubit device.",
        ):
            qubit_device_2_wires.reset()
            qubit_device_2_wires.apply(
                [qml.RZ(0.5, wires=[0]), qml.QubitStateVector(np.array([0, 1, 0, 0]), wires=[0, 1])]
            )

    def test_apply_errors_basis_state(self, qubit_device_2_wires):
        with pytest.raises(
            ValueError, match="BasisState parameter must consist of 0 or 1 integers."
        ):
            qubit_device_2_wires.apply([qml.BasisState(np.array([-0.2, 4.2]), wires=[0, 1])])

        with pytest.raises(
            ValueError, match="BasisState parameter and wires must be of equal length."
        ):
            qubit_device_2_wires.apply([qml.BasisState(np.array([0, 1]), wires=[0])])

        with pytest.raises(
            DeviceError,
            match="Operation BasisState cannot be used after other Operations have already been applied "
            "on a default.qubit device.",
        ):
            qubit_device_2_wires.reset()
            qubit_device_2_wires.apply(
                [qml.RZ(0.5, wires=[0]), qml.BasisState(np.array([1, 1]), wires=[0, 1])]
            )


class TestExpval:
    """Tests that expectation values are properly calculated or that the proper errors are raised."""

    @pytest.mark.parametrize(
        "operation,input,expected_output",
        [
            (qml.PauliX, [1 / math.sqrt(2), 1 / math.sqrt(2)], 1),
            (qml.PauliX, [1 / math.sqrt(2), -1 / math.sqrt(2)], -1),
            (qml.PauliX, [1, 0], 0),
            (qml.PauliY, [1 / math.sqrt(2), 1j / math.sqrt(2)], 1),
            (qml.PauliY, [1 / math.sqrt(2), -1j / math.sqrt(2)], -1),
            (qml.PauliY, [1, 0], 0),
            (qml.PauliZ, [1, 0], 1),
            (qml.PauliZ, [0, 1], -1),
            (qml.PauliZ, [1 / math.sqrt(2), 1 / math.sqrt(2)], 0),
            (qml.Hadamard, [1, 0], 1 / math.sqrt(2)),
            (qml.Hadamard, [0, 1], -1 / math.sqrt(2)),
            (qml.Hadamard, [1 / math.sqrt(2), 1 / math.sqrt(2)], 1 / math.sqrt(2)),
            (qml.Identity, [1, 0], 1),
            (qml.Identity, [0, 1], 1),
            (qml.Identity, [1 / math.sqrt(2), -1 / math.sqrt(2)], 1),
        ],
    )
    def test_expval_single_wire_no_parameters(
        self, qubit_device_1_wire, tol, operation, input, expected_output
    ):
        """Tests that expectation values are properly calculated for single-wire observables without parameters."""

        obs = operation(wires=[0])

        qubit_device_1_wire.reset()
        qubit_device_1_wire.apply(
            [qml.QubitStateVector(np.array(input), wires=[0])], obs.diagonalizing_gates()
        )
        res = qubit_device_1_wire.expval(obs)

        assert np.isclose(res, expected_output, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "operation,input,expected_output,par",
        [
            (qml.Hermitian, [1, 0], 1, [[1, 1j], [-1j, 1]]),
            (qml.Hermitian, [0, 1], 1, [[1, 1j], [-1j, 1]]),
            (qml.Hermitian, [1 / math.sqrt(2), -1 / math.sqrt(2)], 1, [[1, 1j], [-1j, 1]]),
        ],
    )
    def test_expval_single_wire_with_parameters(
        self, qubit_device_1_wire, tol, operation, input, expected_output, par
    ):
        """Tests that expectation values are properly calculated for single-wire observables with parameters."""

        obs = operation(np.array(par), wires=[0])

        qubit_device_1_wire.reset()
        qubit_device_1_wire.apply(
            [qml.QubitStateVector(np.array(input), wires=[0])], obs.diagonalizing_gates()
        )
        res = qubit_device_1_wire.expval(obs)

        assert np.isclose(res, expected_output, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "operation,input,expected_output,par",
        [
            (
                qml.Hermitian,
                [1 / math.sqrt(3), 0, 1 / math.sqrt(3), 1 / math.sqrt(3)],
                5 / 3,
                [[1, 1j, 0, 1], [-1j, 1, 0, 0], [0, 0, 1, -1j], [1, 0, 1j, 1]],
            ),
            (
                qml.Hermitian,
                [0, 0, 0, 1],
                0,
                [[0, 1j, 0, 0], [-1j, 0, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]],
            ),
            (
                qml.Hermitian,
                [1 / math.sqrt(2), 0, -1 / math.sqrt(2), 0],
                1,
                [[1, 1j, 0, 0], [-1j, 1, 0, 0], [0, 0, 1, -1j], [0, 0, 1j, 1]],
            ),
            (
                qml.Hermitian,
                [1 / math.sqrt(3), -1 / math.sqrt(3), 1 / math.sqrt(6), 1 / math.sqrt(6)],
                1,
                [[1, 1j, 0, 0.5j], [-1j, 1, 0, 0], [0, 0, 1, -1j], [-0.5j, 0, 1j, 1]],
            ),
            (
                qml.Hermitian,
                [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)],
                1,
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
            ),
            (
                qml.Hermitian,
                [0, 1 / math.sqrt(2), -1 / math.sqrt(2), 0],
                -1,
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
            ),
        ],
    )
    def test_expval_two_wires_with_parameters(
        self, qubit_device_2_wires, tol, operation, input, expected_output, par
    ):
        """Tests that expectation values are properly calculated for two-wire observables with parameters."""

        obs = operation(np.array(par), wires=[0, 1])

        qubit_device_2_wires.reset()
        qubit_device_2_wires.apply(
            [qml.QubitStateVector(np.array(input), wires=[0, 1])], obs.diagonalizing_gates()
        )
        res = qubit_device_2_wires.expval(obs)

        assert np.isclose(res, expected_output, atol=tol, rtol=0)

    def test_expval_estimate(self):
        """Test that the expectation value is not analytically calculated"""

        dev = qml.device("default.qubit", wires=1, shots=3)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit():
            return qml.expval(qml.PauliX(0))

        expval = circuit()

        # With 3 samples we are guaranteed to see a difference between
        # an estimated variance an an analytically calculated one
        assert expval != 0.0


class TestVar:
    """Tests that variances are properly calculated."""

    @pytest.mark.parametrize(
        "operation,input,expected_output",
        [
            (qml.PauliX, [1 / math.sqrt(2), 1 / math.sqrt(2)], 0),
            (qml.PauliX, [1 / math.sqrt(2), -1 / math.sqrt(2)], 0),
            (qml.PauliX, [1, 0], 1),
            (qml.PauliY, [1 / math.sqrt(2), 1j / math.sqrt(2)], 0),
            (qml.PauliY, [1 / math.sqrt(2), -1j / math.sqrt(2)], 0),
            (qml.PauliY, [1, 0], 1),
            (qml.PauliZ, [1, 0], 0),
            (qml.PauliZ, [0, 1], 0),
            (qml.PauliZ, [1 / math.sqrt(2), 1 / math.sqrt(2)], 1),
            (qml.Hadamard, [1, 0], 1 / 2),
            (qml.Hadamard, [0, 1], 1 / 2),
            (qml.Hadamard, [1 / math.sqrt(2), 1 / math.sqrt(2)], 1 / 2),
            (qml.Identity, [1, 0], 0),
            (qml.Identity, [0, 1], 0),
            (qml.Identity, [1 / math.sqrt(2), -1 / math.sqrt(2)], 0),
        ],
    )
    def test_var_single_wire_no_parameters(
        self, qubit_device_1_wire, tol, operation, input, expected_output
    ):
        """Tests that variances are properly calculated for single-wire observables without parameters."""

        obs = operation(wires=[0])

        qubit_device_1_wire.reset()
        qubit_device_1_wire.apply(
            [qml.QubitStateVector(np.array(input), wires=[0])], obs.diagonalizing_gates()
        )
        res = qubit_device_1_wire.var(obs)

        assert np.isclose(res, expected_output, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "operation,input,expected_output,par",
        [
            (qml.Hermitian, [1, 0], 1, [[1, 1j], [-1j, 1]]),
            (qml.Hermitian, [0, 1], 1, [[1, 1j], [-1j, 1]]),
            (qml.Hermitian, [1 / math.sqrt(2), -1 / math.sqrt(2)], 1, [[1, 1j], [-1j, 1]]),
        ],
    )
    def test_var_single_wire_with_parameters(
        self, qubit_device_1_wire, tol, operation, input, expected_output, par
    ):
        """Tests that variances are properly calculated for single-wire observables with parameters."""

        obs = operation(np.array(par), wires=[0])

        qubit_device_1_wire.reset()
        qubit_device_1_wire.apply(
            [qml.QubitStateVector(np.array(input), wires=[0])], obs.diagonalizing_gates()
        )
        res = qubit_device_1_wire.var(obs)

        assert np.isclose(res, expected_output, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "operation,input,expected_output,par",
        [
            (
                qml.Hermitian,
                [1 / math.sqrt(3), 0, 1 / math.sqrt(3), 1 / math.sqrt(3)],
                11 / 9,
                [[1, 1j, 0, 1], [-1j, 1, 0, 0], [0, 0, 1, -1j], [1, 0, 1j, 1]],
            ),
            (
                qml.Hermitian,
                [0, 0, 0, 1],
                1,
                [[0, 1j, 0, 0], [-1j, 0, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]],
            ),
            (
                qml.Hermitian,
                [1 / math.sqrt(2), 0, -1 / math.sqrt(2), 0],
                1,
                [[1, 1j, 0, 0], [-1j, 1, 0, 0], [0, 0, 1, -1j], [0, 0, 1j, 1]],
            ),
            (
                qml.Hermitian,
                [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)],
                0,
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
            ),
            (
                qml.Hermitian,
                [0, 1 / math.sqrt(2), -1 / math.sqrt(2), 0],
                0,
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
            ),
        ],
    )
    def test_var_two_wires_with_parameters(
        self, qubit_device_2_wires, tol, operation, input, expected_output, par
    ):
        """Tests that variances are properly calculated for two-wire observables with parameters."""

        obs = operation(np.array(par), wires=[0, 1])

        qubit_device_2_wires.reset()
        qubit_device_2_wires.apply(
            [qml.QubitStateVector(np.array(input), wires=[0, 1])], obs.diagonalizing_gates()
        )
        res = qubit_device_2_wires.var(obs)

        assert np.isclose(res, expected_output, atol=tol, rtol=0)

    def test_var_estimate(self):
        """Test that the variance is not analytically calculated"""

        dev = qml.device("default.qubit", wires=1, shots=3)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit():
            return qml.var(qml.PauliX(0))

        var = circuit()

        # With 3 samples we are guaranteed to see a difference between
        # an estimated variance and an analytically calculated one
        assert var != 1.0


class TestSample:
    """Tests that samples are properly calculated."""

    def test_sample_dimensions(self):
        """Tests if the samples returned by the sample function have
        the correct dimensions
        """

        # Explicitly resetting is necessary as the internal
        # state is set to None in __init__ and only properly
        # initialized during reset
        dev = qml.device("default.qubit", wires=2, shots=1000)

        dev.apply([qml.RX(1.5708, wires=[0]), qml.RX(1.5708, wires=[1])])

        dev.shots = 10
        dev._wires_measured = {0}
        dev._samples = dev.generate_samples()
        s1 = dev.sample(qml.PauliZ(wires=[0]))
        assert np.array_equal(s1.shape, (10,))

        dev.reset()
        dev.shots = 12
        dev._wires_measured = {1}
        dev._samples = dev.generate_samples()
        s2 = dev.sample(qml.PauliZ(wires=[1]))
        assert np.array_equal(s2.shape, (12,))

        dev.reset()
        dev.shots = 17
        dev._wires_measured = {0, 1}
        dev._samples = dev.generate_samples()
        s3 = dev.sample(qml.PauliX(0) @ qml.PauliZ(1))
        assert np.array_equal(s3.shape, (17,))

    def test_sample_values(self, qubit_device_2_wires, tol):
        """Tests if the samples returned by sample have
        the correct values
        """

        # Explicitly resetting is necessary as the internal
        # state is set to None in __init__ and only properly
        # initialized during reset
        dev = qml.device("default.qubit", wires=2, shots=1000)

        dev.apply([qml.RX(1.5708, wires=[0])])
        dev._wires_measured = {0}
        dev._samples = dev.generate_samples()

        s1 = dev.sample(qml.PauliZ(0))

        # s1 should only contain 1 and -1, which is guaranteed if
        # they square to 1
        assert np.allclose(s1**2, 1, atol=tol, rtol=0)


class TestDefaultQubitIntegration:
    """Integration tests for default.qubit. This test ensures it integrates
    properly with the PennyLane interface, in particular QNode."""

    def test_defines_correct_capabilities(self):
        """Test that the device defines the right capabilities"""

        dev = qml.device("default.qubit", wires=1)
        cap = dev.capabilities()
        capabilities = {
            "model": "qubit",
            "supports_finite_shots": True,
            "supports_tensor_observables": True,
            "returns_probs": True,
            "returns_state": True,
            "supports_inverse_operations": True,
            "supports_analytic_computation": True,
            "supports_broadcasting": True,
            "passthru_devices": {
                "torch": "default.qubit.torch",
                "tf": "default.qubit.tf",
                "autograd": "default.qubit.autograd",
                "jax": "default.qubit.jax",
            },
        }
        assert cap == capabilities

    @pytest.mark.parametrize("r_dtype", [np.float32, np.float64])
    def test_qubit_circuit(self, qubit_device_1_wire, r_dtype, tol):
        """Test that the default qubit plugin provides correct result for a simple circuit"""

        p = 0.543

        dev = qubit_device_1_wire
        dev.R_DTYPE = r_dtype

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliY(0))

        expected = -np.sin(p)

        res = circuit(p)
        assert np.isclose(res, expected, atol=tol, rtol=0)
        assert res.dtype == r_dtype

    def test_qubit_identity(self, qubit_device_1_wire, tol):
        """Test that the default qubit plugin provides correct result for the Identity expectation"""

        p = 0.543

        @qml.qnode(qubit_device_1_wire)
        def circuit(x):
            """Test quantum function"""
            qml.RX(x, wires=0)
            return qml.expval(qml.Identity(0))

        assert np.isclose(circuit(p), 1, atol=tol, rtol=0)

    def test_nonzero_shots(self, tol):
        """Test that the default qubit plugin provides correct result for high shot number"""

        shots = 10**5
        dev = qml.device("default.qubit", wires=1, shots=shots)

        p = 0.543

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(x):
            """Test quantum function"""
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliY(0))

        runs = []
        for _ in range(100):
            runs.append(circuit(p))

        assert np.isclose(np.mean(runs), -np.sin(p), atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "name,state,expected_output",
        [
            ("PauliX", [1 / math.sqrt(2), 1 / math.sqrt(2)], 1),
            ("PauliX", [1 / math.sqrt(2), -1 / math.sqrt(2)], -1),
            ("PauliX", [1, 0], 0),
            ("PauliY", [1 / math.sqrt(2), 1j / math.sqrt(2)], 1),
            ("PauliY", [1 / math.sqrt(2), -1j / math.sqrt(2)], -1),
            ("PauliY", [1, 0], 0),
            ("PauliZ", [1, 0], 1),
            ("PauliZ", [0, 1], -1),
            ("PauliZ", [1 / math.sqrt(2), 1 / math.sqrt(2)], 0),
            ("Hadamard", [1, 0], 1 / math.sqrt(2)),
            ("Hadamard", [0, 1], -1 / math.sqrt(2)),
            ("Hadamard", [1 / math.sqrt(2), 1 / math.sqrt(2)], 1 / math.sqrt(2)),
        ],
    )
    def test_supported_observable_single_wire_no_parameters(
        self, qubit_device_1_wire, tol, name, state, expected_output
    ):
        """Tests supported observables on single wires without parameters."""

        obs = getattr(qml.ops, name)

        assert qubit_device_1_wire.supports_observable(name)

        @qml.qnode(qubit_device_1_wire)
        def circuit():
            qml.QubitStateVector(np.array(state), wires=[0])
            return qml.expval(obs(wires=[0]))

        assert np.isclose(circuit(), expected_output, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "name,state,expected_output,par",
        [
            ("Identity", [1, 0], 1, []),
            ("Identity", [0, 1], 1, []),
            ("Identity", [1 / math.sqrt(2), -1 / math.sqrt(2)], 1, []),
            ("Hermitian", [1, 0], 1, [np.array([[1, 1j], [-1j, 1]])]),
            ("Hermitian", [0, 1], 1, [np.array([[1, 1j], [-1j, 1]])]),
            (
                "Hermitian",
                [1 / math.sqrt(2), -1 / math.sqrt(2)],
                1,
                [np.array([[1, 1j], [-1j, 1]])],
            ),
        ],
    )
    def test_supported_observable_single_wire_with_parameters(
        self, qubit_device_1_wire, tol, name, state, expected_output, par
    ):
        """Tests supported observables on single wires with parameters."""

        obs = getattr(qml.ops, name)

        assert qubit_device_1_wire.supports_observable(name)

        @qml.qnode(qubit_device_1_wire)
        def circuit():
            qml.QubitStateVector(np.array(state), wires=[0])
            return qml.expval(obs(*par, wires=[0]))

        assert np.isclose(circuit(), expected_output, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "name,state,expected_output,par",
        [
            (
                "Hermitian",
                [1 / math.sqrt(3), 0, 1 / math.sqrt(3), 1 / math.sqrt(3)],
                5 / 3,
                [np.array([[1, 1j, 0, 1], [-1j, 1, 0, 0], [0, 0, 1, -1j], [1, 0, 1j, 1]])],
            ),
            (
                "Hermitian",
                [0, 0, 0, 1],
                0,
                [np.array([[0, 1j, 0, 0], [-1j, 0, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]])],
            ),
            (
                "Hermitian",
                [1 / math.sqrt(2), 0, -1 / math.sqrt(2), 0],
                1,
                [np.array([[1, 1j, 0, 0], [-1j, 1, 0, 0], [0, 0, 1, -1j], [0, 0, 1j, 1]])],
            ),
            (
                "Hermitian",
                [1 / math.sqrt(3), -1 / math.sqrt(3), 1 / math.sqrt(6), 1 / math.sqrt(6)],
                1,
                [np.array([[1, 1j, 0, 0.5j], [-1j, 1, 0, 0], [0, 0, 1, -1j], [-0.5j, 0, 1j, 1]])],
            ),
            (
                "Hermitian",
                [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)],
                1,
                [np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])],
            ),
            (
                "Hermitian",
                [0, 1 / math.sqrt(2), -1 / math.sqrt(2), 0],
                -1,
                [np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])],
            ),
        ],
    )
    def test_supported_observable_two_wires_with_parameters(
        self, qubit_device_2_wires, tol, name, state, expected_output, par
    ):
        """Tests supported observables on two wires with parameters."""

        obs = getattr(qml.ops, name)

        assert qubit_device_2_wires.supports_observable(name)

        @qml.qnode(qubit_device_2_wires)
        def circuit():
            qml.QubitStateVector(np.array(state), wires=[0, 1])
            return qml.expval(obs(*par, wires=[0, 1]))

        assert np.isclose(circuit(), expected_output, atol=tol, rtol=0)

    def test_multi_samples_return_correlated_results(self):
        """Tests if the samples returned by the sample function have
        the correct dimensions
        """

        dev = qml.device("default.qubit", wires=2, shots=1000)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit():
            qml.Hadamard(0)
            qml.CNOT(wires=[0, 1])
            return qml.sample(qml.PauliZ(0)), qml.sample(qml.PauliZ(1))

        outcomes = circuit()

        assert np.array_equal(outcomes[0], outcomes[1])

    @pytest.mark.parametrize("num_wires", [3, 4, 5, 6, 7, 8])
    def test_multi_samples_return_correlated_results_more_wires_than_size_of_observable(
        self, num_wires
    ):
        """Tests if the samples returned by the sample function have
        the correct dimensions
        """

        dev = qml.device("default.qubit", wires=num_wires, shots=1000)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit():
            qml.Hadamard(0)
            qml.CNOT(wires=[0, 1])
            return qml.sample(qml.PauliZ(0)), qml.sample(qml.PauliZ(1))

        outcomes = circuit()

        assert np.array_equal(outcomes[0], outcomes[1])


@pytest.mark.parametrize("theta,phi,varphi", list(zip(THETA, PHI, VARPHI)))
class TestTensorExpval:
    """Test tensor expectation values"""

    def test_paulix_pauliy(self, theta, phi, varphi, tol):
        """Test that a tensor product involving PauliX and PauliY works correctly"""
        dev = qml.device("default.qubit", wires=3)
        dev.reset()

        obs = qml.PauliX(0) @ qml.PauliY(2)

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2]),
            ],
            obs.diagonalizing_gates(),
        )

        res = dev.expval(obs)

        expected = np.sin(theta) * np.sin(phi) * np.sin(varphi)

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_pauliz_identity(self, theta, phi, varphi, tol):
        """Test that a tensor product involving PauliZ and Identity works correctly"""
        dev = qml.device("default.qubit", wires=3)
        dev.reset()

        obs = qml.PauliZ(0) @ qml.Identity(1) @ qml.PauliZ(2)

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2]),
            ],
            obs.diagonalizing_gates(),
        )

        res = dev.expval(obs)

        expected = np.cos(varphi) * np.cos(phi)

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_pauliz_hadamard(self, theta, phi, varphi, tol):
        """Test that a tensor product involving PauliZ and PauliY and hadamard works correctly"""
        dev = qml.device("default.qubit", wires=3)
        obs = qml.PauliZ(0) @ qml.Hadamard(1) @ qml.PauliY(2)

        dev.reset()
        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2]),
            ],
            obs.diagonalizing_gates(),
        )

        res = dev.expval(obs)

        expected = -(np.cos(varphi) * np.sin(phi) + np.sin(varphi) * np.cos(theta)) / np.sqrt(2)

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_hermitian(self, theta, phi, varphi, tol):
        """Test that a tensor product involving qml.Hermitian works correctly"""
        dev = qml.device("default.qubit", wires=3)
        dev.reset()

        A = np.array(
            [
                [-6, 2 + 1j, -3, -5 + 2j],
                [2 - 1j, 0, 2 - 1j, -5 + 4j],
                [-3, 2 + 1j, 0, -4 + 3j],
                [-5 - 2j, -5 - 4j, -4 - 3j, -6],
            ]
        )

        obs = qml.PauliZ(0) @ qml.Hermitian(A, wires=[1, 2])

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2]),
            ],
            obs.diagonalizing_gates(),
        )

        res = dev.expval(obs)

        expected = 0.5 * (
            -6 * np.cos(theta) * (np.cos(varphi) + 1)
            - 2 * np.sin(varphi) * (np.cos(theta) + np.sin(phi) - 2 * np.cos(phi))
            + 3 * np.cos(varphi) * np.sin(phi)
            + np.sin(phi)
        )

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_hermitian_hermitian(self, theta, phi, varphi, tol):
        """Test that a tensor product involving two Hermitian matrices works correctly"""
        dev = qml.device("default.qubit", wires=3)

        A1 = np.array([[1, 2], [2, 4]])

        A2 = np.array(
            [
                [-6, 2 + 1j, -3, -5 + 2j],
                [2 - 1j, 0, 2 - 1j, -5 + 4j],
                [-3, 2 + 1j, 0, -4 + 3j],
                [-5 - 2j, -5 - 4j, -4 - 3j, -6],
            ]
        )

        obs = qml.Hermitian(A1, wires=[0]) @ qml.Hermitian(A2, wires=[1, 2])

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2]),
            ],
            obs.diagonalizing_gates(),
        )

        res = dev.expval(obs)

        expected = 0.25 * (
            -30
            + 4 * np.cos(phi) * np.sin(theta)
            + 3 * np.cos(varphi) * (-10 + 4 * np.cos(phi) * np.sin(theta) - 3 * np.sin(phi))
            - 3 * np.sin(phi)
            - 2
            * (5 + np.cos(phi) * (6 + 4 * np.sin(theta)) + (-3 + 8 * np.sin(theta)) * np.sin(phi))
            * np.sin(varphi)
            + np.cos(theta)
            * (
                18
                + 5 * np.sin(phi)
                + 3 * np.cos(varphi) * (6 + 5 * np.sin(phi))
                + 2 * (3 + 10 * np.cos(phi) - 5 * np.sin(phi)) * np.sin(varphi)
            )
        )

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_hermitian_identity_expectation(self, theta, phi, varphi, tol):
        """Test that a tensor product involving an Hermitian matrix and the identity works correctly"""
        dev = qml.device("default.qubit", wires=2)

        A = np.array(
            [[1.02789352, 1.61296440 - 0.3498192j], [1.61296440 + 0.3498192j, 1.23920938 + 0j]]
        )

        obs = qml.Hermitian(A, wires=[0]) @ qml.Identity(wires=[1])

        dev.apply(
            [qml.RY(theta, wires=[0]), qml.RY(phi, wires=[1]), qml.CNOT(wires=[0, 1])],
            obs.diagonalizing_gates(),
        )

        res = dev.expval(obs)

        a = A[0, 0]
        re_b = A[0, 1].real
        d = A[1, 1]
        expected = ((a - d) * np.cos(theta) + 2 * re_b * np.sin(theta) * np.sin(phi) + a + d) / 2

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_hermitian_two_wires_identity_expectation(self, theta, phi, varphi, tol):
        """Test that a tensor product involving an Hermitian matrix for two wires and the identity works correctly"""
        dev = qml.device("default.qubit", wires=3)

        A = np.array(
            [[1.02789352, 1.61296440 - 0.3498192j], [1.61296440 + 0.3498192j, 1.23920938 + 0j]]
        )
        Identity = np.array([[1, 0], [0, 1]])
        H = np.kron(np.kron(Identity, Identity), A)
        obs = qml.Hermitian(H, wires=[2, 1, 0])

        dev.apply(
            [qml.RY(theta, wires=[0]), qml.RY(phi, wires=[1]), qml.CNOT(wires=[0, 1])],
            obs.diagonalizing_gates(),
        )
        res = dev.expval(obs)

        a = A[0, 0]
        re_b = A[0, 1].real
        d = A[1, 1]

        expected = ((a - d) * np.cos(theta) + 2 * re_b * np.sin(theta) * np.sin(phi) + a + d) / 2
        assert np.allclose(res, expected, atol=tol, rtol=0)


@pytest.mark.parametrize("theta, phi, varphi", list(zip(THETA, PHI, VARPHI)))
class TestTensorVar:
    """Tests for variance of tensor observables"""

    def test_paulix_pauliy(self, theta, phi, varphi, tol):
        """Test that a tensor product involving PauliX and PauliY works correctly"""
        dev = qml.device("default.qubit", wires=3)

        obs = qml.PauliX(0) @ qml.PauliY(2)

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2]),
            ],
            obs.diagonalizing_gates(),
        )

        res = dev.var(obs)

        expected = (
            8 * np.sin(theta) ** 2 * np.cos(2 * varphi) * np.sin(phi) ** 2
            - np.cos(2 * (theta - phi))
            - np.cos(2 * (theta + phi))
            + 2 * np.cos(2 * theta)
            + 2 * np.cos(2 * phi)
            + 14
        ) / 16

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_pauliz_hadamard(self, theta, phi, varphi, tol):
        """Test that a tensor product involving PauliZ and PauliY and hadamard works correctly"""
        dev = qml.device("default.qubit", wires=3)
        obs = qml.PauliZ(0) @ qml.Hadamard(1) @ qml.PauliY(2)

        dev.reset()
        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2]),
            ],
            obs.diagonalizing_gates(),
        )

        res = dev.var(obs)

        expected = (
            3
            + np.cos(2 * phi) * np.cos(varphi) ** 2
            - np.cos(2 * theta) * np.sin(varphi) ** 2
            - 2 * np.cos(theta) * np.sin(phi) * np.sin(2 * varphi)
        ) / 4

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_hermitian(self, theta, phi, varphi, tol):
        """Test that a tensor product involving qml.Hermitian works correctly"""
        dev = qml.device("default.qubit", wires=3)

        A = np.array(
            [
                [-6, 2 + 1j, -3, -5 + 2j],
                [2 - 1j, 0, 2 - 1j, -5 + 4j],
                [-3, 2 + 1j, 0, -4 + 3j],
                [-5 - 2j, -5 - 4j, -4 - 3j, -6],
            ]
        )

        obs = qml.PauliZ(0) @ qml.Hermitian(A, wires=[1, 2])

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2]),
            ],
            obs.diagonalizing_gates(),
        )

        res = dev.var(obs)

        expected = (
            1057
            - np.cos(2 * phi)
            + 12 * (27 + np.cos(2 * phi)) * np.cos(varphi)
            - 2 * np.cos(2 * varphi) * np.sin(phi) * (16 * np.cos(phi) + 21 * np.sin(phi))
            + 16 * np.sin(2 * phi)
            - 8 * (-17 + np.cos(2 * phi) + 2 * np.sin(2 * phi)) * np.sin(varphi)
            - 8 * np.cos(2 * theta) * (3 + 3 * np.cos(varphi) + np.sin(varphi)) ** 2
            - 24 * np.cos(phi) * (np.cos(phi) + 2 * np.sin(phi)) * np.sin(2 * varphi)
            - 8
            * np.cos(theta)
            * (
                4
                * np.cos(phi)
                * (
                    4
                    + 8 * np.cos(varphi)
                    + np.cos(2 * varphi)
                    - (1 + 6 * np.cos(varphi)) * np.sin(varphi)
                )
                + np.sin(phi)
                * (
                    15
                    + 8 * np.cos(varphi)
                    - 11 * np.cos(2 * varphi)
                    + 42 * np.sin(varphi)
                    + 3 * np.sin(2 * varphi)
                )
            )
        ) / 16

        assert np.allclose(res, expected, atol=tol, rtol=0)


@pytest.mark.parametrize("theta, phi, varphi", list(zip(THETA, PHI, VARPHI)))
class TestTensorSample:
    """Test tensor expectation values"""

    def test_paulix_pauliy(self, theta, phi, varphi, tol_stochastic):
        """Test that a tensor product involving PauliX and PauliY works correctly"""
        dev = qml.device("default.qubit", wires=3, shots=int(1e6))

        obs = qml.PauliX(0) @ qml.PauliY(2)

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2]),
            ],
            obs.diagonalizing_gates(),
        )

        dev._wires_measured = {0, 1, 2}
        dev._samples = dev.generate_samples()
        dev.sample(obs)

        s1 = obs.eigvals()
        p = dev.probability(wires=dev.map_wires(obs.wires))

        # s1 should only contain 1 and -1
        assert np.allclose(s1**2, 1, atol=tol_stochastic, rtol=0)

        mean = s1 @ p
        expected = np.sin(theta) * np.sin(phi) * np.sin(varphi)
        assert np.allclose(mean, expected, atol=tol_stochastic, rtol=0)

        var = (s1**2) @ p - (s1 @ p).real ** 2
        expected = (
            8 * np.sin(theta) ** 2 * np.cos(2 * varphi) * np.sin(phi) ** 2
            - np.cos(2 * (theta - phi))
            - np.cos(2 * (theta + phi))
            + 2 * np.cos(2 * theta)
            + 2 * np.cos(2 * phi)
            + 14
        ) / 16
        assert np.allclose(var, expected, atol=tol_stochastic, rtol=0)

    def test_pauliz_hadamard(self, theta, phi, varphi, tol_stochastic):
        """Test that a tensor product involving PauliZ and PauliY and hadamard works correctly"""
        dev = qml.device("default.qubit", wires=3, shots=int(1e6))
        obs = qml.PauliZ(0) @ qml.Hadamard(1) @ qml.PauliY(2)
        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2]),
            ],
            obs.diagonalizing_gates(),
        )

        dev._wires_measured = {0, 1, 2}
        dev._samples = dev.generate_samples()
        dev.sample(obs)

        s1 = obs.eigvals()
        p = dev.marginal_prob(dev.probability(), wires=obs.wires)

        # s1 should only contain 1 and -1
        assert np.allclose(s1**2, 1, atol=tol_stochastic, rtol=0)

        mean = s1 @ p
        expected = -(np.cos(varphi) * np.sin(phi) + np.sin(varphi) * np.cos(theta)) / np.sqrt(2)
        assert np.allclose(mean, expected, atol=tol_stochastic, rtol=0)

        var = (s1**2) @ p - (s1 @ p).real ** 2
        expected = (
            3
            + np.cos(2 * phi) * np.cos(varphi) ** 2
            - np.cos(2 * theta) * np.sin(varphi) ** 2
            - 2 * np.cos(theta) * np.sin(phi) * np.sin(2 * varphi)
        ) / 4
        assert np.allclose(var, expected, atol=tol_stochastic, rtol=0)

    def test_hermitian(self, theta, phi, varphi, tol_stochastic):
        """Test that a tensor product involving qml.Hermitian works correctly"""
        dev = qml.device("default.qubit", wires=3, shots=int(1e6))

        A = 0.1 * np.array(
            [
                [-6, 2 + 1j, -3, -5 + 2j],
                [2 - 1j, 0, 2 - 1j, -5 + 4j],
                [-3, 2 + 1j, 0, -4 + 3j],
                [-5 - 2j, -5 - 4j, -4 - 3j, -6],
            ]
        )

        obs = qml.PauliZ(0) @ qml.Hermitian(A, wires=[1, 2])
        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2]),
            ],
            obs.diagonalizing_gates(),
        )

        dev._wires_measured = {0, 1, 2}
        dev._samples = dev.generate_samples()
        dev.sample(obs)

        s1 = obs.eigvals()
        p = dev.marginal_prob(dev.probability(), wires=obs.wires)

        # s1 should only contain the eigenvalues of
        # the hermitian matrix tensor product Z
        Z = np.diag([1, -1])
        eigvals = np.linalg.eigvalsh(np.kron(Z, A))
        assert set(np.round(s1, 8).tolist()).issubset(set(np.round(eigvals, 8).tolist()))

        mean = s1 @ p
        expected = (
            0.1
            * 0.5
            * (
                -6 * np.cos(theta) * (np.cos(varphi) + 1)
                - 2 * np.sin(varphi) * (np.cos(theta) + np.sin(phi) - 2 * np.cos(phi))
                + 3 * np.cos(varphi) * np.sin(phi)
                + np.sin(phi)
            )
        )
        assert np.allclose(mean, expected, atol=tol_stochastic, rtol=0)

        var = (s1**2) @ p - (s1 @ p).real ** 2
        expected = (
            0.01
            * (
                1057
                - np.cos(2 * phi)
                + 12 * (27 + np.cos(2 * phi)) * np.cos(varphi)
                - 2 * np.cos(2 * varphi) * np.sin(phi) * (16 * np.cos(phi) + 21 * np.sin(phi))
                + 16 * np.sin(2 * phi)
                - 8 * (-17 + np.cos(2 * phi) + 2 * np.sin(2 * phi)) * np.sin(varphi)
                - 8 * np.cos(2 * theta) * (3 + 3 * np.cos(varphi) + np.sin(varphi)) ** 2
                - 24 * np.cos(phi) * (np.cos(phi) + 2 * np.sin(phi)) * np.sin(2 * varphi)
                - 8
                * np.cos(theta)
                * (
                    4
                    * np.cos(phi)
                    * (
                        4
                        + 8 * np.cos(varphi)
                        + np.cos(2 * varphi)
                        - (1 + 6 * np.cos(varphi)) * np.sin(varphi)
                    )
                    + np.sin(phi)
                    * (
                        15
                        + 8 * np.cos(varphi)
                        - 11 * np.cos(2 * varphi)
                        + 42 * np.sin(varphi)
                        + 3 * np.sin(2 * varphi)
                    )
                )
            )
            / 16
        )
        assert np.allclose(var, expected, atol=tol_stochastic, rtol=0)


@pytest.mark.parametrize(
    "r_dtype,c_dtype", [(np.float32, np.complex64), (np.float64, np.complex128)]
)
class TestDtypePreserved:
    """Test that the user-defined dtype of the device is preserved for QNode
    evaluation"""

    @pytest.mark.parametrize(
        "op",
        [
            qml.SingleExcitation,
            qml.SingleExcitationPlus,
            qml.SingleExcitationMinus,
            qml.DoubleExcitation,
            qml.DoubleExcitationPlus,
            qml.DoubleExcitationMinus,
            qml.OrbitalRotation,
            qml.QubitSum,
            qml.QubitCarry,
        ],
    )
    def test_state_dtype_after_op(self, r_dtype, c_dtype, op, tol):
        """Test that the default qubit plugin preserves data types of states when an operation is
        applied. As TestApply class check most of operators, we here only check some subtle
        examples.
        """

        dev = qml.device("default.qubit", wires=4, r_dtype=r_dtype, c_dtype=c_dtype)

        n_wires = op.num_wires
        n_params = op.num_params

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit():
            if n_params == 0:
                op(wires=range(n_wires))
            elif n_params == 1:
                op(0.543, wires=range(n_wires))
            else:
                op([0.543] * n_params, wires=range(n_wires))
            return qml.state()

        res = circuit()
        assert res.dtype == c_dtype

    @pytest.mark.parametrize(
        "measurement",
        [
            qml.expval(qml.PauliY(0)),
            qml.var(qml.PauliY(0)),
            qml.probs(wires=[1]),
            qml.probs(wires=[2, 0]),
        ],
    )
    def test_measurement_real_dtype(self, r_dtype, c_dtype, measurement, tol):
        """Test that the default qubit plugin provides correct result for a simple circuit"""
        p = 0.543

        dev = qml.device("default.qubit", wires=3, r_dtype=r_dtype, c_dtype=c_dtype)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.apply(measurement)

        res = circuit(p)
        assert res.dtype == r_dtype

    @pytest.mark.parametrize(
        "measurement",
        [qml.state(), qml.density_matrix(wires=[1]), qml.density_matrix(wires=[2, 0])],
    )
    def test_measurement_complex_dtype(self, r_dtype, c_dtype, measurement, tol):
        """Test that the default qubit plugin provides correct result for a simple circuit"""
        p = 0.543

        dev = qml.device("default.qubit", wires=3, r_dtype=r_dtype, c_dtype=c_dtype)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.apply(measurement)

        res = circuit(p)
        assert res.dtype == c_dtype


class TestProbabilityIntegration:
    """Test probability method for when analytic is True/False"""

    def mock_analytic_counter(self, wires=None):
        self.analytic_counter += 1
        return np.array([1, 0, 0, 0], dtype=float)

    @pytest.mark.parametrize("x", [[0.2, 0.5], [0.4, 0.9], [0.8, 0.3]])
    def test_probability(self, x, tol):
        """Test that the probability function works for finite and infinite shots"""
        dev = qml.device("default.qubit", wires=2, shots=1000)
        dev_analytic = qml.device("default.qubit", wires=2, shots=None)

        def circuit(x):
            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0, 1])

        prob = qml.QNode(circuit, dev)
        prob_analytic = qml.QNode(circuit, dev_analytic)

        assert np.isclose(prob(x).sum(), 1, atol=tol, rtol=0)
        assert np.allclose(prob_analytic(x), prob(x), atol=0.1, rtol=0)
        assert not np.array_equal(prob_analytic(x), prob(x))

    def test_call_generate_samples(self, monkeypatch):
        """Test analytic_probability call when generating samples"""
        self.analytic_counter = False

        dev = qml.device("default.qubit", wires=2, shots=1000)
        monkeypatch.setattr(dev, "analytic_probability", self.mock_analytic_counter)

        # generate samples through `generate_samples` (using 'analytic_probability')
        dev.generate_samples()

        # should call `analytic_probability` once through `generate_samples`
        assert self.analytic_counter == 1

    def test_stateless_analytic_return(self):
        """Test that analytic_probability returns None if device is stateless"""
        dev = qml.device("default.qubit", wires=2)
        dev._state = None

        assert dev.analytic_probability() is None


class TestWiresIntegration:
    """Test that the device integrates with PennyLane's wire management."""

    def make_circuit_probs(self, wires):
        """Factory for a qnode returning probabilities using arbitrary wire labels."""
        dev = qml.device("default.qubit", wires=wires)
        n_wires = len(wires)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit():
            qml.RX(0.5, wires=wires[0 % n_wires])
            qml.RY(2.0, wires=wires[1 % n_wires])
            if n_wires > 1:
                qml.CNOT(wires=[wires[0], wires[1]])
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
        dev = qml.device("default.qubit", wires=["a", "b"])

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.5, wires="c")

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
        dev = qml.device("default.qubit", wires=dev_wires)

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


class TestGetSlice:
    """Tests for the _get_slice function."""

    def test_get_slice(self):
        """Test that the _get_slice function returns the expected slice and allows us to slice
        correctly into an array."""

        sl = _get_slice(1, 1, 3)
        array = np.arange(27).reshape((3, 3, 3))
        target = array[:, 1, :]

        assert sl == (slice(None, None, None), 1, slice(None, None, None))
        assert np.allclose(array[sl], target)

    def test_get_slice_first(self):
        """Test that the _get_slice function returns the expected slice when accessing the first
        axis of an array."""

        sl = _get_slice(2, 0, 3)
        array = np.arange(27).reshape((3, 3, 3))
        target = array[2]

        assert sl == (2, slice(None, None, None), slice(None, None, None))
        assert np.allclose(array[sl], target)

    def test_get_slice_last(self):
        """Test that the _get_slice function returns the expected slice when accessing the last
        axis of an array."""

        sl = _get_slice(0, 2, 3)
        array = np.arange(27).reshape((3, 3, 3))
        target = array[:, :, 0]

        assert sl == (slice(None, None, None), slice(None, None, None), 0)
        assert np.allclose(array[sl], target)

    def test_get_slice_1d(self):
        """Test that the _get_slice function returns the expected slice when accessing a
        1-dimensional array."""

        sl = _get_slice(2, 0, 1)
        array = np.arange(27)
        target = array[2]

        assert sl == (2,)
        assert np.allclose(array[sl], target)


@pytest.mark.parametrize("inverse", [True, False])
class TestApplyOps:
    """Tests for special methods listed in _apply_ops that use array manipulation tricks to apply
    gates in DefaultQubit."""

    state = np.arange(2**4, dtype=np.complex128).reshape((2, 2, 2, 2))
    dev = qml.device("default.qubit", wires=4)

    single_qubit_ops = [
        (qml.PauliX, dev._apply_x),
        (qml.PauliY, dev._apply_y),
        (qml.PauliZ, dev._apply_z),
        (qml.Hadamard, dev._apply_hadamard),
        (qml.S, dev._apply_s),
        (qml.T, dev._apply_t),
        (qml.SX, dev._apply_sx),
    ]
    two_qubit_ops = [
        (qml.CNOT, dev._apply_cnot),
        (qml.SWAP, dev._apply_swap),
        (qml.CZ, dev._apply_cz),
    ]
    three_qubit_ops = [
        (qml.Toffoli, dev._apply_toffoli),
    ]

    @pytest.mark.parametrize("op, method", single_qubit_ops)
    def test_apply_single_qubit_op(self, op, method, inverse):
        """Test if the application of single qubit operations is correct."""
        state_out = method(self.state, axes=[1], inverse=inverse)
        op = op(wires=[1])
        matrix = op.inv().matrix() if inverse else op.matrix()
        state_out_einsum = np.einsum("ab,ibjk->iajk", matrix, self.state)
        assert np.allclose(state_out, state_out_einsum)

    @pytest.mark.parametrize("op, method", two_qubit_ops)
    def test_apply_two_qubit_op(self, op, method, inverse):
        """Test if the application of two qubit operations is correct."""
        state_out = method(self.state, axes=[0, 1])
        op = op(wires=[0, 1])
        matrix = op.inv().matrix() if inverse else op.matrix()
        matrix = matrix.reshape((2, 2, 2, 2))
        state_out_einsum = np.einsum("abcd,cdjk->abjk", matrix, self.state)
        assert np.allclose(state_out, state_out_einsum)

    @pytest.mark.parametrize("op, method", two_qubit_ops)
    def test_apply_two_qubit_op_reverse(self, op, method, inverse):
        """Test if the application of two qubit operations is correct when the applied wires are
        reversed."""
        state_out = method(self.state, axes=[2, 1])
        op = op(wires=[2, 1])
        matrix = op.inv().matrix() if inverse else op.matrix()
        matrix = matrix.reshape((2, 2, 2, 2))
        state_out_einsum = np.einsum("abcd,idck->ibak", matrix, self.state)
        assert np.allclose(state_out, state_out_einsum)

    @pytest.mark.parametrize("op, method", three_qubit_ops)
    def test_apply_three_qubit_op_controls_smaller(self, op, method, inverse):
        """Test if the application of three qubit operations is correct when both control wires are
        smaller than the target wire."""
        state_out = method(self.state, axes=[0, 2, 3])
        op = op(wires=[0, 2, 3])
        matrix = op.inv().matrix() if inverse else op.matrix()
        matrix = matrix.reshape((2, 2) * 3)
        state_out_einsum = np.einsum("abcdef,dkef->akbc", matrix, self.state)
        assert np.allclose(state_out, state_out_einsum)

    @pytest.mark.parametrize("op, method", three_qubit_ops)
    def test_apply_three_qubit_op_controls_greater(self, op, method, inverse):
        """Test if the application of three qubit operations is correct when both control wires are
        greater than the target wire."""
        state_out = method(self.state, axes=[2, 1, 0])
        op = op(wires=[2, 1, 0])
        matrix = op.inv().matrix() if inverse else op.matrix()
        matrix = matrix.reshape((2, 2) * 3)
        state_out_einsum = np.einsum("abcdef,fedk->cbak", matrix, self.state)
        assert np.allclose(state_out, state_out_einsum)

    @pytest.mark.parametrize("op, method", three_qubit_ops)
    def test_apply_three_qubit_op_controls_split(self, op, method, inverse):
        """Test if the application of three qubit operations is correct when one control wire is smaller
        and one control wire is greater than the target wire."""
        state_out = method(self.state, axes=[3, 1, 2])
        op = op(wires=[3, 1, 2])
        matrix = op.inv().matrix() if inverse else op.matrix()
        matrix = matrix.reshape((2, 2) * 3)
        state_out_einsum = np.einsum("abcdef,kdfe->kacb", matrix, self.state)
        assert np.allclose(state_out, state_out_einsum)


class TestStateVector:
    """Unit tests for the _apply_state_vector method"""

    def test_full_subsystem(self, mocker):
        """Test applying a state vector to the full subsystem"""
        dev = DefaultQubit(wires=["a", "b", "c"])
        state = np.array([1, 0, 0, 0, 1, 0, 1, 1]) / 2.0
        state_wires = qml.wires.Wires(["a", "b", "c"])

        spy = mocker.spy(dev, "_scatter")
        dev._apply_state_vector(state=state, device_wires=state_wires)

        assert np.all(dev._state.flatten() == state)
        spy.assert_not_called()

    def test_partial_subsystem(self, mocker):
        """Test applying a state vector to a subset of wires of the full subsystem"""

        dev = DefaultQubit(wires=["a", "b", "c"])
        state = np.array([1, 0, 1, 0]) / np.sqrt(2.0)
        state_wires = qml.wires.Wires(["a", "c"])

        spy = mocker.spy(dev, "_scatter")
        dev._apply_state_vector(state=state, device_wires=state_wires)
        res = np.sum(dev._state, axis=(1,)).flatten()

        assert np.all(res == state)
        spy.assert_called()


class TestInverseDecomposition:
    """Integration tests for decompositions of inverse gates"""

    def test_inverse_S(self, tol):
        """Test that applying the inverse of the S gate
        works without decomposition"""

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev, diff_method="parameter-shift")
        def test_s():
            qml.Hadamard(wires=0)
            qml.S(wires=0)
            return qml.probs(wires=0)

        test_s()
        operations = test_s.qtape.operations
        assert "S" in [i.name for i in operations]

        expected = np.array([1.0, 1.0j]) / np.sqrt(2)
        assert np.allclose(dev.state, expected, atol=tol, rtol=0)

        @qml.qnode(dev, diff_method="parameter-shift")
        def test_s_inverse():
            qml.Hadamard(wires=0)
            qml.S(wires=0).inv()
            return qml.probs(wires=0)

        test_s_inverse()
        operations = test_s_inverse.qtape.operations
        assert "S.inv" in [i.name for i in operations]

        expected = np.array([1.0, -1.0j]) / np.sqrt(2)
        assert np.allclose(dev.state, expected, atol=tol, rtol=0)

    def test_inverse_decomposition(self, tol, mocker):
        """Test that the expansion is inverted when inverse gate is applied and the gate isn't supported."""
        dev = qml.device("default.qubit", wires=1)
        spy = mocker.spy(dev, "execute")

        class DummyOp(qml.operation.Operation):
            """A Dummy Op with a decomposition defined."""

            num_wires = 1

            def decomposition(self):
                return [qml.PauliX(self.wires), qml.PauliY(self.wires)]

        @qml.qnode(dev, diff_method=None)
        def test_dummy_inv():
            DummyOp(0).inv()
            return qml.probs(wires=0)

        test_dummy_inv()
        operations = spy.call_args[0][0].operations
        assert qml.equal(operations[0], qml.PauliY(0))
        assert qml.equal(operations[1], qml.PauliX(0))


class TestApplyOperationUnit:
    """Unit tests for the internal _apply_operation method."""

    @pytest.mark.parametrize("inverse", [True, False])
    def test_internal_apply_ops_case(self, inverse, monkeypatch):
        """Tests that if we provide an operation that has an internal
        implementation, then we use that specific implementation.

        This test provides a new internal function that `default.qubit` uses to
        apply `PauliX` (rather than redefining the gate itself).
        """
        dev = qml.device("default.qubit", wires=1)

        # Create a dummy operation
        expected_test_output = np.ones(1)
        supported_gate_application = lambda *args, **kwargs: expected_test_output

        with monkeypatch.context() as m:
            # Set the internal ops implementations dict
            m.setattr(dev, "_apply_ops", {"PauliX": supported_gate_application})

            test_state = np.array([1, 0])
            op = qml.PauliX(0) if not inverse else qml.PauliX(0).inv()

            res = dev._apply_operation(test_state, op)
            assert np.allclose(res, expected_test_output)

    def test_diagonal_operation_case(self, mocker, monkeypatch):
        """Tests the case when the operation to be applied is
        diagonal in the computational basis and the _apply_diagonal_unitary method is used."""
        dev = qml.device("default.qubit", wires=1)
        par = 0.3

        test_state = np.array([1, 0])
        wires = 0
        op = qml.PhaseShift(par, wires=wires)
        assert op.name not in dev._apply_ops

        # Set the internal _apply_diagonal_unitary
        history = []
        mock_apply_diag = lambda state, matrix, wires: history.append((state, matrix, wires))
        with monkeypatch.context() as m:
            m.setattr(dev, "_apply_diagonal_unitary", mock_apply_diag)
            assert dev._apply_diagonal_unitary == mock_apply_diag

            dev._apply_operation(test_state, op)

            res_state, res_mat, res_wires = history[0]

            assert np.allclose(res_state, test_state)
            assert np.allclose(res_mat, np.diag(op.matrix()))
            assert np.allclose(res_wires, wires)

    def test_apply_einsum_case(self, mocker, monkeypatch):
        """Tests the case when np.einsum is used to apply an operation in
        default.qubit."""
        dev = qml.device("default.qubit", wires=1)

        test_state = np.array([1, 0])
        wires = 0

        # Redefine the S gate so that it is an example for a one-qubit gate
        # that is not registered in the diagonal_in_z_basis attribute
        class TestSGate(qml.operation.Operation):
            num_wires = 1

            @staticmethod
            def compute_matrix(*params, **hyperparams):
                return np.array([[1, 0], [0, 1j]])

        dev.operations.add("TestSGate")
        op = TestSGate(wires=wires)

        assert op.name in dev.operations
        assert op.name not in dev._apply_ops

        # Set the internal _apply_unitary_einsum
        history = []
        mock_apply_einsum = lambda state, matrix, wires: history.append((state, matrix, wires))
        with monkeypatch.context() as m:
            m.setattr(dev, "_apply_unitary_einsum", mock_apply_einsum)

            dev._apply_operation(test_state, op)

            res_state, res_mat, res_wires = history[0]

            assert np.allclose(res_state, test_state)
            assert np.allclose(res_mat, op.matrix())
            assert np.allclose(res_wires, wires)

    @pytest.mark.parametrize("inverse", [True, False])
    def test_apply_tensordot_case(self, inverse, mocker, monkeypatch):
        """Tests the case when np.tensordot is used to apply an operation in
        default.qubit."""
        dev = qml.device("default.qubit", wires=3)

        test_state = np.array([1, 0])
        wires = [0, 1, 2]

        # Redefine the Toffoli gate so that it is an example for a gate with
        # more than two wires
        class TestToffoli(qml.operation.Operation):
            num_wires = 3

            @staticmethod
            def compute_matrix(*params, **hyperparams):
                return U_toffoli

        dev.operations.add("TestToffoli")
        op = TestToffoli(wires=wires)

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

    def test_apply_unitary_tensordot_double_broadcasting_error(self):
        """Tests that an error is raised if attempting to use _apply_unitary
        with a broadcasted matrix and a broadcasted state simultaneously."""
        dev = qml.device("default.qubit", wires=3)

        class BroadcastedToffoli(qml.operation.Operation):
            num_wires = 3
            batch_size = 3
            num_params = 0

            @staticmethod
            def compute_matrix(*params, **hyperparams):
                return np.array([U_toffoli] * 3)

        state = np.eye(8)[:3]
        wires = qml.wires.Wires([0, 1, 2])

        mat = BroadcastedToffoli(wires=wires).matrix()
        with pytest.raises(NotImplementedError, match="broadcasted unitary to an already"):
            dev._apply_unitary(state, mat, wires=wires)

    def test_identity_skipped(self, mocker):
        """Test that applying the identity operation does not perform any additional computations."""
        dev = qml.device("default.qubit", wires=1)

        starting_state = np.array([1, 0])
        op = qml.Identity(0)

        spy_diagonal = mocker.spy(dev, "_apply_diagonal_unitary")
        spy_einsum = mocker.spy(dev, "_apply_unitary_einsum")
        spy_unitary = mocker.spy(dev, "_apply_unitary")

        res = dev._apply_operation(starting_state, op)
        assert res is starting_state

        spy_diagonal.assert_not_called()
        spy_einsum.assert_not_called()
        spy_unitary.assert_not_called()


class TestHamiltonianSupport:
    """Tests the devices' native support for Hamiltonian observables."""

    def test_do_not_split_analytic(self, mocker):
        """Tests that the Hamiltonian is not split for shots=None."""
        dev = qml.device("default.qubit", wires=2)
        H = qml.Hamiltonian(np.array([0.1, 0.2]), [qml.PauliX(0), qml.PauliZ(1)])

        @qml.qnode(dev, diff_method="parameter-shift", interface=None)
        def circuit():
            return qml.expval(H)

        spy = mocker.spy(dev, "expval")

        circuit()
        # evaluated one expval altogether
        assert spy.call_count == 1

    def test_split_finite_shots(self, mocker):
        """Tests that the Hamiltonian is split for finite shots."""
        dev = qml.device("default.qubit", wires=2, shots=10)
        spy = mocker.spy(dev, "expval")

        H = qml.Hamiltonian(np.array([0.1, 0.2]), [qml.PauliX(0), qml.PauliZ(1)])

        @qml.qnode(dev)
        def circuit():
            return qml.expval(H)

        circuit()

        # evaluated one expval per Pauli observable
        assert spy.call_count == 2

    def test_error_hamiltonian_expval_finite_shots(self):
        """Tests that the Hamiltonian is split for finite shots."""
        dev = qml.device("default.qubit", wires=2, shots=10)
        H = qml.Hamiltonian([0.1, 0.2], [qml.PauliX(0), qml.PauliZ(1)])

        with pytest.raises(AssertionError, match="Hamiltonian must be used with shots=None"):
            dev.expval(H)


class TestGetBatchSize:
    """Tests for the helper method ``_get_batch_size`` of ``QubitDevice``."""

    @pytest.mark.parametrize("shape", [(4, 4), (1, 8), (4,)])
    def test_batch_size_None(self, shape):
        """Test that a ``batch_size=None`` is reported correctly."""
        dev = qml.device("default.qubit", wires=1)
        tensor0 = np.ones(shape, dtype=complex)
        assert dev._get_batch_size(tensor0, shape, qml.math.prod(shape)) is None
        tensor1 = np.arange(np.prod(shape)).reshape(shape)
        assert dev._get_batch_size(tensor1, shape, qml.math.prod(shape)) is None

    @pytest.mark.parametrize("shape", [(4, 4), (1, 8), (4,)])
    @pytest.mark.parametrize("batch_size", [1, 3])
    def test_batch_size_int(self, shape, batch_size):
        """Test that an integral ``batch_size`` is reported correctly."""
        dev = qml.device("default.qubit", wires=1)
        full_shape = (batch_size,) + shape
        tensor0 = np.ones(full_shape, dtype=complex)
        assert dev._get_batch_size(tensor0, shape, qml.math.prod(shape)) == batch_size
        tensor1 = np.arange(np.prod(full_shape)).reshape(full_shape)
        assert dev._get_batch_size(tensor1, shape, qml.math.prod(shape)) == batch_size

    @pytest.mark.filterwarnings("ignore:Creating an ndarray from ragged nested")
    def test_invalid_tensor(self):
        """Test that an error is raised if a tensor is provided that does not
        have a proper shape/ndim."""
        dev = qml.device("default.qubit", wires=1)
        with pytest.raises(ValueError, match="could not broadcast"):
            dev._get_batch_size([qml.math.ones((2, 3)), qml.math.ones((2, 2))], (2, 2, 2), 8)

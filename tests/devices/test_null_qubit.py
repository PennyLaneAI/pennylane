# Copyright 2022 Xanadu Quantum Technologies Inc.

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
from pennylane import numpy as np


@pytest.fixture(scope="function", params=[(np.float32, np.complex64), (np.float64, np.complex128)])
def nullqubit_device_1_wire(request):
    return qml.device("null.qubit", wires=1, r_dtype=request.param[0], c_dtype=request.param[1])


@pytest.fixture(scope="function", params=[(np.float32, np.complex64), (np.float64, np.complex128)])
def nullqubit_device_2_wires(request):
    return qml.device("null.qubit", wires=2, r_dtype=request.param[0], c_dtype=request.param[1])


@pytest.fixture(scope="function", params=[(np.float32, np.complex64), (np.float64, np.complex128)])
def nullqubit_device_3_wires(request):
    return qml.device("null.qubit", wires=3, r_dtype=request.param[0], c_dtype=request.param[1])


class TestApply:
    """Tests that operations and inverses of certain operations are applied correctly or that the proper
    errors are raised.
    """

    test_data_no_parameters = [
        (qml.PauliX, [1 / math.sqrt(5), 2 / math.sqrt(5)]),
        (qml.PauliY, [1 / math.sqrt(5), 2 / math.sqrt(5)]),
        (qml.PauliZ, [1 / math.sqrt(5), 2 / math.sqrt(5)]),
        (qml.S, [1 / math.sqrt(5), 2 / math.sqrt(5)]),
        (qml.T, [1 / math.sqrt(5), 2 / math.sqrt(5)]),
        (qml.Hadamard, [1 / math.sqrt(5), 2 / math.sqrt(5)]),
        (qml.Identity, [1 / math.sqrt(5), 2 / math.sqrt(5)]),
    ]

    @pytest.mark.parametrize("operation,input", test_data_no_parameters)
    def test_apply_operation_single_wire_no_parameters(
        self, nullqubit_device_1_wire, tol, operation, input
    ):
        """Tests that applying an operation yields the expected output state for single wire
        operations that have no parameters."""

        nullqubit_device_1_wire._state = np.array(input, dtype=nullqubit_device_1_wire.C_DTYPE)
        nullqubit_device_1_wire.apply([operation(wires=[0])])

        assert np.allclose(nullqubit_device_1_wire._state, np.array(input), atol=tol, rtol=0)
        assert nullqubit_device_1_wire._state.dtype == nullqubit_device_1_wire.C_DTYPE

    @pytest.mark.parametrize("operation,input", test_data_no_parameters)
    def test_apply_operation_single_wire_no_parameters_inverse(
        self, nullqubit_device_1_wire, tol, operation, input
    ):
        """Tests that applying the inverse of an operation yields the expected output state for
        single wire operations that have no parameters."""

        nullqubit_device_1_wire._state = np.array(input, dtype=nullqubit_device_1_wire.C_DTYPE)
        nullqubit_device_1_wire.apply([operation(wires=[0]).inv()])

        assert np.allclose(nullqubit_device_1_wire._state, np.array(input), atol=tol, rtol=0)
        assert nullqubit_device_1_wire._state.dtype == nullqubit_device_1_wire.C_DTYPE

    test_data_two_wires_no_parameters = [
        (qml.CNOT, [1 / math.sqrt(30), 2 / math.sqrt(30), 3 / math.sqrt(30), 4 / math.sqrt(30)]),
        (qml.SWAP, [1 / math.sqrt(30), 2 / math.sqrt(30), 3 / math.sqrt(30), 4 / math.sqrt(30)]),
        (qml.CZ, [1 / math.sqrt(30), 2 / math.sqrt(30), 3 / math.sqrt(30), 4 / math.sqrt(30)]),
    ]

    test_data_iswap = [
        (qml.ISWAP, [1 / math.sqrt(30), 2 / math.sqrt(30), 3 / math.sqrt(30), 4 / math.sqrt(30)]),
    ]

    test_data_siswap = [
        (qml.SISWAP, [1 / math.sqrt(30), 2 / math.sqrt(30), 3 / math.sqrt(30), 4 / math.sqrt(30)]),
    ]

    test_data_sqisw = [
        (qml.SQISW, [1 / math.sqrt(30), 2 / math.sqrt(30), 3 / math.sqrt(30), 4 / math.sqrt(30)]),
    ]

    all_two_wires_no_parameters = (
        test_data_two_wires_no_parameters + test_data_iswap + test_data_siswap + test_data_sqisw
    )

    @pytest.mark.parametrize("operation,input", all_two_wires_no_parameters)
    def test_apply_operation_two_wires_no_parameters(
        self, nullqubit_device_2_wires, tol, operation, input
    ):
        """Tests that applying an operation yields the expected output state for two wires
        operations that have no parameters."""

        nullqubit_device_2_wires._state = np.array(
            input, dtype=nullqubit_device_2_wires.C_DTYPE
        ).reshape((2, 2))
        nullqubit_device_2_wires.apply([operation(wires=[0, 1])])

        assert np.allclose(
            nullqubit_device_2_wires._state.flatten(), np.array(input), atol=tol, rtol=0
        )
        assert nullqubit_device_2_wires._state.dtype == nullqubit_device_2_wires.C_DTYPE

    @pytest.mark.parametrize("operation,input", all_two_wires_no_parameters)
    def test_apply_operation_two_wires_no_parameters_inverse(
        self, nullqubit_device_2_wires, tol, operation, input
    ):
        """Tests that applying the inverse of an operation yields the expected output state for
        two wires operations that have no parameters."""

        nullqubit_device_2_wires._state = np.array(
            input, dtype=nullqubit_device_2_wires.C_DTYPE
        ).reshape((2, 2))
        nullqubit_device_2_wires.apply([operation(wires=[0, 1]).inv()])

        assert np.allclose(
            nullqubit_device_2_wires._state.flatten(),
            np.array(input),
            atol=tol,
            rtol=0,
        )
        assert nullqubit_device_2_wires._state.dtype == nullqubit_device_2_wires.C_DTYPE

    test_data_three_wires_no_parameters = [
        (
            qml.CSWAP,
            [
                1 / math.sqrt(204),
                2 / math.sqrt(204),
                3 / math.sqrt(204),
                4 / math.sqrt(204),
                5 / math.sqrt(204),
                6 / math.sqrt(204),
                7 / math.sqrt(204),
                8 / math.sqrt(204),
            ],
        ),
    ]

    @pytest.mark.parametrize("operation,input", test_data_three_wires_no_parameters)
    def test_apply_operation_three_wires_no_parameters(
        self, nullqubit_device_3_wires, tol, operation, input
    ):
        """Tests that applying an operation yields the expected output state for three wires
        operations that have no parameters."""

        nullqubit_device_3_wires._state = np.array(
            input, dtype=nullqubit_device_3_wires.C_DTYPE
        ).reshape((2, 2, 2))
        nullqubit_device_3_wires.apply([operation(wires=[0, 1, 2])])

        assert np.allclose(
            nullqubit_device_3_wires._state.flatten(), np.array(input), atol=tol, rtol=0
        )
        assert nullqubit_device_3_wires._state.dtype == nullqubit_device_3_wires.C_DTYPE

    @pytest.mark.parametrize("operation,input", test_data_three_wires_no_parameters)
    def test_apply_operation_three_wires_no_parameters_inverse(
        self, nullqubit_device_3_wires, tol, operation, input
    ):
        """Tests that applying the inverse of an operation yields the expected output state for
        three wires operations that have no parameters."""

        nullqubit_device_3_wires._state = np.array(
            input, dtype=nullqubit_device_3_wires.C_DTYPE
        ).reshape((2, 2, 2))
        nullqubit_device_3_wires.apply([operation(wires=[0, 1, 2]).inv()])

        assert np.allclose(
            nullqubit_device_3_wires._state.flatten(), np.array(input), atol=tol, rtol=0
        )
        assert nullqubit_device_3_wires._state.dtype == nullqubit_device_3_wires.C_DTYPE

    @pytest.mark.parametrize(
        "operation,input,expected_output",
        [
            (
                qml.BasisState,
                [1 / math.sqrt(30), 2 / math.sqrt(30), 3 / math.sqrt(30), 4 / math.sqrt(30)],
                None,
            ),
            (
                qml.QubitStateVector,
                [1 / math.sqrt(30), 2 / math.sqrt(30), 3 / math.sqrt(30), 4 / math.sqrt(30)],
                None,
            ),
        ],
    )
    def test_apply_operation_state_preparation(
        self, nullqubit_device_2_wires, tol, operation, input, expected_output
    ):
        """Tests that the null.qubit does nothing regarding state initialization."""

        input = np.array(input)
        nullqubit_device_2_wires.reset()
        nullqubit_device_2_wires.apply([operation(input, wires=[0, 1])])
        print(expected_output)
        assert nullqubit_device_2_wires._state == expected_output

    test_data_single_wire_with_parameters = [
        (qml.PhaseShift, [1 / math.sqrt(5), 2 / math.sqrt(5)], [math.pi / 4]),
        (qml.RX, [1 / math.sqrt(5), 2 / math.sqrt(5)], [math.pi / 4]),
        (qml.RY, [1 / math.sqrt(5), 2 / math.sqrt(5)], [math.pi / 4]),
        (qml.RZ, [1 / math.sqrt(5), 2 / math.sqrt(5)], [math.pi / 4]),
        (qml.MultiRZ, [1 / math.sqrt(5), 2 / math.sqrt(5)], [math.pi / 4]),
        (qml.Rot, [1 / math.sqrt(5), 2 / math.sqrt(5)], [math.pi / 2, math.pi / 4, math.pi / 8]),
        (
            qml.QubitUnitary,
            [1 / math.sqrt(5), 2 / math.sqrt(5)],
            [
                np.array(
                    [
                        [1j / math.sqrt(2), 1j / math.sqrt(2)],
                        [1j / math.sqrt(2), -1j / math.sqrt(2)],
                    ]
                )
            ],
        ),
        (qml.DiagonalQubitUnitary, [1 / math.sqrt(5), 2 / math.sqrt(5)], [np.array([-1, 1])]),
    ]

    @pytest.mark.parametrize("operation,input,par", test_data_single_wire_with_parameters)
    def test_apply_operation_single_wire_with_parameters(
        self, nullqubit_device_1_wire, tol, operation, input, par
    ):
        """Tests that applying an operation yields the expected output state for single wire
        operations that have parameters."""

        nullqubit_device_1_wire._state = np.array(input, dtype=nullqubit_device_1_wire.C_DTYPE)
        nullqubit_device_1_wire.apply([operation(*par, wires=[0])])
        assert np.allclose(nullqubit_device_1_wire._state, np.array(input), atol=tol, rtol=0)
        assert nullqubit_device_1_wire._state.dtype == nullqubit_device_1_wire.C_DTYPE

    @pytest.mark.parametrize("operation,input,par", test_data_single_wire_with_parameters)
    def test_apply_operation_single_wire_with_parameters_inverse(
        self, nullqubit_device_1_wire, tol, operation, input, par
    ):
        """Tests that applying the inverse of an operation yields the expected output state for single wire
        operations that have parameters."""

        nullqubit_device_1_wire._state = np.array(input, dtype=nullqubit_device_1_wire.C_DTYPE)
        nullqubit_device_1_wire.apply([operation(*par, wires=[0]).inv()])

        assert np.allclose(nullqubit_device_1_wire._state, np.array(input), atol=tol, rtol=0)
        assert nullqubit_device_1_wire._state.dtype == nullqubit_device_1_wire.C_DTYPE

    test_data_two_wires_with_parameters = [
        (
            qml.CRX,
            [1 / math.sqrt(30), 2 / math.sqrt(30), 3 / math.sqrt(30), 4 / math.sqrt(30)],
            [math.pi / 2],
        ),
        (
            qml.CRY,
            [1 / math.sqrt(30), 2 / math.sqrt(30), 3 / math.sqrt(30), 4 / math.sqrt(30)],
            [math.pi / 2],
        ),
        (
            qml.CRZ,
            [1 / math.sqrt(30), 2 / math.sqrt(30), 3 / math.sqrt(30), 4 / math.sqrt(30)],
            [math.pi / 2],
        ),
        (
            qml.MultiRZ,
            [1 / math.sqrt(30), 2 / math.sqrt(30), 3 / math.sqrt(30), 4 / math.sqrt(30)],
            [math.pi / 2],
        ),
        (
            qml.CRot,
            [1 / math.sqrt(30), 2 / math.sqrt(30), 3 / math.sqrt(30), 4 / math.sqrt(30)],
            [math.pi / 2, 0, 0],
        ),
        (
            qml.QubitUnitary,
            [1 / math.sqrt(30), 2 / math.sqrt(30), 3 / math.sqrt(30), 4 / math.sqrt(30)],
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
            qml.DiagonalQubitUnitary,
            [1 / math.sqrt(30), 2 / math.sqrt(30), 3 / math.sqrt(30), 4 / math.sqrt(30)],
            [np.array([-1, 1, 1, -1])],
        ),
        (
            qml.IsingXX,
            [1 / math.sqrt(30), 2 / math.sqrt(30), 3 / math.sqrt(30), 4 / math.sqrt(30)],
            [math.pi / 2],
        ),
        (
            qml.IsingYY,
            [1 / math.sqrt(30), 2 / math.sqrt(30), 3 / math.sqrt(30), 4 / math.sqrt(30)],
            [math.pi / 2],
        ),
        (
            qml.IsingZZ,
            [1 / math.sqrt(30), 2 / math.sqrt(30), 3 / math.sqrt(30), 4 / math.sqrt(30)],
            [math.pi / 2],
        ),
    ]

    @pytest.mark.parametrize("operation,input,par", test_data_two_wires_with_parameters)
    def test_apply_operation_two_wires_with_parameters(
        self, nullqubit_device_2_wires, tol, operation, input, par
    ):
        """Tests that applying an operation yields the expected output state for two wire
        operations that have parameters."""

        nullqubit_device_2_wires._state = np.array(
            input, dtype=nullqubit_device_2_wires.C_DTYPE
        ).reshape((2, 2))
        nullqubit_device_2_wires.apply([operation(*par, wires=[0, 1])])

        assert np.allclose(
            nullqubit_device_2_wires._state.flatten(), np.array(input), atol=tol, rtol=0
        )
        assert nullqubit_device_2_wires._state.dtype == nullqubit_device_2_wires.C_DTYPE

    @pytest.mark.parametrize("operation,input,par", test_data_two_wires_with_parameters)
    def test_apply_operation_two_wires_with_parameters_inverse(
        self, nullqubit_device_2_wires, tol, operation, input, par
    ):
        """Tests that applying the inverse of an operation yields the expected output state for two wires
        operations that have parameters."""

        nullqubit_device_2_wires._state = np.array(
            input, dtype=nullqubit_device_2_wires.C_DTYPE
        ).reshape((2, 2))
        nullqubit_device_2_wires.apply([operation(*par, wires=[0, 1]).inv()])

        assert np.allclose(
            nullqubit_device_2_wires._state.flatten(), np.array(input), atol=tol, rtol=0
        )
        assert nullqubit_device_2_wires._state.dtype == nullqubit_device_2_wires.C_DTYPE


class TestExpval:
    """Tests that expectation values are properly calculated or that the proper errors are raised."""

    @pytest.mark.parametrize(
        "operation,input",
        [
            (qml.PauliX, [1 / math.sqrt(5), 2 / math.sqrt(5)]),
            (qml.PauliY, [1 / math.sqrt(5), 2 / math.sqrt(5)]),
            (qml.PauliZ, [1 / math.sqrt(5), 2 / math.sqrt(5)]),
            (qml.Hadamard, [1 / math.sqrt(5), 2 / math.sqrt(5)]),
            (qml.Identity, [1 / math.sqrt(5), 2 / math.sqrt(5)]),
        ],
    )
    def test_expval_single_wire_no_parameters(self, nullqubit_device_1_wire, operation, input):
        """Tests that expectation values are properly calculated for single-wire observables without parameters."""

        obs = operation(wires=[0])

        nullqubit_device_1_wire.reset()
        nullqubit_device_1_wire.apply(
            [qml.QubitStateVector(np.array(input), wires=[0])], obs.diagonalizing_gates()
        )
        res = nullqubit_device_1_wire.expval(obs)
        assert res == None

    @pytest.mark.parametrize(
        "operation,input,par",
        [
            (qml.Hermitian, [1, 0], [[1, 1j], [-1j, 1]]),
            (qml.Hermitian, [0, 1], [[1, 1j], [-1j, 1]]),
            (qml.Hermitian, [1 / math.sqrt(2), -1 / math.sqrt(2)], [[1, 1j], [-1j, 1]]),
        ],
    )
    def test_expval_single_wire_with_parameters(
        self, nullqubit_device_1_wire, operation, input, par
    ):
        """Tests that expectation values are properly calculated for single-wire observables with parameters."""

        obs = operation(np.array(par), wires=[0])

        nullqubit_device_1_wire.reset()
        nullqubit_device_1_wire.apply(
            [qml.QubitStateVector(np.array(input), wires=[0])], obs.diagonalizing_gates()
        )
        res = nullqubit_device_1_wire.expval(obs)

        assert res == None

    @pytest.mark.parametrize(
        "operation,input,par",
        [
            (
                qml.Hermitian,
                [1 / math.sqrt(3), 0, 1 / math.sqrt(3), 1 / math.sqrt(3)],
                [[1, 1j, 0, 1], [-1j, 1, 0, 0], [0, 0, 1, -1j], [1, 0, 1j, 1]],
            ),
            (
                qml.Hermitian,
                [1 / math.sqrt(3), 0, 1 / math.sqrt(3), 1 / math.sqrt(3)],
                [[0, 1j, 0, 0], [-1j, 0, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]],
            ),
            (
                qml.Hermitian,
                [1 / math.sqrt(3), 0, 1 / math.sqrt(3), 1 / math.sqrt(3)],
                [[1, 1j, 0, 0.5j], [-1j, 1, 0, 0], [0, 0, 1, -1j], [-0.5j, 0, 1j, 1]],
            ),
        ],
    )
    def test_expval_two_wires_with_parameters(
        self, nullqubit_device_2_wires, tol, operation, input, par
    ):
        """Tests that expectation values are properly calculated for two-wire observables with parameters."""

        obs = operation(np.array(par), wires=[0, 1])

        nullqubit_device_2_wires.reset()
        nullqubit_device_2_wires.apply(
            [qml.QubitStateVector(np.array(input), wires=[0, 1])], obs.diagonalizing_gates()
        )
        res = nullqubit_device_2_wires.expval(obs)

        assert res == None

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
Unit tests for the :mod:`pennylane.plugin.NullQubit` device.
"""
import cmath

# pylint: disable=protected-access,cell-var-from-loop
import math

import pytest
import pennylane as qml
from pennylane import numpy as np, DeviceError
from pennylane.devices.null_qubit import NullQubit

from collections import defaultdict


@pytest.fixture(scope="function", params=[(np.float32, np.complex64), (np.float64, np.complex128)])
def nullqubit_device_1_wire(request):
    return qml.device("null.qubit", wires=1, r_dtype=request.param[0], c_dtype=request.param[1])


@pytest.fixture(scope="function", params=[(np.float32, np.complex64), (np.float64, np.complex128)])
def nullqubit_device_2_wires(request):
    return qml.device("null.qubit", wires=2, r_dtype=request.param[0], c_dtype=request.param[1])


@pytest.fixture(scope="function", params=[(np.float32, np.complex64), (np.float64, np.complex128)])
def nullqubit_device_3_wires(request):
    return qml.device("null.qubit", wires=3, r_dtype=request.param[0], c_dtype=request.param[1])


def test_analytic_deprecation():
    """Tests if the kwarg `analytic` is used and displays error message."""
    msg = "The analytic argument has been replaced by shots=None. "
    msg += "Please use shots=None instead of analytic=True."

    with pytest.raises(
        DeviceError,
        match=msg,
    ):
        qml.device("null.qubit", wires=1, shots=1, analytic=True)


def test_dtype_errors():
    """Test that if an incorrect dtype is provided to the device then an error is raised."""
    with pytest.raises(DeviceError, match="Real datatype must be a floating point type."):
        qml.device("null.qubit", wires=1, r_dtype=np.complex128)
    with pytest.raises(
        DeviceError, match="Complex datatype must be a complex floating point type."
    ):
        qml.device("null.qubit", wires=1, c_dtype=np.float64)


def test_custom_op_with_matrix():
    """Test that a dummy op with a matrix is supported."""

    class DummyOp(qml.operation.Operation):
        num_wires = 1

        def compute_matrix(self):
            return np.eye(2)

    with qml.tape.QuantumTape() as tape:
        DummyOp(0)
        qml.state()

    dev = qml.device("null.qubit", wires=1)
    assert dev.execute(tape) == None


class TestApply:
    """Tests that operations and inverses of certain operations are applied correctly."""

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
        "operation,input",
        [
            (
                qml.BasisState,
                [1 / math.sqrt(30), 2 / math.sqrt(30), 3 / math.sqrt(30), 4 / math.sqrt(30)],
            ),
            (
                qml.QubitStateVector,
                [1 / math.sqrt(30), 2 / math.sqrt(30), 3 / math.sqrt(30), 4 / math.sqrt(30)],
            ),
        ],
    )
    def test_apply_operation_state_preparation(
        self, nullqubit_device_2_wires, tol, operation, input
    ):
        """Tests that the null.qubit does nothing regarding state initialization."""

        input = np.array(input)
        nullqubit_device_2_wires.reset()
        nullqubit_device_2_wires.apply([operation(input, wires=[0, 1])])
        assert nullqubit_device_2_wires._state == None

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
        """Tests that applying an operation yields the expected output state for two wires
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

    @pytest.mark.parametrize(
        "r_dtype,c_dtype", [(np.float32, np.complex64), (np.float64, np.complex128)]
    )
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
    def test_advanced_op(self, r_dtype, c_dtype, op, tol):
        """Test some advanced operations."""

        dev = qml.device("null.qubit", wires=4, r_dtype=r_dtype, c_dtype=c_dtype)

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

        assert circuit() == None


class TestExpval:
    """Tests that expectation values are properly (not) calculated."""

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
                [1 / math.sqrt(30), 2 / math.sqrt(30), 3 / math.sqrt(30), 4 / math.sqrt(30)],
                [[1, 1j, 0, 1], [-1j, 1, 0, 0], [0, 0, 1, -1j], [1, 0, 1j, 1]],
            ),
            (
                qml.Hermitian,
                [1 / math.sqrt(30), 2 / math.sqrt(30), 3 / math.sqrt(30), 4 / math.sqrt(30)],
                [[0, 1j, 0, 0], [-1j, 0, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]],
            ),
            (
                qml.Hermitian,
                [1 / math.sqrt(30), 2 / math.sqrt(30), 3 / math.sqrt(30), 4 / math.sqrt(30)],
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


class TestVar:
    """Tests that variances are properly (not) calculated."""

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
    def test_var_single_wire_no_parameters(self, nullqubit_device_1_wire, operation, input):
        """Tests that variances are properly (not) calculated for single-wire observables without parameters."""

        obs = operation(wires=[0])

        nullqubit_device_1_wire.reset()
        nullqubit_device_1_wire.apply(
            [qml.QubitStateVector(np.array(input), wires=[0])], obs.diagonalizing_gates()
        )
        res = nullqubit_device_1_wire.var(obs)
        assert res == None

    @pytest.mark.parametrize(
        "operation,input,par",
        [
            (qml.Hermitian, [1, 0], [[1, 1j], [-1j, 1]]),
            (qml.Hermitian, [0, 1], [[1, 1j], [-1j, 1]]),
            (qml.Hermitian, [1 / math.sqrt(2), -1 / math.sqrt(2)], [[1, 1j], [-1j, 1]]),
        ],
    )
    def test_var_single_wire_with_parameters(self, nullqubit_device_1_wire, operation, input, par):
        """Tests that variances are properly (not) calculated for single-wire observables with parameters."""

        obs = operation(np.array(par), wires=[0])

        nullqubit_device_1_wire.reset()
        nullqubit_device_1_wire.apply(
            [qml.QubitStateVector(np.array(input), wires=[0])], obs.diagonalizing_gates()
        )
        res = nullqubit_device_1_wire.var(obs)

        assert res == None

    @pytest.mark.parametrize(
        "operation,input,par",
        [
            (
                qml.Hermitian,
                [1 / math.sqrt(30), 2 / math.sqrt(30), 3 / math.sqrt(30), 4 / math.sqrt(30)],
                [[1, 1j, 0, 1], [-1j, 1, 0, 0], [0, 0, 1, -1j], [1, 0, 1j, 1]],
            ),
            (
                qml.Hermitian,
                [1 / math.sqrt(30), 2 / math.sqrt(30), 3 / math.sqrt(30), 4 / math.sqrt(30)],
                [[0, 1j, 0, 0], [-1j, 0, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]],
            ),
            (
                qml.Hermitian,
                [1 / math.sqrt(30), 2 / math.sqrt(30), 3 / math.sqrt(30), 4 / math.sqrt(30)],
                [[1, 1j, 0, 0.5j], [-1j, 1, 0, 0], [0, 0, 1, -1j], [-0.5j, 0, 1j, 1]],
            ),
        ],
    )
    def test_var_two_wires_with_parameters(
        self, nullqubit_device_2_wires, tol, operation, input, par
    ):
        """Tests that variances are properly (not) calculated for two-wire observables with parameters."""

        obs = operation(np.array(par), wires=[0, 1])

        nullqubit_device_2_wires.reset()
        nullqubit_device_2_wires.apply(
            [qml.QubitStateVector(np.array(input), wires=[0, 1])], obs.diagonalizing_gates()
        )
        res = nullqubit_device_2_wires.var(obs)

        assert res == None


class TestSample:
    """Tests that samples are properly (not) calculated."""

    def test_sample_values(self):
        """Tests if the samples returned by sample have
        the correct values
        """
        dev = qml.device("null.qubit", wires=2, shots=1000)

        dev.apply([qml.RX(1.5708, wires=[0])])
        dev._wires_measured = {0}
        dev._samples = dev.generate_samples()

        s1 = dev.sample(qml.PauliZ(0))

        assert s1 == None


class TestNullQubitIntegration:
    """Integration tests for null.qubit. These tests ensure it integrates
    properly with the PennyLane interface, in particular QNode."""

    def test_defines_correct_capabilities(self):
        """Test that the device defines the right capabilities"""

        dev = qml.device("null.qubit", wires=1)
        cap = dev.capabilities()
        capabilities = {
            "model": "qubit",
            "supports_broadcasting": True,
            "supports_finite_shots": True,
            "supports_tensor_observables": True,
            "returns_probs": True,
            "supports_inverse_operations": True,
            "supports_analytic_computation": True,
            "returns_state": True,
            "passthru_devices": {
                "tf": "null.qubit.tf",
                "torch": "null.qubit.torch",
                "autograd": "null.qubit.autograd",
                "jax": "null.qubit.jax",
            },
        }
        assert cap == capabilities

    @pytest.mark.parametrize("r_dtype", [np.float32, np.float64])
    def test_qubit_circuit_state(self, nullqubit_device_1_wire, r_dtype):
        """Test that the null qubit plugin provides the correct state for a simple circuit"""

        p = 0.543

        dev = nullqubit_device_1_wire
        dev.R_DTYPE = r_dtype

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.state()

        assert circuit(p) == None

    @pytest.mark.parametrize("r_dtype", [np.float32, np.float64])
    def test_qubit_circuit_expval(self, nullqubit_device_1_wire, r_dtype):
        """Test that the null qubit plugin provides the correct expval for a simple circuit"""

        p = 0.543

        dev = nullqubit_device_1_wire
        dev.R_DTYPE = r_dtype

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliY(0))

        assert circuit(p) == np.array(None, dtype=object)

    @pytest.mark.parametrize("r_dtype", [np.float32, np.float64])
    def test_qubit_circuit_var(self, nullqubit_device_1_wire, r_dtype):
        """Test that the null qubit plugin provides the correct var for a simple circuit"""

        p = 0.543

        dev = nullqubit_device_1_wire
        dev.R_DTYPE = r_dtype

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.var(qml.PauliY(0))

        assert circuit(p) == np.array(None, dtype=object)

    def test_qubit_identity(self, nullqubit_device_1_wire, tol):
        """Test that the null qubit plugin provides correct result for the Identity expectation"""

        p = 0.543

        @qml.qnode(nullqubit_device_1_wire, diff_method="parameter-shift")
        def circuit(x):
            """Test quantum function"""
            qml.RX(x, wires=0)
            return qml.expval(qml.Identity(0))

        assert circuit(p) == np.array(None, dtype=object)

    def test_nonzero_shots(self):
        """Test that the null qubit plugin provides correct result for high shot number"""
        shots = 10**5
        dev = qml.device("null.qubit", wires=1, shots=shots)

        p = 0.543

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(x):
            """Test quantum function"""
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliY(0))

        runs = []
        for _ in range(100):
            runs.append(circuit(p))

        assert np.all(runs == np.array(None, dtype=object))

    @pytest.mark.parametrize(
        "name,state",
        [
            ("PauliX", [1 / math.sqrt(5), 2 / math.sqrt(5)]),
            ("PauliY", [1 / math.sqrt(5), 2 / math.sqrt(5)]),
            ("PauliZ", [1 / math.sqrt(5), 2 / math.sqrt(5)]),
            ("Hadamard", [1 / math.sqrt(5), 2 / math.sqrt(5)]),
        ],
    )
    def test_supported_observable_single_wire_no_parameters(
        self, nullqubit_device_1_wire, name, state
    ):
        """Tests supported observables on single wires without parameters."""

        obs = getattr(qml.ops, name)

        assert nullqubit_device_1_wire.supports_observable(name)

        @qml.qnode(nullqubit_device_1_wire, diff_method="parameter-shift")
        def circuit():
            qml.QubitStateVector(np.array(state), wires=[0])
            return qml.expval(obs(wires=[0]))

        assert circuit() == np.array(None, dtype=object)

    @pytest.mark.parametrize(
        "name,state,par",
        [
            ("Identity", [1 / math.sqrt(5), 2 / math.sqrt(5)], []),
            ("Hermitian", [1 / math.sqrt(5), 2 / math.sqrt(5)], [np.array([[1, 1j], [-1j, 1]])]),
        ],
    )
    def test_supported_observable_single_wire_with_parameters(
        self, nullqubit_device_1_wire, name, state, par
    ):
        """Tests supported observables on single wires with parameters."""

        obs = getattr(qml.ops, name)

        assert nullqubit_device_1_wire.supports_observable(name)

        @qml.qnode(nullqubit_device_1_wire, diff_method="parameter-shift")
        def circuit():
            qml.QubitStateVector(np.array(state), wires=[0])
            return qml.expval(obs(*par, wires=[0]))

        assert circuit() == np.array(None, dtype=object)

    @pytest.mark.parametrize(
        "name,state,par",
        [
            (
                "Hermitian",
                [1 / math.sqrt(30), 2 / math.sqrt(30), 3 / math.sqrt(30), 4 / math.sqrt(30)],
                [np.array([[1, 1j, 0, 1], [-1j, 1, 0, 0], [0, 0, 1, -1j], [1, 0, 1j, 1]])],
            ),
            (
                "Hermitian",
                [1 / math.sqrt(30), 2 / math.sqrt(30), 3 / math.sqrt(30), 4 / math.sqrt(30)],
                [np.array([[1, 1j, 0, 0.5j], [-1j, 1, 0, 0], [0, 0, 1, -1j], [-0.5j, 0, 1j, 1]])],
            ),
            (
                "Hermitian",
                [1 / math.sqrt(30), 2 / math.sqrt(30), 3 / math.sqrt(30), 4 / math.sqrt(30)],
                [np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])],
            ),
        ],
    )
    def test_supported_observable_two_wires_with_parameters(
        self, nullqubit_device_2_wires, tol, name, state, par
    ):
        """Tests supported observables on two wires with parameters."""

        obs = getattr(qml.ops, name)

        assert nullqubit_device_2_wires.supports_observable(name)

        @qml.qnode(nullqubit_device_2_wires, diff_method="parameter-shift")
        def circuit():
            qml.QubitStateVector(np.array(state), wires=[0, 1])
            return qml.expval(obs(*par, wires=[0, 1]))

        assert circuit() == np.array(None, dtype=object)


THETA = np.linspace(0.11, 1, 3)
PHI = np.linspace(0.32, 1, 3)
VARPHI = np.linspace(0.02, 1, 3)


@pytest.mark.parametrize("theta,phi,varphi", list(zip(THETA, PHI, VARPHI)))
class TestTensorExpval:
    """Test if tensor expectation values returns None"""

    def test_paulix_pauliy(self, theta, phi, varphi):
        """Test that a tensor product involving PauliX and PauliY works correctly"""
        dev = qml.device("null.qubit", wires=3)
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

        assert dev.expval(obs) == None

    def test_pauliz_identity(self, theta, phi, varphi):
        """Test that a tensor product involving PauliZ and Identity works correctly"""
        dev = qml.device("null.qubit", wires=3)
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

        assert dev.expval(obs) == None

    def test_pauliz_hadamard(self, theta, phi, varphi):
        """Test that a tensor product involving PauliZ and PauliY and hadamard works correctly"""
        dev = qml.device("null.qubit", wires=3)
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

        assert dev.expval(obs) == None

    def test_hermitian(self, theta, phi, varphi, tol):
        """Test that a tensor product involving qml.Hermitian works correctly"""
        dev = qml.device("null.qubit", wires=3)
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

        assert dev.expval(obs) == None

    def test_hermitian_hermitian(self, theta, phi, varphi, tol):
        """Test that a tensor product involving two Hermitian matrices works correctly"""
        dev = qml.device("null.qubit", wires=3)

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

        assert dev.expval(obs) == None

    def test_hermitian_identity_expectation(self, theta, phi, varphi, tol):
        """Test that a tensor product involving an Hermitian matrix and the identity works correctly"""
        dev = qml.device("null.qubit", wires=2)

        A = np.array(
            [[1.02789352, 1.61296440 - 0.3498192j], [1.61296440 + 0.3498192j, 1.23920938 + 0j]]
        )

        obs = qml.Hermitian(A, wires=[0]) @ qml.Identity(wires=[1])

        dev.apply(
            [qml.RY(theta, wires=[0]), qml.RY(phi, wires=[1]), qml.CNOT(wires=[0, 1])],
            obs.diagonalizing_gates(),
        )

        assert dev.expval(obs) == None

    def test_hermitian_two_wires_identity_expectation(self, theta, phi, varphi, tol):
        """Test that a tensor product involving an Hermitian matrix for two wires and the identity works correctly"""
        dev = qml.device("null.qubit", wires=3)

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

        assert dev.expval(obs) == None


@pytest.mark.parametrize("theta,phi,varphi", list(zip(THETA, PHI, VARPHI)))
class TestTensorVar:
    """Test if tensor variance returns None"""

    def test_paulix_pauliy(self, theta, phi, varphi):
        """Test that a tensor product involving PauliX and PauliY works correctly"""
        dev = qml.device("null.qubit", wires=3)
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

        assert dev.var(obs) == None

    def test_pauliz_identity(self, theta, phi, varphi):
        """Test that a tensor product involving PauliZ and Identity works correctly"""
        dev = qml.device("null.qubit", wires=3)
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

        assert dev.var(obs) == None

    def test_pauliz_hadamard(self, theta, phi, varphi):
        """Test that a tensor product involving PauliZ and PauliY and hadamard works correctly"""
        dev = qml.device("null.qubit", wires=3)
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

        assert dev.var(obs) == None

    def test_hermitian(self, theta, phi, varphi, tol):
        """Test that a tensor product involving qml.Hermitian works correctly"""
        dev = qml.device("null.qubit", wires=3)
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

        assert dev.var(obs) == None

    def test_hermitian_hermitian(self, theta, phi, varphi, tol):
        """Test that a tensor product involving two Hermitian matrices works correctly"""
        dev = qml.device("null.qubit", wires=3)

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

        assert dev.var(obs) == None

    def test_hermitian_identity_expectation(self, theta, phi, varphi, tol):
        """Test that a tensor product involving an Hermitian matrix and the identity works correctly"""
        dev = qml.device("null.qubit", wires=2)

        A = np.array(
            [[1.02789352, 1.61296440 - 0.3498192j], [1.61296440 + 0.3498192j, 1.23920938 + 0j]]
        )

        obs = qml.Hermitian(A, wires=[0]) @ qml.Identity(wires=[1])

        dev.apply(
            [qml.RY(theta, wires=[0]), qml.RY(phi, wires=[1]), qml.CNOT(wires=[0, 1])],
            obs.diagonalizing_gates(),
        )

        assert dev.var(obs) == None

    def test_hermitian_two_wires_identity_expectation(self, theta, phi, varphi, tol):
        """Test that a tensor product involving an Hermitian matrix for two wires and the identity works correctly"""
        dev = qml.device("null.qubit", wires=3)

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

        assert dev.var(obs) == None


@pytest.mark.parametrize("inverse", [True, False])
class TestApplyOps:
    """Tests special methods to apply gates in NullQubit."""

    state = np.arange(2**4, dtype=np.complex128).reshape((2, 2, 2, 2))
    dev = qml.device("null.qubit", wires=4)

    single_qubit_ops = [
        (dev._apply_x),
        (dev._apply_y),
        (dev._apply_z),
        (dev._apply_hadamard),
        (dev._apply_s),
        (dev._apply_t),
        (dev._apply_sx),
    ]
    two_qubit_ops = [
        (dev._apply_cnot),
        (dev._apply_swap),
        (dev._apply_cz),
    ]
    three_qubit_ops = [
        (dev._apply_toffoli),
    ]

    @pytest.mark.parametrize("method", single_qubit_ops)
    def test_apply_single_qubit_op(self, method, inverse):
        """Test if the application of single qubit operations is correct."""
        state_out = method(self.state, axes=[1], inverse=inverse)
        assert state_out == None

    @pytest.mark.parametrize("method", two_qubit_ops)
    def test_apply_two_qubit_op(self, method, inverse):
        """Test if the application of two qubit operations is correct."""
        state_out = method(self.state, axes=[0, 1])
        assert state_out == None

    @pytest.mark.parametrize("method", two_qubit_ops)
    def test_apply_two_qubit_op_reverse(self, method, inverse):
        """Test if the application of two qubit operations is correct when the applied wires are
        reversed."""
        state_out = method(self.state, axes=[2, 1])
        assert state_out == None

    @pytest.mark.parametrize("method", three_qubit_ops)
    def test_apply_three_qubit_op_controls_smaller(self, method, inverse):
        """Test if the application of three qubit operations is correct when both control wires are
        smaller than the target wire."""
        state_out = method(self.state, axes=[0, 2, 3])
        assert state_out == None

    @pytest.mark.parametrize("method", three_qubit_ops)
    def test_apply_three_qubit_op_controls_greater(self, method, inverse):
        """Test if the application of three qubit operations is correct when both control wires are
        greater than the target wire."""
        state_out = method(self.state, axes=[2, 1, 0])
        assert state_out == None

    @pytest.mark.parametrize("method", three_qubit_ops)
    def test_apply_three_qubit_op_controls_split(self, method, inverse):
        """Test if the application of three qubit operations is correct when one control wire is smaller
        and one control wire is greater than the target wire."""
        state_out = method(self.state, axes=[3, 1, 2])
        assert state_out == None

    single_qubit_ops_param = [
        (dev._apply_phase, [1.0]),
    ]

    @pytest.mark.parametrize("method,par", single_qubit_ops_param)
    def test_apply_single_qubit_op_(self, method, par, inverse):
        """Test if the application of single qubit operations (with parameter) is correct."""
        state_out = method(self.state, axes=[1], parameters=par, inverse=inverse)
        assert state_out == None


class TestStateInitialization:
    """Unit tests for state initialization methods"""

    def test_state_vector_full_system(self, mocker):
        """Test applying a state vector to the full system"""
        state_wires = qml.wires.Wires(["a", "b", "c"])
        dev = NullQubit(wires=state_wires)
        state = np.array(
            [
                1 / math.sqrt(204),
                2 / math.sqrt(204),
                3 / math.sqrt(204),
                4 / math.sqrt(204),
                5 / math.sqrt(204),
                6 / math.sqrt(204),
                7 / math.sqrt(204),
                8 / math.sqrt(204),
            ]
        )

        spy = mocker.spy(dev, "_scatter")
        dev._apply_state_vector(state=state, device_wires=state_wires)

        assert dev._state == None
        spy.assert_not_called()

    def test_basis_state_full_system(self, mocker):
        """Test applying a state vector to the full system"""
        state_wires = qml.wires.Wires(["a", "b", "c"])
        dev = NullQubit(wires=state_wires)
        state = np.array(
            [
                1 / math.sqrt(204),
                2 / math.sqrt(204),
                3 / math.sqrt(204),
                4 / math.sqrt(204),
                5 / math.sqrt(204),
                6 / math.sqrt(204),
                7 / math.sqrt(204),
                8 / math.sqrt(204),
            ]
        )

        spy = mocker.spy(dev, "_scatter")
        dev._apply_basis_state(state=state, wires=state_wires)

        assert dev._state == None
        spy.assert_not_called()


class TestOpCallDirect:
    """Tests the operation call statistics with direct calls to special methods."""

    state = np.arange(2**4, dtype=np.complex128).reshape((2, 2, 2, 2))
    dev = qml.device("null.qubit", wires=4)

    single_qubit_ops = [
        (dev._apply_x, {"PauliX": 1}),
        (dev._apply_y, {"PauliX": 1, "PauliY": 1}),
        (dev._apply_z, {"PauliX": 1, "PauliY": 1, "PauliZ": 1}),
        (dev._apply_hadamard, {"PauliX": 1, "PauliY": 1, "PauliZ": 1, "Hadamard": 1}),
        (dev._apply_s, {"PauliX": 1, "PauliY": 1, "PauliZ": 1, "Hadamard": 1, "S": 1}),
        (dev._apply_t, {"PauliX": 1, "PauliY": 1, "PauliZ": 1, "Hadamard": 1, "S": 1, "T": 1}),
        (
            dev._apply_sx,
            {"PauliX": 1, "PauliY": 1, "PauliZ": 1, "Hadamard": 1, "S": 1, "T": 1, "SX": 1},
        ),
    ]
    two_qubit_ops = [
        (
            dev._apply_cnot,
            {
                "PauliX": 1,
                "PauliY": 1,
                "PauliZ": 1,
                "Hadamard": 1,
                "S": 1,
                "T": 1,
                "SX": 1,
                "CNOT": 1,
            },
        ),
        (
            dev._apply_swap,
            {
                "PauliX": 1,
                "PauliY": 1,
                "PauliZ": 1,
                "Hadamard": 1,
                "S": 1,
                "T": 1,
                "SX": 1,
                "CNOT": 1,
                "SWAP": 1,
            },
        ),
        (
            dev._apply_cz,
            {
                "PauliX": 1,
                "PauliY": 1,
                "PauliZ": 1,
                "Hadamard": 1,
                "S": 1,
                "T": 1,
                "SX": 1,
                "CNOT": 1,
                "SWAP": 1,
                "CZ": 1,
            },
        ),
    ]
    three_qubit_ops = [
        (
            dev._apply_toffoli,
            {
                "PauliX": 1,
                "PauliY": 1,
                "PauliZ": 1,
                "Hadamard": 1,
                "S": 1,
                "T": 1,
                "SX": 1,
                "CNOT": 1,
                "SWAP": 1,
                "CZ": 1,
                "Toffoli": 1,
            },
        ),
    ]

    @pytest.mark.parametrize("method,expected", single_qubit_ops)
    def test_single_qubit_op(self, method, expected):
        """Test if the application of single qubit operations is being accounted for."""
        method(self.state, axes=[1])
        expected_dict = defaultdict(int, **expected)
        assert self.dev.operation_calls() == expected_dict

    @pytest.mark.parametrize("method,expected", two_qubit_ops)
    def test_two_qubit_op(self, method, expected):
        """Test if the application of single qubit operations is being accounted for."""
        method(self.state, axes=[0, 1])
        expected_dict = defaultdict(int, **expected)
        assert self.dev.operation_calls() == expected_dict

    @pytest.mark.parametrize("method,expected", three_qubit_ops)
    def test_three_qubit_op(self, method, expected):
        """Test if the application of single qubit operations is being accounted for."""
        method(self.state, axes=[0, 1, 2])
        expected_dict = defaultdict(int, **expected)
        assert self.dev.operation_calls() == expected_dict


class TestOpCallIntegration:
    """Integration tests for operation call statistics."""

    dev = qml.device("null.qubit", wires=2)

    single_qubit_ops = [
        (qml.PauliX, {"PauliX": 1}),
        (qml.PauliY, {"PauliX": 1, "PauliY": 1}),
        (qml.PauliZ, {"PauliX": 1, "PauliY": 1, "PauliZ": 1}),
        (qml.Hadamard, {"PauliX": 1, "PauliY": 1, "PauliZ": 1, "Hadamard": 1}),
        (qml.S, {"PauliX": 1, "PauliY": 1, "PauliZ": 1, "Hadamard": 1, "S": 1}),
        (qml.T, {"PauliX": 1, "PauliY": 1, "PauliZ": 1, "Hadamard": 1, "S": 1, "T": 1}),
    ]
    two_qubit_ops = [
        (
            qml.CNOT,
            {"PauliX": 1, "PauliY": 1, "PauliZ": 1, "Hadamard": 1, "S": 1, "T": 1, "CNOT": 1},
        ),
        (
            qml.SWAP,
            {
                "PauliX": 1,
                "PauliY": 1,
                "PauliZ": 1,
                "Hadamard": 1,
                "S": 1,
                "T": 1,
                "CNOT": 1,
                "SWAP": 1,
            },
        ),
        (
            qml.CZ,
            {
                "PauliX": 1,
                "PauliY": 1,
                "PauliZ": 1,
                "Hadamard": 1,
                "S": 1,
                "T": 1,
                "CNOT": 1,
                "SWAP": 1,
                "CZ": 1,
            },
        ),
    ]

    @pytest.mark.parametrize("operation,expected", single_qubit_ops)
    def test_single_qubit_op(self, operation, expected):
        """Test if the application of single qubit operations, without parameters,
        is being accounted for."""

        @qml.qnode(self.dev, diff_method="parameter-shift")
        def circuit():
            operation(wires=[0])
            return qml.state()

        circuit()

        expected_dict = defaultdict(int, **expected)
        assert self.dev.operation_calls() == expected_dict

    @pytest.mark.parametrize("operation,expected", two_qubit_ops)
    def test_two_qubit_op(self, operation, expected):
        """Test if the application of two qubit operations, without parameters,
        is being accounted for."""

        @qml.qnode(self.dev, diff_method="parameter-shift")
        def circuit():
            operation(wires=[0, 1])
            return qml.state()

        circuit()

        expected_dict = defaultdict(int, **expected)
        assert self.dev.operation_calls() == expected_dict

    single_qubit_ops_par = [
        (
            qml.RX,
            [math.pi / 4],
            {
                "PauliX": 1,
                "PauliY": 1,
                "PauliZ": 1,
                "Hadamard": 1,
                "S": 1,
                "T": 1,
                "CNOT": 1,
                "SWAP": 1,
                "CZ": 1,
                "RX": 1,
            },
        ),
        (
            qml.RY,
            [math.pi / 4],
            {
                "PauliX": 1,
                "PauliY": 1,
                "PauliZ": 1,
                "Hadamard": 1,
                "S": 1,
                "T": 1,
                "CNOT": 1,
                "SWAP": 1,
                "CZ": 1,
                "RX": 1,
                "RY": 1,
            },
        ),
        (
            qml.RZ,
            [math.pi / 4],
            {
                "PauliX": 1,
                "PauliY": 1,
                "PauliZ": 1,
                "Hadamard": 1,
                "S": 1,
                "T": 1,
                "CNOT": 1,
                "SWAP": 1,
                "CZ": 1,
                "RX": 1,
                "RY": 1,
                "RZ": 1,
            },
        ),
        (
            qml.MultiRZ,
            [math.pi / 2],
            {
                "PauliX": 1,
                "PauliY": 1,
                "PauliZ": 1,
                "Hadamard": 1,
                "S": 1,
                "T": 1,
                "CNOT": 1,
                "SWAP": 1,
                "CZ": 1,
                "RX": 1,
                "RY": 1,
                "RZ": 1,
                "MultiRZ": 1,
            },
        ),
        (
            qml.DiagonalQubitUnitary,
            [np.array([-1, 1])],
            {
                "PauliX": 1,
                "PauliY": 1,
                "PauliZ": 1,
                "Hadamard": 1,
                "S": 1,
                "T": 1,
                "CNOT": 1,
                "SWAP": 1,
                "CZ": 1,
                "RX": 1,
                "RY": 1,
                "RZ": 1,
                "MultiRZ": 1,
                "DiagonalQubitUnitary": 1,
            },
        ),
    ]
    two_qubit_ops_par = [
        (
            qml.CRX,
            [math.pi / 2],
            {
                "PauliX": 1,
                "PauliY": 1,
                "PauliZ": 1,
                "Hadamard": 1,
                "S": 1,
                "T": 1,
                "CNOT": 1,
                "SWAP": 1,
                "CZ": 1,
                "RX": 1,
                "RY": 1,
                "RZ": 1,
                "MultiRZ": 1,
                "DiagonalQubitUnitary": 1,
                "CRX": 1,
            },
        ),
        (
            qml.CRY,
            [math.pi / 2],
            {
                "PauliX": 1,
                "PauliY": 1,
                "PauliZ": 1,
                "Hadamard": 1,
                "S": 1,
                "T": 1,
                "CNOT": 1,
                "SWAP": 1,
                "CZ": 1,
                "RX": 1,
                "RY": 1,
                "RZ": 1,
                "MultiRZ": 1,
                "DiagonalQubitUnitary": 1,
                "CRX": 1,
                "CRY": 1,
            },
        ),
        (
            qml.CRZ,
            [math.pi / 2],
            {
                "PauliX": 1,
                "PauliY": 1,
                "PauliZ": 1,
                "Hadamard": 1,
                "S": 1,
                "T": 1,
                "CNOT": 1,
                "SWAP": 1,
                "CZ": 1,
                "RX": 1,
                "RY": 1,
                "RZ": 1,
                "MultiRZ": 1,
                "DiagonalQubitUnitary": 1,
                "CRX": 1,
                "CRY": 1,
                "CRZ": 1,
            },
        ),
        (
            qml.MultiRZ,
            [math.pi / 2],
            {
                "PauliX": 1,
                "PauliY": 1,
                "PauliZ": 1,
                "Hadamard": 1,
                "S": 1,
                "T": 1,
                "CNOT": 1,
                "SWAP": 1,
                "CZ": 1,
                "RX": 1,
                "RY": 1,
                "RZ": 1,
                "MultiRZ": 2,
                "DiagonalQubitUnitary": 1,
                "CRX": 1,
                "CRY": 1,
                "CRZ": 1,
            },
        ),
        (
            qml.DiagonalQubitUnitary,
            [np.array([-1, 1, -1, 1])],
            {
                "PauliX": 1,
                "PauliY": 1,
                "PauliZ": 1,
                "Hadamard": 1,
                "S": 1,
                "T": 1,
                "CNOT": 1,
                "SWAP": 1,
                "CZ": 1,
                "RX": 1,
                "RY": 1,
                "RZ": 1,
                "MultiRZ": 2,
                "DiagonalQubitUnitary": 2,
                "CRX": 1,
                "CRY": 1,
                "CRZ": 1,
            },
        ),
        (
            qml.IsingXX,
            [math.pi / 2],
            {
                "PauliX": 1,
                "PauliY": 1,
                "PauliZ": 1,
                "Hadamard": 1,
                "S": 1,
                "T": 1,
                "CNOT": 1,
                "SWAP": 1,
                "CZ": 1,
                "RX": 1,
                "RY": 1,
                "RZ": 1,
                "MultiRZ": 2,
                "DiagonalQubitUnitary": 2,
                "CRX": 1,
                "CRY": 1,
                "CRZ": 1,
                "IsingXX": 1,
            },
        ),
        (
            qml.IsingYY,
            [math.pi / 2],
            {
                "PauliX": 1,
                "PauliY": 1,
                "PauliZ": 1,
                "Hadamard": 1,
                "S": 1,
                "T": 1,
                "CNOT": 1,
                "SWAP": 1,
                "CZ": 1,
                "RX": 1,
                "RY": 1,
                "RZ": 1,
                "MultiRZ": 2,
                "DiagonalQubitUnitary": 2,
                "CRX": 1,
                "CRY": 1,
                "CRZ": 1,
                "IsingXX": 1,
                "IsingYY": 1,
            },
        ),
        (
            qml.IsingZZ,
            [math.pi / 2],
            {
                "PauliX": 1,
                "PauliY": 1,
                "PauliZ": 1,
                "Hadamard": 1,
                "S": 1,
                "T": 1,
                "CNOT": 1,
                "SWAP": 1,
                "CZ": 1,
                "RX": 1,
                "RY": 1,
                "RZ": 1,
                "MultiRZ": 2,
                "DiagonalQubitUnitary": 2,
                "CRX": 1,
                "CRY": 1,
                "CRZ": 1,
                "IsingXX": 1,
                "IsingYY": 1,
                "IsingZZ": 1,
            },
        ),
        (
            qml.QubitStateVector,
            [1 / math.sqrt(30), 2 / math.sqrt(30), 3 / math.sqrt(30), 4 / math.sqrt(30)],
            {
                "PauliX": 1,
                "PauliY": 1,
                "PauliZ": 1,
                "Hadamard": 1,
                "S": 1,
                "T": 1,
                "CNOT": 1,
                "SWAP": 1,
                "CZ": 1,
                "RX": 1,
                "RY": 1,
                "RZ": 1,
                "MultiRZ": 2,
                "DiagonalQubitUnitary": 2,
                "CRX": 1,
                "CRY": 1,
                "CRZ": 1,
                "IsingXX": 1,
                "IsingYY": 1,
                "IsingZZ": 1,
                "QubitStateVector": 1,
            },
        ),
        (
            qml.BasisState,
            [1 / math.sqrt(30), 2 / math.sqrt(30), 3 / math.sqrt(30), 4 / math.sqrt(30)],
            {
                "PauliX": 1,
                "PauliY": 1,
                "PauliZ": 1,
                "Hadamard": 1,
                "S": 1,
                "T": 1,
                "CNOT": 1,
                "SWAP": 1,
                "CZ": 1,
                "RX": 1,
                "RY": 1,
                "RZ": 1,
                "MultiRZ": 2,
                "DiagonalQubitUnitary": 2,
                "CRX": 1,
                "CRY": 1,
                "CRZ": 1,
                "IsingXX": 1,
                "IsingYY": 1,
                "IsingZZ": 1,
                "QubitStateVector": 1,
                "BasisState": 1,
            },
        ),
        (
            qml.BasisState,
            [1 / math.sqrt(30), 2 / math.sqrt(30), 3 / math.sqrt(30), 4 / math.sqrt(30)],
            {
                "PauliX": 1,
                "PauliY": 1,
                "PauliZ": 1,
                "Hadamard": 1,
                "S": 1,
                "T": 1,
                "CNOT": 1,
                "SWAP": 1,
                "CZ": 1,
                "RX": 1,
                "RY": 1,
                "RZ": 1,
                "MultiRZ": 2,
                "DiagonalQubitUnitary": 2,
                "CRX": 1,
                "CRY": 1,
                "CRZ": 1,
                "IsingXX": 1,
                "IsingYY": 1,
                "IsingZZ": 1,
                "QubitStateVector": 1,
                "BasisState": 2,
            },
        ),
    ]

    @pytest.mark.parametrize("operation,input,expected", single_qubit_ops_par)
    def test_single_qubit_op_with_par(self, operation, input, expected):
        """Test if the application of single qubit operations, with parameters,
        is being accounted for."""

        @qml.qnode(self.dev, diff_method="parameter-shift")
        def circuit(input):
            operation(input, wires=[0])
            return qml.state()

        circuit(input)

        expected_dict = defaultdict(int, **expected)
        assert self.dev.operation_calls() == expected_dict

    @pytest.mark.parametrize("operation,input,expected", two_qubit_ops_par)
    def test_two_qubit_op_with_par(self, operation, input, expected):
        """Test if the application of two qubit operations, with parameters,
        is being accounted for."""

        @qml.qnode(self.dev, diff_method="parameter-shift")
        def circuit(input):
            operation(input, wires=[0, 1])
            return qml.state()

        circuit(input)

        expected_dict = defaultdict(int, **expected)
        assert self.dev.operation_calls() == expected_dict


class TestState:
    "Unit test for state and density_matrix operations."
    dev = qml.device("null.qubit", wires=3)

    @pytest.mark.parametrize(
        "measurement",
        [
            dev.state,
            dev.density_matrix(wires=[1]),
            dev.density_matrix(wires=[2, 0]),
            dev.density_matrix(wires=[2, 1, 0]),
        ],
    )
    def test_state_measurement(self, measurement):
        """Test that the null qubit plugin provides correct state results for a simple circuit"""
        assert measurement == None

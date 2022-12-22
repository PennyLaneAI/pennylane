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
Unit tests for the :mod:`pennylane.plugin.DefaultQubit` device when using broadcasting.
"""
from itertools import product
import cmath

# pylint: disable=protected-access,cell-var-from-loop
import math

import pytest
import pennylane as qml
from pennylane import numpy as np, DeviceError
from pennylane.devices.default_qubit import _get_slice, DefaultQubit
from pennylane.wires import Wires, WireError
from gate_data import *

THETA = np.linspace(0.11, 1, 3)
PHI = np.linspace(0.32, 1, 3)
VARPHI = np.linspace(0.02, 1, 3)

INVSQ2 = 1 / math.sqrt(2)
T_PHASE = np.exp(1j * np.pi / 4)
T_PHASE_C = np.exp(-1j * np.pi / 4)

# Variant of diag that does not take the diagonal of a 2d array, but broadcasts diag.
diag = lambda x: np.array([np.diag(_x) for _x in x]) if np.ndim(x) == 2 else np.diag(x)


def mat_vec(mat, vec, par=None, inv=False):
    if par is not None:
        scalar = [np.isscalar(p) for p in par]
        if not all(scalar):
            batch_size = len(par[scalar.index(False)])
            par = [tuple(p if s else p[i] for p, s in zip(par, scalar)) for i in range(batch_size)]
            mat = np.array([mat(*_par) for _par in par])
        else:
            mat = mat(*par)

    if inv:
        mat = np.moveaxis(mat.conj(), -2, -1)

    return np.einsum("...ij,...j->...i", mat, vec)


class TestApplyBroadcasted:
    """Tests that operations and inverses of certain operations are applied to a broadcasted
    state/with broadcasted parameters (or both) correctly, or that the proper errors are raised.
    """

    triple_state = np.array([[1, 0], [INVSQ2, INVSQ2], [0, 1]])
    test_data_no_parameters = [
        (qml.PauliX, triple_state, mat_vec(X, triple_state)),
        (qml.PauliY, triple_state, mat_vec(Y, triple_state)),
        (qml.PauliZ, triple_state, mat_vec(Z, triple_state)),
        (qml.S, triple_state, mat_vec(S, triple_state)),
        (qml.T, triple_state, mat_vec(T, triple_state)),
        (qml.Hadamard, triple_state, mat_vec(H, triple_state)),
        (qml.Identity, triple_state, triple_state),
    ]

    test_data_no_parameters_inverses = [
        (qml.PauliX, triple_state, mat_vec(X, triple_state, inv=True)),
        (qml.PauliY, triple_state, mat_vec(Y, triple_state, inv=True)),
        (qml.PauliZ, triple_state, mat_vec(Z, triple_state, inv=True)),
        (qml.S, triple_state, mat_vec(S, triple_state, inv=True)),
        (qml.T, triple_state, mat_vec(T, triple_state, inv=True)),
        (qml.Hadamard, triple_state, mat_vec(H, triple_state, inv=True)),
        (qml.Identity, triple_state, triple_state),
    ]

    @pytest.mark.parametrize("operation,input,expected_output", test_data_no_parameters)
    def test_apply_operation_single_wire_no_parameters_broadcasted(
        self, qubit_device_1_wire, tol, operation, input, expected_output
    ):
        """Tests that applying an operation yields the expected output state for single wire
        operations that have no parameters."""

        qubit_device_1_wire._state = np.array(input, dtype=qubit_device_1_wire.C_DTYPE)
        qubit_device_1_wire.apply([operation(wires=[0])])

        assert np.allclose(qubit_device_1_wire._state, np.array(expected_output), atol=tol, rtol=0)
        assert qubit_device_1_wire._state.dtype == qubit_device_1_wire.C_DTYPE

    @pytest.mark.parametrize("operation,input,expected_output", test_data_no_parameters_inverses)
    def test_apply_operation_single_wire_no_parameters_inverse_broadcasted(
        self, qubit_device_1_wire, tol, operation, input, expected_output
    ):
        """Tests that applying an operation yields the expected output state for single wire
        operations that have no parameters."""

        qubit_device_1_wire._state = np.array(input, dtype=qubit_device_1_wire.C_DTYPE)
        qubit_device_1_wire.apply([operation(wires=[0]).inv()])

        assert np.allclose(qubit_device_1_wire._state, np.array(expected_output), atol=tol, rtol=0)
        assert qubit_device_1_wire._state.dtype == qubit_device_1_wire.C_DTYPE

    single_state = np.array([[0, 0.6, 0, 0.8]])
    triple_state = np.array([[1, 0, 0, 0], [0, 0, INVSQ2, -INVSQ2], [0, 0.6, 0, 0.8]])

    test_data_two_wires_no_param = [
        (qml_op, state, mat_vec(mat_op, state))
        for (qml_op, mat_op), state in product(
            zip(
                [qml.CNOT, qml.SWAP, qml.CZ, qml.ISWAP, qml.SISWAP, qml.SQISW],
                [CNOT, SWAP, CZ, ISWAP, SISWAP, SISWAP],
            ),
            [single_state, triple_state],
        )
    ]

    @pytest.mark.parametrize("operation,input,expected_output", test_data_two_wires_no_param)
    def test_apply_operation_two_wires_no_parameters_broadcasted(
        self, qubit_device_2_wires, tol, operation, input, expected_output
    ):
        """Tests that applying an operation yields the expected output state for two wire
        operations that have no parameters."""

        qubit_device_2_wires._state = np.array(input, dtype=qubit_device_2_wires.C_DTYPE).reshape(
            (-1, 2, 2)
        )
        qubit_device_2_wires.apply([operation(wires=[0, 1])])

        assert np.allclose(
            qubit_device_2_wires._state.reshape((-1, 4)),
            np.array(expected_output),
            atol=tol,
            rtol=0,
        )
        assert qubit_device_2_wires._state.dtype == qubit_device_2_wires.C_DTYPE

    test_data_two_wires_no_param_inv = [
        (qml_op, state, mat_vec(mat_op, state, inv=True))
        for (qml_op, mat_op), state in product(
            zip(
                [qml.CNOT, qml.SWAP, qml.CZ, qml.ISWAP, qml.SISWAP, qml.SQISW],
                [CNOT, SWAP, CZ, ISWAP, SISWAP, SISWAP],
            ),
            [single_state, triple_state],
        )
    ]

    @pytest.mark.parametrize("operation,input,expected_output", test_data_two_wires_no_param_inv)
    def test_apply_operation_two_wires_no_parameters_inverse_broadcasted(
        self, qubit_device_2_wires, tol, operation, input, expected_output
    ):
        """Tests that applying an operation yields the expected output state for two wire
        operations that have no parameters."""

        qubit_device_2_wires._state = np.array(input, dtype=qubit_device_2_wires.C_DTYPE).reshape(
            (-1, 2, 2)
        )
        qubit_device_2_wires.apply([operation(wires=[0, 1]).inv()])

        assert np.allclose(
            qubit_device_2_wires._state.reshape((-1, 4)),
            np.array(expected_output),
            atol=tol,
            rtol=0,
        )
        assert qubit_device_2_wires._state.dtype == qubit_device_2_wires.C_DTYPE

    quad_state = np.array(
        [
            [0.6, 0, 0, 0, 0, 0, 0.8, 0],
            [-INVSQ2, INVSQ2, 0, 0, 0, 0, 0, 0],
            [0, 0, 0.5, 0.5, 0.5, -0.5, 0, 0],
            [0, 0, 0.5, 0, 0.5, -0.5, 0, 0.5],
        ]
    )
    test_data_three_wires_no_parameters = [(qml.CSWAP, quad_state, mat_vec(CSWAP, quad_state))]

    @pytest.mark.parametrize("operation,input,expected_output", test_data_three_wires_no_parameters)
    def test_apply_operation_three_wires_no_parameters_broadcasted(
        self, qubit_device_3_wires, tol, operation, input, expected_output
    ):
        """Tests that applying an operation yields the expected output state for three wire
        operations that have no parameters."""

        qubit_device_3_wires._state = np.array(input, dtype=qubit_device_3_wires.C_DTYPE).reshape(
            (-1, 2, 2, 2)
        )
        qubit_device_3_wires.apply([operation(wires=[0, 1, 2])])

        assert np.allclose(
            qubit_device_3_wires._state.reshape((-1, 8)),
            np.array(expected_output),
            atol=tol,
            rtol=0,
        )
        assert qubit_device_3_wires._state.dtype == qubit_device_3_wires.C_DTYPE

    test_data_three_wires_no_param_inv = [
        (qml.CSWAP, quad_state, mat_vec(CSWAP, quad_state, inv=True))
    ]

    @pytest.mark.parametrize("operation,input,expected_output", test_data_three_wires_no_param_inv)
    def test_apply_operation_three_wires_no_parameters_inverse_broadcasted(
        self, qubit_device_3_wires, tol, operation, input, expected_output
    ):
        """Tests that applying the inverse of an operation yields the expected output state for three wire
        operations that have no parameters."""

        qubit_device_3_wires._state = np.array(input, dtype=qubit_device_3_wires.C_DTYPE).reshape(
            (-1, 2, 2, 2)
        )
        qubit_device_3_wires.apply([operation(wires=[0, 1, 2]).inv()])

        assert np.allclose(
            qubit_device_3_wires._state.reshape((-1, 8)),
            np.array(expected_output),
            atol=tol,
            rtol=0,
        )
        assert qubit_device_3_wires._state.dtype == qubit_device_3_wires.C_DTYPE

    single_state = np.array([[0, 0, 1, 0]])
    triple_state = np.array(
        [
            [0, 0, 1, 0],
            [1 / math.sqrt(3), 0, 1 / math.sqrt(3), 1 / math.sqrt(3)],
            [0.5, -0.5, 0.5j, -0.5j],
        ]
    )

    # TODO[dwierichs]: add tests with qml.BaisState once `_apply_basis_state` supports broadcasting
    @pytest.mark.parametrize(
        "operation,expected_output,par",
        [(qml.QubitStateVector, s, s) for s in [single_state, triple_state]],
    )
    def test_apply_operation_state_preparation_broadcasted(
        self, qubit_device_2_wires, tol, operation, expected_output, par
    ):
        """Tests that applying an operation yields the expected output state for single wire
        operations that have no parameters."""

        par = np.array(par)
        qubit_device_2_wires.reset()
        qubit_device_2_wires.apply([operation(par, wires=[0, 1])])

        assert np.allclose(
            qubit_device_2_wires._state.reshape((-1, 4)),
            np.array(expected_output),
            atol=tol,
            rtol=0,
        )

    # Collect test cases for single-scalar-parameter single-wire operations and their inverses
    # For each operation, we choose broadcasted state, broadcasted params, or both
    state_1 = np.array([0.6, 0.8j])
    state_5 = np.array([[INVSQ2, INVSQ2], [0.6, 0.8], [0, 1j], [-1, 0], [-INVSQ2, INVSQ2]])
    scalar_par_1 = [np.pi / 2]
    scalar_par_5 = [[np.pi / 3, np.pi, 0.5, -1.2, -3 * np.pi / 2]]
    test_data_single_wire_with_parameters = [
        (qml_op, state, mat_vec(mat_op, state, par=par), par)
        for (qml_op, mat_op), (state, par) in product(
            zip(
                [qml.PhaseShift, qml.RX, qml.RY, qml.RZ, qml.MultiRZ],
                [Rphi, Rotx, Roty, Rotz, MultiRZ1],
            ),
            [(state_1, scalar_par_5), (state_5, scalar_par_1), (state_5, scalar_par_5)],
        )
    ]
    test_data_single_wire_with_param_inv = [
        (qml_op, state, mat_vec(mat_op, state, par=par, inv=True), par)
        for (qml_op, mat_op), (state, par) in product(
            zip(
                [qml.PhaseShift, qml.RX, qml.RY, qml.RZ, qml.MultiRZ],
                [Rphi, Rotx, Roty, Rotz, MultiRZ1],
            ),
            [(state_1, scalar_par_5), (state_5, scalar_par_1), (state_5, scalar_par_5)],
        )
    ]

    # Add qml.QubitUnitary test cases
    matrix_1_par_1 = [np.array([[1, 1j], [1j, 1]]) * INVSQ2]
    matrix_1_par_5 = [
        np.array(
            [
                np.array([[1, -1j], [-1j, 1]]) * INVSQ2,
                np.array([[1, -1], [1, 1]]) * INVSQ2,
                np.array([[T_PHASE_C, 0], [0, T_PHASE]]),
                np.array([[1, 0], [0, -1]]),
                T,
            ]
        )
    ]
    test_data_single_wire_with_parameters += [
        (qml.QubitUnitary, s, mat_vec(par[0], s), par)
        for s, par in [
            (state_1, matrix_1_par_5),
            (state_5, matrix_1_par_1),
            (state_5, matrix_1_par_5),
        ]
    ]
    test_data_single_wire_with_param_inv += [
        (qml.QubitUnitary, s, mat_vec(par[0], s, inv=True), par)
        for s, par in [
            (state_1, matrix_1_par_5),
            (state_5, matrix_1_par_1),
            (state_5, matrix_1_par_5),
        ]
    ]

    # Add qml.DiagonalQubitUnitary test cases
    diag_par_1 = [[np.exp(1j * 0.1), np.exp(1j * np.pi)]]
    diag_par_5 = [
        np.array(
            [
                [1, -1j],
                [np.exp(1j * 0.4), np.exp(1j * -0.4)],
                [np.exp(1j * 0.1), np.exp(1j * np.pi)],
                [1.0, np.exp(1j * np.pi / 2)],
                [1, 1],
            ]
        )
    ]
    test_data_single_wire_with_parameters += [
        (qml.DiagonalQubitUnitary, s, mat_vec(diag(par[0]), s), par)
        for s, par in [(state_1, diag_par_5), (state_5, diag_par_1), (state_5, diag_par_5)]
    ]
    test_data_single_wire_with_param_inv += [
        (qml.DiagonalQubitUnitary, s, mat_vec(diag(par[0]), s, inv=True), par)
        for s, par in [(state_1, diag_par_5), (state_5, diag_par_1), (state_5, diag_par_5)]
    ]

    # Add qml.Rot test cases
    multi_par_1 = {
        "rz_0": [0.632, 0, 0],
        "ry": [0, 0.632, 0],
        "rz_1": [0, 0, 0.632],
        "mixed": [0.12, -2.468, 0.812],
    }
    multi_par_5 = {
        "rz_0": [[np.pi / 2 * i for i in range(5)], 0, 0],
        "ry": [0, [np.pi / 2 * i for i in range(5)], 0],
        "rz_1": [0, 0, [np.pi / 2 * i for i in range(5)]],
        "mixed": [[np.pi / 2 * i for i in range(5)], [np.pi / 2 * i for i in range(5)], np.pi],
    }
    for like in ["rz_0", "ry", "rz_1", "mixed"]:
        states_and_pars = [
            (state_1, multi_par_5[like]),
            (state_5, multi_par_1[like]),
            (state_5, multi_par_5[like]),
        ]
        test_data_single_wire_with_parameters += [
            (qml.Rot, s, mat_vec(Rot3, s, par=par), par) for s, par in states_and_pars
        ]
        test_data_single_wire_with_param_inv += [
            (qml.Rot, s, mat_vec(Rot3, s, par=par, inv=True), par) for s, par in states_and_pars
        ]

    @pytest.mark.parametrize(
        "operation,input,expected_output,par", test_data_single_wire_with_parameters
    )
    def test_apply_operation_single_wire_with_parameters_broadcasted(
        self, qubit_device_1_wire, tol, operation, input, expected_output, par
    ):
        """Tests that applying an operation yields the expected output state for single wire
        operations that have parameters."""

        qubit_device_1_wire._state = np.array(input, dtype=qubit_device_1_wire.C_DTYPE)

        par = tuple(np.array(p) for p in par)
        qubit_device_1_wire.apply([operation(*par, wires=[0])])

        assert np.allclose(qubit_device_1_wire._state, np.array(expected_output), atol=tol, rtol=0)
        assert qubit_device_1_wire._state.dtype == qubit_device_1_wire.C_DTYPE

    @pytest.mark.parametrize(
        "operation,input,expected_output,par", test_data_single_wire_with_param_inv
    )
    def test_apply_operation_single_wire_with_parameters_inverse_broadcasted(
        self, qubit_device_1_wire, tol, operation, input, expected_output, par
    ):
        """Tests that applying the inverse of an operation yields the expected output state for single wire
        operations that have parameters."""

        qubit_device_1_wire._state = np.array(input, dtype=qubit_device_1_wire.C_DTYPE)

        par = tuple(np.array(p) for p in par)
        qubit_device_1_wire.apply([operation(*par, wires=[0]).inv()])

        assert np.allclose(qubit_device_1_wire._state, np.array(expected_output), atol=tol, rtol=0)
        assert qubit_device_1_wire._state.dtype == qubit_device_1_wire.C_DTYPE

    # Collect test cases for single-scalar-parameter two-wires operations and their inverses
    # For each operation, we choose broadcasted state, broadcasted params, or both
    state_1 = np.array([0.6, 0.8j, -0.6, -0.8j]) * INVSQ2
    state_5 = np.array(
        [
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, INVSQ2, INVSQ2, 0],
            [0.5, 0.5j, -0.5, 0.5],
            [0.6, 0, -0.8j, 0],
        ]
    )
    scalar_par_1 = [np.pi / 2]
    scalar_par_5 = [[np.pi / 3, np.pi, 0.5, -1.2, -3 * np.pi / 2]]
    two_wires_scalar_par_ops = [
        qml.CRX,
        qml.CRY,
        qml.CRZ,
        qml.MultiRZ,
        qml.IsingXX,
        qml.IsingYY,
        qml.IsingZZ,
    ]
    two_wires_scalar_par_mats = [CRotx, CRoty, CRotz, MultiRZ2, IsingXX, IsingYY, IsingZZ]
    test_data_two_wires_with_parameters = [
        (qml_op, state, mat_vec(mat_op, state, par=par), par)
        for (qml_op, mat_op), (state, par) in product(
            zip(two_wires_scalar_par_ops, two_wires_scalar_par_mats),
            [(state_1, scalar_par_5), (state_5, scalar_par_1), (state_5, scalar_par_5)],
        )
    ]
    test_data_two_wires_with_param_inv = [
        (qml_op, state, mat_vec(mat_op, state, par=par, inv=True), par)
        for (qml_op, mat_op), (state, par) in product(
            zip(two_wires_scalar_par_ops, two_wires_scalar_par_mats),
            [(state_1, scalar_par_5), (state_5, scalar_par_1), (state_5, scalar_par_5)],
        )
    ]

    # Add qml.CRot test cases
    multi_par_1 = {
        "rz_0": [0.632, 0, 0],
        "ry": [0, 0.632, 0],
        "rz_1": [0, 0, 0.632],
        "mixed": [0.12, -2.468, 0.812],
    }
    multi_par_5 = {
        "rz_0": [[np.pi / 2 * i for i in range(5)], 0, 0],
        "ry": [0, [np.pi / 2 * i for i in range(5)], 0],
        "rz_1": [0, 0, [np.pi / 2 * i for i in range(5)]],
        "mixed": [[np.pi / 2 * i for i in range(5)], [np.pi / 2 * i for i in range(5)], np.pi],
    }
    for like in ["rz_0", "ry", "rz_1", "mixed"]:
        states_and_pars = [
            (state_1, multi_par_5[like]),
            (state_5, multi_par_1[like]),
            (state_5, multi_par_5[like]),
        ]
        test_data_two_wires_with_parameters += [
            (qml.CRot, s, mat_vec(CRot3, s, par=par), par) for s, par in states_and_pars
        ]
        test_data_two_wires_with_param_inv += [
            (qml.CRot, s, mat_vec(CRot3, s, par=par, inv=True), par) for s, par in states_and_pars
        ]

    # Add qml.QubitUnitary test cases
    matrix_2_par_1 = [SISWAP]
    matrix_2_par_5 = [
        np.array(
            [
                np.eye(4),
                np.array([[1, -1j, 0, 0], [-1j, 1, 0, 0], [0, 0, 1, -1j], [0, 0, -1j, 1]]) * INVSQ2,
                np.array([[1, -1, 0, 0], [1, 1, 0, 0], [0, 0, 1, -1j], [0, 0, 1j, -1]]) * INVSQ2,
                np.array([[T_PHASE_C, 0, 0, 0], [0, T_PHASE, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1j]]),
                SISWAP,
            ]
        )
    ]
    test_data_two_wires_with_parameters += [
        (qml.QubitUnitary, s, mat_vec(par[0], s), par)
        for s, par in [
            (state_1, matrix_2_par_5),
            (state_5, matrix_2_par_1),
            (state_5, matrix_2_par_5),
        ]
    ]
    test_data_two_wires_with_param_inv += [
        (qml.QubitUnitary, s, mat_vec(par[0], s, inv=True), par)
        for s, par in [
            (state_1, matrix_2_par_5),
            (state_5, matrix_2_par_1),
            (state_5, matrix_2_par_5),
        ]
    ]

    # Add qml.DiagonalQubitUnitary test cases
    diag_par_1 = [np.exp(1j * np.array([0.1, np.pi, 0.2, -2.4]))]
    diag_par_5 = [
        np.array(
            [
                np.ones(4),
                [1, -1j, 1, 1j],
                [np.exp(1j * 0.4), np.exp(1j * -0.4), 1j, 1],
                [np.exp(1j * 0.1), np.exp(1j * np.pi), INVSQ2 * (1 + 1j), INVSQ2 * (1 - 1j)],
                [1.0, np.exp(1j * np.pi / 2), 1, 1],
            ]
        )
    ]
    test_data_two_wires_with_parameters += [
        (qml.DiagonalQubitUnitary, s, mat_vec(diag(par[0]), s), par)
        for s, par in [(state_1, diag_par_5), (state_5, diag_par_1), (state_5, diag_par_5)]
    ]
    test_data_two_wires_with_param_inv += [
        (qml.DiagonalQubitUnitary, s, mat_vec(diag(par[0]), s, inv=True), par)
        for s, par in [(state_1, diag_par_5), (state_5, diag_par_1), (state_5, diag_par_5)]
    ]

    @pytest.mark.parametrize(
        "operation,input,expected_output,par", test_data_two_wires_with_parameters
    )
    def test_apply_operation_two_wires_with_parameters_broadcasted(
        self, qubit_device_2_wires, tol, operation, input, expected_output, par
    ):
        """Tests that applying an operation yields the expected output state for two wire
        operations that have parameters."""

        shape = (5, 2, 2) if np.array(input).size == 20 else (2, 2)
        dtype = qubit_device_2_wires.C_DTYPE
        qubit_device_2_wires._state = np.array(input, dtype=dtype).reshape(shape)
        par = tuple(np.array(p) for p in par)
        qubit_device_2_wires.apply([operation(*par, wires=[0, 1])])

        assert np.allclose(
            qubit_device_2_wires._state.reshape((5, 4)), expected_output, atol=tol, rtol=0
        )
        assert qubit_device_2_wires._state.dtype == qubit_device_2_wires.C_DTYPE

    @pytest.mark.parametrize(
        "operation,input,expected_output,par", test_data_two_wires_with_param_inv
    )
    def test_apply_operation_two_wires_with_parameters_inverse_broadcasted(
        self, qubit_device_2_wires, tol, operation, input, expected_output, par
    ):
        """Tests that applying the inverse of an operation yields the expected output state for two wire
        operations that have parameters."""

        shape = (5, 2, 2) if np.array(input).size == 20 else (2, 2)
        dtype = qubit_device_2_wires.C_DTYPE
        qubit_device_2_wires._state = np.array(input, dtype=dtype).reshape(shape)
        par = tuple(np.array(p) for p in par)
        qubit_device_2_wires.apply([operation(*par, wires=[0, 1]).inv()])

        assert np.allclose(
            qubit_device_2_wires._state.reshape((5, 4)), expected_output, atol=tol, rtol=0
        )
        assert qubit_device_2_wires._state.dtype == qubit_device_2_wires.C_DTYPE

    def test_apply_errors_qubit_state_vector_broadcasted(self, qubit_device_2_wires):
        """Test that apply fails for incorrect state preparation, and > 2 qubit gates"""
        with pytest.raises(ValueError, match="Sum of amplitudes-squared does not equal one."):
            qubit_device_2_wires.apply(
                [qml.QubitStateVector(np.array([[1, -1], [0, 2]]), wires=[0])]
            )

        # Also test that the sum-check is *not* performed along the broadcasting dimension
        qubit_device_2_wires.apply(
            [qml.QubitStateVector(np.array([[0.6, 0.8], [0.6, 0.8]]), wires=[0])]
        )

        with pytest.raises(ValueError, match=r"State vector must have shape \(2\*\*wires,\)."):
            # Second dimension does not match 2**num_wires
            p = np.array([[1, 0, 1, 1, 0], [0, 1, 1, 0, 1]]) / np.sqrt(3)
            qubit_device_2_wires.apply([qml.QubitStateVector(p, wires=[0, 1])])

        with pytest.raises(ValueError, match=r"State vector must have shape \(2\*\*wires,\)."):
            # Broadcasting dimension is not first dimension
            p = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1], [0, 1, 1]]) / np.sqrt(2)
            qubit_device_2_wires.apply([qml.QubitStateVector(p, wires=[0, 1])])

        qubit_device_2_wires.reset()
        vec = qml.QubitStateVector(np.array([[0, 1, 0, 0], [0, 0, 1, 0]]), wires=[0, 1])
        with pytest.raises(
            DeviceError,
            match="Operation QubitStateVector cannot be used after other Operations have already been applied "
            "on a default.qubit device.",
        ):
            qubit_device_2_wires.apply([qml.RZ(0.5, wires=[0]), vec])

    @pytest.mark.skip("Applying a BasisState does not support broadcasting yet")
    def test_apply_errors_basis_state_broadcasted(self, qubit_device_2_wires):
        """Test that applying the BasisState operation raises the correct errors."""
        with pytest.raises(
            ValueError, match="BasisState parameter must consist of 0 or 1 integers."
        ):
            op = qml.BasisState(np.array([[-0.2, 4.2], [0.5, 1.2]]), wires=[0, 1])
            qubit_device_2_wires.apply([op])

        with pytest.raises(
            ValueError, match="BasisState parameter and wires must be of equal length."
        ):
            # Test that the error is raised
            qubit_device_2_wires.apply(
                [qml.BasisState(np.array([[0, 1], [1, 1], [1, 0]]), wires=[0])]
            )
        # Test that the broadcasting dimension is allowed to mismatch the length of the wires
        qubit_device_2_wires.apply([qml.BasisState(np.array([[0], [1], [0]]), wires=[0])])

        qubit_device_2_wires.reset()
        qubit_device_2_wires.apply([qml.RZ(0.5, wires=[0])])
        vec = qml.BasisState(np.array([[0, 0], [1, 0], [1, 1]]), wires=[0, 1])
        with pytest.raises(
            DeviceError,
            match="Operation BasisState cannot be used after other Operations have already been applied "
            "on a default.qubit device.",
        ):
            qubit_device_2_wires.apply([vec])


zero = [1, 0]
one = [0, 1]
plus = [INVSQ2, INVSQ2]
minus = [INVSQ2, -INVSQ2]
y_plus = [INVSQ2, 1j * INVSQ2]
y_minus = [INVSQ2, -1j * INVSQ2]


class TestExpvalBroadcasted:
    """Tests that expectation values are properly calculated for broadcasted states
    or that the proper errors are raised."""

    @pytest.mark.parametrize(
        "operation,input,expected_output",
        [
            (qml.PauliX, np.array([plus, zero, minus]), [1, 0, -1]),
            (qml.PauliY, np.array([y_plus, zero, y_minus]), [1, 0, -1]),
            (qml.PauliZ, np.array([plus, zero, one]), [0, 1, -1]),
            (qml.Hadamard, np.array([plus, zero, one]), [INVSQ2, INVSQ2, -INVSQ2]),
            (qml.Identity, np.array([minus, zero, one]), [1, 1, 1]),
        ],
    )
    def test_expval_single_wire_no_parameters_broadcasted(
        self, qubit_device_1_wire, tol, operation, input, expected_output
    ):
        """Tests that expectation values are properly calculated for single-wire observables without parameters."""

        obs = operation(wires=[0])

        qubit_device_1_wire.reset()
        qubit_device_1_wire.apply(
            [qml.QubitStateVector(np.array(input), wires=[0])], obs.diagonalizing_gates()
        )
        res = qubit_device_1_wire.expval(obs)

        assert np.allclose(res, expected_output, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "operation,input,expected_output,par",
        [(qml.Hermitian, [zero, one, minus, y_plus], [1, 1, 1, 0], I - Y)],
    )
    def test_expval_single_wire_with_parameters_broadcasted(
        self, qubit_device_1_wire, tol, operation, input, expected_output, par
    ):
        """Tests that expectation values are properly calculated for single-wire observables with parameters."""

        obs = operation(np.array(par), wires=[0])

        qubit_device_1_wire.reset()
        qubit_device_1_wire.apply(
            [qml.QubitStateVector(np.array(input), wires=[0])], obs.diagonalizing_gates()
        )
        res = qubit_device_1_wire.expval(obs)

        assert np.allclose(res, expected_output, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "operation,input,expected_output,par",
        [
            (
                qml.Hermitian,
                [
                    [1 / math.sqrt(3), 0, 1 / math.sqrt(3), 1 / math.sqrt(3)],
                    [0, 0, 0, 1],
                    [1 / math.sqrt(2), 0, -1 / math.sqrt(2), 0],
                ],
                [4 / 3, 0, 1],
                [[1, 1j, 0, 1], [-1j, 1, 0, 0], [0, 0, 1, -1j], [1, 0, 1j, 0]],
            ),
            (
                qml.Hermitian,
                [[INVSQ2, 0, 0, INVSQ2], [0, INVSQ2, -INVSQ2, 0]],
                [1, -1],
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
            ),
        ],
    )
    def test_expval_two_wires_with_parameters_broadcasted(
        self, qubit_device_2_wires, tol, operation, input, expected_output, par
    ):
        """Tests that expectation values are properly calculated for two-wire observables with parameters."""

        obs = operation(np.array(par), wires=[0, 1])

        qubit_device_2_wires.reset()
        qubit_device_2_wires.apply(
            [qml.QubitStateVector(np.array(input), wires=[0, 1])], obs.diagonalizing_gates()
        )
        res = qubit_device_2_wires.expval(obs)

        assert np.allclose(res, expected_output, atol=tol, rtol=0)

    def test_expval_estimate_broadcasted(self):
        """Test that the expectation value is not analytically calculated"""

        dev = qml.device("default.qubit", wires=1, shots=3)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit():
            qml.RX(np.zeros(5), wires=0)  # Broadcast the tape without applying an op
            return qml.expval(qml.PauliX(0))

        expval = circuit()

        # With 3 samples we are guaranteed to see a difference between
        # an estimated variance an an analytically calculated one
        assert np.all(expval != 0.0)


class TestVarBroadcasted:
    """Tests that variances are properly calculated for broadcasted states."""

    @pytest.mark.parametrize(
        "operation,input,expected_output",
        [
            (qml.PauliX, [plus, zero, minus], [0, 1, 0]),
            (qml.PauliY, [y_plus, zero, y_minus], [0, 1, 0]),
            (qml.PauliZ, [plus, zero, one], [1, 0, 0]),
            (qml.Hadamard, [plus, zero, one], [0.5, 0.5, 0.5]),
            (qml.Identity, [minus, zero, one], [0, 0, 0]),
        ],
    )
    def test_var_single_wire_no_parameters_broadcasted(
        self, qubit_device_1_wire, tol, operation, input, expected_output
    ):
        """Tests that variances are properly calculated for single-wire observables without parameters."""

        obs = operation(wires=[0])

        qubit_device_1_wire.reset()
        qubit_device_1_wire.apply(
            [qml.QubitStateVector(np.array(input), wires=[0])], obs.diagonalizing_gates()
        )
        res = qubit_device_1_wire.var(obs)

        assert np.allclose(res, expected_output, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "operation,input,expected_output,par",
        [(qml.Hermitian, [zero, one, minus, y_plus], [1, 1, 1, 0], I - Y)],
    )
    def test_var_single_wire_with_parameters_broadcasted(
        self, qubit_device_1_wire, tol, operation, input, expected_output, par
    ):
        """Tests that variances are properly calculated for single-wire observables with parameters."""

        obs = operation(np.array(par), wires=[0])

        qubit_device_1_wire.reset()
        qubit_device_1_wire.apply(
            [qml.QubitStateVector(np.array(input), wires=[0])], obs.diagonalizing_gates()
        )
        res = qubit_device_1_wire.var(obs)

        assert np.allclose(res, expected_output, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "operation,input,expected_output,par",
        [
            (
                qml.Hermitian,
                [
                    [1 / math.sqrt(3), 0, 1 / math.sqrt(3), 1 / math.sqrt(3)],
                    [0, 0, 0, 1],
                    [1 / math.sqrt(2), 0, -1 / math.sqrt(2), 0],
                ],
                [11 / 9, 2, 3 / 2],
                [[1, 1j, 0, 1], [-1j, 1, 0, 0], [0, 0, 1, -1j], [1, 0, 1j, 1]],
            ),
            (
                qml.Hermitian,
                [[INVSQ2, 0, 0, INVSQ2], [0, INVSQ2, -INVSQ2, 0]],
                [0, 0],
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
            ),
        ],
    )
    def test_var_two_wires_with_parameters_broadcasted(
        self, qubit_device_2_wires, tol, operation, input, expected_output, par
    ):
        """Tests that variances are properly calculated for two-wire observables with parameters."""

        obs = operation(np.array(par), wires=[0, 1])

        qubit_device_2_wires.reset()
        qubit_device_2_wires.apply(
            [qml.QubitStateVector(np.array(input), wires=[0, 1])], obs.diagonalizing_gates()
        )
        res = qubit_device_2_wires.var(obs)

        assert np.allclose(res, expected_output, atol=tol, rtol=0)

    def test_var_estimate_broadcasted(self):
        """Test that the variance is not analytically calculated"""

        dev = qml.device("default.qubit", wires=1, shots=3)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit():
            qml.RX(np.zeros(5), wires=0)  # Broadcast the tape without applying an op
            return qml.var(qml.PauliX(0))

        var = circuit()

        # With 3 samples we are guaranteed to see a difference between
        # an estimated variance and an analytically calculated one
        assert np.all(var != 1.0)


class TestSampleBroadcasted:
    """Tests that samples are properly calculated for broadcasted states."""

    def test_sample_dimensions_broadcasted(self):
        """Tests if the samples returned by the sample function have
        the correct dimensions
        """

        # Explicitly resetting is necessary as the internal
        # state is set to None in __init__ and only properly
        # initialized during reset
        dev = qml.device("default.qubit", wires=2, shots=1000)

        dev.apply([qml.RX(np.array([np.pi / 2, 0.0]), 0), qml.RX(np.array([np.pi / 2, 0.0]), 1)])

        dev.shots = 10
        dev._wires_measured = {0}
        dev._samples = dev.generate_samples()
        s1 = dev.sample(qml.PauliZ(0))
        assert s1.shape == (
            2,
            10,
        )

        dev.reset()
        dev.shots = 12
        dev._wires_measured = {1}
        dev._samples = dev.generate_samples()
        s2 = dev.sample(qml.PauliZ(wires=[1]))
        assert s2.shape == (12,)

        dev.reset()
        dev.apply([qml.RX(np.ones(5), 0), qml.RX(np.ones(5), 1)])
        dev.shots = 17
        dev._wires_measured = {0, 1}
        dev._samples = dev.generate_samples()
        s3 = dev.sample(qml.PauliX(0) @ qml.PauliZ(1))
        assert s3.shape == (5, 17)

    def test_sample_values_broadcasted(self, qubit_device_2_wires, tol):
        """Tests if the samples returned by sample have
        the correct values
        """

        # Explicitly resetting is necessary as the internal
        # state is set to None in __init__ and only properly
        # initialized during reset
        dev = qml.device("default.qubit", wires=2, shots=1000)

        dev.apply([qml.RX(np.ones(3), wires=[0])])
        dev._wires_measured = {0}
        dev._samples = dev.generate_samples()

        s1 = dev.sample(qml.PauliZ(0))

        # s1 should only contain 1 and -1, which is guaranteed if
        # they square to 1
        assert np.allclose(s1**2, 1, atol=tol, rtol=0)


class TestDefaultQubitIntegrationBroadcasted:
    """Integration tests for default.qubit. This test ensures it integrates
    properly with the PennyLane interface, in particular QNode."""

    @pytest.mark.parametrize("r_dtype", [np.float32, np.float64])
    def test_qubit_circuit_broadcasted(self, qubit_device_1_wire, r_dtype, tol):
        """Test that the default qubit plugin provides correct result for a simple circuit"""

        p = np.array([0.543, np.pi / 2, 0.0, 1.0])

        dev = qubit_device_1_wire
        dev.R_DTYPE = r_dtype

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliY(0))

        expected = -np.sin(p)

        res = circuit(p)
        assert np.allclose(res, expected, atol=tol, rtol=0)
        assert res.dtype == r_dtype

    def test_qubit_identity_broadcasted(self, qubit_device_1_wire, tol):
        """Test that the default qubit plugin provides correct result for the Identity expectation"""

        p = np.array([0.543, np.pi / 2, 0.0, 1.0])

        @qml.qnode(qubit_device_1_wire)
        def circuit(x):
            """Test quantum function"""
            qml.RX(x, wires=0)
            return qml.expval(qml.Identity(0))

        assert np.allclose(circuit(p), 1, atol=tol, rtol=0)

    def test_nonzero_shots_broadcasted(self, tol):
        """Test that the default qubit plugin provides correct result for high shot number"""

        shots = 10**5
        dev = qml.device("default.qubit", wires=1, shots=shots)

        p = np.array([0.543, np.pi / 2, 0.0, 1.0])

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(x):
            """Test quantum function"""
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliY(0))

        runs = []
        for _ in range(100):
            runs.append(circuit(p))

        assert np.allclose(np.mean(runs, axis=0), -np.sin(p), atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "name,state,expected_output",
        [
            ("PauliX", [plus, minus, zero], [1, -1, 0]),
            ("PauliY", [y_plus, y_minus, zero], [1, -1, 0]),
            ("PauliZ", [plus, one, zero], [0, -1, 1]),
            ("Hadamard", [plus, one, zero], [INVSQ2, -INVSQ2, INVSQ2]),
        ],
    )
    def test_supported_observable_single_wire_no_parameters_broadcasted(
        self, qubit_device_1_wire, tol, name, state, expected_output
    ):
        """Tests supported observables on single wires without parameters."""

        obs = getattr(qml.ops, name)

        assert qubit_device_1_wire.supports_observable(name)

        @qml.qnode(qubit_device_1_wire)
        def circuit():
            qml.QubitStateVector(np.array(state), wires=[0])
            return qml.expval(obs(wires=[0]))

        assert np.allclose(circuit(), expected_output, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "name,state,expected_output,par",
        [
            ("Identity", [zero, one, plus], [1, 1, 1], []),
            ("Hermitian", [zero, one, minus], [1, 1, 1], [I - Y]),
        ],
    )
    def test_supported_observable_single_wire_with_parameters_broadcasted(
        self, qubit_device_1_wire, tol, name, state, expected_output, par
    ):
        """Tests supported observables on single wires with parameters."""

        obs = getattr(qml.ops, name)

        assert qubit_device_1_wire.supports_observable(name)

        @qml.qnode(qubit_device_1_wire)
        def circuit():
            qml.QubitStateVector(np.array(state), wires=[0])
            return qml.expval(obs(*par, wires=[0]))

        assert np.allclose(circuit(), expected_output, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "name,state,expected_output,par",
        [
            (
                "Hermitian",
                [
                    [1 / math.sqrt(3), 0, 1 / math.sqrt(3), 1 / math.sqrt(3)],
                    [0, 0, 0, 1],
                    [1 / math.sqrt(2), 0, -1 / math.sqrt(2), 0],
                ],
                [4 / 3, 0, 1],
                ([[1, 1j, 0, 1], [-1j, 1, 0, 0], [0, 0, 1, -1j], [1, 0, 1j, 0]],),
            ),
            (
                "Hermitian",
                [[INVSQ2, 0, 0, INVSQ2], [0, INVSQ2, -INVSQ2, 0]],
                [1, -1],
                ([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],),
            ),
        ],
    )
    def test_supported_observable_two_wires_with_parameters_broadcasted(
        self, qubit_device_2_wires, tol, name, state, expected_output, par
    ):
        """Tests supported observables on two wires with parameters."""

        obs = getattr(qml.ops, name)

        assert qubit_device_2_wires.supports_observable(name)

        @qml.qnode(qubit_device_2_wires)
        def circuit():
            qml.QubitStateVector(np.array(state), wires=[0, 1])
            return qml.expval(obs(*par, wires=[0, 1]))

        assert np.allclose(circuit(), expected_output, atol=tol, rtol=0)

    def test_multi_samples_return_correlated_results_broadcasted(self):
        """Tests if the samples returned by the sample function are correlated
        correctly for correlated observables.
        """

        dev = qml.device("default.qubit", wires=2, shots=1000)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit():
            qml.Hadamard(0)
            qml.RX(np.zeros(5), 0)
            qml.CNOT(wires=[0, 1])
            return qml.sample(qml.PauliZ(0)), qml.sample(qml.PauliZ(1))

        outcomes = circuit()

        assert np.array_equal(outcomes[0], outcomes[1])

    @pytest.mark.parametrize("num_wires", [3, 4, 5, 6, 7, 8])
    def test_multi_samples_correlated_results_more_wires_than_observable_broadcasted(
        self, num_wires
    ):
        """Tests if the samples returned by the sample function are correlated
        correctly for correlated observables on larger devices than the observables
        """

        dev = qml.device("default.qubit", wires=num_wires, shots=1000)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit():
            qml.Hadamard(0)
            qml.RX(np.zeros(5), 0)
            qml.CNOT(wires=[0, 1])
            return qml.sample(qml.PauliZ(0)), qml.sample(qml.PauliZ(1))

        outcomes = circuit()

        assert np.array_equal(outcomes[0], outcomes[1])


@pytest.mark.parametrize(
    "theta,phi,varphi", [(THETA, PHI, VARPHI), (THETA, PHI[0], VARPHI), (THETA[0], PHI, VARPHI[1])]
)
class TestTensorExpvalBroadcasted:
    """Test tensor expectation values for broadcasted states"""

    def test_paulix_pauliy_broadcasted(self, theta, phi, varphi, tol):
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

    def test_pauliz_identity_broadcasted(self, theta, phi, varphi, tol):
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

    def test_pauliz_hadamard_broadcasted(self, theta, phi, varphi, tol):
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

    def test_hermitian_broadcasted(self, theta, phi, varphi, tol):
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

    def test_hermitian_hermitian_broadcasted(self, theta, phi, varphi, tol):
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

    def test_hermitian_identity_expectation_broadcasted(self, theta, phi, varphi, tol):
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

    def test_hermitian_two_wires_identity_expectation_broadcasted(self, theta, phi, varphi, tol):
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


@pytest.mark.parametrize(
    "theta,phi,varphi", [(THETA, PHI, VARPHI), (THETA, PHI[0], VARPHI), (THETA[0], PHI, VARPHI[1])]
)
class TestTensorVarBroadcasted:
    """Tests for variance of tensor observables for broadcasted states"""

    def test_paulix_pauliy_broadcasted(self, theta, phi, varphi, tol):
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

    def test_pauliz_hadamard_broadcasted(self, theta, phi, varphi, tol):
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

    def test_hermitian_broadcasted(self, theta, phi, varphi, tol):
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


@pytest.mark.parametrize(
    "theta,phi,varphi", [(THETA, PHI, VARPHI), (THETA, PHI[0], VARPHI), (THETA[0], PHI, VARPHI[1])]
)
class TestTensorSampleBroadcasted:
    """Test tensor sampling for broadcated states"""

    def test_paulix_pauliy_broadcasted(self, theta, phi, varphi, tol_stochastic):
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

        mean = p @ s1
        expected = np.sin(theta) * np.sin(phi) * np.sin(varphi)
        assert np.allclose(mean, expected, atol=tol_stochastic, rtol=0)

        var = p @ (s1**2) - (p @ s1).real ** 2
        expected = (
            8 * np.sin(theta) ** 2 * np.cos(2 * varphi) * np.sin(phi) ** 2
            - np.cos(2 * (theta - phi))
            - np.cos(2 * (theta + phi))
            + 2 * np.cos(2 * theta)
            + 2 * np.cos(2 * phi)
            + 14
        ) / 16
        assert np.allclose(var, expected, atol=tol_stochastic, rtol=0)

    def test_pauliz_hadamard_broadcasted(self, theta, phi, varphi, tol_stochastic):
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

        mean = p @ s1
        expected = -(np.cos(varphi) * np.sin(phi) + np.sin(varphi) * np.cos(theta)) / np.sqrt(2)
        assert np.allclose(mean, expected, atol=tol_stochastic, rtol=0)

        var = p @ (s1**2) - (p @ s1).real ** 2
        expected = (
            3
            + np.cos(2 * phi) * np.cos(varphi) ** 2
            - np.cos(2 * theta) * np.sin(varphi) ** 2
            - 2 * np.cos(theta) * np.sin(phi) * np.sin(2 * varphi)
        ) / 4
        assert np.allclose(var, expected, atol=tol_stochastic, rtol=0)

    def test_hermitian_broadcasted(self, theta, phi, varphi, tol_stochastic):
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

        mean = p @ s1
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

        var = p @ (s1**2) - (p @ s1).real ** 2
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
class TestDtypePreservedBroadcasted:
    """Test that the user-defined dtype of the device is preserved for QNode
    evaluation"""

    @pytest.mark.parametrize(
        "op",
        [
            # TODO[dwierichs]: Include the following test cases once the operations support
            # broadcasting.
            # qml.SingleExcitation,
            # qml.SingleExcitationPlus,
            # qml.SingleExcitationMinus,
            # qml.DoubleExcitation,
            # qml.DoubleExcitationPlus,
            # qml.DoubleExcitationMinus,
            # qml.OrbitalRotation,
            # qml.FermionicSWAP,
            qml.QubitSum,
            qml.QubitCarry,
        ],
    )
    def test_state_dtype_after_op_broadcasted(self, r_dtype, c_dtype, op, tol):
        """Test that the default qubit plugin preserves data types of states when an operation is
        applied. As TestApply class check most of operators, we here only check some subtle
        examples.
        """

        dev = qml.device("default.qubit", wires=4, r_dtype=r_dtype, c_dtype=c_dtype)

        n_wires = op.num_wires
        n_params = op.num_params

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit():
            x = np.array([0.543, 0.622, 1.3])
            if n_params == 0:
                op(wires=range(n_wires))
            elif n_params == 1:
                op(x, wires=range(n_wires))
            else:
                op([x] * n_params, wires=range(n_wires))
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
    def test_measurement_real_dtype_broadcasted(self, r_dtype, c_dtype, measurement, tol):
        """Test that the default qubit plugin provides correct result for a simple circuit"""
        p = np.array([0.543, 0.622, 1.3])

        dev = qml.device("default.qubit", wires=3, r_dtype=r_dtype, c_dtype=c_dtype)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.apply(measurement)

        res = circuit(p)
        assert res.dtype == r_dtype

    def test_measurement_complex_dtype_broadcasted(self, r_dtype, c_dtype, tol):
        """Test that the default qubit plugin provides correct result for a simple circuit"""
        p = np.array([0.543, 0.622, 1.3])
        m = qml.state()

        dev = qml.device("default.qubit", wires=3, r_dtype=r_dtype, c_dtype=c_dtype)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.apply(m)

        res = circuit(p)
        assert res.dtype == c_dtype


class TestProbabilityIntegrationBroadcasted:
    """Test probability method for when analytic is True/False"""

    def mock_analytic_counter(self, wires=None):
        self.analytic_counter += 1
        return np.array([1, 0, 0, 0], dtype=float)

    def test_probability_broadcasted(self, tol):
        """Test that the probability function works for finite and infinite shots"""
        dev = qml.device("default.qubit", wires=2, shots=1000)
        dev_analytic = qml.device("default.qubit", wires=2, shots=None)

        x = np.array([[0.2, 0.5, 0.4], [0.9, 0.8, 0.3]])

        def circuit(x):
            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0, 1])

        prob = qml.QNode(circuit, dev)
        prob_analytic = qml.QNode(circuit, dev_analytic)

        assert np.allclose(prob(x).sum(axis=-1), 1, atol=tol, rtol=0)
        assert np.allclose(prob_analytic(x), prob(x), atol=0.1, rtol=0)
        assert not np.array_equal(prob_analytic(x), prob(x))


class TestWiresIntegrationBroadcasted:
    """Test that the device integrates with PennyLane's wire management."""

    def make_circuit_probs(self, wires):
        """Factory for a qnode returning probabilities using arbitrary wire labels."""
        dev = qml.device("default.qubit", wires=wires)
        n_wires = len(wires)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit():
            qml.RX(np.array([0.5, 1.2, -0.6]), wires=wires[0 % n_wires])
            qml.RY(np.array([2.0, 0.4, 1.2]), wires=wires[1 % n_wires])
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
    def test_wires_probs_broadcasted(self, wires1, wires2, tol):
        """Test that the probability vector of a circuit is independent from the wire labels used."""

        circuit1 = self.make_circuit_probs(wires1)
        circuit2 = self.make_circuit_probs(wires2)

        assert np.allclose(circuit1(), circuit2(), tol)


@pytest.mark.parametrize("inverse", [True, False])
class TestApplyOpsBroadcasted:
    """Tests for special methods listed in _apply_ops that use array manipulation tricks to apply
    gates in DefaultQubit."""

    broadcasted_state = np.arange(2**4 * 3, dtype=np.complex128).reshape((3, 2, 2, 2, 2))
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
    def test_apply_single_qubit_op_broadcasted_state(self, op, method, inverse):
        """Test if the application of single qubit operations to a
        broadcasted state is correct."""
        state_out = method(self.broadcasted_state, axes=[2], inverse=inverse)
        op = op(wires=[1])
        matrix = op.inv().matrix() if inverse else op.matrix()
        state_out_einsum = np.einsum("ab,mibjk->miajk", matrix, self.broadcasted_state)
        assert np.allclose(state_out, state_out_einsum)

    @pytest.mark.parametrize("op, method", two_qubit_ops)
    def test_apply_two_qubit_op_broadcasted_state(self, op, method, inverse):
        """Test if the application of two qubit operations to a
        broadcasted state is correct."""
        state_out = method(self.broadcasted_state, axes=[1, 2])
        op = op(wires=[0, 1])
        matrix = op.inv().matrix() if inverse else op.matrix()
        matrix = matrix.reshape((2, 2, 2, 2))
        state_out_einsum = np.einsum("abcd,mcdjk->mabjk", matrix, self.broadcasted_state)
        assert np.allclose(state_out, state_out_einsum)

    @pytest.mark.parametrize("op, method", two_qubit_ops)
    def test_apply_two_qubit_op_reverse_broadcasted_state(self, op, method, inverse):
        """Test if the application of two qubit operations to a
        broadcasted state is correct when the applied wires are reversed."""
        state_out = method(self.broadcasted_state, axes=[3, 2])
        op = op(wires=[2, 1])
        matrix = op.inv().matrix() if inverse else op.matrix()
        matrix = matrix.reshape((2, 2, 2, 2))
        state_out_einsum = np.einsum("abcd,midck->mibak", matrix, self.broadcasted_state)
        assert np.allclose(state_out, state_out_einsum)

    @pytest.mark.parametrize("op, method", three_qubit_ops)
    def test_apply_three_qubit_op_controls_smaller_broadcasted_state(self, op, method, inverse):
        """Test if the application of three qubit operations to a broadcasted
        state is correct when both control wires are smaller than the target wire."""
        state_out = method(self.broadcasted_state, axes=[1, 3, 4])
        op = op(wires=[0, 2, 3])
        matrix = op.inv().matrix() if inverse else op.matrix()
        matrix = matrix.reshape((2, 2) * 3)
        state_out_einsum = np.einsum("abcdef,mdkef->makbc", matrix, self.broadcasted_state)
        assert np.allclose(state_out, state_out_einsum)

    @pytest.mark.parametrize("op, method", three_qubit_ops)
    def test_apply_three_qubit_op_controls_greater_broadcasted_state(self, op, method, inverse):
        """Test if the application of three qubit operations to a broadcasted
        state is correct when both control wires are greater than the target wire."""
        state_out = method(self.broadcasted_state, axes=[3, 2, 1])
        op = op(wires=[2, 1, 0])
        matrix = op.inv().matrix() if inverse else op.matrix()
        matrix = matrix.reshape((2, 2) * 3)
        state_out_einsum = np.einsum("abcdef,mfedk->mcbak", matrix, self.broadcasted_state)
        assert np.allclose(state_out, state_out_einsum)

    @pytest.mark.parametrize("op, method", three_qubit_ops)
    def test_apply_three_qubit_op_controls_split_broadcasted_state(self, op, method, inverse):
        """Test if the application of three qubit operations to a broadcasted state is correct
        when one control wire is smaller and one control wire is greater than the target wire."""
        state_out = method(self.broadcasted_state, axes=[4, 2, 3])
        op = op(wires=[3, 1, 2])
        matrix = op.inv().matrix() if inverse else op.matrix()
        matrix = matrix.reshape((2, 2) * 3)
        state_out_einsum = np.einsum("abcdef,mkdfe->mkacb", matrix, self.broadcasted_state)
        assert np.allclose(state_out, state_out_einsum)


class TestStateVectorBroadcasted:
    """Unit tests for the _apply_state_vector method with broadcasting"""

    def test_full_subsystem_broadcasted(self, mocker):
        """Test applying a state vector to the full subsystem"""
        dev = DefaultQubit(wires=["a", "b", "c"])
        state = np.array([[0, 1, 1, 0, 1, 1, 0, 0], [1, 0, 0, 0, 1, 0, 1, 1]]) / 2.0
        state_wires = qml.wires.Wires(["a", "b", "c"])

        spy = mocker.spy(dev, "_scatter")
        dev._apply_state_vector(state=state, device_wires=state_wires)

        assert np.all(dev._state.reshape((2, 8)) == state)
        spy.assert_not_called()

    def test_partial_subsystem_broadcasted(self, mocker):
        """Test applying a state vector to a subset of wires of the full subsystem"""

        dev = DefaultQubit(wires=["a", "b", "c"])
        state = np.array([[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 0]]) / np.sqrt(2.0)
        state_wires = qml.wires.Wires(["a", "c"])

        spy = mocker.spy(dev, "_scatter")
        dev._apply_state_vector(state=state, device_wires=state_wires)
        # Axes are (broadcasting, wire "a", wire "b", wire "c"), so we sum over axis=2
        res = np.sum(dev._state, axis=(2,)).reshape((3, 4))

        assert np.all(res == state)
        spy.assert_called()


class TestApplyOperationBroadcasted:
    """Unit tests for the internal _apply_operation method."""

    @pytest.mark.parametrize("inverse", [True, False])
    def test_internal_apply_ops_case_broadcasted(self, inverse, monkeypatch):
        """Tests that if we provide an operation that has an internal
        implementation, then we use that specific implementation.

        This test provides a new internal function that `default.qubit` uses to
        apply `PauliX` (rather than redefining the gate itself).
        """
        dev = qml.device("default.qubit", wires=1)

        test_state = np.array([[1, 0], [INVSQ2, INVSQ2], [0, 1]])
        # Create a dummy operation
        expected_test_output = np.ones(1)
        supported_gate_application = lambda *args, **kwargs: expected_test_output

        with monkeypatch.context() as m:
            # Set the internal ops implementations dict
            m.setattr(dev, "_apply_ops", {"PauliX": supported_gate_application})

            op = qml.PauliX(0) if not inverse else qml.PauliX(0).inv()

            res = dev._apply_operation(test_state, op)
            assert np.allclose(res, expected_test_output)

    def test_diagonal_operation_case_broadcasted(self, mocker, monkeypatch):
        """Tests the case when the operation to be applied is
        diagonal in the computational basis and the _apply_diagonal_unitary method is used."""
        dev = qml.device("default.qubit", wires=1)
        par = 0.3

        test_state = np.array([[1, 0], [INVSQ2, INVSQ2], [0, 1]])
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

    def test_apply_einsum_case_broadcasted(self, mocker, monkeypatch):
        """Tests the case when np.einsum is used to apply an operation in
        default.qubit."""
        dev = qml.device("default.qubit", wires=1)

        test_state = np.array([[1, 0], [INVSQ2, INVSQ2], [0, 1]])
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
    def test_apply_tensordot_case_broadcasted(self, inverse, mocker, monkeypatch):
        """Tests the case when np.tensordot is used to apply an operation in
        default.qubit."""
        dev = qml.device("default.qubit", wires=3)

        test_state = np.array([[1, 0], [INVSQ2, INVSQ2], [0, 1]])
        wires = [0, 1, 2]

        # Redefine the Toffoli gate so that it is an example for a gate with
        # more than two wires
        class TestToffoli(qml.operation.Operation):
            num_wires = 3

            @staticmethod
            def compute_matrix(*params, **hyperparams):
                return Toffoli

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

    def test_identity_skipped_broadcasted(self, mocker):
        """Test that applying the identity operation does not perform any additional computations."""
        dev = qml.device("default.qubit", wires=1)

        starting_state = np.array([[1, 0], [INVSQ2, INVSQ2], [0, 1]])
        op = qml.Identity(0)

        spy_diagonal = mocker.spy(dev, "_apply_diagonal_unitary")
        spy_einsum = mocker.spy(dev, "_apply_unitary_einsum")
        spy_unitary = mocker.spy(dev, "_apply_unitary")

        res = dev._apply_operation(starting_state, op)
        assert res is starting_state

        spy_diagonal.assert_not_called()
        spy_einsum.assert_not_called()
        spy_unitary.assert_not_called()


class TestHamiltonianSupportBroadcasted:
    """Tests the devices' native support for Hamiltonian observables."""

    def test_do_not_split_analytic_broadcasted(self, mocker):
        """Tests that the Hamiltonian is not split for shots=None."""
        dev = qml.device("default.qubit", wires=2)
        H = qml.Hamiltonian(np.array([0.1, 0.2]), [qml.PauliX(0), qml.PauliZ(1)])

        @qml.qnode(dev, diff_method="parameter-shift", interface=None)
        def circuit():
            qml.RX(np.zeros(5), 0)  # Broadcast the state by applying a broadcasted identity
            return qml.expval(H)

        spy = mocker.spy(dev, "expval")

        circuit()
        # evaluated one expval altogether
        assert spy.call_count == 1

    def test_split_finite_shots_broadcasted(self, mocker):
        """Tests that the Hamiltonian is split for finite shots."""
        dev = qml.device("default.qubit", wires=2, shots=10)
        spy = mocker.spy(dev, "expval")

        H = qml.Hamiltonian(np.array([0.1, 0.2]), [qml.PauliX(0), qml.PauliZ(1)])

        @qml.qnode(dev)
        def circuit():
            qml.RX(np.zeros(5), 0)  # Broadcast the state by applying a broadcasted identity
            return qml.expval(H)

        circuit()

        # evaluated one expval per Pauli observable
        assert spy.call_count == 2


original_capabilities = qml.devices.DefaultQubit.capabilities()


@pytest.fixture(scope="function")
def mock_default_qubit(monkeypatch):
    """A function to create a mock device that mocks the broadcasting support flag
    to be False, so that default support via broadcast_expand transform can be tested"""

    def overwrite_support(*cls):
        capabilities = original_capabilities.copy()
        capabilities.update(supports_broadcasting=False)
        return capabilities

    with monkeypatch.context() as m:
        m.setattr(qml.devices.DefaultQubit, "capabilities", overwrite_support)

        def get_default_qubit(wires=1, shots=None):
            dev = qml.devices.DefaultQubit(wires=wires, shots=shots)
            return dev

        yield get_default_qubit


@pytest.mark.parametrize("shots", [None, 100000])
class TestBroadcastingSupportViaExpansion:
    """Tests that the device correctly makes use of ``broadcast_expand`` to
    execute broadcasted tapes if its capability to execute broadcasted tapes
    is artificially deactivated."""

    @pytest.mark.parametrize("x", [0.2, np.array([0.1, 0.6, 0.3]), np.array([0.1])])
    def test_with_single_broadcasted_par(self, x, shots, mock_default_qubit):
        """Test that broadcasting on a circuit with a
        single parametrized operation works."""
        dev = mock_default_qubit(wires=2, shots=shots)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        circuit.construct((np.array(x),), {})
        out = circuit(np.array(x))

        assert circuit.device.num_executions == (1 if isinstance(x, float) else len(x))
        tol = 1e-10 if shots is None else 1e-2
        assert qml.math.allclose(out, qml.math.cos(x), atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "x, y", [(0.2, np.array([0.4])), (np.array([0.1, 5.1]), np.array([0.1, -0.3]))]
    )
    def test_with_multiple_pars(self, x, y, shots, mock_default_qubit):
        """Test that broadcasting on a circuit with a
        single parametrized operation works."""
        dev = mock_default_qubit(wires=2, shots=shots)

        @qml.qnode(dev)
        def circuit(x, y):
            qml.RX(x, wires=0)
            qml.RX(y, wires=1)
            return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))]

        out = circuit(x, y)
        expected = qml.math.stack([qml.math.cos(x) * qml.math.ones_like(y), -qml.math.sin(y)]).T

        assert circuit.device.num_executions == len(y)
        tol = 1e-10 if shots is None else 1e-2
        assert qml.math.allclose(out, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "x, y", [(0.2, np.array([0.4])), (np.array([0.1, 5.1]), np.array([0.1, -0.3]))]
    )
    def test_with_Hamiltonian(self, x, y, shots, mock_default_qubit):
        """Test that broadcasting on a circuit with a
        single parametrized operation works."""
        dev = mock_default_qubit(wires=2, shots=shots)

        H = qml.Hamiltonian([0.3, 0.9], [qml.PauliZ(0), qml.PauliY(1)])
        H.compute_grouping()

        @qml.qnode(dev)
        def circuit(x, y):
            qml.RX(x, wires=0)
            qml.RX(y, wires=1)
            return qml.expval(H)

        out = circuit(x, y)
        expected = 0.3 * qml.math.cos(x) * qml.math.ones_like(y) - 0.9 * qml.math.sin(y)

        assert circuit.device.num_executions == len(y)
        tol = 1e-10 if shots is None else 1e-2
        assert qml.math.allclose(out, expected, atol=tol, rtol=0)

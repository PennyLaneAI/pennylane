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
from pennylane.operation import Operation

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

U_cswap = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1]])


H = np.array(
    [[1.02789352, 1.61296440 - 0.3498192j], [1.61296440 + 0.3498192j, 1.23920938 + 0j]]
)


THETA = np.linspace(0.11, 1, 3)
PHI = np.linspace(0.32, 1, 3)
VARPHI = np.linspace(0.02, 1, 3)


def prep_par(par, op):
    "Convert par into a list of parameters that op expects."
    if op.par_domain == "A":
        return [np.diag([x, 1]) for x in par]
    return par


def include_inverses_with_test_data(test_data):
    return test_data + [(item[0] + ".inv", item[1], item[2]) for item in test_data]


class TestApply:
    """Tests that operations and inverses of certain operations are applied correctly or that the proper
    errors are raised.
    """

    test_data_no_parameters = [
        (qml.PauliX, [1, 0], np.array([0, 1])),
        (qml.PauliX, [1/math.sqrt(2), 1/math.sqrt(2)], [1/math.sqrt(2), 1/math.sqrt(2)]),
        (qml.PauliY, [1, 0], [0, 1j]),
        (qml.PauliY, [1/math.sqrt(2), 1/math.sqrt(2)], [-1j/math.sqrt(2), 1j/math.sqrt(2)]),
        (qml.PauliZ, [1, 0], [1, 0]),
        (qml.PauliZ, [1/math.sqrt(2), 1/math.sqrt(2)], [1/math.sqrt(2), -1/math.sqrt(2)]),
        (qml.S, [1, 0], [1, 0]),
        (qml.S, [1/math.sqrt(2), 1/math.sqrt(2)], [1/math.sqrt(2), 1j/math.sqrt(2)]),
        (qml.T, [1, 0], [1, 0]),
        (qml.T, [1 / math.sqrt(2), 1 / math.sqrt(2)], [1 / math.sqrt(2), np.exp(1j * np.pi / 4) / math.sqrt(2)]),
        (qml.Hadamard, [1, 0], [1/math.sqrt(2), 1/math.sqrt(2)]),
        (qml.Hadamard, [1/math.sqrt(2), -1/math.sqrt(2)], [0, 1]),
    ]

    test_data_no_parameters_inverses  = [
        (qml.PauliX, [1, 0], np.array([0, 1])),
        (qml.PauliX, [1/math.sqrt(2), 1/math.sqrt(2)], [1/math.sqrt(2), 1/math.sqrt(2)]),
        (qml.PauliY, [1, 0], [0, 1j]),
        (qml.PauliY, [1/math.sqrt(2), 1/math.sqrt(2)], [-1j/math.sqrt(2), 1j/math.sqrt(2)]),
        (qml.PauliZ, [1, 0], [1, 0]),
        (qml.PauliZ, [1/math.sqrt(2), 1/math.sqrt(2)], [1/math.sqrt(2), -1/math.sqrt(2)]),
        (qml.S, [1, 0], [1, 0]),
        (qml.S, [1/math.sqrt(2), 1/math.sqrt(2)], [1/math.sqrt(2), -1j/math.sqrt(2)]),
        (qml.T, [1, 0], [1, 0]),
        (qml.T, [1 / math.sqrt(2), 1 / math.sqrt(2)], [1 / math.sqrt(2), np.exp(-1j * np.pi / 4) / math.sqrt(2)]),
        (qml.Hadamard, [1, 0], [1/math.sqrt(2), 1/math.sqrt(2)]),
        (qml.Hadamard, [1/math.sqrt(2), -1/math.sqrt(2)], [0, 1]),
    ]

    @pytest.mark.parametrize("operation,input,expected_output", test_data_no_parameters)
    def test_apply_operation_single_wire_no_parameters(self, qubit_device_1_wire, tol, operation, input, expected_output):
        """Tests that applying an operation yields the expected output state for single wire
           operations that have no parameters."""

        qubit_device_1_wire._state = np.array(input)
        qubit_device_1_wire.apply([operation(wires=[0])])

        assert np.allclose(qubit_device_1_wire._state, np.array(expected_output), atol=tol, rtol=0)

    @pytest.mark.parametrize("operation,input,expected_output", test_data_no_parameters_inverses)
    def test_apply_operation_single_wire_no_parameters_inverse(self, qubit_device_1_wire, tol, operation, input, expected_output):
        """Tests that applying an operation yields the expected output state for single wire
           operations that have no parameters."""

        qubit_device_1_wire._state = np.array(input)
        qubit_device_1_wire.apply([operation(wires=[0]).inv()])

        assert np.allclose(qubit_device_1_wire._state, np.array(expected_output), atol=tol, rtol=0)

    test_data_two_wires_no_parameters = [
        (qml.CNOT, [1, 0, 0, 0], [1, 0, 0, 0]),
        (qml.CNOT, [0, 0, 1, 0], [0, 0, 0, 1]),
        (qml.CNOT, [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)], [1 / math.sqrt(2), 0, 1 / math.sqrt(2), 0]),
        (qml.SWAP, [1, 0, 0, 0], [1, 0, 0, 0]),
        (qml.SWAP, [0, 0, 1, 0], [0, 1, 0, 0]),
        (qml.SWAP, [1 / math.sqrt(2), 0, -1 / math.sqrt(2), 0], [1 / math.sqrt(2), -1 / math.sqrt(2), 0, 0]),
        (qml.CZ, [1, 0, 0, 0], [1, 0, 0, 0]),
        (qml.CZ, [0, 0, 0, 1], [0, 0, 0, -1]),
        (qml.CZ, [1 / math.sqrt(2), 0, 0, -1 / math.sqrt(2)], [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)]),
    ]

    @pytest.mark.parametrize("operation,input,expected_output", test_data_two_wires_no_parameters)
    def test_apply_operation_two_wires_no_parameters(self, qubit_device_2_wires, tol, operation, input, expected_output):
        """Tests that applying an operation yields the expected output state for two wire
           operations that have no parameters."""

        qubit_device_2_wires._state = np.array(input).reshape((2, 2))
        qubit_device_2_wires.apply([operation(wires=[0, 1])])

        assert np.allclose(qubit_device_2_wires._state.flatten(), np.array(expected_output), atol=tol, rtol=0)

    @pytest.mark.parametrize("operation,input,expected_output", test_data_two_wires_no_parameters)
    def test_apply_operation_two_wires_no_parameters_inverse(self, qubit_device_2_wires, tol, operation, input, expected_output):
        """Tests that applying an operation yields the expected output state for two wire
           operations that have no parameters."""

        qubit_device_2_wires._state = np.array(input).reshape((2, 2))
        qubit_device_2_wires.apply([operation(wires=[0, 1]).inv()])

        assert np.allclose(qubit_device_2_wires._state.flatten(), np.array(expected_output), atol=tol, rtol=0)

    test_data_three_wires_no_parameters = [
        (qml.CSWAP, [1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0]),
        (qml.CSWAP, [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0]),
        (qml.CSWAP, [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0, 0]),
    ]

    @pytest.mark.parametrize("operation,input,expected_output", test_data_three_wires_no_parameters)
    def test_apply_operation_three_wires_no_parameters(self, qubit_device_3_wires, tol, operation, input, expected_output):
        """Tests that applying an operation yields the expected output state for three wire
           operations that have no parameters."""

        qubit_device_3_wires._state = np.array(input).reshape((2, 2, 2))
        qubit_device_3_wires.apply([operation(wires=[0, 1, 2])])

        assert np.allclose(qubit_device_3_wires._state.flatten(), np.array(expected_output), atol=tol, rtol=0)

    @pytest.mark.parametrize("operation,input,expected_output", test_data_three_wires_no_parameters)
    def test_apply_operation_three_wires_no_parameters_inverse(self, qubit_device_3_wires, tol, operation, input, expected_output):
        """Tests that applying the inverse of an operation yields the expected output state for three wire
           operations that have no parameters."""

        qubit_device_3_wires._state = np.array(input).reshape((2, 2, 2))
        qubit_device_3_wires.apply([operation(wires=[0, 1, 2]).inv()])

        assert np.allclose(qubit_device_3_wires._state.flatten(), np.array(expected_output), atol=tol, rtol=0)


    @pytest.mark.parametrize("operation,expected_output,par", [
        (qml.BasisState, [0, 0, 1, 0], [1, 0]),
        (qml.BasisState, [0, 0, 1, 0], [1, 0]),
        (qml.BasisState, [0, 0, 0, 1], [1, 1]),
        (qml.QubitStateVector, [0, 0, 1, 0], [0, 0, 1, 0]),
        (qml.QubitStateVector, [0, 0, 1, 0], [0, 0, 1, 0]),
        (qml.QubitStateVector, [0, 0, 0, 1], [0, 0, 0, 1]),
        (qml.QubitStateVector, [1/math.sqrt(3), 0, 1/math.sqrt(3), 1/math.sqrt(3)], [1/math.sqrt(3), 0, 1/math.sqrt(3), 1/math.sqrt(3)]),
        (qml.QubitStateVector, [1/math.sqrt(3), 0, -1/math.sqrt(3), 1/math.sqrt(3)], [1/math.sqrt(3), 0, -1/math.sqrt(3), 1/math.sqrt(3)]),
    ])
    def test_apply_operation_state_preparation(self, qubit_device_2_wires, tol, operation, expected_output, par):
        """Tests that applying an operation yields the expected output state for single wire
           operations that have no parameters."""

        par = np.array(par)
        qubit_device_2_wires.reset()
        qubit_device_2_wires.apply([operation(par, wires=[0, 1])])

        assert np.allclose(qubit_device_2_wires._state.flatten(), np.array(expected_output), atol=tol, rtol=0)

    test_data_single_wire_with_parameters = [
        (qml.PhaseShift, [1, 0], [1, 0], [math.pi / 2]),
        (qml.PhaseShift, [0, 1], [0, 1j], [math.pi / 2]),
        (qml.PhaseShift, [1 / math.sqrt(2), 1 / math.sqrt(2)], [1 / math.sqrt(2), 1 / 2 + 1j / 2], [math.pi / 4]),
        (qml.RX, [1, 0], [1 / math.sqrt(2), -1j * 1 / math.sqrt(2)], [math.pi / 2]),
        (qml.RX, [1, 0], [0, -1j], [math.pi]),
        (qml.RX, [1 / math.sqrt(2), 1 / math.sqrt(2)], [1 / 2 - 1j / 2, 1 / 2 - 1j / 2], [math.pi / 2]),
        (qml.RY, [1, 0], [1 / math.sqrt(2), 1 / math.sqrt(2)], [math.pi / 2]),
        (qml.RY, [1, 0], [0, 1], [math.pi]),
        (qml.RY, [1 / math.sqrt(2), 1 / math.sqrt(2)], [0, 1], [math.pi / 2]),
        (qml.RZ, [1, 0], [1 / math.sqrt(2) - 1j / math.sqrt(2), 0], [math.pi / 2]),
        (qml.RZ, [0, 1], [0, 1j], [math.pi]),
        (qml.RZ, [1 / math.sqrt(2), 1 / math.sqrt(2)], [1 / 2 - 1j / 2, 1 / 2 + 1j / 2], [math.pi / 2]),
        (qml.MultiRZ, [1, 0], [1 / math.sqrt(2) - 1j / math.sqrt(2), 0], [math.pi / 2]),
        (qml.MultiRZ, [0, 1], [0, 1j], [math.pi]),
        (qml.MultiRZ, [1 / math.sqrt(2), 1 / math.sqrt(2)], [1 / 2 - 1j / 2, 1 / 2 + 1j / 2], [math.pi / 2]),
        (qml.Rot, [1, 0], [1 / math.sqrt(2) - 1j / math.sqrt(2), 0], [math.pi / 2, 0, 0]),
        (qml.Rot, [1, 0], [1 / math.sqrt(2), 1 / math.sqrt(2)], [0, math.pi / 2, 0]),
        (qml.Rot, [1 / math.sqrt(2), 1 / math.sqrt(2)], [1 / 2 - 1j / 2, 1 / 2 + 1j / 2], [0, 0, math.pi / 2]),
        (qml.Rot, [1, 0], [-1j / math.sqrt(2), -1 / math.sqrt(2)], [math.pi / 2, -math.pi / 2, math.pi / 2]),
        (qml.Rot, [1 / math.sqrt(2), 1 / math.sqrt(2)], [1 / 2 + 1j / 2, -1 / 2 + 1j / 2],
         [-math.pi / 2, math.pi, math.pi]),
        (qml.QubitUnitary, [1, 0], [1j / math.sqrt(2), 1j / math.sqrt(2)],
         [np.array([[1j / math.sqrt(2), 1j / math.sqrt(2)], [1j / math.sqrt(2), -1j / math.sqrt(2)]])]),
        (qml.QubitUnitary, [0, 1], [1j / math.sqrt(2), -1j / math.sqrt(2)],
         [np.array([[1j / math.sqrt(2), 1j / math.sqrt(2)], [1j / math.sqrt(2), -1j / math.sqrt(2)]])]),
        (qml.QubitUnitary, [1 / math.sqrt(2), -1 / math.sqrt(2)], [0, 1j],
         [np.array([[1j / math.sqrt(2), 1j / math.sqrt(2)], [1j / math.sqrt(2), -1j / math.sqrt(2)]])]),
        (qml.DiagonalQubitUnitary, [1, 0], [-1, 0], [np.array([-1, 1])]),
        (qml.DiagonalQubitUnitary, [1 / math.sqrt(2), 1 / math.sqrt(2)], [1 / math.sqrt(2), 1j / math.sqrt(2)], [np.array([1, 1j])]),
        (qml.DiagonalQubitUnitary, [1 / 2, math.sqrt(3) / 4], [cmath.exp(1j * 0.4) / 2, cmath.exp(1j * -0.4) * math.sqrt(3) / 4], [np.array([cmath.exp(1j * 0.4), cmath.exp(1j * -0.4)])]),
    ]

    test_data_single_wire_with_parameters_inverses = [
        (qml.PhaseShift, [1, 0], [1, 0], [math.pi / 2]),
        (qml.PhaseShift, [0, 1], [0, -1j], [math.pi / 2]),
        (qml.PhaseShift, [1 / math.sqrt(2), 1 / math.sqrt(2)],
         [1 / math.sqrt(2), 1 / 2 - 1j / 2], [math.pi / 4]),
        (qml.RX, [1, 0], [1 / math.sqrt(2), 1j * 1 / math.sqrt(2)], [math.pi / 2]),
        (qml.RX, [1, 0], [0, 1j], [math.pi]),
        (qml.RX, [1 / math.sqrt(2), 1 / math.sqrt(2)], [1 / 2 + 1j / 2, 1 / 2 + 1j / 2], [math.pi / 2]),
        (qml.RY, [1, 0], [1 / math.sqrt(2), -1 / math.sqrt(2)], [math.pi / 2]),
        (qml.RY, [1, 0], [0, -1], [math.pi]),
        (qml.RY, [1 / math.sqrt(2), 1 / math.sqrt(2)], [1, 0], [math.pi / 2]),
        (qml.RZ, [1, 0], [1 / math.sqrt(2) + 1j / math.sqrt(2), 0], [math.pi / 2]),
        (qml.RZ, [0, 1], [0, -1j], [math.pi]),
        (qml.RZ, [1 / math.sqrt(2), 1 / math.sqrt(2)],
         [1 / 2 + 1/2*1j, 1 / 2 - 1/2*1j], [math.pi / 2]),
        (qml.MultiRZ, [1, 0], [1 / math.sqrt(2) + 1j / math.sqrt(2), 0], [math.pi / 2]),
        (qml.MultiRZ, [0, 1], [0, -1j], [math.pi]),
        (qml.MultiRZ, [1 / math.sqrt(2), 1 / math.sqrt(2)],
         [1 / 2 + 1/2*1j, 1 / 2 - 1/2*1j], [math.pi / 2]),
        (qml.DiagonalQubitUnitary, [1, 0], [-1, 0], [np.array([-1, 1])]),
        (qml.DiagonalQubitUnitary, [1 / math.sqrt(2), 1 / math.sqrt(2)], [1 / math.sqrt(2), -1j / math.sqrt(2)], [np.array([1, 1j])]),
        (qml.DiagonalQubitUnitary, [1 / 2, math.sqrt(3) / 4], [cmath.exp(-1j * 0.4) / 2, cmath.exp(1j * 0.4) * math.sqrt(3) / 4], [np.array([cmath.exp(1j * 0.4), cmath.exp(1j * -0.4)])]),
    ]

    @pytest.mark.parametrize("operation,input,expected_output,par", test_data_single_wire_with_parameters)
    def test_apply_operation_single_wire_with_parameters(self, qubit_device_1_wire, tol, operation, input, expected_output, par):
        """Tests that applying an operation yields the expected output state for single wire
           operations that have parameters."""

        #parameter = par[0]
        qubit_device_1_wire._state = np.array(input)

        qubit_device_1_wire.apply([operation(*par, wires=[0])])

        assert np.allclose(qubit_device_1_wire._state, np.array(expected_output), atol=tol, rtol=0)

    @pytest.mark.parametrize("operation,input,expected_output,par", test_data_single_wire_with_parameters_inverses)
    def test_apply_operation_single_wire_with_parameters_inverse(self, qubit_device_1_wire, tol, operation, input, expected_output, par):
        """Tests that applying the inverse of an operation yields the expected output state for single wire
           operations that have parameters."""

        qubit_device_1_wire._state = np.array(input)
        qubit_device_1_wire.apply([operation(*par, wires=[0]).inv()])

        assert np.allclose(qubit_device_1_wire._state, np.array(expected_output), atol=tol, rtol=0)

    test_data_two_wires_with_parameters = [
        (qml.CRX, [0, 1, 0, 0], [0, 1, 0, 0], [math.pi / 2]),
        (qml.CRX, [0, 0, 0, 1], [0, 0, -1j, 0], [math.pi]),
        (qml.CRX, [0, 1 / math.sqrt(2), 1 / math.sqrt(2), 0], [0, 1 / math.sqrt(2), 1 / 2, -1j / 2], [math.pi / 2]),
        (qml.CRY, [0, 0, 0, 1], [0, 0, -1 / math.sqrt(2), 1 / math.sqrt(2)], [math.pi / 2]),
        (qml.CRY, [0, 0, 0, 1], [0, 0, -1, 0], [math.pi]),
        (qml.CRY, [1 / math.sqrt(2), 1 / math.sqrt(2), 0, 0], [1 / math.sqrt(2), 1 / math.sqrt(2), 0, 0], [math.pi / 2]),
        (qml.CRZ, [0, 0, 0, 1], [0, 0, 0, 1 / math.sqrt(2) + 1j / math.sqrt(2)], [math.pi / 2]),
        (qml.CRZ, [0, 0, 0, 1], [0, 0, 0, 1j], [math.pi]),
        (qml.CRZ, [1 / math.sqrt(2), 1 / math.sqrt(2), 0, 0], [1 / math.sqrt(2), 1 / math.sqrt(2), 0, 0], [math.pi / 2]),
        (qml.MultiRZ, [0, 0, 0, 1], [0, 0, 0, 1 / math.sqrt(2) - 1j / math.sqrt(2)], [math.pi / 2]),
        (qml.MultiRZ, [0, 0, 1, 0], [0, 0, 1j, 0], [math.pi]),
        (qml.MultiRZ, [1 / math.sqrt(2), 1 / math.sqrt(2), 0, 0], [1 / 2 - 1j / 2, 1 / 2 + 1j / 2, 0, 0], [math.pi / 2]),
        (qml.CRot, [0, 0, 0, 1], [0, 0, 0, 1 / math.sqrt(2) + 1j / math.sqrt(2)], [math.pi / 2, 0, 0]),
        (qml.CRot, [0, 0, 0, 1], [0, 0, -1 / math.sqrt(2), 1 / math.sqrt(2)], [0, math.pi / 2, 0]),
        (qml.CRot, [0, 0, 1 / math.sqrt(2), 1 / math.sqrt(2)], [0, 0, 1 / 2 - 1j / 2, 1 / 2 + 1j / 2],
         [0, 0, math.pi / 2]),
        (qml.CRot, [0, 0, 0, 1], [0, 0, 1 / math.sqrt(2), 1j / math.sqrt(2)], [math.pi / 2, -math.pi / 2, math.pi / 2]),
        (qml.CRot, [0, 1 / math.sqrt(2), 1 / math.sqrt(2), 0], [0, 1 / math.sqrt(2), 0, -1 / 2 + 1j / 2],
         [-math.pi / 2, math.pi, math.pi]),
        (qml.QubitUnitary, [1, 0, 0, 0], [1, 0, 0, 0], [np.array(
            [[1, 0, 0, 0], [0, 1 / math.sqrt(2), 1 / math.sqrt(2), 0], [0, 1 / math.sqrt(2), -1 / math.sqrt(2), 0],
             [0, 0, 0, 1]])]),
        (qml.QubitUnitary, [0, 1, 0, 0], [0, 1 / math.sqrt(2), 1 / math.sqrt(2), 0], [np.array(
            [[1, 0, 0, 0], [0, 1 / math.sqrt(2), 1 / math.sqrt(2), 0], [0, 1 / math.sqrt(2), -1 / math.sqrt(2), 0],
             [0, 0, 0, 1]])]),
        (qml.QubitUnitary, [1 / 2, 1 / 2, -1 / 2, 1 / 2], [1 / 2, 0, 1 / math.sqrt(2), 1 / 2], [np.array(
            [[1, 0, 0, 0], [0, 1 / math.sqrt(2), 1 / math.sqrt(2), 0], [0, 1 / math.sqrt(2), -1 / math.sqrt(2), 0],
             [0, 0, 0, 1]])]),
        (qml.DiagonalQubitUnitary, [1, 0, 0, 0], [-1, 0, 0, 0], [np.array([-1, 1, 1, -1])]),
        (qml.DiagonalQubitUnitary, [1/math.sqrt(2), 0, 0, 1/math.sqrt(2)], [1/math.sqrt(2), 0, 0, -1/math.sqrt(2)], [np.array([1, 1, 1, -1])]),
        (qml.DiagonalQubitUnitary, [0, 0, 1, 0], [0, 0, 1j, 0], [np.array([-1, 1j, 1j, -1])]),
    ]

    test_data_two_wires_with_parameters_inverses = [
        (qml.CRX, [0, 1, 0, 0], [0, 1, 0, 0], [math.pi / 2]),
        (qml.CRX, [0, 0, 0, 1], [0, 0, 1j, 0], [math.pi]),
        (qml.CRX, [0, 1 / math.sqrt(2), 1 / math.sqrt(2), 0],
         [0, 1 / math.sqrt(2), 1 / 2, 1j / 2], [math.pi / 2]),
        (qml.MultiRZ, [0, 0, 0, 1], [0, 0, 0, 1 / math.sqrt(2) + 1j / math.sqrt(2)], [math.pi / 2]),
        (qml.MultiRZ, [0, 0, 1, 0], [0, 0, -1j, 0], [math.pi]),
        (qml.MultiRZ, [1 / math.sqrt(2), 1 / math.sqrt(2), 0, 0], [1 / 2 + 1j / 2, 1 / 2 - 1j / 2, 0, 0], [math.pi / 2]),
        (qml.DiagonalQubitUnitary, [1, 0, 0, 0], [-1, 0, 0, 0], [np.array([-1, 1, 1, -1])]),
        (qml.DiagonalQubitUnitary, [1/math.sqrt(2), 0, 0, 1/math.sqrt(2)], [1/math.sqrt(2), 0, 0, -1/math.sqrt(2)], [np.array([1, 1, 1, -1])]),
        (qml.DiagonalQubitUnitary, [0, 0, 1, 0], [0, 0, -1j, 0], [np.array([-1, 1j, 1j, -1])]),
    ]

    @pytest.mark.parametrize("operation,input,expected_output,par", test_data_two_wires_with_parameters)
    def test_apply_operation_two_wires_with_parameters(self, qubit_device_2_wires, tol, operation, input, expected_output, par):
        """Tests that applying an operation yields the expected output state for two wire
           operations that have parameters."""

        qubit_device_2_wires._state = np.array(input).reshape((2, 2))
        qubit_device_2_wires.apply([operation(*par, wires=[0, 1])])

        assert np.allclose(qubit_device_2_wires._state.flatten(), np.array(expected_output), atol=tol, rtol=0)

    @pytest.mark.parametrize("operation,input,expected_output,par", test_data_two_wires_with_parameters_inverses)
    def test_apply_operation_two_wires_with_parameters_inverse(self, qubit_device_2_wires, tol, operation, input, expected_output, par):
        """Tests that applying the inverse of an operation yields the expected output state for two wire
           operations that have parameters."""

        qubit_device_2_wires._state = np.array(input).reshape((2, 2))
        qubit_device_2_wires.apply([operation(*par, wires=[0, 1]).inv()])

        assert np.allclose(qubit_device_2_wires._state.flatten(), np.array(expected_output), atol=tol, rtol=0)

    def test_apply_errors_qubit_state_vector(self, qubit_device_2_wires):
        """Test that apply fails for incorrect state preparation, and > 2 qubit gates"""
        with pytest.raises(
            ValueError,
            match="Sum of amplitudes-squared does not equal one."
        ):
            qubit_device_2_wires.apply([qml.QubitStateVector(np.array([1, -1]), wires=[0])])

        with pytest.raises(
            ValueError,
            match=r"State vector must be of length 2\*\*wires."
        ):
            p = np.array([1, 0, 1, 1, 0]) / np.sqrt(3)
            qubit_device_2_wires.apply([qml.QubitStateVector(p, wires=[0, 1])])

        with pytest.raises(
            DeviceError,
            match="Operation QubitStateVector cannot be used after other Operations have already been applied "
                                  "on a default.qubit device."
        ):
            qubit_device_2_wires.reset()
            qubit_device_2_wires.apply([
                qml.RZ(0.5, wires=[0]),
                qml.QubitStateVector(np.array([0, 1, 0, 0]), wires=[0, 1])
            ])

    def test_apply_errors_basis_state(self, qubit_device_2_wires):
        with pytest.raises(
            ValueError,
            match="BasisState parameter must consist of 0 or 1 integers."
        ):
            qubit_device_2_wires.apply([qml.BasisState(np.array([-0.2, 4.2]), wires=[0, 1])])

        with pytest.raises(
            ValueError,
            match="BasisState parameter and wires must be of equal length."
        ):
            qubit_device_2_wires.apply([qml.BasisState(np.array([0, 1]), wires=[0])])

        with pytest.raises(
            DeviceError,
            match="Operation BasisState cannot be used after other Operations have already been applied "
                                  "on a default.qubit device."
        ):
            qubit_device_2_wires.reset()
            qubit_device_2_wires.apply([
                qml.RZ(0.5, wires=[0]),
                qml.BasisState(np.array([1, 1]), wires=[0, 1])
            ])

class TestExpval:
    """Tests that expectation values are properly calculated or that the proper errors are raised."""

    @pytest.mark.parametrize("operation,input,expected_output", [
        (qml.PauliX, [1/math.sqrt(2), 1/math.sqrt(2)], 1),
        (qml.PauliX, [1/math.sqrt(2), -1/math.sqrt(2)], -1),
        (qml.PauliX, [1, 0], 0),
        (qml.PauliY, [1/math.sqrt(2), 1j/math.sqrt(2)], 1),
        (qml.PauliY, [1/math.sqrt(2), -1j/math.sqrt(2)], -1),
        (qml.PauliY, [1, 0], 0),
        (qml.PauliZ, [1, 0], 1),
        (qml.PauliZ, [0, 1], -1),
        (qml.PauliZ, [1/math.sqrt(2), 1/math.sqrt(2)], 0),
        (qml.Hadamard, [1, 0], 1/math.sqrt(2)),
        (qml.Hadamard, [0, 1], -1/math.sqrt(2)),
        (qml.Hadamard, [1/math.sqrt(2), 1/math.sqrt(2)], 1/math.sqrt(2)),
        (qml.Identity, [1, 0], 1),
        (qml.Identity, [0, 1], 1),
        (qml.Identity, [1/math.sqrt(2), -1/math.sqrt(2)], 1),
    ])
    def test_expval_single_wire_no_parameters(self, qubit_device_1_wire, tol, operation, input, expected_output):
        """Tests that expectation values are properly calculated for single-wire observables without parameters."""

        obs = operation(wires=[0])

        qubit_device_1_wire.reset()
        qubit_device_1_wire.apply(
            [qml.QubitStateVector(np.array(input), wires=[0])],
            obs.diagonalizing_gates()
        )
        res = qubit_device_1_wire.expval(obs)

        assert np.isclose(res, expected_output, atol=tol, rtol=0)

    @pytest.mark.parametrize("operation,input,expected_output,par", [
        (qml.Hermitian, [1, 0], 1, [[1, 1j], [-1j, 1]]),
        (qml.Hermitian, [0, 1], 1, [[1, 1j], [-1j, 1]]),
        (qml.Hermitian, [1/math.sqrt(2), -1/math.sqrt(2)], 1, [[1, 1j], [-1j, 1]]),
    ])
    def test_expval_single_wire_with_parameters(self, qubit_device_1_wire, tol, operation, input, expected_output, par):
        """Tests that expectation values are properly calculated for single-wire observables with parameters."""

        obs = operation(np.array(par), wires=[0])

        qubit_device_1_wire.reset()
        qubit_device_1_wire.apply(
            [qml.QubitStateVector(np.array(input), wires=[0])],
            obs.diagonalizing_gates()
        )
        res = qubit_device_1_wire.expval(obs)

        assert np.isclose(res, expected_output, atol=tol, rtol=0)

    @pytest.mark.parametrize("operation,input,expected_output,par", [
        (qml.Hermitian, [1/math.sqrt(3), 0, 1/math.sqrt(3), 1/math.sqrt(3)], 5/3, [[1, 1j, 0, 1], [-1j, 1, 0, 0], [0, 0, 1, -1j], [1, 0, 1j, 1]]),
        (qml.Hermitian, [0, 0, 0, 1], 0, [[0, 1j, 0, 0], [-1j, 0, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]]),
        (qml.Hermitian, [1/math.sqrt(2), 0, -1/math.sqrt(2), 0], 1, [[1, 1j, 0, 0], [-1j, 1, 0, 0], [0, 0, 1, -1j], [0, 0, 1j, 1]]),
        (qml.Hermitian, [1/math.sqrt(3), -1/math.sqrt(3), 1/math.sqrt(6), 1/math.sqrt(6)], 1, [[1, 1j, 0, .5j], [-1j, 1, 0, 0], [0, 0, 1, -1j], [-.5j, 0, 1j, 1]]),
        (qml.Hermitian, [1/math.sqrt(2), 0, 0, 1/math.sqrt(2)], 1, [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]),
        (qml.Hermitian, [0, 1/math.sqrt(2), -1/math.sqrt(2), 0], -1, [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]),
    ])
    def test_expval_two_wires_with_parameters(self, qubit_device_2_wires, tol, operation, input, expected_output, par):
        """Tests that expectation values are properly calculated for two-wire observables with parameters."""

        obs = operation(np.array(par), wires=[0, 1])

        qubit_device_2_wires.reset()
        qubit_device_2_wires.apply(
            [qml.QubitStateVector(np.array(input), wires=[0, 1])],
            obs.diagonalizing_gates()
        )
        res = qubit_device_2_wires.expval(obs)

        assert np.isclose(res, expected_output, atol=tol, rtol=0)

    def test_expval_estimate(self):
        """Test that the expectation value is not analytically calculated"""

        dev = qml.device("default.qubit", wires=1, shots=3, analytic=False)

        @qml.qnode(dev)
        def circuit():
            return qml.expval(qml.PauliX(0))

        expval = circuit()

        # With 3 samples we are guaranteed to see a difference between
        # an estimated variance an an analytically calculated one
        assert expval != 0.0

class TestVar:
    """Tests that variances are properly calculated."""

    @pytest.mark.parametrize("operation,input,expected_output", [
        (qml.PauliX, [1/math.sqrt(2), 1/math.sqrt(2)], 0),
        (qml.PauliX, [1/math.sqrt(2), -1/math.sqrt(2)], 0),
        (qml.PauliX, [1, 0], 1),
        (qml.PauliY, [1/math.sqrt(2), 1j/math.sqrt(2)], 0),
        (qml.PauliY, [1/math.sqrt(2), -1j/math.sqrt(2)], 0),
        (qml.PauliY, [1, 0], 1),
        (qml.PauliZ, [1, 0], 0),
        (qml.PauliZ, [0, 1], 0),
        (qml.PauliZ, [1/math.sqrt(2), 1/math.sqrt(2)], 1),
        (qml.Hadamard, [1, 0], 1/2),
        (qml.Hadamard, [0, 1], 1/2),
        (qml.Hadamard, [1/math.sqrt(2), 1/math.sqrt(2)], 1/2),
        (qml.Identity, [1, 0], 0),
        (qml.Identity, [0, 1], 0),
        (qml.Identity, [1/math.sqrt(2), -1/math.sqrt(2)], 0),

    ])
    def test_var_single_wire_no_parameters(self, qubit_device_1_wire, tol, operation, input, expected_output):
        """Tests that variances are properly calculated for single-wire observables without parameters."""

        obs = operation(wires=[0])

        qubit_device_1_wire.reset()
        qubit_device_1_wire.apply(
            [qml.QubitStateVector(np.array(input), wires=[0])],
            obs.diagonalizing_gates()
        )
        res = qubit_device_1_wire.var(obs)

        assert np.isclose(res, expected_output, atol=tol, rtol=0)

    @pytest.mark.parametrize("operation,input,expected_output,par", [
        (qml.Hermitian, [1, 0], 1, [[1, 1j], [-1j, 1]]),
        (qml.Hermitian, [0, 1], 1, [[1, 1j], [-1j, 1]]),
        (qml.Hermitian, [1/math.sqrt(2), -1/math.sqrt(2)], 1, [[1, 1j], [-1j, 1]]),
    ])
    def test_var_single_wire_with_parameters(self, qubit_device_1_wire, tol, operation, input, expected_output, par):
        """Tests that variances are properly calculated for single-wire observables with parameters."""

        obs = operation(np.array(par), wires=[0])

        qubit_device_1_wire.reset()
        qubit_device_1_wire.apply(
            [qml.QubitStateVector(np.array(input), wires=[0])],
            obs.diagonalizing_gates()
        )
        res = qubit_device_1_wire.var(obs)

        assert np.isclose(res, expected_output, atol=tol, rtol=0)

    @pytest.mark.parametrize("operation,input,expected_output,par", [
        (qml.Hermitian, [1/math.sqrt(3), 0, 1/math.sqrt(3), 1/math.sqrt(3)], 11/9, [[1, 1j, 0, 1], [-1j, 1, 0, 0], [0, 0, 1, -1j], [1, 0, 1j, 1]]),
        (qml.Hermitian, [0, 0, 0, 1], 1, [[0, 1j, 0, 0], [-1j, 0, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]]),
        (qml.Hermitian, [1/math.sqrt(2), 0, -1/math.sqrt(2), 0], 1, [[1, 1j, 0, 0], [-1j, 1, 0, 0], [0, 0, 1, -1j], [0, 0, 1j, 1]]),
        (qml.Hermitian, [1/math.sqrt(2), 0, 0, 1/math.sqrt(2)], 0, [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]),
        (qml.Hermitian, [0, 1/math.sqrt(2), -1/math.sqrt(2), 0], 0, [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]),
    ])
    def test_var_two_wires_with_parameters(self, qubit_device_2_wires, tol, operation, input, expected_output, par):
        """Tests that variances are properly calculated for two-wire observables with parameters."""

        obs = operation(np.array(par), wires=[0, 1])

        qubit_device_2_wires.reset()
        qubit_device_2_wires.apply(
            [qml.QubitStateVector(np.array(input), wires=[0, 1])],
            obs.diagonalizing_gates()
        )
        res = qubit_device_2_wires.var(obs)

        assert np.isclose(res, expected_output, atol=tol, rtol=0)

    def test_var_estimate(self):
        """Test that the variance is not analytically calculated"""

        dev = qml.device("default.qubit", wires=1, shots=3, analytic=False)

        @qml.qnode(dev)
        def circuit():
            return qml.var(qml.PauliX(0))

        var = circuit()

        # With 3 samples we are guaranteed to see a difference between
        # an estimated variance and an analytically calculated one
        assert var != 1.0

class TestSample:
    """Tests that samples are properly calculated."""

    def test_sample_dimensions(self, qubit_device_2_wires):
        """Tests if the samples returned by the sample function have
        the correct dimensions
        """

        # Explicitly resetting is necessary as the internal
        # state is set to None in __init__ and only properly
        # initialized during reset
        qubit_device_2_wires.reset()

        qubit_device_2_wires.apply(
            [qml.RX(1.5708, wires=[0]), qml.RX(1.5708, wires=[1])]
        )

        qubit_device_2_wires.shots = 10
        qubit_device_2_wires._wires_measured = {0}
        qubit_device_2_wires._samples = qubit_device_2_wires.generate_samples()
        s1 = qubit_device_2_wires.sample(qml.PauliZ(wires=[0]))
        assert np.array_equal(s1.shape, (10,))

        qubit_device_2_wires.reset()
        qubit_device_2_wires.shots = 12
        qubit_device_2_wires._wires_measured = {1}
        qubit_device_2_wires._samples = qubit_device_2_wires.generate_samples()
        s2 = qubit_device_2_wires.sample(qml.PauliZ(wires=[1]))
        assert np.array_equal(s2.shape, (12,))

        qubit_device_2_wires.reset()
        qubit_device_2_wires.shots = 17
        qubit_device_2_wires._wires_measured = {0, 1}
        qubit_device_2_wires._samples = qubit_device_2_wires.generate_samples()
        s3 = qubit_device_2_wires.sample(qml.PauliX(0) @ qml.PauliZ(1))
        assert np.array_equal(s3.shape, (17,))

    def test_sample_values(self, qubit_device_2_wires, tol):
        """Tests if the samples returned by sample have
        the correct values
        """

        # Explicitly resetting is necessary as the internal
        # state is set to None in __init__ and only properly
        # initialized during reset
        qubit_device_2_wires.reset()

        qubit_device_2_wires.apply([qml.RX(1.5708, wires=[0])])
        qubit_device_2_wires._wires_measured = {0}
        qubit_device_2_wires._samples = qubit_device_2_wires.generate_samples()

        s1 = qubit_device_2_wires.sample(qml.PauliZ(0))

        # s1 should only contain 1 and -1, which is guaranteed if
        # they square to 1
        assert np.allclose(s1**2, 1, atol=tol, rtol=0)

class TestDefaultQubitIntegration:
    """Integration tests for default.qubit. This test ensures it integrates
    properly with the PennyLane interface, in particular QNode."""

    def test_load_default_qubit_device(self):
        """Test that the default plugin loads correctly"""

        dev = qml.device("default.qubit", wires=2)
        assert dev.num_wires == 2
        assert dev.shots == 1000
        assert dev.analytic
        assert dev.short_name == "default.qubit"

    def test_args(self):
        """Test that the plugin requires correct arguments"""

        with pytest.raises(
            TypeError, match="missing 1 required positional argument: 'wires'"
        ):
            qml.device("default.qubit")

    def test_qubit_circuit(self, qubit_device_1_wire, tol):
        """Test that the default qubit plugin provides correct result for a simple circuit"""

        p = 0.543

        @qml.qnode(qubit_device_1_wire)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliY(0))

        expected = -np.sin(p)

        assert np.isclose(circuit(p), expected, atol=tol, rtol=0)

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

        shots = 10 ** 5
        dev = qml.device("default.qubit", wires=1)

        p = 0.543

        @qml.qnode(dev)
        def circuit(x):
            """Test quantum function"""
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliY(0))

        runs = []
        for _ in range(100):
            runs.append(circuit(p))

        assert np.isclose(np.mean(runs), -np.sin(p), atol=tol, rtol=0)

    @pytest.mark.parametrize("name,expected_output", [
        ("PauliX", 1),
        ("PauliY", 1),
        ("S", -1),
    ])
    def test_inverse_circuit(self, qubit_device_1_wire, tol, name, expected_output):
        """Tests the inverse of supported gates that act on a single wire and are not parameterized"""

        op = getattr(qml.ops, name)

        @qml.qnode(qubit_device_1_wire)
        def circuit():
            qml.BasisState(np.array([1]), wires=[0])
            op(wires=0).inv()
            return qml.expval(qml.PauliZ(0))

        assert np.isclose(circuit(), expected_output, atol=tol, rtol=0)

    @pytest.mark.parametrize("name,expected_output", [
        ("PauliX", 1),
        ("PauliY", 1),
        ("S", -1),
    ])
    def test_inverse_circuit_calling_inv_multiple_times(self, qubit_device_1_wire, tol, name, expected_output):
        """Tests that multiple calls to the inverse of an operation works"""

        op = getattr(qml.ops, name)

        @qml.qnode(qubit_device_1_wire)
        def circuit():
            qml.BasisState(np.array([1]), wires=[0])
            op(wires=0).inv().inv().inv()
            return qml.expval(qml.PauliZ(0))

        assert np.isclose(circuit(), expected_output, atol=tol, rtol=0)

    @pytest.mark.parametrize("name,expected_output,phi", [("RX", 1,
                                                           multiplier * 0.5432) for multiplier in range(8)
                                                          ])
    def test_inverse_circuit_with_parameters(self, qubit_device_1_wire, tol, name, expected_output, phi):
        """Tests the inverse of supported gates that act on a single wire and are parameterized"""

        @qml.qnode(qubit_device_1_wire)
        def circuit():
            qml.RX(phi, wires=0)
            qml.RX(phi, wires=0).inv()
            return qml.expval(qml.PauliZ(0))

        assert np.isclose(circuit(), expected_output, atol=tol, rtol=0)



    @pytest.mark.parametrize("name,expected_output,phi", [("RX", 1,
                                                           multiplier * 0.5432) for multiplier in range(8)
                                                          ])
    def test_inverse_circuit_with_parameters_expectation(self, qubit_device_1_wire, tol, name, expected_output, phi):
        @qml.qnode(qubit_device_1_wire)
        def circuit():
            qml.RX(phi, wires=0)
            qml.RX(phi, wires=0).inv()
            return qml.expval(qml.PauliZ(0).inv())

        assert np.isclose(circuit(), expected_output, atol=tol, rtol=0)

    # This test is ran against the state |0> with one Z expval
    @pytest.mark.parametrize("name,expected_output", [
        ("PauliX", -1),
        ("PauliY", -1),
        ("PauliZ", 1),
        ("Hadamard", 0),
    ])
    def test_supported_gate_single_wire_no_parameters(self, qubit_device_1_wire, tol, name, expected_output):
        """Tests supported gates that act on a single wire that are not parameterized"""

        op = getattr(qml.ops, name)

        assert qubit_device_1_wire.supports_operation(name)

        @qml.qnode(qubit_device_1_wire)
        def circuit():
            op(wires=0)
            return qml.expval(qml.PauliZ(0))

        assert np.isclose(circuit(), expected_output, atol=tol, rtol=0)

    # This test is ran against the state |Phi+> with two Z expvals
    @pytest.mark.parametrize("name,expected_output", [
        ("CNOT", [-1/2, 1]),
        ("SWAP", [-1/2, -1/2]),
        ("CZ", [-1/2, -1/2]),
    ])
    def test_supported_gate_two_wires_no_parameters(self, qubit_device_2_wires, tol, name, expected_output):
        """Tests supported gates that act on two wires that are not parameterized"""

        op = getattr(qml.ops, name)

        assert qubit_device_2_wires.supports_operation(name)

        @qml.qnode(qubit_device_2_wires)
        def circuit():
            qml.QubitStateVector(np.array([1/2, 0, 0, math.sqrt(3)/2]), wires=[0, 1])
            op(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        assert np.allclose(circuit(), expected_output, atol=tol, rtol=0)

    @pytest.mark.parametrize("name,expected_output", [
        ("CSWAP", [-1, -1, 1]),
    ])
    def test_supported_gate_three_wires_no_parameters(self, qubit_device_3_wires, tol, name, expected_output):
        """Tests supported gates that act on three wires that are not parameterized"""

        op = getattr(qml.ops, name)

        assert qubit_device_3_wires.supports_operation(name)

        @qml.qnode(qubit_device_3_wires)
        def circuit():
            qml.BasisState(np.array([1, 0, 1]), wires=[0, 1, 2])
            op(wires=[0, 1, 2])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2))

        assert np.allclose(circuit(), expected_output, atol=tol, rtol=0)

    # This test is ran with two Z expvals
    @pytest.mark.parametrize("name,par,expected_output", [
        ("BasisState", [0, 0], [1, 1]),
        ("BasisState", [1, 0], [-1, 1]),
        ("BasisState", [0, 1], [1, -1]),
        ("QubitStateVector", [1, 0, 0, 0], [1, 1]),
        ("QubitStateVector", [0, 0, 1, 0], [-1, 1]),
        ("QubitStateVector", [0, 1, 0, 0], [1, -1]),
    ])
    def test_supported_state_preparation(self, qubit_device_2_wires, tol, name, par, expected_output):
        """Tests supported state preparations"""

        op = getattr(qml.ops, name)

        assert qubit_device_2_wires.supports_operation(name)

        @qml.qnode(qubit_device_2_wires)
        def circuit():
            op(np.array(par), wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        assert np.allclose(circuit(), expected_output, atol=tol, rtol=0)

    # This test is ran with two Z expvals
    @pytest.mark.parametrize("name,par,wires,expected_output", [
        ("BasisState", [1, 1], [0, 1], [-1, -1]),
        ("BasisState", [1], [0], [-1, 1]),
        ("BasisState", [1], [1], [1, -1])
    ])
    def test_basis_state_2_qubit_subset(self, qubit_device_2_wires, tol, name, par, wires, expected_output):
        """Tests qubit basis state preparation on subsets of qubits"""

        op = getattr(qml.ops, name)

        @qml.qnode(qubit_device_2_wires)
        def circuit():
            op(np.array(par), wires=wires)
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        assert np.allclose(circuit(), expected_output, atol=tol, rtol=0)

    # This test is run with two expvals
    @pytest.mark.parametrize("name,par,wires,expected_output", [
        ("QubitStateVector", [0, 1], [1], [1, -1]),
        ("QubitStateVector", [0, 1], [0], [-1, 1]),
        ("QubitStateVector", [1./np.sqrt(2), 1./np.sqrt(2)], [1], [1, 0]),
        ("QubitStateVector", [1j/2., np.sqrt(3)/2.], [1], [1, -0.5]),
        ("QubitStateVector", [(2-1j)/3., 2j/3.], [0], [1/9., 1])
    ])
    def test_state_vector_2_qubit_subset(self, qubit_device_2_wires, tol, name, par, wires, expected_output):
        """Tests qubit state vector preparation on subsets of 2 qubits"""

        op = getattr(qml.ops, name)

        par = np.array(par)

        @qml.qnode(qubit_device_2_wires)
        def circuit():
            op(par, wires=wires)
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        assert np.allclose(circuit(), expected_output, atol=tol, rtol=0)

    # This test is run with three expvals
    @pytest.mark.parametrize("name,par,wires,expected_output", [
        ("QubitStateVector", [1j/np.sqrt(10), (1-2j)/np.sqrt(10), 0, 0, 0, 2/np.sqrt(10), 0, 0],
         [0, 1, 2], [1/5., 1., -4/5.]),
        ("QubitStateVector", [1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], [0, 2], [0., 1., 0.]),
        ("QubitStateVector", [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)], [0, 1], [0., 0., 1.]),
        ("QubitStateVector", [0, 1, 0, 0, 0, 0, 0, 0], [2, 1, 0], [-1., 1., 1.]),
        ("QubitStateVector", [0, 1j, 0, 0, 0, 0, 0, 0], [0, 2, 1], [1., -1., 1.]),
        ("QubitStateVector", [0, 1/np.sqrt(2), 0, 1/np.sqrt(2)], [1, 0], [-1., 0., 1.]),
        ("QubitStateVector", [0, 1 / np.sqrt(2), 0, 1 / np.sqrt(2)], [0, 1], [0., -1., 1.])
    ])
    def test_state_vector_3_qubit_subset(self, qubit_device_3_wires, tol, name, par, wires, expected_output):
        """Tests qubit state vector preparation on subsets of 3 qubits"""

        op = getattr(qml.ops, name)

        par = np.array(par)

        @qml.qnode(qubit_device_3_wires)
        def circuit():
            op(par, wires=wires)
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2))

        assert np.allclose(circuit(), expected_output, atol=tol, rtol=0)

    # This test is ran on the state |0> with one Z expvals
    @pytest.mark.parametrize("name,par,expected_output", [
        ("PhaseShift", [math.pi/2], 1),
        ("PhaseShift", [-math.pi/4], 1),
        ("RX", [math.pi/2], 0),
        ("RX", [-math.pi/4], 1/math.sqrt(2)),
        ("RY", [math.pi/2], 0),
        ("RY", [-math.pi/4], 1/math.sqrt(2)),
        ("RZ", [math.pi/2], 1),
        ("RZ", [-math.pi/4], 1),
        ("MultiRZ", [math.pi/2], 1),
        ("MultiRZ", [-math.pi/4], 1),
        ("Rot", [math.pi/2, 0, 0], 1),
        ("Rot", [0, math.pi/2, 0], 0),
        ("Rot", [0, 0, math.pi/2], 1),
        ("Rot", [math.pi/2, -math.pi/4, -math.pi/4], 1/math.sqrt(2)),
        ("Rot", [-math.pi/4, math.pi/2, math.pi/4], 0),
        ("Rot", [-math.pi/4, math.pi/4, math.pi/2], 1/math.sqrt(2)),
        ("QubitUnitary", [np.array([[1j/math.sqrt(2), 1j/math.sqrt(2)], [1j/math.sqrt(2), -1j/math.sqrt(2)]])], 0),
        ("QubitUnitary", [np.array([[-1j/math.sqrt(2), 1j/math.sqrt(2)], [1j/math.sqrt(2), 1j/math.sqrt(2)]])], 0),
    ])
    def test_supported_gate_single_wire_with_parameters(self, qubit_device_1_wire, tol, name, par, expected_output):
        """Tests supported gates that act on a single wire that are parameterized"""

        op = getattr(qml.ops, name)

        assert qubit_device_1_wire.supports_operation(name)

        @qml.qnode(qubit_device_1_wire)
        def circuit():
            op(*par, wires=0)
            return qml.expval(qml.PauliZ(0))

        assert np.isclose(circuit(), expected_output, atol=tol, rtol=0)

    # This test is ran against the state 1/2|00>+sqrt(3)/2|11> with two Z expvals
    @pytest.mark.parametrize("name,par,expected_output", [
        ("CRX", [0], [-1/2, -1/2]),
        ("CRX", [-math.pi], [-1/2, 1]),
        ("CRX", [math.pi/2], [-1/2, 1/4]),
        ("CRY", [0], [-1/2, -1/2]),
        ("CRY", [-math.pi], [-1/2, 1]),
        ("CRY", [math.pi/2], [-1/2, 1/4]),
        ("CRZ", [0], [-1/2, -1/2]),
        ("CRZ", [-math.pi], [-1/2, -1/2]),
        ("CRZ", [math.pi/2], [-1/2, -1/2]),
        ("MultiRZ", [0], [-1/2, -1/2]),
        ("MultiRZ", [-math.pi], [-1/2, -1/2]),
        ("MultiRZ", [math.pi/2], [-1/2, -1/2]),
        ("CRot", [math.pi/2, 0, 0], [-1/2, -1/2]),
        ("CRot", [0, math.pi/2, 0], [-1/2, 1/4]),
        ("CRot", [0, 0, math.pi/2], [-1/2, -1/2]),
        ("CRot", [math.pi/2, 0, -math.pi], [-1/2, -1/2]),
        ("CRot", [0, math.pi/2, -math.pi], [-1/2, 1/4]),
        ("CRot", [-math.pi, 0, math.pi/2], [-1/2, -1/2]),
        ("QubitUnitary", [np.array([[1, 0, 0, 0], [0, 1/math.sqrt(2), 1/math.sqrt(2), 0], [0, 1/math.sqrt(2), -1/math.sqrt(2), 0], [0, 0, 0, 1]])], [-1/2, -1/2]),
        ("QubitUnitary", [np.array([[-1, 0, 0, 0], [0, 1/math.sqrt(2), 1/math.sqrt(2), 0], [0, 1/math.sqrt(2), -1/math.sqrt(2), 0], [0, 0, 0, -1]])], [-1/2, -1/2]),
    ])
    def test_supported_gate_two_wires_with_parameters(self, qubit_device_2_wires, tol, name, par, expected_output):
        """Tests supported gates that act on two wires wires that are parameterized"""

        op = getattr(qml.ops, name)

        assert qubit_device_2_wires.supports_operation(name)

        @qml.qnode(qubit_device_2_wires)
        def circuit():
            qml.QubitStateVector(np.array([1/2, 0, 0, math.sqrt(3)/2]), wires=[0, 1])
            op(*par, wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        assert np.allclose(circuit(), expected_output, atol=tol, rtol=0)

    @pytest.mark.parametrize("name,state,expected_output", [
        ("PauliX", [1/math.sqrt(2), 1/math.sqrt(2)], 1),
        ("PauliX", [1/math.sqrt(2), -1/math.sqrt(2)], -1),
        ("PauliX", [1, 0], 0),
        ("PauliY", [1/math.sqrt(2), 1j/math.sqrt(2)], 1),
        ("PauliY", [1/math.sqrt(2), -1j/math.sqrt(2)], -1),
        ("PauliY", [1, 0], 0),
        ("PauliZ", [1, 0], 1),
        ("PauliZ", [0, 1], -1),
        ("PauliZ", [1/math.sqrt(2), 1/math.sqrt(2)], 0),
        ("Hadamard", [1, 0], 1/math.sqrt(2)),
        ("Hadamard", [0, 1], -1/math.sqrt(2)),
        ("Hadamard", [1/math.sqrt(2), 1/math.sqrt(2)], 1/math.sqrt(2)),
    ])
    def test_supported_observable_single_wire_no_parameters(self, qubit_device_1_wire, tol, name, state, expected_output):
        """Tests supported observables on single wires without parameters."""

        obs = getattr(qml.ops, name)

        assert qubit_device_1_wire.supports_observable(name)

        @qml.qnode(qubit_device_1_wire)
        def circuit():
            qml.QubitStateVector(np.array(state), wires=[0])
            return qml.expval(obs(wires=[0]))

        assert np.isclose(circuit(), expected_output, atol=tol, rtol=0)

    @pytest.mark.parametrize("name,state,expected_output,par", [
        ("Identity", [1, 0], 1, []),
        ("Identity", [0, 1], 1, []),
        ("Identity", [1/math.sqrt(2), -1/math.sqrt(2)], 1, []),
        ("Hermitian", [1, 0], 1, [np.array([[1, 1j], [-1j, 1]])]),
        ("Hermitian", [0, 1], 1, [np.array([[1, 1j], [-1j, 1]])]),
        ("Hermitian", [1/math.sqrt(2), -1/math.sqrt(2)], 1, [np.array([[1, 1j], [-1j, 1]])]),
    ])
    def test_supported_observable_single_wire_with_parameters(self, qubit_device_1_wire, tol, name, state, expected_output, par):
        """Tests supported observables on single wires with parameters."""

        obs = getattr(qml.ops, name)

        assert qubit_device_1_wire.supports_observable(name)

        @qml.qnode(qubit_device_1_wire)
        def circuit():
            qml.QubitStateVector(np.array(state), wires=[0])
            return qml.expval(obs(*par, wires=[0]))

        assert np.isclose(circuit(), expected_output, atol=tol, rtol=0)

    @pytest.mark.parametrize("name,state,expected_output,par", [
        ("Hermitian", [1/math.sqrt(3), 0, 1/math.sqrt(3), 1/math.sqrt(3)], 5/3, [np.array([[1, 1j, 0, 1], [-1j, 1, 0, 0], [0, 0, 1, -1j], [1, 0, 1j, 1]])]),
        ("Hermitian", [0, 0, 0, 1], 0, [np.array([[0, 1j, 0, 0], [-1j, 0, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]])]),
        ("Hermitian", [1/math.sqrt(2), 0, -1/math.sqrt(2), 0], 1, [np.array([[1, 1j, 0, 0], [-1j, 1, 0, 0], [0, 0, 1, -1j], [0, 0, 1j, 1]])]),
        ("Hermitian", [1/math.sqrt(3), -1/math.sqrt(3), 1/math.sqrt(6), 1/math.sqrt(6)], 1, [np.array([[1, 1j, 0, .5j], [-1j, 1, 0, 0], [0, 0, 1, -1j], [-.5j, 0, 1j, 1]])]),
        ("Hermitian", [1/math.sqrt(2), 0, 0, 1/math.sqrt(2)], 1, [np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])]),
        ("Hermitian", [0, 1/math.sqrt(2), -1/math.sqrt(2), 0], -1, [np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])]),
    ])
    def test_supported_observable_two_wires_with_parameters(self, qubit_device_2_wires, tol, name, state, expected_output, par):
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

        dev = qml.device('default.qubit', wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            qml.CNOT(wires=[0, 1])
            return qml.sample(qml.PauliZ(0)), qml.sample(qml.PauliZ(1))

        outcomes = circuit()

        assert np.array_equal(outcomes[0], outcomes[1])

    @pytest.mark.parametrize("num_wires", [3, 4, 5, 6, 7, 8])
    def test_multi_samples_return_correlated_results_more_wires_than_size_of_observable(self, num_wires):
        """Tests if the samples returned by the sample function have
        the correct dimensions
        """

        dev = qml.device('default.qubit', wires=num_wires)

        @qml.qnode(dev)
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
                qml.CNOT(wires=[1, 2])
            ],
            obs.diagonalizing_gates()
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
                qml.CNOT(wires=[1, 2])
            ],
            obs.diagonalizing_gates()
        )

        res = dev.expval(obs)

        expected = np.cos(varphi)*np.cos(phi)

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
                qml.CNOT(wires=[1, 2])
            ],
            obs.diagonalizing_gates()
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
                qml.CNOT(wires=[1, 2])
            ],
            obs.diagonalizing_gates()
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

        A1 = np.array([[1, 2],
                       [2, 4]])

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
                qml.CNOT(wires=[1, 2])
            ],
            obs.diagonalizing_gates()
        )

        res = dev.expval(obs)

        expected = 0.25 * (
            -30
            + 4 * np.cos(phi) * np.sin(theta)
            + 3 * np.cos(varphi) * (-10 + 4 * np.cos(phi) * np.sin(theta) - 3 * np.sin(phi))
            - 3 * np.sin(phi)
            - 2 * (5 + np.cos(phi) * (6 + 4 * np.sin(theta)) + (-3 + 8 * np.sin(theta)) * np.sin(phi))
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

        A = np.array([[1.02789352, 1.61296440 - 0.3498192j], [1.61296440 + 0.3498192j, 1.23920938 + 0j]])

        obs = qml.Hermitian(A, wires=[0]) @ qml.Identity(wires=[1])

        dev.apply(
            [
                qml.RY(theta, wires=[0]),
                qml.RY(phi, wires=[1]),
                qml.CNOT(wires=[0, 1])
            ],
            obs.diagonalizing_gates()
        )

        res = dev.expval(obs)

        a = A[0, 0]
        re_b = A[0, 1].real
        d = A[1, 1]
        expected = ((a - d) * np.cos(theta) + 2 * re_b * np.sin(theta) * np.sin(phi) + a + d) / 2

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_hermitian_two_wires_identity_expectation(self, theta, phi, varphi, tol):
        """Test that a tensor product involving an Hermitian matrix for two wires and the identity works correctly"""
        dev = qml.device("default.qubit", wires=3, analytic=True)

        A = np.array([[1.02789352, 1.61296440 - 0.3498192j], [1.61296440 + 0.3498192j, 1.23920938 + 0j]])
        Identity = np.array([[1, 0],[0, 1]])
        H = np.kron(np.kron(Identity,Identity), A)
        obs = qml.Hermitian(H, wires=[2, 1, 0])

        dev.apply(
            [
                qml.RY(theta, wires=[0]),
                qml.RY(phi, wires=[1]),
                qml.CNOT(wires=[0, 1])
            ],
            obs.diagonalizing_gates()
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
                qml.CNOT(wires=[1, 2])
            ],
            obs.diagonalizing_gates()
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
                qml.CNOT(wires=[1, 2])
            ],
            obs.diagonalizing_gates()
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
                qml.CNOT(wires=[1, 2])
            ],
            obs.diagonalizing_gates()
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

    def test_paulix_pauliy(self, theta, phi, varphi, monkeypatch, tol):
        """Test that a tensor product involving PauliX and PauliY works correctly"""
        dev = qml.device("default.qubit", wires=3, shots=10000)

        obs = qml.PauliX(0) @ qml.PauliY(2)

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2])
            ],
            obs.diagonalizing_gates()
        )

        dev._wires_measured = {0, 1, 2}
        dev._samples = dev.generate_samples()
        dev.sample(obs)

        s1 = obs.eigvals
        p = dev.probability(wires=obs.wires)

        # s1 should only contain 1 and -1
        assert np.allclose(s1 ** 2, 1, atol=tol, rtol=0)

        mean = s1 @ p
        expected = np.sin(theta) * np.sin(phi) * np.sin(varphi)
        assert np.allclose(mean, expected, atol=tol, rtol=0)

        var = (s1 ** 2) @ p - (s1 @ p).real ** 2
        expected = (
            8 * np.sin(theta) ** 2 * np.cos(2 * varphi) * np.sin(phi) ** 2
            - np.cos(2 * (theta - phi))
            - np.cos(2 * (theta + phi))
            + 2 * np.cos(2 * theta)
            + 2 * np.cos(2 * phi)
            + 14
        ) / 16
        assert np.allclose(var, expected, atol=tol, rtol=0)

    def test_pauliz_hadamard(self, theta, phi, varphi, monkeypatch, tol):
        """Test that a tensor product involving PauliZ and PauliY and hadamard works correctly"""
        dev = qml.device("default.qubit", wires=3)
        obs = qml.PauliZ(0) @ qml.Hadamard(1) @ qml.PauliY(2)
        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2])
            ],
            obs.diagonalizing_gates()
        )

        dev._wires_measured = {0, 1, 2}
        dev._samples = dev.generate_samples()
        dev.sample(obs)

        s1 = obs.eigvals
        p = dev.marginal_prob(dev.probability(), wires=obs.wires)

        # s1 should only contain 1 and -1
        assert np.allclose(s1 ** 2, 1, atol=tol, rtol=0)

        mean = s1 @ p
        expected = -(np.cos(varphi) * np.sin(phi) + np.sin(varphi) * np.cos(theta)) / np.sqrt(2)
        assert np.allclose(mean, expected, atol=tol, rtol=0)

        var = (s1 ** 2) @ p - (s1 @ p).real ** 2
        expected = (
            3
            + np.cos(2 * phi) * np.cos(varphi) ** 2
            - np.cos(2 * theta) * np.sin(varphi) ** 2
            - 2 * np.cos(theta) * np.sin(phi) * np.sin(2 * varphi)
        ) / 4
        assert np.allclose(var, expected, atol=tol, rtol=0)

    def test_hermitian(self, theta, phi, varphi, monkeypatch, tol):
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
                qml.CNOT(wires=[1, 2])
            ],
            obs.diagonalizing_gates()
        )

        dev._wires_measured = {0, 1, 2}
        dev._samples = dev.generate_samples()
        dev.sample(obs)

        s1 = obs.eigvals
        p = dev.marginal_prob(dev.probability(), wires=obs.wires)

        # s1 should only contain the eigenvalues of
        # the hermitian matrix tensor product Z
        Z = np.diag([1, -1])
        eigvals = np.linalg.eigvalsh(np.kron(Z, A))
        assert set(np.round(s1, 8)).issubset(set(np.round(eigvals, 8)))

        mean = s1 @ p
        expected = 0.5 * (
            -6 * np.cos(theta) * (np.cos(varphi) + 1)
            - 2 * np.sin(varphi) * (np.cos(theta) + np.sin(phi) - 2 * np.cos(phi))
            + 3 * np.cos(varphi) * np.sin(phi)
            + np.sin(phi)
        )
        assert np.allclose(mean, expected, atol=tol, rtol=0)

        var = (s1 ** 2) @ p - (s1 @ p).real ** 2
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
        assert np.allclose(var, expected, atol=tol, rtol=0)

class TestProbabilityIntegration:
    """Test probability method for when analytic is True/False"""

    def mock_analytic_counter(self, wires=None):
        self.analytic_counter += 1
        return np.array([1, 0, 0, 0], dtype=float)

    @pytest.mark.parametrize("x", [[0.2, 0.5], [0.4, 0.9], [0.8, 0.3]])
    def test_probability(self, x, tol):
        """Test that the probability function works when analytic=False"""
        dev = qml.device("default.qubit", wires=2, analytic=False)
        dev_analytic = qml.device("default.qubit", wires=2, analytic=True)

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

    @pytest.mark.parametrize("analytic", [True, False])
    def test_call_generate_samples(self, analytic, monkeypatch):
        """Test analytic_probability call when generating samples"""
        self.analytic_counter = False

        dev = qml.device("default.qubit", wires=2, analytic=analytic)
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

# Copyright 2018 Xanadu Quantum Technologies Inc.

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
import logging as log
import math

import pennylane as qml
import pytest
from pennylane import numpy as np
from pennylane.plugins.default_qubit import (CNOT, CRot3, CRotx, CRoty, CRotz,
                                             DefaultQubit, I, Rot3, Rotx, Roty,
                                             Rotz, Rphi, X, Z, hermitian,
                                             spectral_decomposition, unitary)

log.getLogger("defaults")


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


H = np.array(
    [[1.02789352, 1.61296440 - 0.3498192j], [1.61296440 + 0.3498192j, 1.23920938 + 0j]]
)


def prep_par(par, op):
    "Convert par into a list of parameters that op expects."
    if op.par_domain == "A":
        return [np.diag([x, 1]) for x in par]
    return par


class TestAuxillaryFunctions:
    """Test auxillary functions."""

    def test_spectral_decomposition(self, tol):
        """Test that the correct spectral decomposition is returned."""

        a, P = spectral_decomposition(H)

        # verify that H = \sum_k a_k P_k
        assert np.allclose(H, np.einsum("i,ijk->jk", a, P), atol=tol, rtol=0)

    def test_phase_shift(self, tol):
        """Test phase shift is correct"""

        # test identity for theta=0
        assert np.allclose(Rphi(0), np.identity(2), atol=tol, rtol=0)

        # test arbitrary phase shift
        phi = 0.5432
        expected = np.array([[1, 0], [0, np.exp(1j * phi)]])
        assert np.allclose(Rphi(phi), expected, atol=tol, rtol=0)

    def test_x_rotation(self, tol):
        """Test x rotation is correct"""

        # test identity for theta=0
        assert np.allclose(Rotx(0), np.identity(2), atol=tol, rtol=0)

        # test identity for theta=pi/2
        expected = np.array([[1, -1j], [-1j, 1]]) / np.sqrt(2)
        assert np.allclose(Rotx(np.pi / 2), expected, atol=tol, rtol=0)

        # test identity for theta=pi
        expected = -1j * np.array([[0, 1], [1, 0]])
        assert np.allclose(Rotx(np.pi), expected, atol=tol, rtol=0)

    def test_y_rotation(self, tol):
        """Test y rotation is correct"""

        # test identity for theta=0
        assert np.allclose(Roty(0), np.identity(2), atol=tol, rtol=0)

        # test identity for theta=pi/2
        expected = np.array([[1, -1], [1, 1]]) / np.sqrt(2)
        assert np.allclose(Roty(np.pi / 2), expected, atol=tol, rtol=0)

        # test identity for theta=pi
        expected = np.array([[0, -1], [1, 0]])
        assert np.allclose(Roty(np.pi), expected, atol=tol, rtol=0)

    def test_z_rotation(self, tol):
        """Test z rotation is correct"""

        # test identity for theta=0
        assert np.allclose(Rotz(0), np.identity(2), atol=tol, rtol=0)

        # test identity for theta=pi/2
        expected = np.diag(np.exp([-1j * np.pi / 4, 1j * np.pi / 4]))
        assert np.allclose(Rotz(np.pi / 2), expected, atol=tol, rtol=0)

        # test identity for theta=pi
        assert np.allclose(Rotz(np.pi), -1j * Z, atol=tol, rtol=0)

    def test_arbitrary_rotation(self, tol):
        """Test arbitrary single qubit rotation is correct"""

        # test identity for phi,theta,omega=0
        assert np.allclose(Rot3(0, 0, 0), np.identity(2), atol=tol, rtol=0)

        # expected result
        def arbitrary_rotation(x, y, z):
            """arbitrary single qubit rotation"""
            c = np.cos(y / 2)
            s = np.sin(y / 2)
            return np.array(
                [
                    [np.exp(-0.5j * (x + z)) * c, -np.exp(0.5j * (x - z)) * s],
                    [np.exp(-0.5j * (x - z)) * s, np.exp(0.5j * (x + z)) * c],
                ]
            )

        a, b, c = 0.432, -0.152, 0.9234
        assert np.allclose(Rot3(a, b, c), arbitrary_rotation(a, b, c), atol=tol, rtol=0)

    def test_C_x_rotation(self, tol):
        """Test controlled x rotation is correct"""

        # test identity for theta=0
        assert np.allclose(CRotx(0), np.identity(4), atol=tol, rtol=0)

        # test identity for theta=pi/2
        expected = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1/np.sqrt(2), -1j/np.sqrt(2)], [0, 0, -1j/np.sqrt(2), 1/np.sqrt(2)]])
        assert np.allclose(CRotx(np.pi / 2), expected, atol=tol, rtol=0)

        # test identity for theta=pi
        expected = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, -1j, 0]])
        assert np.allclose(CRotx(np.pi), expected, atol=tol, rtol=0)

    def test_C_y_rotation(self, tol):
        """Test controlled y rotation is correct"""

        # test identity for theta=0
        assert np.allclose(CRoty(0), np.identity(4), atol=tol, rtol=0)

        # test identity for theta=pi/2
        expected = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1/np.sqrt(2), -1/np.sqrt(2)], [0, 0, 1/np.sqrt(2), 1/np.sqrt(2)]])
        assert np.allclose(CRoty(np.pi / 2), expected, atol=tol, rtol=0)

        # test identity for theta=pi
        expected = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]])
        assert np.allclose(CRoty(np.pi), expected, atol=tol, rtol=0)

    def test_C_z_rotation(self, tol):
        """Test controlled z rotation is correct"""

        # test identity for theta=0
        assert np.allclose(CRotz(0), np.identity(4), atol=tol, rtol=0)

        # test identity for theta=pi/2
        expected = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, np.exp(-1j * np.pi / 4), 0], [0, 0, 0, np.exp(1j * np.pi / 4)]])
        assert np.allclose(CRotz(np.pi / 2), expected, atol=tol, rtol=0)

        # test identity for theta=pi
        expected = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1j, 0], [0, 0, 0, 1j]])
        assert np.allclose(CRotz(np.pi), expected, atol=tol, rtol=0)

    def test_controlled_arbitrary_rotation(self, tol):
        """Test controlled arbitrary rotation is correct"""

        # test identity for phi,theta,omega=0
        assert np.allclose(CRot3(0, 0, 0), np.identity(4), atol=tol, rtol=0)

        # test identity for phi,theta,omega=pi
        expected = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]])
        assert np.allclose(CRot3(np.pi, np.pi, np.pi), expected, atol=tol, rtol=0)

        def arbitrary_Crotation(x, y, z):
            """controlled arbitrary single qubit rotation"""
            c = np.cos(y / 2)
            s = np.sin(y / 2)
            return np.array(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, np.exp(-0.5j * (x + z)) * c, -np.exp(0.5j * (x - z)) * s],
                    [0, 0, np.exp(-0.5j * (x - z)) * s, np.exp(0.5j * (x + z)) * c]
                ]
            )

        a, b, c = 0.432, -0.152, 0.9234
        assert np.allclose(CRot3(a, b, c), arbitrary_Crotation(a, b, c), atol=tol, rtol=0)

class TestStateFunctions:
    """Arbitrary state and operator tests."""

    def test_unitary(self, tol):
        """Test that the unitary function produces the correct output."""

        out = unitary(U)

        # verify output type
        assert isinstance(out, np.ndarray)

        # verify equivalent to input state
        assert np.allclose(out, U, atol=tol, rtol=0)

    def test_unitary_exceptions(self):
        """Tests that the unitary function raises the proper errors."""

        # test non-square matrix
        with pytest.raises(ValueError, match="must be a square matrix"):
            unitary(U[1:])

        # test non-unitary matrix
        U3 = U.copy()
        U3[0, 0] += 0.5
        with pytest.raises(ValueError, match="must be unitary"):
            unitary(U3)

    def test_hermitian(self, tol):
        """Test that the hermitian function produces the correct output."""
        
        out = hermitian(H)

        # verify output type
        assert isinstance(out, np.ndarray)

        # verify equivalent to input state
        assert np.allclose(out, H, atol=tol, rtol=0)

    def test_hermitian_exceptions(self):
        """Tests that the hermitian function raises the proper errors."""

        # test non-square matrix
        with pytest.raises(ValueError, match="must be a square matrix"):
            hermitian(H[1:])

        # test non-Hermitian matrix
        H2 = H.copy()
        H2[0, 1] = H2[0, 1].conj()
        with pytest.raises(ValueError, match="must be Hermitian"):
            hermitian(H2)


class TestOperatorMatrices:
    """Tests that get_operator_matrix returns the correct matrix."""

    @pytest.mark.parametrize("name,expected", [
        ("PauliX", np.array([[0, 1], [1, 0]])),
        ("PauliY", np.array([[0, -1j], [1j, 0]])),
        ("PauliZ", np.array([[1, 0], [0, -1]])),
        ("Hadamard", np.array([[1, 1], [1, -1]])/np.sqrt(2)),
        ("CNOT", np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])),
        ("SWAP", np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])),
        ("CZ", np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])),
    ])
    def test_get_operator_matrix_no_parameters(self, qubit_device_2_wires, tol, name, expected):
        """Tests that get_operator_matrix returns the correct matrix."""

        res = qubit_device_2_wires._get_operator_matrix(name, ())

        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("name,expected,par", [
        ('PhaseShift', lambda phi: np.array([[1, 0], [0, np.exp(1j*phi)]]), [0.223]),
        ('RX', lambda phi: np.array([[math.cos(phi/2), -1j*math.sin(phi/2)], [-1j*math.sin(phi/2), math.cos(phi/2)]]), [0.223]),
        ('RY', lambda phi: np.array([[math.cos(phi/2), -math.sin(phi/2)], [math.sin(phi/2), math.cos(phi/2)]]), [0.223]),
        ('RZ', lambda phi: np.array([[cmath.exp(-1j*phi/2), 0], [0, cmath.exp(1j*phi/2)]]), [0.223]),
        ('Rot', lambda phi, theta, omega: np.array([[cmath.exp(-1j*(phi+omega)/2)*math.cos(theta/2), -cmath.exp(1j*(phi-omega)/2)*math.sin(theta/2)], [cmath.exp(-1j*(phi-omega)/2)*math.sin(theta/2), cmath.exp(1j*(phi+omega)/2)*math.cos(theta/2)]]), [0.223, 0.153, 1.212]),
        ('CRX', lambda phi: np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, math.cos(phi/2), -1j*math.sin(phi/2)], [0, 0, -1j*math.sin(phi/2), math.cos(phi/2)]]), [0.223]),
        ('CRY', lambda phi: np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, math.cos(phi/2), -math.sin(phi/2)], [0, 0, math.sin(phi/2), math.cos(phi/2)]]), [0.223]),
        ('CRZ', lambda phi: np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, cmath.exp(-1j*phi/2), 0], [0, 0, 0, cmath.exp(1j*phi/2)]]), [0.223]),
        ('CRot', lambda phi, theta, omega: np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, cmath.exp(-1j*(phi+omega)/2)*math.cos(theta/2), -cmath.exp(1j*(phi-omega)/2)*math.sin(theta/2)], [0, 0, cmath.exp(-1j*(phi-omega)/2)*math.sin(theta/2), cmath.exp(1j*(phi+omega)/2)*math.cos(theta/2)]]), [0.223, 0.153, 1.212]),
        ('QubitUnitary', lambda U: np.asarray(U), [np.array([[0.83645892 - 0.40533293j, -0.20215326 + 0.30850569j], [-0.23889780 - 0.28101519j, -0.88031770 - 0.29832709j]])]),
        ('Hermitian', lambda H: np.asarray(H), [np.array([[1.02789352, 1.61296440 - 0.3498192j], [1.61296440 + 0.3498192j, 1.23920938 + 0j]])]),
        # Identity will always return a 2x2 Identity, but is still parameterized
        ('Identity', lambda n: np.eye(2), [2])
    ])
    def test_get_operator_matrix_with_parameters(self, qubit_device_2_wires, tol, name, expected, par):
        """Tests that get_operator_matrix returns the correct matrix building functions."""

        res = qubit_device_2_wires._get_operator_matrix(name, par)

        assert np.allclose(res, expected(*par), atol=tol, rtol=0)

    @pytest.mark.parametrize("name", ["BasisState", "QubitStateVector"])
    def test_get_operator_matrix_none(self, qubit_device_2_wires, name):
        """Tests that get_operator_matrix returns none for direct state manipulations."""

        res = qubit_device_2_wires._get_operator_matrix(name, ())

        assert res is None

class TestApply:
    """Tests that operations are applied correctly or that the proper errors are raised."""

    @pytest.mark.parametrize("name,input,expected_output", [
        ("PauliX", [1, 0], np.array([0, 1])),
        ("PauliX", [1/math.sqrt(2), 1/math.sqrt(2)], [1/math.sqrt(2), 1/math.sqrt(2)]),
        ("PauliY", [1, 0], [0, 1j]),
        ("PauliY", [1/math.sqrt(2), 1/math.sqrt(2)], [-1j/math.sqrt(2), 1j/math.sqrt(2)]),
        ("PauliZ", [1, 0], [1, 0]),
        ("PauliZ", [1/math.sqrt(2), 1/math.sqrt(2)], [1/math.sqrt(2), -1/math.sqrt(2)]),
        ("Hadamard", [1, 0], [1/math.sqrt(2), 1/math.sqrt(2)]),
        ("Hadamard", [1/math.sqrt(2), -1/math.sqrt(2)], [0, 1]),
    ])
    def test_apply_operation_single_wire_no_parameters(self, qubit_device_1_wire, tol, name, input, expected_output):
        """Tests that applying an operation yields the expected output state for single wire
           operations that have no parameters."""

        qubit_device_1_wire._state = np.array(input)
        qubit_device_1_wire.apply(name, wires=[0], par=[])

        assert np.allclose(qubit_device_1_wire._state, np.array(expected_output), atol=tol, rtol=0)

    @pytest.mark.parametrize("name,input,expected_output", [
        ("CNOT", [1, 0, 0, 0], [1, 0, 0, 0]),
        ("CNOT", [0, 0, 1, 0], [0, 0, 0, 1]),
        ("CNOT", [1/math.sqrt(2), 0, 0, 1/math.sqrt(2)], [1/math.sqrt(2), 0, 1/math.sqrt(2), 0]),
        ("SWAP", [1, 0, 0, 0], [1, 0, 0, 0]),
        ("SWAP", [0, 0, 1, 0], [0, 1, 0, 0]),
        ("SWAP", [1/math.sqrt(2), 0, -1/math.sqrt(2), 0], [1/math.sqrt(2), -1/math.sqrt(2), 0, 0]),
        ("CZ", [1, 0, 0, 0], [1, 0, 0, 0]),
        ("CZ", [0, 0, 0, 1], [0, 0, 0, -1]),
        ("CZ", [1/math.sqrt(2), 0, 0, -1/math.sqrt(2)], [1/math.sqrt(2), 0, 0, 1/math.sqrt(2)]),
    ])
    def test_apply_operation_two_wires_no_parameters(self, qubit_device_2_wires, tol, name, input, expected_output):
        """Tests that applying an operation yields the expected output state for two wire
           operations that have no parameters."""

        qubit_device_2_wires._state = np.array(input)
        qubit_device_2_wires.apply(name, wires=[0, 1], par=[])

        assert np.allclose(qubit_device_2_wires._state, np.array(expected_output), atol=tol, rtol=0)

    @pytest.mark.parametrize("name,input,expected_output,par", [
        ("BasisState", [1, 0, 0, 0], [0, 0, 1, 0], [[1, 0]]),
        ("BasisState", [1/math.sqrt(2), 0, 1/math.sqrt(2), 0], [0, 0, 1, 0], [[1, 0]]),
        ("BasisState", [1/math.sqrt(2), 0, 1/math.sqrt(2), 0], [0, 0, 0, 1], [[1, 1]]),
        ("QubitStateVector", [1, 0, 0, 0], [0, 0, 1, 0], [[0, 0, 1, 0]]),
        ("QubitStateVector", [1/math.sqrt(2), 0, 1/math.sqrt(2), 0], [0, 0, 1, 0], [[0, 0, 1, 0]]),
        ("QubitStateVector", [1/math.sqrt(2), 0, 1/math.sqrt(2), 0], [0, 0, 0, 1], [[0, 0, 0, 1]]),
        ("QubitStateVector", [1, 0, 0, 0], [1/math.sqrt(3), 0, 1/math.sqrt(3), 1/math.sqrt(3)], [[1/math.sqrt(3), 0, 1/math.sqrt(3), 1/math.sqrt(3)]]),
        ("QubitStateVector", [1, 0, 0, 0], [1/math.sqrt(3), 0, -1/math.sqrt(3), 1/math.sqrt(3)], [[1/math.sqrt(3), 0, -1/math.sqrt(3), 1/math.sqrt(3)]]),
    ])
    def test_apply_operation_state_preparation(self, qubit_device_2_wires, tol, name, input, expected_output, par):
        """Tests that applying an operation yields the expected output state for single wire
           operations that have no parameters."""

        qubit_device_2_wires._state = np.array(input)
        qubit_device_2_wires.apply(name, wires=[0, 1], par=par)

        assert np.allclose(qubit_device_2_wires._state, np.array(expected_output), atol=tol, rtol=0)

    @pytest.mark.parametrize("name,input,expected_output,par", [
        ("PhaseShift", [1, 0], [1, 0], [math.pi/2]),
        ("PhaseShift", [0, 1], [0, 1j], [math.pi/2]),
        ("PhaseShift", [1/math.sqrt(2), 1/math.sqrt(2)], [1/math.sqrt(2), 1/2 + 1j/2], [math.pi/4]),
        ("RX", [1, 0], [1/math.sqrt(2), -1j*1/math.sqrt(2)], [math.pi/2]),
        ("RX", [1, 0], [0, -1j], [math.pi]),
        ("RX", [1/math.sqrt(2), 1/math.sqrt(2)], [1/2 - 1j/2, 1/2 -1j/2], [math.pi/2]),
        ("RY", [1, 0], [1/math.sqrt(2), 1/math.sqrt(2)], [math.pi/2]),
        ("RY", [1, 0], [0, 1], [math.pi]),
        ("RY", [1/math.sqrt(2), 1/math.sqrt(2)], [0, 1], [math.pi/2]),
        ("RZ", [1, 0], [1/math.sqrt(2) - 1j/math.sqrt(2), 0], [math.pi/2]),
        ("RZ", [0, 1], [0, 1j], [math.pi]),
        ("RZ", [1/math.sqrt(2), 1/math.sqrt(2)], [1/2 - 1j/2, 1/2 + 1j/2], [math.pi/2]),
        ("Rot", [1, 0], [1/math.sqrt(2) - 1j/math.sqrt(2), 0], [math.pi/2, 0, 0]),
        ("Rot", [1, 0], [1/math.sqrt(2), 1/math.sqrt(2)], [0, math.pi/2, 0]),
        ("Rot", [1/math.sqrt(2), 1/math.sqrt(2)], [1/2 - 1j/2, 1/2 + 1j/2], [0, 0, math.pi/2]),
        ("Rot", [1, 0], [-1j/math.sqrt(2), -1/math.sqrt(2)], [math.pi/2, -math.pi/2, math.pi/2]),
        ("Rot", [1/math.sqrt(2), 1/math.sqrt(2)], [1/2 + 1j/2, -1/2 + 1j/2], [-math.pi/2, math.pi, math.pi]),
        ("QubitUnitary", [1, 0], [1j/math.sqrt(2), 1j/math.sqrt(2)], [np.array([[1j/math.sqrt(2), 1j/math.sqrt(2)], [1j/math.sqrt(2), -1j/math.sqrt(2)]])]),
        ("QubitUnitary", [0, 1], [1j/math.sqrt(2), -1j/math.sqrt(2)], [np.array([[1j/math.sqrt(2), 1j/math.sqrt(2)], [1j/math.sqrt(2), -1j/math.sqrt(2)]])]),
        ("QubitUnitary", [1/math.sqrt(2), -1/math.sqrt(2)], [0, 1j], [np.array([[1j/math.sqrt(2), 1j/math.sqrt(2)], [1j/math.sqrt(2), -1j/math.sqrt(2)]])]),
    ])
    def test_apply_operation_single_wire_with_parameters(self, qubit_device_1_wire, tol, name, input, expected_output, par):
        """Tests that applying an operation yields the expected output state for single wire
           operations that have no parameters."""

        qubit_device_1_wire._state = np.array(input)
        qubit_device_1_wire.apply(name, wires=[0], par=par)

        assert np.allclose(qubit_device_1_wire._state, np.array(expected_output), atol=tol, rtol=0)

    @pytest.mark.parametrize("name,input,expected_output,par", [
        ("CRX", [0, 1, 0, 0], [0, 1, 0, 0], [math.pi/2]),
        ("CRX", [0, 0, 0, 1], [0, 0, -1j, 0], [math.pi]),
        ("CRX", [0, 1/math.sqrt(2), 1/math.sqrt(2), 0], [0, 1/math.sqrt(2), 1/2, -1j/2], [math.pi/2]),
        ("CRY", [0, 0, 0, 1], [0, 0, -1/math.sqrt(2), 1/math.sqrt(2)], [math.pi/2]),
        ("CRY", [0, 0, 0, 1], [0, 0, -1, 0], [math.pi]),
        ("CRY", [1/math.sqrt(2), 1/math.sqrt(2), 0, 0], [1/math.sqrt(2), 1/math.sqrt(2), 0, 0], [math.pi/2]),
        ("CRZ", [0, 0, 0, 1], [0, 0, 0, 1/math.sqrt(2) + 1j/math.sqrt(2)], [math.pi/2]),
        ("CRZ", [0, 0, 0, 1], [0, 0, 0, 1j], [math.pi]),
        ("CRZ", [1/math.sqrt(2), 1/math.sqrt(2), 0, 0], [1/math.sqrt(2), 1/math.sqrt(2), 0, 0], [math.pi/2]),
        ("CRot", [0, 0, 0, 1], [0, 0, 0, 1/math.sqrt(2) + 1j/math.sqrt(2)], [math.pi/2, 0, 0]),
        ("CRot", [0, 0, 0, 1], [0, 0, -1/math.sqrt(2), 1/math.sqrt(2)], [0, math.pi/2, 0]),
        ("CRot", [0, 0, 1/math.sqrt(2), 1/math.sqrt(2)], [0, 0, 1/2 - 1j/2, 1/2 + 1j/2], [0, 0, math.pi/2]),
        ("CRot", [0, 0, 0, 1], [0, 0, 1/math.sqrt(2), 1j/math.sqrt(2)], [math.pi/2, -math.pi/2, math.pi/2]),
        ("CRot", [0, 1/math.sqrt(2), 1/math.sqrt(2), 0], [0, 1/math.sqrt(2), 0, -1/2 + 1j/2], [-math.pi/2, math.pi, math.pi]),
        ("QubitUnitary", [1, 0, 0, 0], [1, 0, 0, 0], [np.array([[1, 0, 0, 0], [0, 1/math.sqrt(2), 1/math.sqrt(2), 0], [0, 1/math.sqrt(2), -1/math.sqrt(2), 0], [0, 0, 0, 1]])]),
        ("QubitUnitary", [0, 1, 0, 0], [0, 1/math.sqrt(2), 1/math.sqrt(2), 0], [np.array([[1, 0, 0, 0], [0, 1/math.sqrt(2), 1/math.sqrt(2), 0], [0, 1/math.sqrt(2), -1/math.sqrt(2), 0], [0, 0, 0, 1]])]),
        ("QubitUnitary", [1/2, 1/2, -1/2, 1/2], [1/2, 0, 1/math.sqrt(2), 1/2], [np.array([[1, 0, 0, 0], [0, 1/math.sqrt(2), 1/math.sqrt(2), 0], [0, 1/math.sqrt(2), -1/math.sqrt(2), 0], [0, 0, 0, 1]])]),
    ])
    def test_apply_operation_two_wires_with_parameters(self, qubit_device_2_wires, tol, name, input, expected_output, par):
        """Tests that applying an operation yields the expected output state for single wire
           operations that have no parameters."""

        qubit_device_2_wires._state = np.array(input)
        qubit_device_2_wires.apply(name, wires=[0, 1], par=par)

        assert np.allclose(qubit_device_2_wires._state, np.array(expected_output), atol=tol, rtol=0)

    def test_apply_errors(self, qubit_device_2_wires):
        """Test that apply fails for incorrect state preparation, and > 2 qubit gates"""

        with pytest.raises(
            ValueError,
            match=r"State vector must be of length 2\*\*wires."
        ):
            p = [np.array([1, 0, 1, 1, 1]) / np.sqrt(3)]
            qubit_device_2_wires.apply("QubitStateVector", wires=[0, 1], par=[p])

        with pytest.raises(
            ValueError,
            match="BasisState parameter must be an array of 0 or 1 integers of length at most 2."
        ):
            qubit_device_2_wires.apply("BasisState", wires=[0, 1], par=[np.array([-0.2, 4.2])])

        with pytest.raises(
            ValueError,
            match="The default.qubit plugin can apply BasisState only to all of the 2 wires."
        ):
            qubit_device_2_wires.apply("BasisState", wires=[0, 1, 2], par=[np.array([0, 1])])


class TestExpval:
    """Tests that expectation values are properly calculated or that the proper errors are raised."""

    @pytest.mark.parametrize("name,input,expected_output", [
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
    def test_expval_single_wire_no_parameters(self, qubit_device_1_wire, tol, name, input, expected_output):
        """Tests that expectation values are properly calculated for single-wire observables without parameters."""

        qubit_device_1_wire._state = np.array(input)
        res = qubit_device_1_wire.expval(name, wires=[0], par=[])

        assert np.isclose(res, expected_output, atol=tol, rtol=0) 

    @pytest.mark.parametrize("name,input,expected_output,par", [
        ("Identity", [1, 0], 1, []),
        ("Identity", [0, 1], 1, []),
        ("Identity", [1/math.sqrt(2), -1/math.sqrt(2)], 1, []),
        ("Hermitian", [1, 0], 1, [[[1, 1j], [-1j, 1]]]),
        ("Hermitian", [0, 1], 1, [[[1, 1j], [-1j, 1]]]),
        ("Hermitian", [1/math.sqrt(2), -1/math.sqrt(2)], 1, [[[1, 1j], [-1j, 1]]]),
    ])
    def test_expval_single_wire_with_parameters(self, qubit_device_1_wire, tol, name, input, expected_output, par):
        """Tests that expectation values are properly calculated for single-wire observables with parameters."""

        qubit_device_1_wire._state = np.array(input)
        res = qubit_device_1_wire.expval(name, wires=[0], par=par)

        assert np.isclose(res, expected_output, atol=tol, rtol=0)

    @pytest.mark.parametrize("name,input,expected_output,par", [
        ("Hermitian", [1/math.sqrt(3), 0, 1/math.sqrt(3), 1/math.sqrt(3)], 5/3, [[[1, 1j, 0, 1], [-1j, 1, 0, 0], [0, 0, 1, -1j], [1, 0, 1j, 1]]]),
        ("Hermitian", [0, 0, 0, 1], 0, [[[0, 1j, 0, 0], [-1j, 0, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]]]),
        ("Hermitian", [1/math.sqrt(2), 0, -1/math.sqrt(2), 0], 1, [[[1, 1j, 0, 0], [-1j, 1, 0, 0], [0, 0, 1, -1j], [0, 0, 1j, 1]]]),
        ("Hermitian", [1/math.sqrt(3), -1/math.sqrt(3), 1/math.sqrt(6), 1/math.sqrt(6)], 1, [[[1, 1j, 0, .5j], [-1j, 1, 0, 0], [0, 0, 1, -1j], [-.5j, 0, 1j, 1]]]),
        ("Hermitian", [1/math.sqrt(2), 0, 0, 1/math.sqrt(2)], 1, [[[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]]),
        ("Hermitian", [0, 1/math.sqrt(2), -1/math.sqrt(2), 0], -1, [[[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]]),
    ])
    def test_expval_two_wires_with_parameters(self, qubit_device_2_wires, tol, name, input, expected_output, par):
        """Tests that expectation values are properly calculated for two-wire observables with parameters."""

        qubit_device_2_wires._state = np.array(input)
        res = qubit_device_2_wires.expval(name, wires=[0, 1], par=par)

        assert np.isclose(res, expected_output, atol=tol, rtol=0)

    def test_expval_warnings(self, qubit_device_1_wire):
        """Tests that expval raises a warning if the given observable is complex."""
        
        qubit_device_1_wire.reset()

        # text warning raised if matrix is complex
        with pytest.warns(RuntimeWarning, match='Nonvanishing imaginary part'):
            qubit_device_1_wire.ev(np.array([[1+1j, 0], [0, 1+1j]]), wires=[0])

class TestVar:
    """Tests that variances are properly calculated."""

    @pytest.mark.parametrize("name,input,expected_output", [
        ("PauliX", [1/math.sqrt(2), 1/math.sqrt(2)], 0),
        ("PauliX", [1/math.sqrt(2), -1/math.sqrt(2)], 0),
        ("PauliX", [1, 0], 1),
        ("PauliY", [1/math.sqrt(2), 1j/math.sqrt(2)], 0),
        ("PauliY", [1/math.sqrt(2), -1j/math.sqrt(2)], 0),
        ("PauliY", [1, 0], 1),
        ("PauliZ", [1, 0], 0),
        ("PauliZ", [0, 1], 0),
        ("PauliZ", [1/math.sqrt(2), 1/math.sqrt(2)], 1),
        ("Hadamard", [1, 0], 1/2),
        ("Hadamard", [0, 1], 1/2),
        ("Hadamard", [1/math.sqrt(2), 1/math.sqrt(2)], 1/2),
    ])
    def test_var_single_wire_no_parameters(self, qubit_device_1_wire, tol, name, input, expected_output):
        """Tests that variances are properly calculated for single-wire observables without parameters."""

        qubit_device_1_wire._state = np.array(input)
        res = qubit_device_1_wire.var(name, wires=[0], par=[])

        assert np.isclose(res, expected_output, atol=tol, rtol=0) 

    @pytest.mark.parametrize("name,input,expected_output,par", [
        ("Identity", [1, 0], 0, []),
        ("Identity", [0, 1], 0, []),
        ("Identity", [1/math.sqrt(2), -1/math.sqrt(2)], 0, []),
        ("Hermitian", [1, 0], 1, [[[1, 1j], [-1j, 1]]]),
        ("Hermitian", [0, 1], 1, [[[1, 1j], [-1j, 1]]]),
        ("Hermitian", [1/math.sqrt(2), -1/math.sqrt(2)], 1, [[[1, 1j], [-1j, 1]]]),
    ])
    def test_var_single_wire_with_parameters(self, qubit_device_1_wire, tol, name, input, expected_output, par):
        """Tests that expectation values are properly calculated for single-wire observables with parameters."""

        qubit_device_1_wire._state = np.array(input)
        res = qubit_device_1_wire.var(name, wires=[0], par=par)

        assert np.isclose(res, expected_output, atol=tol, rtol=0)

    @pytest.mark.parametrize("name,input,expected_output,par", [
        ("Hermitian", [1/math.sqrt(3), 0, 1/math.sqrt(3), 1/math.sqrt(3)], 11/9, [[[1, 1j, 0, 1], [-1j, 1, 0, 0], [0, 0, 1, -1j], [1, 0, 1j, 1]]]),
        ("Hermitian", [0, 0, 0, 1], 1, [[[0, 1j, 0, 0], [-1j, 0, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]]]),
        ("Hermitian", [1/math.sqrt(2), 0, -1/math.sqrt(2), 0], 1, [[[1, 1j, 0, 0], [-1j, 1, 0, 0], [0, 0, 1, -1j], [0, 0, 1j, 1]]]),
        ("Hermitian", [1/math.sqrt(2), 0, 0, 1/math.sqrt(2)], 0, [[[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]]),
        ("Hermitian", [0, 1/math.sqrt(2), -1/math.sqrt(2), 0], 0, [[[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]]),
    ])
    def test_var_two_wires_with_parameters(self, qubit_device_2_wires, tol, name, input, expected_output, par):
        """Tests that variances are properly calculated for two-wire observables with parameters."""

        qubit_device_2_wires._state = np.array(input)
        res = qubit_device_2_wires.var(name, wires=[0, 1], par=par)

        assert np.isclose(res, expected_output, atol=tol, rtol=0)

    def test_var_estimate(self):
        """Test that the variance is not analytically calculated"""

        dev = qml.device("default.qubit", wires=1, shots=3)

        @qml.qnode(dev)
        def circuit():
            return qml.var(qml.PauliX(0))

        var = circuit()

        # With 3 samples we are guaranteed to see a difference between
        # an estimated variance an an analytically calculated one
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

        qubit_device_2_wires.apply('RX', wires=[0], par=[1.5708])
        qubit_device_2_wires.apply('RX', wires=[1], par=[1.5708])

        s1 = qubit_device_2_wires.sample('PauliZ', [0], [], 10)
        assert np.array_equal(s1.shape, (10,))

        s2 = qubit_device_2_wires.sample('PauliZ', [1], [], 12)
        assert np.array_equal(s2.shape, (12,))

        s3 = qubit_device_2_wires.sample('CZ', [0,1], [], 17)
        assert np.array_equal(s3.shape, (17,))

    def test_sample_values(self, qubit_device_2_wires, tol):
        """Tests if the samples returned by sample have
        the correct values
        """

        # Explicitly resetting is necessary as the internal
        # state is set to None in __init__ and only properly
        # initialized during reset
        qubit_device_2_wires.reset()

        qubit_device_2_wires.apply('RX', wires=[0], par=[1.5708])

        s1 = qubit_device_2_wires.sample('PauliZ', [0], [], 10)

        # s1 should only contain 1 and -1, which is guaranteed if
        # they square to 1
        assert np.allclose(s1**2, 1, atol=tol, rtol=0)

    def test_sample_exception_analytic_mode(self, qubit_device_2_wires):
        """Tests if the sampling raises an error for sample size n=0"""

        with pytest.raises(
            ValueError, match="Calling sample with n = 0 is not possible."
        ):
            qubit_device_2_wires.sample('PauliZ', [0], [], n = 0)

        # self.def.shots = 0, so this should also fail
        with pytest.raises(
            ValueError, match="Calling sample with n = 0 is not possible."
        ):
            qubit_device_2_wires.sample('PauliZ', [0], [])

    def test_sample_exception_wrong_n(self, qubit_device_2_wires):
        """Tests if the sampling raises an error for sample size n<0
        or non-integer n
        """

        with pytest.raises(
            ValueError, match="The number of samples must be a positive integer."
        ):
            qubit_device_2_wires.sample('PauliZ', [0], [], n = -12)

        qubit_device_2_wires.shots = 0
        with pytest.raises(
            ValueError, match="The number of samples must be a positive integer."
        ):
            qubit_device_2_wires.sample('PauliZ', [0], [], n = 12.3)


class TestMaps:
    """Tests the maps for operations and observables."""

    def test_operation_map(self, qubit_device_2_wires):
        """Test that default qubit device supports all PennyLane discrete gates."""
        
        assert set(qml.ops._qubit__ops__) ==  set(qubit_device_2_wires._operation_map)

    def test_observable_map(self, qubit_device_2_wires):
        """Test that default qubit device supports all PennyLane discrete observables."""
        
        assert set(qml.ops._qubit__obs__) | {"Identity"} == set(qubit_device_2_wires._observable_map)

    
class TestDefaultQubitIntegration:
    """Integration tests for default.qubit. This test ensures it integrates
    properly with the PennyLane interface, in particular QNode."""

    def test_load_default_qubit_device(self):
        """Test that the default plugin loads correctly"""
        
        dev = qml.device("default.qubit", wires=2)
        assert dev.num_wires ==  2
        assert dev.shots ==  0
        assert dev.short_name ==  "default.qubit"

    def test_args(self):
        """Test that the plugin requires correct arguments"""

        with pytest.raises(
            TypeError,  match="missing 1 required positional argument: 'wires'"
        ):
            qml.device("default.qubit")

    def test_unsupported_gates(self, qubit_device_2_wires, tol):
        """Test error is raised with unsupported gates"""

        gates = set(qubit_device_2_wires._operation_map.keys())
        all_gates = set(qml.ops.__all_ops__)

        for g in all_gates - gates:
            op = getattr(qml.ops, g)

            if op.num_wires == 0:
                wires = [0]
            else:
                wires = list(range(op.num_wires))

            @qml.qnode(qubit_device_2_wires)
            def circuit(*x):
                """Test quantum function"""
                x = prep_par(x, op)
                op(*x, wires=wires)

                if issubclass(op, qml.operation.CV):
                    return qml.expval(qml.X(0))

                return qml.expval(qml.PauliZ(0))

            with pytest.raises(
                qml.DeviceError,
                match="Gate {} not supported on device default.qubit".format(g),
            ):
                x = np.random.random([op.num_params])
                circuit(*x)

    def test_unsupported_observables(self, qubit_device_2_wires):
        """Test error is raised with unsupported observables"""

        obs = set(qubit_device_2_wires._observable_map.keys())
        all_obs = set(qml.ops.__all_obs__)

        for g in all_obs - obs:
            op = getattr(qml.ops, g)

            if op.num_wires == 0:
                wires = [0]
            else:
                wires = list(range(op.num_wires))

            @qml.qnode(qubit_device_2_wires)
            def circuit(*x):
                """Test quantum function"""
                x = prep_par(x, op)
                return qml.expval(op(*x, wires=wires))

            with pytest.raises(
                qml.DeviceError,
                match="Observable {} not supported on device default.qubit".format(g),
            ):
                x = np.random.random([op.num_params])
                circuit(*x)

    def test_qubit_circuit(self, qubit_device_1_wire, tol):
        """Test that the default qubit plugin provides correct result for simple circuit"""

        p = 0.543

        @qml.qnode(qubit_device_1_wire)
        def circuit(x):
            """Test quantum function"""
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliY(0))

        # <0|RX(p)^\dagger.PauliY.RX(p)|0> = -sin(p)
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
        dev = qml.device("default.qubit", wires=1, shots=shots)

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

    def test_supported_gates(self, qubit_device_2_wires):
        """Test that all supported gates work correctly"""
        
        a = 0.312
        b = 0.123

        for g, qop in qubit_device_2_wires._operation_map.items():
            log.debug("\tTesting gate %s...", g)
            assert qubit_device_2_wires.supports_operation(g)
            qubit_device_2_wires.reset()

            op = getattr(qml.ops, g)
            if op.num_wires <= 0:
                if g == "BasisState" or g == "QubitStateVector":
                    wires = [0, 1]
                else:
                    wires = [0]
            else:
                wires = list(range(op.num_wires))

            @qml.qnode(qubit_device_2_wires)
            def circuit(*x):
                """Reference quantum function"""
                op(*x, wires=wires)
                return qml.expval(qml.PauliX(0))

            # compare to reference result
            def reference(*x):
                """reference circuit"""
                if callable(qop):
                    # if the default.qubit is an operation accepting parameters,
                    # initialise it using the parameters generated above.
                    O = qop(*x)
                else:
                    # otherwise, the operation is simply an array.
                    O = qop

                # calculate the expected output
                if op.num_wires == 1 or g == "QubitUnitary":
                    out_state = np.kron(O @ np.array([1, 0]), np.array([1, 0]))
                elif g == "QubitStateVector":
                    out_state = x[0]
                elif g == "BasisState":
                    out_state = np.array([0, 0, 0, 1])
                else:
                    out_state = O @ qubit_device_2_wires._state

                expectation = out_state.conj() @ np.kron(X, np.identity(2)) @ out_state
                return expectation

            if g == "QubitStateVector":
                p = np.array([1, 0, 0, 1]) / np.sqrt(2)
                assert np.array_equal(circuit(p), reference(p))
            elif g == "BasisState":
                p = np.array([1, 1])
                assert np.array_equal(circuit(p), reference(p))
            elif g == "QubitUnitary":
                assert np.array_equal(circuit(U), reference(U))
            elif op.num_params == 1:
                assert np.array_equal(circuit(a), reference(a))
            elif op.num_params == 2:
                assert np.array_equal(circuit(a, b))

    def test_supported_observables(self, qubit_device_2_wires, tol):
        """Test that all supported observables work correctly"""
        
        a = 0.312

        for g, qop in qubit_device_2_wires._observable_map.items():
            log.debug("\tTesting observable %s...", g)
            assert qubit_device_2_wires.supports_observable(g)
            qubit_device_2_wires.reset()

            op = getattr(qml.ops, g)
            if op.num_wires < 0:
                wires = [0]
            else:
                wires = list(range(op.num_wires))

            @qml.qnode(qubit_device_2_wires)
            def circuit(*x):
                """Reference quantum function"""
                qml.RX(a, wires=0)
                return qml.expval(op(*x, wires=wires))

            # compare to reference result
            def reference(*x):
                """reference circuit"""
                if callable(qop):
                    # if the default.qubit is an operation accepting parameters,
                    # initialise it using the parameters generated above.
                    O = qop(*x)
                else:
                    # otherwise, the operation is simply an array.
                    O = qop

                # calculate the expected output
                out_state = np.kron(Rotx(a) @ np.array([1, 0]), np.array([1, 0]))
                expectation = out_state.conj() @ np.kron(O, np.identity(2)) @ out_state
                return expectation

            if op.num_params == 0:
                assert np.allclose(circuit(), reference(), atol=tol, rtol=0)
            elif g == "Hermitian":
                assert np.allclose(circuit(H), reference(H), atol=tol, rtol=0)

    def test_two_qubit_observable(self, qubit_device_2_wires, tol):
        """Tests expval for two-qubit observables """

        @qml.qnode(qubit_device_2_wires)
        def circuit(x, target_observable=None):
            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=0)
            qml.RZ(x[2], wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Hermitian(target_observable, wires=[0, 1]))

        target_state = 1 / np.sqrt(2) * np.array([1, 0, 0, 1])
        target_herm_op = np.outer(target_state.conj(), target_state)
        weights = np.array([0.5, 0.1, 0.2])
        expval = circuit(weights, target_observable=target_herm_op)
        assert np.isclose(expval, 0.590556, atol=tol, rtol=0)

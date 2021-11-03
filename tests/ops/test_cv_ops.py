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
Unit tests for the :mod:`pennylane.plugin.DefaultGaussian` device.
"""
# pylint: disable=protected-access,cell-var-from-loop

from scipy.linalg import block_diag
import pytest
from copy import copy

import pennylane as qml
from pennylane import numpy as np
import numpy.testing as np_testing
from pennylane.ops import cv
from pennylane.wires import Wires

s_vals = np.linspace(-3, 3, 13)
phis = np.linspace(-2 * np.pi, 2 * np.pi, 11)
mags = np.linspace(0.0, 1.0, 7)

GAUSSIAN_GATES = [
    qml.Squeezing(1.0, 1.0, wires=0),
    qml.TwoModeSqueezing(1.0, 1.0, wires=[0, 1]),
    qml.Beamsplitter(1.0, 1.0, wires=[0, 1]),
    qml.Displacement(1.0, 1.0, wires=0),
    qml.ControlledPhase(1.0, wires=[0, 1]),
    qml.Rotation(1.0, wires=0),
    qml.ControlledAddition(1.0, wires=[0, 1]),
    qml.InterferometerUnitary(
        np.array(
            [
                [0.83645892 - 0.40533293j, -0.20215326 + 0.30850569j],
                [-0.23889780 - 0.28101519j, -0.88031770 - 0.29832709j],
            ]
        ),
        wires=[0, 1],
    ),
    qml.QuadraticPhase(1.0, wires=0),
]

GAUSSIAN_OBS = [
    qml.QuadOperator(1.0, wires=0),
    qml.P(wires=0),
    qml.X(wires=0),
    qml.NumberOperator(wires=0),
    qml.PolyXP(np.array([0.1, 0.2, 0.3]), wires=0),
]


class TestGaussian:
    """Tests the continuous variable based operations."""

    @pytest.mark.parametrize("op", GAUSSIAN_GATES)
    def test_gradient_recipe_operations(self, op, tol):
        "Heisenberg picture adjoint actions of CV Operations."

        if op.grad_recipe is not None:
            # compare gradient recipe to numerical gradient
            h = 1e-7
            U = op.heisenberg_tr(op.wires)
            temp = copy(op.data)

            for k in range(op.num_params):
                D = op.heisenberg_pd(k)  # using the recipe
                # using finite difference
                op.data[k] += h
                Up = op.heisenberg_tr(op.wires)
                op.data = temp
                G = (Up - U) / h
                assert D == pytest.approx(G, abs=tol)

    @pytest.mark.parametrize("op", GAUSSIAN_OBS)
    def test_ev_order(self, op, tol):
        """Test that for observables, ev_order equals the number of dimensions of the H-rep array"""
        if op.name != "PolyXP":  # PolyXP is an exception
            Q = op.heisenberg_obs(op.wires)
            assert Q.ndim == op.ev_order


class TestGaussianHeisenbergReps:
    """Test the Heisenberg representation of Gaussian gates"""

    @pytest.mark.parametrize("phi", phis)
    def test_rotation_heisenberg(self, phi):
        """ops: Tests the Heisenberg representation of the Rotation gate."""
        matrix = cv.Rotation._heisenberg_rep([phi])
        true_matrix = np.array(
            [[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]]
        )
        assert np.allclose(matrix, true_matrix)

    @pytest.mark.parametrize("phi", phis)
    @pytest.mark.parametrize("mag", mags)
    def test_squeezing_heisenberg(self, phi, mag):
        """ops: Tests the Heisenberg representation of the Squeezing gate."""
        r = mag
        matrix = cv.Squeezing._heisenberg_rep([r, phi])
        true_matrix = np.array(
            [
                [1, 0, 0],
                [0, np.cosh(r) - np.cos(phi) * np.sinh(r), -np.sin(phi) * np.sinh(r)],
                [0, -np.sin(phi) * np.sinh(r), np.cosh(r) + np.cos(phi) * np.sinh(r)],
            ]
        )
        assert np.allclose(matrix, true_matrix)

    @pytest.mark.parametrize("phi", phis)
    @pytest.mark.parametrize("mag", mags)
    def test_displacement_heisenberg(self, phi, mag):
        """ops: Tests the Heisenberg representation of the Displacement gate."""
        r = mag
        hbar = 2
        matrix = cv.Displacement._heisenberg_rep([r, phi])
        true_matrix = np.array(
            [
                [1, 0, 0],
                [np.sqrt(2 * hbar) * r * np.cos(phi), 1, 0],
                [np.sqrt(2 * hbar) * r * np.sin(phi), 0, 1],
            ]
        )
        assert np.allclose(matrix, true_matrix)

    @pytest.mark.parametrize("phi", phis)
    @pytest.mark.parametrize("theta", phis)
    def test_beamsplitter_heisenberg(self, phi, theta):
        """ops: Tests the Heisenberg representation of the Beamsplitter gate."""
        matrix = cv.Beamsplitter._heisenberg_rep([theta, phi])
        true_matrix = np.array(
            [
                [1, 0, 0, 0, 0],
                [0, np.cos(theta), 0, -np.cos(phi) * np.sin(theta), -np.sin(phi) * np.sin(theta)],
                [0, 0, np.cos(theta), np.sin(phi) * np.sin(theta), -np.cos(phi) * np.sin(theta)],
                [0, np.cos(phi) * np.sin(theta), -np.sin(phi) * np.sin(theta), np.cos(theta), 0],
                [0, np.sin(phi) * np.sin(theta), np.cos(phi) * np.sin(theta), 0, np.cos(theta)],
            ]
        )
        assert np.allclose(matrix, true_matrix)

    @pytest.mark.parametrize("phi", phis)
    @pytest.mark.parametrize("mag", mags)
    def test_two_mode_squeezing_heisenberg(self, phi, mag):
        """ops: Tests the Heisenberg representation of the Beamsplitter gate."""
        r = mag
        matrix = cv.TwoModeSqueezing._heisenberg_rep([r, phi])
        true_matrix = np.array(
            [
                [1, 0, 0, 0, 0],
                [0, np.cosh(r), 0, np.cos(phi) * np.sinh(r), np.sin(phi) * np.sinh(r)],
                [0, 0, np.cosh(r), np.sin(phi) * np.sinh(r), -np.cos(phi) * np.sinh(r)],
                [0, np.cos(phi) * np.sinh(r), np.sin(phi) * np.sinh(r), np.cosh(r), 0],
                [0, np.sin(phi) * np.sinh(r), -np.cos(phi) * np.sinh(r), 0, np.cosh(r)],
            ]
        )
        assert np.allclose(matrix, true_matrix)

    @pytest.mark.parametrize("s", s_vals)
    def test_quadratic_phase_heisenberg(self, s):
        """ops: Tests the Heisenberg representation of the QuadraticPhase gate."""
        matrix = cv.QuadraticPhase._heisenberg_rep([s])
        true_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, s, 1]])
        assert np.allclose(matrix, true_matrix)

    @pytest.mark.parametrize("s", s_vals)
    def test_controlled_addition_heisenberg(self, s):
        """ops: Tests the Heisenberg representation of ControlledAddition gate."""
        matrix = cv.ControlledAddition._heisenberg_rep([s])
        true_matrix = np.array(
            [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, -s], [0, s, 0, 1, 0], [0, 0, 0, 0, 1]]
        )
        assert np.allclose(matrix, true_matrix)

    @pytest.mark.parametrize("s", s_vals)
    def test_controlled_phase_heisenberg(self, s):
        """Tests the Heisenberg representation of the ControlledPhase gate."""
        matrix = cv.ControlledPhase._heisenberg_rep([s])
        true_matrix = np.array(
            [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, s, 0], [0, 0, 0, 1, 0], [0, s, 0, 0, 1]]
        )
        assert np.allclose(matrix, true_matrix)


class TestNonGaussian:
    """Tests that non-Gaussian gates are properly handled."""

    @pytest.mark.parametrize("gate", [cv.Kerr, cv.CrossKerr, cv.CubicPhase])
    def test_heisenberg_rep_nonguassian(self, gate):
        """ops: Tests that the `_heisenberg_rep` for a non-Gaussian gates is
        None
        """
        assert gate._heisenberg_rep(0.1) is None

    def test_heisenberg_transformation_nongaussian(self):
        """ops: Tests that proper exceptions are raised if we try to call the
        Heisenberg transformation of non-Gaussian gates."""
        op = cv.Kerr
        with pytest.raises(RuntimeError, match=r"not a Gaussian operation"):
            op_ = op(0.1, wires=range(op.num_wires))
            op_.heisenberg_tr(Wires(range(op.num_wires)))

        op = cv.CrossKerr
        with pytest.raises(RuntimeError):
            op_ = op(0.1, wires=range(op.num_wires))
            op_.heisenberg_tr(Wires(range(op.num_wires)))

        op = cv.CubicPhase
        with pytest.raises(RuntimeError):
            op_ = op(0.1, wires=range(op.num_wires))
            op_.heisenberg_tr(Wires(range(op.num_wires)))


class TestAdjoint:
    """Test the adjoint method"""

    @pytest.mark.parametrize(
        "op,size",
        [
            (cv.Squeezing(0.123, -0.456, wires=1), 3),
            (cv.Squeezing(0.668, 10.0, wires=0), 3),  # phi > 2pi
            (cv.Squeezing(1.992, -9.782, wires=0), 3),  # phi < -2pi
            (cv.Rotation(2.005, wires=1), 3),
            (cv.Rotation(-1.365, wires=1), 3),
            (cv.Displacement(2.841, 0.456, wires=0), 3),
            (cv.Displacement(3.142, -7.221, wires=0), 3),  # phi < -2pi
            (cv.Displacement(2.004, 8.673, wires=0), 3),  # phi > 2pi
            (cv.Beamsplitter(0.456, -0.789, wires=[0, 2]), 5),
            (cv.TwoModeSqueezing(2.532, 1.778, wires=[1, 2]), 5),
            (
                cv.InterferometerUnitary(
                    np.array([[1, 1], [1, -1]]) * -1.0j / np.sqrt(2.0), wires=1
                ),
                5,
            ),
            (cv.ControlledAddition(2.551, wires=[0, 2]), 5),
            (cv.ControlledPhase(2.189, wires=[3, 1]), 5),
        ],
    )
    def test_adjoint_cv_ops(self, op, size, tol):
        """Verify that the product of the heisenberg representation of an operator and its adjoint
        is the identity"""
        op_d = op.adjoint()
        op_heis = op._heisenberg_rep(op.parameters)
        op_d_heis = op_d._heisenberg_rep(op_d.parameters)
        res1 = np.dot(op_heis, op_d_heis)
        res2 = np.dot(op_d_heis, op_heis)
        np_testing.assert_allclose(res1, np.eye(size), atol=tol)
        np_testing.assert_allclose(res2, np.eye(size), atol=tol)
        assert op.wires == op_d.wires

    @pytest.mark.parametrize(
        "op",
        [
            cv.CrossKerr(-1.724, wires=[2, 0]),
            cv.CubicPhase(0.997, wires=2),
            cv.Kerr(2.568, wires=2),
        ],
    )
    def test_adjoint_no_heisenberg_rep_defined(self, op, tol):
        """Verify that for some operations, the adjoint simply negates the parameters."""
        op_d = op.adjoint()
        assert op.parameters[0] == -op_d.parameters[0]


class TestInverse:
    """Test the inverse of Gaussian operations"""

    @pytest.mark.parametrize("op", GAUSSIAN_GATES)
    def test_inverse(self, op, tol):
        """Check that the heisenberg representation and the inverse heisenberg representation
        multiply to the identity"""
        V = op.heisenberg_tr(op.wires, inverse=True)
        U = op.heisenberg_tr(op.wires)
        I = np.eye(*U.shape)

        # first row is always (1,0,0...)
        assert np.all(U[0, :] == I[:, 0])

        assert np.linalg.norm(U @ V - I) == pytest.approx(0, abs=tol)
        assert np.linalg.norm(V @ U - I) == pytest.approx(0, abs=tol)


label_data = [
    (cv.Rotation(1.2345, wires=0), "R", "R\n(1.23)", "R⁻¹\n(1)"),
    (cv.Squeezing(1.234, 2.345, wires=0), "S", "S\n(1.23,2.35)", "S⁻¹\n(1,2)"),
    (cv.Displacement(1.234, 2.345, wires=0), "D", "D\n(1.23,2.35)", "D⁻¹\n(1,2)"),
    (cv.Beamsplitter(1.234, 2.345, wires=(0, 1)), "BS", "BS\n(1.23,2.35)", "BS⁻¹\n(1,2)"),
    (cv.TwoModeSqueezing(1.2345, 2.3456, wires=(0, 1)), "S", "S\n(1.23,2.35)", "S⁻¹\n(1,2)"),
    (cv.QuadraticPhase(1.2345, wires=0), "P", "P\n(1.23)", "P⁻¹\n(1)"),
    (cv.ControlledAddition(1.234, wires=(0, 1)), "X", "X\n(1.23)", "X⁻¹\n(1)"),
    (cv.ControlledPhase(1.2345, wires=(0, 1)), "Z", "Z\n(1.23)", "Z⁻¹\n(1)"),
    (cv.Kerr(1.234, wires=0), "Kerr", "Kerr\n(1.23)", "Kerr⁻¹\n(1)"),
    (cv.CrossKerr(1.234, wires=(0, 1)), "CrossKerr", "CrossKerr\n(1.23)", "CrossKerr⁻¹\n(1)"),
    (cv.CubicPhase(1.234, wires=0), "V", "V\n(1.23)", "V⁻¹\n(1)"),
    (cv.InterferometerUnitary(np.eye(2), wires=(0)), "U", "U", "U⁻¹"),
    (cv.ThermalState(1.234, wires=0), "Thermal", "Thermal\n(1.23)", "Thermal⁻¹\n(1)"),
    (
        cv.GaussianState(np.array([[2, 0], [0, 2]]), np.array([1, 2]), wires=[1]),
        "Gaussian",
        "Gaussian",
        "Gaussian⁻¹",
    ),
    (cv.FockState(7, wires=0), "|7⟩", "|7⟩", "|7⟩"),
    (cv.FockStateVector([1, 2, 3], wires=(0, 1, 2)), "|123⟩", "|123⟩", "|123⟩"),
    (cv.NumberOperator(wires=0), "n", "n", None),
    (cv.TensorN(wires=(0, 1, 2)), "n⊗n⊗n", "n⊗n⊗n", None),
    (cv.QuadOperator(1.234, wires=0), "cos(φ)x\n+sin(φ)p", "cos(1.23)x\n+sin(1.23)p", None),
    (cv.FockStateProjector([1, 2, 3], wires=(0, 1, 2)), "|123⟩⟨123|", "|123⟩⟨123|", None),
]


label_data_base_name = [
    (cv.FockState(7, wires=0), "name", "name\n(7)"),
    (cv.FockStateVector([1, 2, 3], wires=(0, 1, 2)), "name", "name"),
    (cv.TensorN(wires=(0, 1, 2)), "name", "name"),
    (cv.QuadOperator(1.234, wires=0), "name", "name\n(1.23)"),
    (cv.FockStateProjector([1, 2, 3], wires=(0, 1, 2)), "name", "name"),
]


class TestLabel:
    @pytest.mark.parametrize("op, label1, label2, label3", label_data)
    def test_label_method(self, op, label1, label2, label3):
        """Tests the label method for formatting in drawings"""
        assert op.label() == label1
        assert op.label(decimals=2) == label2

        # exclude observables
        if label3 is not None:
            op.inv()
            assert op.label(decimals=0) == label3
            op.inv()

    @pytest.mark.parametrize("op, label1, label2", label_data_base_name)
    def test_label_base_name(self, op, label1, label2):
        """Test label method with custom base label."""

        assert op.label(base_label="name") == label1
        assert op.label(base_label="name", decimals=2) == label2

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

import pennylane
from pennylane import numpy as np
from pennylane.ops import cv
from pennylane.wires import Wires

s_vals = np.linspace(-3, 3, 13)
phis = np.linspace(-2 * np.pi, 2 * np.pi, 11)
mags = np.linspace(0.0, 1.0, 7)

class TestCV:
    """Tests the continuous variable based operations."""

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
                [
                    0,
                    np.cos(theta),
                    0,
                    -np.cos(phi) * np.sin(theta),
                    -np.sin(phi) * np.sin(theta),
                ],
                [
                    0,
                    0,
                    np.cos(theta),
                    np.sin(phi) * np.sin(theta),
                    -np.cos(phi) * np.sin(theta),
                ],
                [
                    0,
                    np.cos(phi) * np.sin(theta),
                    -np.sin(phi) * np.sin(theta),
                    np.cos(theta),
                    0,
                ],
                [
                    0,
                    np.sin(phi) * np.sin(theta),
                    np.cos(phi) * np.sin(theta),
                    0,
                    np.cos(theta),
                ],
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
        """ops: Tests the Heisenberg representation of ControlledAddition gate.
        """
        matrix = cv.ControlledAddition._heisenberg_rep([s])
        true_matrix = np.array(
            [
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, -s],
                [0, s, 0, 1, 0],
                [0, 0, 0, 0, 1],
            ]
        )
        assert np.allclose(matrix, true_matrix)


    @pytest.mark.parametrize("s", s_vals)
    def test_controlled_phase_heisenberg(self, s):
        """Tests the Heisenberg representation of the ControlledPhase gate."""
        matrix = cv.ControlledPhase._heisenberg_rep([s])
        true_matrix = np.array(
            [
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, s, 0],
                [0, 0, 0, 1, 0],
                [0, s, 0, 0, 1],
            ]
        )
        assert np.allclose(matrix, true_matrix)


class TestNonGaussian:
    """Tests that non-Gaussian gates are properly handled."""

    @pytest.mark.parametrize("gate", [cv.Kerr, cv.CrossKerr, cv.CubicPhase])
    def test_heisenberg_rep_nonguassian(self, gate):
        """ops: Tests that the `_heisenberg_rep` for a non-Gaussian gates is
        None
        """
        assert gate._heisenberg_rep(*[0.1] * gate.num_params) is None

    def test_heisenberg_transformation_nongaussian(self):
        """ops: Tests that proper exceptions are raised if we try to call the
        Heisenberg transformation of non-Gaussian gates."""
        op = cv.Kerr
        with pytest.raises(RuntimeError, match=r"not a Gaussian operation"):
            op_ = op(*[0.1] * op.num_params, wires=range(op.num_wires))
            op_.heisenberg_tr(Wires(range(op.num_wires)))

        op = cv.CrossKerr
        with pytest.raises(RuntimeError):
            op_ = op(*[0.1] * op.num_params, wires=range(op.num_wires))
            op_.heisenberg_tr(Wires(range(op.num_wires)))

        op = cv.CubicPhase
        with pytest.raises(RuntimeError):
            op_ = op(*[0.1] * op.num_params, wires=range(op.num_wires))
            op_.heisenberg_tr(Wires(range(op.num_wires)))

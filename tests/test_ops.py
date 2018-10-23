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
Unit tests for the :mod:`openqml.plugin.DefaultGaussian` device.
"""
# pylint: disable=protected-access,cell-var-from-loop
import unittest
import inspect
import logging as log

from openqml import numpy as np
from scipy.linalg import block_diag

from defaults import openqml as qm, BaseTest
from openqml.ops import cv


log.getLogger('defaults')

hbar = 2
phis = np.linspace(-2 * np.pi, 2 * np.pi, 11)
mags = np.linspace(0., 1., 7)
s_vals = np.linspace(-3,3,13)

class TestHeisenberg(BaseTest):
    """Tests for the Heisenberg representation of gates."""

    def test_rotation_heisenberg(self):
        """Tests the Heisenberg representation of the Rotation gate."""
        self.logTestName()

        for phi in phis:
            matrix = cv.Rotation._heisenberg_rep([phi])
            true_matrix = np.array([[1, 0, 0],
                                    [0, np.cos(phi), -np.sin(phi)],
                                    [0, np.sin(phi), np.cos(phi)]])
            self.assertAllAlmostEqual(matrix, true_matrix, delta=self.tol)


    def test_squeezing_heisenberg(self):
        """Tests the Heisenberg representation of the Squeezing gate."""
        self.logTestName()

        for r in mags:
            for phi in phis:
                matrix = cv.Squeezing._heisenberg_rep([r,phi])
                true_matrix = np.array([[1, 0, 0],
                                        [0, np.cosh(r) - np.cos(phi) * np.sinh(r), - np.sin(phi) * np.sinh(r)],
                                        [0, -np.sin(phi) * np.sinh(r), np.cosh(r) + np.cos(phi) * np.sinh(r)]])
                self.assertAllAlmostEqual(matrix, true_matrix, delta=self.tol)


    def test_displacement_heisenberg(self):
        """Tests the Heisenberg representation of the Displacement gate."""
        self.logTestName()

        for r in mags:
            for phi in phis:
                matrix = cv.Displacement._heisenberg_rep([r,phi])
                true_matrix = np.array([[1, 0, 0],
                                        [np.sqrt(2 * hbar) * r * np.cos(phi), 1, 0],
                                        [np.sqrt(2 * hbar) * r * np.sin(phi), 0, 1]])
                self.assertAllAlmostEqual(matrix, true_matrix, delta=self.tol)


    def test_beamsplitter_heisenberg(self):
        """Tests the Heisenberg representation of the Beamsplitter gate."""
        self.logTestName()

        for theta in phis:
            for phi in phis:
                matrix = cv.Beamsplitter._heisenberg_rep([theta,phi])
                true_matrix = np.array([[1, 0, 0, 0, 0],
                                        [0, np.cos(theta), 0, -np.cos(phi) * np.sin(theta), -np.sin(phi) * np.sin(theta)],
                                        [0, 0, np.cos(theta), np.sin(phi) * np.sin(theta), -np.cos(phi) * np.sin(theta)],
                                        [0, np.cos(phi) * np.sin(theta), -np.sin(phi) * np.sin(theta), np.cos(theta), 0],
                                        [0, np.sin(phi) * np.sin(theta), np.cos(phi) * np.sin(theta), 0, np.cos(theta)]])
                self.assertAllAlmostEqual(matrix, true_matrix, delta=self.tol)


    def test_two_mode_squeezing_heisenberg(self):
        """Tests the Heisenberg representation of the Beamsplitter gate."""
        self.logTestName()

        for r in mags:
            for phi in phis:
                matrix = cv.TwoModeSqueezing._heisenberg_rep([r,phi])
                true_matrix = np.array([[1, 0, 0, 0, 0],
                                        [0, np.cosh(r), 0, np.cos(phi) * np.sinh(r), np.sin(phi) * np.sinh(r)],
                                        [0, 0, np.cosh(r), np.sin(phi) * np.sinh(r), -np.cos(phi) * np.sinh(r)],
                                        [0, np.cos(phi) * np.sinh(r), np.sin(phi) * np.sinh(r), np.cosh(r), 0],
                                        [0, np.sin(phi) * np.sinh(r), -np.cos(phi) * np.sinh(r), 0, np.cosh(r)]])
                self.assertAllAlmostEqual(matrix, true_matrix, delta=self.tol)


    def test_quadratic_phase_heisenberg(self):
        """Tests the Heisenberg representation of the QuadraticPhase gate."""
        self.logTestName()

        for s in s_vals:
            matrix = cv.QuadraticPhase._heisenberg_rep([s])
            true_matrix = np.array([[1, 0, 0],
                                    [0, 1, 0],
                                    [0, s, 1]])
            self.assertAllAlmostEqual(matrix, true_matrix, delta=self.tol)


    def test_controlled_addition_heisenberg(self):
        """Tests the Heisenberg representation of the ControlledAddition gate."""
        self.logTestName()

        for s in s_vals:
            matrix = cv.ControlledAddition._heisenberg_rep([s])
            true_matrix = np.array([[1, 0, 0, 0, 0],
                                    [0, 1, 0, 0, 0],
                                    [0, 0, 1, 0, -s],
                                    [0, s, 0, 1, 0],
                                    [0, 0, 0, 0, 1]])
            self.assertAllAlmostEqual(matrix, true_matrix, delta=self.tol)


    def test_controlled_phase_heisenberg(self):
        """Tests the Heisenberg representation of the ControlledPhase gate."""
        self.logTestName()

        for s in s_vals:
            matrix = cv.ControlledPhase._heisenberg_rep([s])
            true_matrix = np.array([[1, 0, 0, 0, 0],
                                    [0, 1, 0, 0, 0],
                                    [0, 0, 1, s, 0],
                                    [0, 0, 0, 1, 0],
                                    [0, s, 0, 0, 1]])
            self.assertAllAlmostEqual(matrix, true_matrix, delta=self.tol)


if __name__ == '__main__':
    print('Testing OpenQML version ' + qm.version() + ', default.gaussian plugin.')
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (TestHeisenberg,):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)
    unittest.TextTestRunner().run(suite)

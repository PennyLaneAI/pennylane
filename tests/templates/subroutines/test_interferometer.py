# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
Tests for the Interferometer template.
"""

import numpy as np
import pytest

import pennylane as qp
from pennylane.wires import Wires


class TestInterferometer:
    """Tests for the Interferometer from the pennylane.template.layers module."""

    def test_clements_beamsplitter_convention(self):
        """test the beamsplitter convention"""
        N = 2
        wires = range(N)

        theta = [0.321]
        phi = [0.234]
        varphi = [0.42342, 0.1121]

        op_rect = qp.Interferometer(
            theta, phi, varphi, mesh="rectangular", beamsplitter="clements", wires=wires
        )

        op_tria = qp.Interferometer(
            theta, phi, varphi, mesh="triangular", beamsplitter="clements", wires=wires
        )

        for rec in [op_rect.decomposition(), op_tria.decomposition()]:
            assert len(rec) == 4

            assert isinstance(rec[0], qp.Rotation)
            assert rec[0].parameters == phi

            assert isinstance(rec[1], qp.Beamsplitter)
            assert rec[1].parameters == [theta[0], 0]

            assert isinstance(rec[2], qp.Rotation)
            assert rec[2].parameters == [varphi[0]]

            assert isinstance(rec[3], qp.Rotation)
            assert rec[3].parameters == [varphi[1]]

    def test_one_mode(self, tol):
        """Test that a one mode interferometer correctly gives a rotation gate"""
        varphi = [0.42342]

        op = qp.Interferometer(theta=[], phi=[], varphi=varphi, wires=0)
        rec = op.decomposition()

        assert len(rec) == 1
        assert isinstance(rec[0], qp.Rotation)
        assert np.allclose(rec[0].parameters, varphi, atol=tol)

    def test_two_mode_rect(self):
        """Test that a two mode interferometer using the rectangular mesh
        correctly gives a beamsplitter+rotation gate"""
        N = 2
        wires = range(N)

        theta = [0.321]
        phi = [0.234]
        varphi = [0.42342, 0.1121]

        op = qp.Interferometer(theta, phi, varphi, wires=wires)
        rec = op.decomposition()

        isinstance(rec[0], qp.Beamsplitter)
        assert rec[0].parameters == theta + phi

        assert isinstance(rec[1], qp.Rotation)
        assert rec[1].parameters == [varphi[0]]

        assert isinstance(rec[2], qp.Rotation)
        assert rec[2].parameters == [varphi[1]]

    def test_two_mode_triangular(self):
        """Test that a two mode interferometer using the triangular mesh
        correctly gives a beamsplitter+rotation gate"""
        N = 2
        wires = range(N)

        theta = [0.321]
        phi = [0.234]
        varphi = [0.42342, 0.1121]

        op = qp.Interferometer(theta, phi, varphi, mesh="triangular", wires=wires)
        rec = op.decomposition()

        assert len(rec) == 3

        assert isinstance(rec[0], qp.Beamsplitter)
        assert rec[0].parameters == theta + phi

        assert isinstance(rec[1], qp.Rotation)
        assert rec[1].parameters == [varphi[0]]

        assert isinstance(rec[2], qp.Rotation)
        assert rec[2].parameters == [varphi[1]]

    def test_three_mode(self):
        """Test that a three mode interferometer using either mesh gives the correct gates"""
        N = 3
        wires = range(N)

        theta = [0.321, 0.4523, 0.21321]
        phi = [0.234, 0.324, 0.234]
        varphi = [0.42342, 0.234, 0.1121]

        op_rect = qp.Interferometer(theta, phi, varphi, wires=wires, mesh="rectangular")
        op_tria = qp.Interferometer(theta, phi, varphi, wires=wires, mesh="triangular")

        # Test rectangular mesh
        rec = op_rect.decomposition()
        assert len(rec) == 6

        expected_bs_wires = [[0, 1], [1, 2], [0, 1]]

        for idx, op in enumerate(rec[:3]):
            assert isinstance(op, qp.Beamsplitter)
            assert op.parameters == [theta[idx], phi[idx]]
            assert op.wires == Wires(expected_bs_wires[idx])

        for idx, op in enumerate(rec[3:]):
            assert isinstance(op, qp.Rotation)
            assert op.parameters == [varphi[idx]]
            assert op.wires == Wires([idx])

        # Test triangular mesh
        rec = op_tria.decomposition()
        assert len(rec) == 6

        expected_bs_wires = [[1, 2], [0, 1], [1, 2]]

        for idx, op in enumerate(rec[:3]):
            assert isinstance(op, qp.Beamsplitter)
            assert op.parameters == [theta[idx], phi[idx]]
            assert op.wires == Wires(expected_bs_wires[idx])

        for idx, op in enumerate(rec[3:]):
            assert isinstance(op, qp.Rotation)
            assert op.parameters == [varphi[idx]]
            assert op.wires == Wires([idx])

    def test_four_mode_rect(self):
        """Test that a 4 mode interferometer using rectangular mesh gives the correct gates"""
        N = 4
        wires = range(N)

        theta = [0.321, 0.4523, 0.21321, 0.123, 0.5234, 1.23]
        phi = [0.234, 0.324, 0.234, 1.453, 1.42341, -0.534]
        varphi = [0.42342, 0.234, 0.4523, 0.1121]

        op = qp.Interferometer(theta, phi, varphi, wires=wires)
        rec = op.decomposition()

        assert len(rec) == 10

        expected_bs_wires = [[0, 1], [2, 3], [1, 2], [0, 1], [2, 3], [1, 2]]

        for idx, op in enumerate(rec[:6]):
            assert isinstance(op, qp.Beamsplitter)
            assert op.parameters == [theta[idx], phi[idx]]
            assert op.wires == Wires(expected_bs_wires[idx])

        for idx, op in enumerate(rec[6:]):
            assert isinstance(op, qp.Rotation)
            assert op.parameters == [varphi[idx]]
            assert op.wires == Wires([idx])

    def test_four_mode_triangular(self):
        """Test that a 4 mode interferometer using triangular mesh gives the correct gates"""
        N = 4
        wires = range(N)

        theta = [0.321, 0.4523, 0.21321, 0.123, 0.5234, 1.23]
        phi = [0.234, 0.324, 0.234, 1.453, 1.42341, -0.534]
        varphi = [0.42342, 0.234, 0.4523, 0.1121]

        op = qp.Interferometer(theta, phi, varphi, wires=wires, mesh="triangular")
        rec = op.decomposition()

        assert len(rec) == 10

        expected_bs_wires = [[2, 3], [1, 2], [0, 1], [2, 3], [1, 2], [2, 3]]

        for idx, op in enumerate(rec[:6]):
            assert isinstance(op, qp.Beamsplitter)
            assert op.parameters == [theta[idx], phi[idx]]
            assert op.wires == Wires(expected_bs_wires[idx])

        for idx, op in enumerate(rec[6:]):
            assert isinstance(op, qp.Rotation)
            assert op.parameters == [varphi[idx]]
            assert op.wires == Wires([idx])

    @pytest.mark.parametrize(
        "n_wires, expected",
        [
            (3, [(3,), (3,), (3,)]),
            (1, [(0,), (0,), (1,)]),
            (2, [(1,), (1,), (2,)]),
        ],
    )
    def test_shapes(self, n_wires, tol, expected):
        """Test that the shape method returns the correct shapes for
        the weight tensors"""

        shapes = qp.Interferometer.shape(n_wires)
        assert np.allclose(shapes, expected, atol=tol, rtol=0)

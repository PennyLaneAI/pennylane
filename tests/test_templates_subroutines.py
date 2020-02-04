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
Unit tests for the :mod:`pennylane.template.subroutines` module.
Integration tests should be placed into ``test_templates.py``.
"""
# pylint: disable=protected-access,cell-var-from-loop
import pytest
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates.subroutines import Interferometer


class TestInterferometer:
    """Tests for the Interferometer from the pennylane.template.layers module."""

    def test_invalid_mesh_exception(self):
        """Test that Interferometer() raises correct exception when mesh not recognized."""
        dev = qml.device('default.gaussian', wires=1)
        varphi = [0.42342]

        @qml.qnode(dev)
        def circuit(varphi, mesh=None):
            Interferometer(theta=[], phi=[], varphi=varphi, mesh=mesh, wires=0)
            return qml.expval(qml.NumberOperator(0))

        with pytest.raises(ValueError, match="Mesh option"):
            circuit(varphi, mesh='a')

    def test_invalid_mesh_exception(self):
        """Test that Interferometer() raises correct exception when beamsplitter not recognized."""
        dev = qml.device('default.gaussian', wires=1)
        varphi = [0.42342]

        @qml.qnode(dev)
        def circuit(varphi, bs=None):
            Interferometer(theta=[], phi=[], varphi=varphi, beamsplitter=bs, wires=0)
            return qml.expval(qml.NumberOperator(0))

        with pytest.raises(ValueError, match="did not recognize option"):
            circuit(varphi, bs='a')

    def test_clements_beamsplitter_convention(self, tol):
        """test the beamsplitter convention"""
        N = 2
        wires = range(N)

        theta = [0.321]
        phi = [0.234]
        varphi = [0.42342, 0.1121]

        with qml.utils.OperationRecorder() as rec_rect:
            Interferometer(theta, phi, varphi, mesh='rectangular', beamsplitter='clements', wires=wires)

        with qml.utils.OperationRecorder() as rec_tria:
            Interferometer(theta, phi, varphi, mesh='triangular', beamsplitter='clements', wires=wires)

        for rec in [rec_rect, rec_tria]:

            assert len(rec.queue) == 4

            assert isinstance(rec.queue[0], qml.Rotation)
            assert rec.queue[0].parameters == phi

            assert isinstance(rec.queue[1], qml.Beamsplitter)
            assert rec.queue[1].parameters == [theta[0], 0]

            assert isinstance(rec.queue[2], qml.Rotation)
            assert rec.queue[2].parameters == [varphi[0]]

            assert isinstance(rec.queue[3], qml.Rotation)
            assert rec.queue[3].parameters == [varphi[1]]

    def test_one_mode(self, tol):
        """Test that a one mode interferometer correctly gives a rotation gate"""
        varphi = [0.42342]

        with qml.utils.OperationRecorder() as rec:
            Interferometer(theta=[], phi=[], varphi=varphi, wires=0)

        assert len(rec.queue) == 1
        assert isinstance(rec.queue[0], qml.Rotation)
        assert np.allclose(rec.queue[0].parameters, varphi, atol=tol)

    def test_two_mode_rect(self, tol):
        """Test that a two mode interferometer using the rectangular mesh
        correctly gives a beamsplitter+rotation gate"""
        N = 2
        wires = range(N)

        theta = [0.321]
        phi = [0.234]
        varphi = [0.42342, 0.1121]

        with qml.utils.OperationRecorder() as rec:
            Interferometer(theta, phi, varphi, wires=wires)

        isinstance(rec.queue[0], qml.Beamsplitter)
        assert rec.queue[0].parameters == theta+phi

        assert isinstance(rec.queue[1], qml.Rotation)
        assert rec.queue[1].parameters == [varphi[0]]

        assert isinstance(rec.queue[2], qml.Rotation)
        assert rec.queue[2].parameters == [varphi[1]]

    def test_two_mode_triangular(self, tol):
        """Test that a two mode interferometer using the triangular mesh
        correctly gives a beamsplitter+rotation gate"""
        N = 2
        wires = range(N)

        theta = [0.321]
        phi = [0.234]
        varphi = [0.42342, 0.1121]

        with qml.utils.OperationRecorder() as rec:
            Interferometer(theta, phi, varphi, mesh='triangular', wires=wires)

        assert len(rec.queue) == 3

        assert isinstance(rec.queue[0], qml.Beamsplitter)
        assert rec.queue[0].parameters == theta+phi

        assert isinstance(rec.queue[1], qml.Rotation)
        assert rec.queue[1].parameters == [varphi[0]]

        assert isinstance(rec.queue[2], qml.Rotation)
        assert rec.queue[2].parameters == [varphi[1]]

    def test_three_mode(self, tol):
        """Test that a three mode interferometer using either mesh gives the correct gates"""
        N = 3
        wires = range(N)

        theta = [0.321, 0.4523, 0.21321]
        phi = [0.234, 0.324, 0.234]
        varphi = [0.42342, 0.234, 0.1121]

        with qml.utils.OperationRecorder() as rec_rect:
            Interferometer(theta, phi, varphi, wires=wires)

        with qml.utils.OperationRecorder() as rec_tria:
            Interferometer(theta, phi, varphi, wires=wires)

        for rec in [rec_rect, rec_tria]:
            # test both meshes (both give identical results for the 3 mode case).
            assert len(rec.queue) == 6

            expected_bs_wires = [[0, 1], [1, 2], [0, 1]]

            for idx, op in enumerate(rec_rect.queue[:3]):
                assert isinstance(op, qml.Beamsplitter)
                assert op.parameters == [theta[idx], phi[idx]]
                assert op.wires == expected_bs_wires[idx]

            for idx, op in enumerate(rec.queue[3:]):
                assert isinstance(op, qml.Rotation)
                assert op.parameters == [varphi[idx]]
                assert op.wires == [idx]

    def test_four_mode_rect(self, tol):
        """Test that a 4 mode interferometer using rectangular mesh gives the correct gates"""
        N = 4
        wires = range(N)

        theta = [0.321, 0.4523, 0.21321, 0.123, 0.5234, 1.23]
        phi = [0.234, 0.324, 0.234, 1.453, 1.42341, -0.534]
        varphi = [0.42342, 0.234, 0.4523, 0.1121]

        with qml.utils.OperationRecorder() as rec:
            Interferometer(theta, phi, varphi, wires=wires)

        assert len(rec.queue) == 10

        expected_bs_wires = [[0, 1], [2, 3], [1, 2], [0, 1], [2, 3], [1, 2]]

        for idx, op in enumerate(rec.queue[:6]):
            assert isinstance(op, qml.Beamsplitter)
            assert op.parameters == [theta[idx], phi[idx]]
            assert op.wires == expected_bs_wires[idx]

        for idx, op in enumerate(rec.queue[6:]):
            assert isinstance(op, qml.Rotation)
            assert op.parameters == [varphi[idx]]
            assert op.wires == [idx]

    def test_four_mode_triangular(self, tol):
        """Test that a 4 mode interferometer using triangular mesh gives the correct gates"""
        N = 4
        wires = range(N)

        theta = [0.321, 0.4523, 0.21321, 0.123, 0.5234, 1.23]
        phi = [0.234, 0.324, 0.234, 1.453, 1.42341, -0.534]
        varphi = [0.42342, 0.234, 0.4523, 0.1121]

        with qml.utils.OperationRecorder() as rec:
            Interferometer(theta, phi, varphi, mesh='triangular', wires=wires)

        assert len(rec.queue) == 10

        expected_bs_wires = [[2, 3], [1, 2], [0, 1], [2, 3], [1, 2], [2, 3]]

        for idx, op in enumerate(rec.queue[:6]):
            assert isinstance(op, qml.Beamsplitter)
            assert op.parameters == [theta[idx], phi[idx]]
            assert op.wires == expected_bs_wires[idx]

        for idx, op in enumerate(rec.queue[6:]):
            assert isinstance(op, qml.Rotation)
            assert op.parameters == [varphi[idx]]
            assert op.wires == [idx]

    def test_integration(self, tol):
        """test integration with PennyLane and gradient calculations"""
        N = 4
        wires = range(N)
        dev = qml.device('default.gaussian', wires=N)

        sq = np.array([[0.8734294, 0.96854066],
                       [0.86919454, 0.53085569],
                       [0.23272833, 0.0113988 ],
                       [0.43046882, 0.40235136]])

        theta = np.array([3.28406182, 3.0058243, 3.48940764, 3.41419504, 4.7808479, 4.47598146])
        phi = np.array([3.89357744, 2.67721355, 1.81631197, 6.11891294, 2.09716418, 1.37476761])
        varphi = np.array([0.4134863, 6.17555778, 0.80334114, 2.02400747])

        @qml.qnode(dev)
        def circuit(theta, phi, varphi):
            for w in wires:
                qml.Squeezing(sq[w][0], sq[w][1], wires=w)

            Interferometer(theta=theta, phi=phi, varphi=varphi, wires=wires)
            return [qml.expval(qml.NumberOperator(w)) for w in wires]

        res = circuit(theta, phi, varphi)
        expected = np.array([0.96852694, 0.23878521, 0.82310606, 0.16547786])
        assert np.allclose(res, expected, atol=tol)

        # compare the two methods of computing the Jacobian
        jac_A = circuit.jacobian((theta, phi, varphi), method="A")
        jac_F = circuit.jacobian((theta, phi, varphi), method="F")
        assert jac_A == pytest.approx(jac_F, abs=tol)

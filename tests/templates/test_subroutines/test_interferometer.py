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

import pennylane as qml
from pennylane.wires import Wires


class TestInterferometer:
    """Tests for the Interferometer from the pennylane.template.layers module."""

    def test_invalid_mesh_exception(self):
        """Test that Interferometer() raises correct exception when mesh not recognized."""
        dev = qml.device("default.gaussian", wires=2)
        varphi = [0.42342, 0.234]

        @qml.qnode(dev)
        def circuit(varphi, mesh=None):
            qml.Interferometer(theta=[0.21], phi=[0.53], varphi=varphi, mesh=mesh, wires=[0, 1])
            return qml.expval(qml.NumberOperator(0))

        with pytest.raises(ValueError, match="did not recognize mesh"):
            circuit(varphi, mesh="a")

    @pytest.mark.parametrize("mesh", ["rectangular", "triangular"])
    def test_invalid_beamsplitter_exception(self, mesh):
        """Test that Interferometer() raises correct exception when beamsplitter not recognized."""
        dev = qml.device("default.gaussian", wires=2)
        varphi = [0.42342, 0.234]

        @qml.qnode(dev)
        def circuit(varphi, bs=None):
            qml.Interferometer(
                theta=[0.21], phi=[0.53], varphi=varphi, beamsplitter=bs, mesh=mesh, wires=[0, 1]
            )
            return qml.expval(qml.NumberOperator(0))

        with pytest.raises(ValueError, match="did not recognize beamsplitter"):
            circuit(varphi, bs="a")

    def test_clements_beamsplitter_convention(self):
        """test the beamsplitter convention"""
        N = 2
        wires = range(N)

        theta = [0.321]
        phi = [0.234]
        varphi = [0.42342, 0.1121]

        op_rect = qml.Interferometer(
            theta, phi, varphi, mesh="rectangular", beamsplitter="clements", wires=wires
        )

        op_tria = qml.Interferometer(
            theta, phi, varphi, mesh="triangular", beamsplitter="clements", wires=wires
        )

        for rec in [op_rect.decomposition(), op_tria.decomposition()]:
            assert len(rec) == 4

            assert isinstance(rec[0], qml.Rotation)
            assert rec[0].parameters == phi

            assert isinstance(rec[1], qml.Beamsplitter)
            assert rec[1].parameters == [theta[0], 0]

            assert isinstance(rec[2], qml.Rotation)
            assert rec[2].parameters == [varphi[0]]

            assert isinstance(rec[3], qml.Rotation)
            assert rec[3].parameters == [varphi[1]]

    def test_one_mode(self, tol):
        """Test that a one mode interferometer correctly gives a rotation gate"""
        varphi = [0.42342]

        op = qml.Interferometer(theta=[], phi=[], varphi=varphi, wires=0)
        rec = op.decomposition()

        assert len(rec) == 1
        assert isinstance(rec[0], qml.Rotation)
        assert np.allclose(rec[0].parameters, varphi, atol=tol)

    def test_two_mode_rect(self):
        """Test that a two mode interferometer using the rectangular mesh
        correctly gives a beamsplitter+rotation gate"""
        N = 2
        wires = range(N)

        theta = [0.321]
        phi = [0.234]
        varphi = [0.42342, 0.1121]

        op = qml.Interferometer(theta, phi, varphi, wires=wires)
        rec = op.decomposition()

        isinstance(rec[0], qml.Beamsplitter)
        assert rec[0].parameters == theta + phi

        assert isinstance(rec[1], qml.Rotation)
        assert rec[1].parameters == [varphi[0]]

        assert isinstance(rec[2], qml.Rotation)
        assert rec[2].parameters == [varphi[1]]

    def test_two_mode_triangular(self):
        """Test that a two mode interferometer using the triangular mesh
        correctly gives a beamsplitter+rotation gate"""
        N = 2
        wires = range(N)

        theta = [0.321]
        phi = [0.234]
        varphi = [0.42342, 0.1121]

        op = qml.Interferometer(theta, phi, varphi, mesh="triangular", wires=wires)
        rec = op.decomposition()

        assert len(rec) == 3

        assert isinstance(rec[0], qml.Beamsplitter)
        assert rec[0].parameters == theta + phi

        assert isinstance(rec[1], qml.Rotation)
        assert rec[1].parameters == [varphi[0]]

        assert isinstance(rec[2], qml.Rotation)
        assert rec[2].parameters == [varphi[1]]

    def test_three_mode(self):
        """Test that a three mode interferometer using either mesh gives the correct gates"""
        N = 3
        wires = range(N)

        theta = [0.321, 0.4523, 0.21321]
        phi = [0.234, 0.324, 0.234]
        varphi = [0.42342, 0.234, 0.1121]

        op_rect = qml.Interferometer(theta, phi, varphi, wires=wires, mesh="rectangular")
        op_tria = qml.Interferometer(theta, phi, varphi, wires=wires, mesh="triangular")

        # Test rectangular mesh
        rec = op_rect.decomposition()
        assert len(rec) == 6

        expected_bs_wires = [[0, 1], [1, 2], [0, 1]]

        for idx, op in enumerate(rec[:3]):
            assert isinstance(op, qml.Beamsplitter)
            assert op.parameters == [theta[idx], phi[idx]]
            assert op.wires == Wires(expected_bs_wires[idx])

        for idx, op in enumerate(rec[3:]):
            assert isinstance(op, qml.Rotation)
            assert op.parameters == [varphi[idx]]
            assert op.wires == Wires([idx])

        # Test triangular mesh
        rec = op_tria.decomposition()
        assert len(rec) == 6

        expected_bs_wires = [[1, 2], [0, 1], [1, 2]]

        for idx, op in enumerate(rec[:3]):
            assert isinstance(op, qml.Beamsplitter)
            assert op.parameters == [theta[idx], phi[idx]]
            assert op.wires == Wires(expected_bs_wires[idx])

        for idx, op in enumerate(rec[3:]):
            assert isinstance(op, qml.Rotation)
            assert op.parameters == [varphi[idx]]
            assert op.wires == Wires([idx])

    def test_four_mode_rect(self):
        """Test that a 4 mode interferometer using rectangular mesh gives the correct gates"""
        N = 4
        wires = range(N)

        theta = [0.321, 0.4523, 0.21321, 0.123, 0.5234, 1.23]
        phi = [0.234, 0.324, 0.234, 1.453, 1.42341, -0.534]
        varphi = [0.42342, 0.234, 0.4523, 0.1121]

        op = qml.Interferometer(theta, phi, varphi, wires=wires)
        rec = op.decomposition()

        assert len(rec) == 10

        expected_bs_wires = [[0, 1], [2, 3], [1, 2], [0, 1], [2, 3], [1, 2]]

        for idx, op in enumerate(rec[:6]):
            assert isinstance(op, qml.Beamsplitter)
            assert op.parameters == [theta[idx], phi[idx]]
            assert op.wires == Wires(expected_bs_wires[idx])

        for idx, op in enumerate(rec[6:]):
            assert isinstance(op, qml.Rotation)
            assert op.parameters == [varphi[idx]]
            assert op.wires == Wires([idx])

    def test_four_mode_triangular(self):
        """Test that a 4 mode interferometer using triangular mesh gives the correct gates"""
        N = 4
        wires = range(N)

        theta = [0.321, 0.4523, 0.21321, 0.123, 0.5234, 1.23]
        phi = [0.234, 0.324, 0.234, 1.453, 1.42341, -0.534]
        varphi = [0.42342, 0.234, 0.4523, 0.1121]

        op = qml.Interferometer(theta, phi, varphi, wires=wires, mesh="triangular")
        rec = op.decomposition()

        assert len(rec) == 10

        expected_bs_wires = [[2, 3], [1, 2], [0, 1], [2, 3], [1, 2], [2, 3]]

        for idx, op in enumerate(rec[:6]):
            assert isinstance(op, qml.Beamsplitter)
            assert op.parameters == [theta[idx], phi[idx]]
            assert op.wires == Wires(expected_bs_wires[idx])

        for idx, op in enumerate(rec[6:]):
            assert isinstance(op, qml.Rotation)
            assert op.parameters == [varphi[idx]]
            assert op.wires == Wires([idx])

    def test_integration(self, tol):
        """test integration with PennyLane and gradient calculations"""
        N = 4
        wires = range(N)
        dev = qml.device("default.gaussian", wires=N)

        sq = np.array(
            [
                [0.8734294, 0.96854066],
                [0.86919454, 0.53085569],
                [0.23272833, 0.0113988],
                [0.43046882, 0.40235136],
            ]
        )

        theta = np.array([3.28406182, 3.0058243, 3.48940764, 3.41419504, 4.7808479, 4.47598146])
        phi = np.array([3.89357744, 2.67721355, 1.81631197, 6.11891294, 2.09716418, 1.37476761])
        varphi = np.array([0.4134863, 6.17555778, 0.80334114, 2.02400747])

        @qml.qnode(dev)
        def circuit(theta, phi, varphi):
            for w in wires:
                qml.Squeezing(sq[w][0], sq[w][1], wires=w)

            qml.Interferometer(theta=theta, phi=phi, varphi=varphi, wires=wires)
            return qml.expval(qml.NumberOperator(0))

        res = circuit(theta, phi, varphi)
        expected = np.array(0.96852694)
        assert np.allclose(res, expected, atol=tol)

    def test_interferometer_wrong_dim(self):
        """Integration test for the CVNeuralNetLayers method."""
        dev = qml.device("default.gaussian", wires=4)

        @qml.qnode(dev)
        def circuit(theta, phi, varphi):
            qml.Interferometer(theta=theta, phi=phi, varphi=varphi, wires=range(4))
            return qml.expval(qml.QuadX(0))

        theta = np.array([3.28406182, 3.0058243, 3.48940764, 3.41419504, 4.7808479, 4.47598146])
        phi = np.array([3.89357744, 2.67721355, 1.81631197, 6.11891294, 2.09716418, 1.37476761])
        varphi = np.array([0.4134863, 6.17555778, 0.80334114, 2.02400747])

        with pytest.raises(ValueError, match=r"Theta must be of shape \(6,\)"):
            wrong_theta = np.array([0.1, 0.2])
            circuit(wrong_theta, phi, varphi)

        with pytest.raises(ValueError, match=r"Phi must be of shape \(6,\)"):
            wrong_phi = np.array([0.1, 0.2])
            circuit(theta, wrong_phi, varphi)

        with pytest.raises(ValueError, match=r"Varphi must be of shape \(4,\)"):
            wrong_varphi = np.array([0.1, 0.2])
            circuit(theta, phi, wrong_varphi)

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

        shapes = qml.Interferometer.shape(n_wires)
        assert np.allclose(shapes, expected, atol=tol, rtol=0)

    @pytest.mark.jax
    @pytest.mark.xfail(reason="JIT not supported.")
    def test_jit(self):
        import jax
        import jax.numpy as jnp

        dev = qml.device("default.gaussian", wires=4)

        @jax.jit
        @qml.qnode(dev)
        def circuit(*params):
            qml.Interferometer(*params, wires=range(4))
            return qml.state()

        shapes = [[6], [6], [4]]
        params = []
        for shape in shapes:
            params.append(np.random.random(shape))

        params = (
            jnp.array([0.5, 0.5, 0.5, 0.5, 0.2, 0.1]),
            jnp.array([0.5, 0.5, 0.5, 0.5, 0.2, 0.1]),
            jnp.array([0.5, 0.5, 0.2, 0.1]),
        )
        circuit(*params)

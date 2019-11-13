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
Unit tests for the :mod:`pennylane.template.layers` module.
"""
# pylint: disable=protected-access,cell-var-from-loop
import pytest

import logging as log
import pennylane as qml
from pennylane import numpy as np
from pennylane.qnode import QuantumFunctionError
from pennylane.templates.layers import (Interferometer)


class TestInterferometer:
    """Tests for the Interferometer from the pennylane.template.layers module."""

    def test_exceptions(self):
        """test that exceptions are correctly raised"""
        dev = qml.device('default.gaussian', wires=1)
        varphi = [0.42342]

        @qml.qnode(dev)
        def circuit(varphi, mesh):
            Interferometer(theta=None, phi=None, varphi=varphi, mesh=mesh, wires=0)
            return qml.expval(qml.NumberOperator(0))

        with pytest.raises(QuantumFunctionError):
            circuit(varphi, 'rectangular')

        @qml.qnode(dev)
        def circuit(varphi, bs):
            Interferometer(theta=None, phi=None, varphi=varphi, beamsplitter=bs, wires=0)
            return qml.expval(qml.NumberOperator(0))

        with pytest.raises(QuantumFunctionError):
            circuit(varphi, 'clements')

    def test_clements_beamsplitter_convention(self, tol):
        """test the beamsplitter convention"""
        N = 2
        wires = range(N)
        dev = qml.device('default.gaussian', wires=N)

        theta = [0.321]
        phi = [0.234]
        varphi = [0.42342]

        def circuit_rect(varphi):
            Interferometer(theta, phi, varphi, mesh='rectangular', beamsplitter='clements', wires=wires)
            return [qml.expval(qml.NumberOperator(w)) for w in wires]

        def circuit_tria(varphi):
            Interferometer(theta, phi, varphi, mesh='triangular', beamsplitter='clements', wires=wires)
            return [qml.expval(qml.NumberOperator(w)) for w in wires]

        for c in [circuit_rect, circuit_tria]:
            qnode = qml.QNode(c, dev)
            assert np.allclose(qnode(varphi), [0, 0], atol=tol)

            queue = qnode.queue
            assert len(queue) == 3

            assert isinstance(qnode.queue[0], qml.Rotation)
            assert qnode.queue[0].parameters == phi

            assert isinstance(qnode.queue[1], qml.Beamsplitter)
            assert qnode.queue[1].parameters == [theta[0], 0]

            assert isinstance(qnode.queue[2], qml.Rotation)
            assert qnode.queue[2].parameters == varphi

    def test_one_mode(self, tol):
        """Test that a one mode interferometer correctly gives a rotation gate"""
        dev = qml.device('default.gaussian', wires=1)
        varphi = [0.42342]

        def circuit(varphi):
            Interferometer(theta=None, phi=None, varphi=varphi, wires=0)
            return qml.expval(qml.NumberOperator(0))

        qnode = qml.QNode(circuit, dev)
        assert np.allclose(qnode(varphi), 0, atol=tol)

        queue = qnode.queue
        assert len(queue) == 1
        assert isinstance(qnode.queue[0], qml.Rotation)
        assert np.allclose(qnode.queue[0].parameters, varphi, atol=tol)

    def test_two_mode_rect(self, tol):
        """Test that a two mode interferometer using the rectangular mesh
        correctly gives a beamsplitter+rotation gate"""
        N = 2
        wires = range(N)
        dev = qml.device('default.gaussian', wires=N)

        theta = [0.321]
        phi = [0.234]
        varphi = [0.42342]

        def circuit(varphi):
            Interferometer(theta, phi, varphi, wires=wires)
            return [qml.expval(qml.NumberOperator(w)) for w in wires]

        qnode = qml.QNode(circuit, dev)
        assert np.allclose(qnode(varphi), [0, 0], atol=tol)

        queue = qnode.queue
        assert len(queue) == 2

        assert isinstance(qnode.queue[0], qml.Beamsplitter)
        assert qnode.queue[0].parameters == theta+phi

        assert isinstance(qnode.queue[1], qml.Rotation)
        assert qnode.queue[1].parameters == varphi

    def test_two_mode_triangular(self, tol):
        """Test that a two mode interferometer using the triangular mesh
        correctly gives a beamsplitter+rotation gate"""
        N = 2
        wires = range(N)
        dev = qml.device('default.gaussian', wires=N)

        theta = [0.321]
        phi = [0.234]
        varphi = [0.42342]

        def circuit(varphi):
            Interferometer(theta, phi, varphi, mesh='triangular', wires=wires)
            return [qml.expval(qml.NumberOperator(w)) for w in wires]

        qnode = qml.QNode(circuit, dev)
        assert np.allclose(qnode(varphi), [0, 0], atol=tol)

        queue = qnode.queue
        assert len(queue) == 2

        assert isinstance(qnode.queue[0], qml.Beamsplitter)
        assert qnode.queue[0].parameters == theta+phi

        assert isinstance(qnode.queue[1], qml.Rotation)
        assert qnode.queue[1].parameters == varphi

    def test_two_mode_rect_overparameterised(self, tol):
        """Test that a two mode interferometer using the rectangular mesh
        correctly gives a beamsplitter+2 rotation gates"""
        N = 2
        wires = range(N)
        dev = qml.device('default.gaussian', wires=N)

        theta = [0.321]
        phi = [0.234]
        varphi = [0.42342, 0.543]

        def circuit(varphi):
            Interferometer(theta, phi, varphi, wires=wires)
            return [qml.expval(qml.NumberOperator(w)) for w in wires]

        qnode = qml.QNode(circuit, dev)
        assert np.allclose(qnode(varphi), [0, 0], atol=tol)

        queue = qnode.queue
        assert len(queue) == 3

        assert isinstance(qnode.queue[0], qml.Beamsplitter)
        assert qnode.queue[0].parameters == theta+phi

        assert isinstance(qnode.queue[1], qml.Rotation)
        assert qnode.queue[1].parameters == [varphi[0]]

        assert isinstance(qnode.queue[2], qml.Rotation)
        assert qnode.queue[2].parameters == [varphi[1]]

    def test_three_mode(self, tol):
        """Test that a three mode interferometer using either mesh gives the correct gates"""
        N = 3
        wires = range(N)
        dev = qml.device('default.gaussian', wires=N)

        theta = [0.321, 0.4523, 0.21321]
        phi = [0.234, 0.324, 0.234]
        varphi = [0.42342, 0.234]

        def circuit_rect(varphi):
            Interferometer(theta, phi, varphi, wires=wires)
            return [qml.expval(qml.NumberOperator(w)) for w in wires]

        def circuit_tria(varphi):
            Interferometer(theta, phi, varphi, wires=wires)
            return [qml.expval(qml.NumberOperator(w)) for w in wires]

        for c in [circuit_rect, circuit_tria]:
            # test both meshes (both give identical results for the 3 mode case).
            qnode = qml.QNode(c, dev)
            assert np.allclose(qnode(varphi), [0]*N, atol=tol)

            queue = qnode.queue
            assert len(queue) == 5

            expected_bs_wires = [[0, 1], [1, 2], [0, 1]]

            for idx, op in enumerate(qnode.queue[:3]):
                assert isinstance(op, qml.Beamsplitter)
                assert op.parameters == [theta[idx], phi[idx]]
                assert op.wires == expected_bs_wires[idx]

            for idx, op in enumerate(qnode.queue[3:]):
                assert isinstance(op, qml.Rotation)
                assert op.parameters == [varphi[idx]]
                assert op.wires == [idx]

    def test_four_mode_rect(self, tol):
        """Test that a 4 mode interferometer using rectangular mesh gives the correct gates"""
        N = 4
        wires = range(N)
        dev = qml.device('default.gaussian', wires=N)

        theta = [0.321, 0.4523, 0.21321, 0.123, 0.5234, 1.23]
        phi = [0.234, 0.324, 0.234, 1.453, 1.42341, -0.534]
        varphi = [0.42342, 0.234, 0.4523]

        def circuit_rect(varphi):
            Interferometer(theta, phi, varphi, wires=wires)
            return [qml.expval(qml.NumberOperator(w)) for w in wires]

        qnode = qml.QNode(circuit_rect, dev)
        assert np.allclose(qnode(varphi), [0]*N, atol=tol)

        queue = qnode.queue
        assert len(queue) == 9

        expected_bs_wires = [[0, 1], [2, 3], [1, 2], [0, 1], [2, 3], [1, 2]]

        for idx, op in enumerate(qnode.queue[:6]):
            assert isinstance(op, qml.Beamsplitter)
            assert op.parameters == [theta[idx], phi[idx]]
            assert op.wires == expected_bs_wires[idx]

        for idx, op in enumerate(qnode.queue[6:]):
            assert isinstance(op, qml.Rotation)
            assert op.parameters == [varphi[idx]]
            assert op.wires == [idx]

    def test_four_mode_triangular(self, tol):
        """Test that a 4 mode interferometer using triangular mesh gives the correct gates"""
        N = 4
        wires = range(N)
        dev = qml.device('default.gaussian', wires=N)

        theta = [0.321, 0.4523, 0.21321, 0.123, 0.5234, 1.23]
        phi = [0.234, 0.324, 0.234, 1.453, 1.42341, -0.534]
        varphi = [0.42342, 0.234, 0.4523]

        def circuit_tria(varphi):
            Interferometer(theta, phi, varphi, mesh='triangular', wires=wires)
            return [qml.expval(qml.NumberOperator(w)) for w in wires]

        qnode = qml.QNode(circuit_tria, dev)
        assert np.allclose(qnode(varphi), [0]*N, atol=tol)

        queue = qnode.queue
        assert len(queue) == 9

        expected_bs_wires = [[2, 3], [1, 2], [0, 1], [2, 3], [1, 2], [2, 3]]

        for idx, op in enumerate(qnode.queue[:6]):
            assert isinstance(op, qml.Beamsplitter)
            assert op.parameters == [theta[idx], phi[idx]]
            assert op.wires == expected_bs_wires[idx]

        for idx, op in enumerate(qnode.queue[6:]):
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

            Interferometer(theta, phi, varphi, wires=wires)
            return [qml.expval(qml.NumberOperator(w)) for w in wires]

        res = circuit(theta, phi, varphi)
        expected = np.array([0.96852694, 0.23878521, 0.82310606, 0.16547786])
        assert np.allclose(res, expected, atol=tol)

        res = qml.jacobian(circuit, 0)(theta, phi, varphi)
        expected = np.array([[-6.18547248e-03, -3.20488426e-04, -4.20274087e-02, -6.21819638e-02,
                              9.68526932e-01, 9.68526932e-01],
                             [ 3.55439246e-04,  3.89820238e-02, -3.35281306e-03,  7.93009027e-04,
                               8.30347888e-02,-3.45150707e-01],
                             [ 5.44893380e-03,  9.30878007e-03, -5.33374094e-01,  6.13889548e-02,
                               -1.16931385e-01, 3.45150707e-01],
                             [ 3.81099442e-04, -4.79703154e-02,  5.78754316e-01,  1.65477867e-01,
                               3.38965967e-02, 1.65477867e-01]])
        assert np.allclose(res, expected, atol=tol)


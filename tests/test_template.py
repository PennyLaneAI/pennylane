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
Unit tests for the :mod:`pennylane.plugin.DefaultGaussian` templates.
"""
# pylint: disable=protected-access,cell-var-from-loop
import unittest
import inspect
import logging as log
import scipy as sp

from pennylane import numpy as np

from defaults import pennylane as qml, BaseTest

from unittest.mock import patch
from pennylane.ops import Kerr

log.getLogger('defaults')

class TestInterferometer(BaseTest):
    """Tests for the Interferometer from the pennylane.template module."""

    num_subsystems = 4

    def randnc(self, *arg):
        """Normally distributed array of random complex numbers."""
        return np.random.randn(*arg) + 1j*np.random.randn(*arg)

    def random_interferometer(self, N):
        r"""Returns a random unitary matrix representing an interferometer.

        For more details, see :cite:`mezzadri2006`.

        Args:
            N (int): number of modes

        Returns:
            array: random :math:`N\times N` unitary distributed according to the Haar measure
        """
        z = self.randnc(N, N)/np.sqrt(2.0)
        q, r = sp.linalg.qr(z)
        d = sp.diagonal(r)
        ph = d/np.abs(d)
        U = np.multiply(q, ph, q)
        return U

    def setUp(self):
        super().setUp()
        np.random.seed(35)
        self.u1 = self.random_interferometer(self.num_subsystems)
        self.u2 = self.random_interferometer(self.num_subsystems-1)

    def test_interferometer(self):
        dev = qml.device('default.gaussian', wires=self.num_subsystems)
        squeezings = np.random.rand(self.num_subsystems, 2)

        @qml.qnode(dev)
        def circuit():
            for wire in range(self.num_subsystems):
                qml.Squeezing(squeezings[wire][0], squeezings[wire][1], wires=wire)

            qml.template.Interferometer(U=self.u1, wires=range(self.num_subsystems))
            qml.template.Interferometer(U=self.u2, wires=range(self.num_subsystems-1))#also apply a non-trivial Interferometer with different wire number parity
            return tuple(qml.expval.MeanPhoton(wires=wires) for wires in range(self.num_subsystems))

        self.assertAllAlmostEqual(circuit(), [0.40805773, 0.56736047, 0.7724671, 0.39767841], 0.00001)

        num_params = int(self.num_subsystems*(self.num_subsystems-1)/2)
        theta = np.random.uniform(0, 2*np.pi, num_params)
        phi = np.random.uniform(0, 2*np.pi, num_params)

        @qml.qnode(dev)
        def circuit():
            for wire in range(self.num_subsystems):
                qml.Squeezing(squeezings[wire][0], squeezings[wire][1], wires=wire)

            qml.template.Interferometer(theta=theta, phi=phi, wires=range(self.num_subsystems))
            return tuple(qml.expval.MeanPhoton(wires=wires) for wires in range(self.num_subsystems))

        self.assertAllAlmostEqual(circuit(), [0.39156747, 0.52844207, 0.82817202, 0.39738214], 0.00001)


    def test_identity_interferometer(self):
        dev = qml.device('default.gaussian', wires=self.num_subsystems)

        displacements = np.random.randint(1, 3, self.num_subsystems)

        @qml.qnode(dev)
        def circuit():
            for wire in range(self.num_subsystems):
                qml.Displacement(displacements[wire], 0, wires=wire)
            qml.template.Interferometer(U=np.identity(self.num_subsystems), wires=range(self.num_subsystems))
            qml.template.Interferometer(U=np.identity(self.num_subsystems-1), wires=range(self.num_subsystems-1))#also apply a trivial Interferometer with different wire number parity
            return tuple(qml.expval.X(wires=wire) for wire in range(self.num_subsystems))

        self.assertAllAlmostEqual(circuit(), 2*displacements, 0.00001)


    def test_interferometer_argument_error (self):
        dev = qml.device('default.gaussian', wires=self.num_subsystems)

        @qml.qnode(dev)
        def circuit():
            qml.template.Interferometer(wires=range(self.num_subsystems))
            return qml.expval.X(wires=0)

        with self.assertRaises(ValueError):
            circuit()

        @qml.qnode(dev)
        def circuit():
            qml.template.Interferometer(U=np.identity(self.num_subsystems), theta=np.zeros(int(self.num_subsystems*(self.num_subsystems-1)/2)), wires=range(self.num_subsystems))
            return qml.expval.X(wires=0)

        with self.assertRaises(ValueError):
            circuit()

        @qml.qnode(dev)
        def circuit():
            qml.template.Interferometer(U=np.identity(self.num_subsystems), phi=np.zeros(int(self.num_subsystems*(self.num_subsystems-1)/2)), wires=range(self.num_subsystems))
            return qml.expval.X(wires=0)

        with self.assertRaises(ValueError):
            circuit()

        @qml.qnode(dev)
        def circuit():
            qml.template.Interferometer(phi=np.zeros(int(self.num_subsystems*(self.num_subsystems-1)/2)), wires=range(self.num_subsystems))
            return qml.expval.X(wires=0)

        with self.assertRaises(ValueError):
            circuit()

        @qml.qnode(dev)
        def circuit():
            qml.template.Interferometer(theta=np.zeros(int(self.num_subsystems*(self.num_subsystems-1)/2)), wires=range(self.num_subsystems))
            return qml.expval.X(wires=0)

        with self.assertRaises(ValueError):
            circuit()

    def test_clements_non_square_error(self):
        V_non_square = np.random.randn(2, 3)
        with self.assertRaises(ValueError):
            qml.template.clements(V_non_square)

class TestCVNeuralNet(BaseTest):
    """Tests for the CVNeuralNet and CVNeuralNetLayer from the pennylane.template module."""

    def setUp(self):
        super().setUp()
        np.random.seed(8)
        self.num_wires = 4

    def test_cvqnn(self):

        dev = qml.device('default.gaussian', wires=self.num_wires)

        weights = [[
            np.random.uniform(0, 2*np.pi, int(self.num_wires*(self.num_wires-1)/2)), #transmittivity angles
            np.random.uniform(0, 2*np.pi, int(self.num_wires*(self.num_wires-1)/2)), #phase angles
            np.random.uniform(0.1, 0.7, self.num_wires), #squeezing amounts
            np.random.uniform(0, 2*np.pi, int(self.num_wires*(self.num_wires-1)/2)), #transmittivity angles
            np.random.uniform(0, 2*np.pi, int(self.num_wires*(self.num_wires-1)/2)), #phase angles
            np.random.uniform(0.1, 2.0, self.num_wires), #displacement magnitudes
            np.random.uniform(0.1, 0.3, self.num_wires) #kerr parameters
        ] for l in range(2)]

        with patch.object(Kerr, '__init__', return_value=None) as _: #Kerr() does not work on any core device, so we have to mock it here with a trivial class
            @qml.qnode(dev)
            def circuit(weights):
                qml.template.CVNeuralNet(weights, wires=range(self.num_wires))
                return qml.expval.X(wires=0)

            circuit(weights)

class TestVariationalClassifiyer(BaseTest):
    """Tests for the VariationalClassifiyer from the pennylane.template module."""

    def setUp(self):
        super().setUp()
        np.random.seed(0)

    def test_variational_classifyer(self):
        """Tests the VariationalClassifyer for various parameters."""
        outcomes = []
        for num_wires in range(2,4):
            for num_layers in range(1,3):

                dev = qml.device('default.qubit', wires=num_wires)

                @qml.qnode(dev)
                def circuit(weights, x=None):
                    qml.BasisState(x, wires=range(num_wires))
                    qml.template.CircuitCentric(weights, True, wires=range(num_wires))
                    return qml.expval.PauliZ(0)

                weights=np.random.randn(num_layers, num_wires, 3)
                outcomes.append(circuit(weights, x=np.array(np.random.randint(0,1,num_wires))))
        self.assertAllAlmostEqual(np.array(outcomes), np.array([-0.29242496,  0.22129055,  0.07540091, -0.77626557]), delta=self.tol)




if __name__ == '__main__':
    print('Testing PennyLane version ' + qml.version() + ', pennylane.template.')
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (TestInterferometer, TestCVNeuralNet, TestVariationalClassifiyer):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)
    unittest.TextTestRunner().run(suite)

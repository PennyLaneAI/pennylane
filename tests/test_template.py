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
Unit tests for the :mod:`pennylane.template` module.
"""
# pylint: disable=protected-access,cell-var-from-loop
import unittest
import inspect
import logging as log
import scipy as sp
from scipy.linalg import qr
from types import MethodType

from pennylane import numpy as np

from defaults import pennylane as qml, BaseTest

from unittest.mock import patch
from pennylane.ops import Kerr

log.getLogger('defaults')

class TestInterferometer(BaseTest):
    """Tests for the Interferometer from the pennylane.template module."""

    num_subsystems = 4

    def test_integration(self):
        """integration test for the Interferomerter."""
        dev = qml.device('default.gaussian', wires=self.num_subsystems)
        np.random.seed(88)
        num_params = int(self.num_subsystems*(self.num_subsystems-1)/2)
        theta = np.random.uniform(0, 2*np.pi, num_params)
        phi = np.random.uniform(0, 2*np.pi, num_params)

        # to test whether the correct circuit is produces we inspect the
        # Device._op_queue by patching post_apply() of the device
        test = self
        def new_post_apply(self):
            print([ op.parameters for op in  self._op_queue])
            print([ [t, p] for t, p in zip(theta, phi)])

            test.assertAllEqual([[t, p] for t, p in zip(theta, phi)], [op.parameters for op in self._op_queue])
            test.assertAllEqual([qml.Beamsplitter]*len(theta), [type(op) for op in self._op_queue])

        dev.post_apply = MethodType(new_post_apply, dev)

        @qml.qnode(dev)
        def circuit(theta, phi):
            qml.template.Interferometer(theta=theta, phi=phi, wires=range(self.num_subsystems))
            return tuple(qml.expval.MeanPhoton(wires=wires) for wires in range(self.num_subsystems))

        circuit(theta, phi)

    def test_execution(self):
        """execution test for the Interferomerter."""
        dev = qml.device('default.gaussian', wires=self.num_subsystems)
        np.random.seed(8)
        squeezings = np.random.rand(self.num_subsystems, 2)
        num_params = int(self.num_subsystems*(self.num_subsystems-1)/2)
        theta = np.random.uniform(0, 2*np.pi, num_params)
        phi = np.random.uniform(0, 2*np.pi, num_params)

        @qml.qnode(dev)
        def circuit(theta, phi):
            for wire in range(self.num_subsystems):
                qml.Squeezing(squeezings[wire][0], squeezings[wire][1], wires=wire)

            qml.template.Interferometer(theta=theta, phi=phi, wires=range(self.num_subsystems))
            return tuple(qml.expval.MeanPhoton(wires=wires) for wires in range(self.num_subsystems))

        self.assertAllAlmostEqual(circuit(theta, phi), [0.96852694, 0.23878521, 0.82310606, 0.16547786], delta=self.tol)

        self.assertAllAlmostEqual(qml.jacobian(circuit, 0)(theta, phi),
                                  np.array([[-6.18547621e-03, -3.20488416e-04, -4.20274088e-02, -6.21819677e-02, 0.00000000e+00,  0.00000000e+00],
                                            [ 3.55439469e-04,  3.89820231e-02, -3.35281297e-03,  7.93009278e-04, 8.30347900e-02, -3.45150712e-01],
                                            [ 5.44893709e-03,  9.30878023e-03, -5.33374090e-01,  6.13889584e-02,-1.16931386e-01,  3.45150712e-01],
                                            [ 3.81099656e-04, -4.79703149e-02,  5.78754312e-01,  0.00000000e+00, 3.38965962e-02,  0.00000000e+00]]
                                  ), delta=self.tol)

class TestCVNeuralNet(BaseTest):
    """Tests for the Interferometer from the pennylane.template module."""

    num_subsystems = 4

    def setUp(self):
        super().setUp()
        np.random.seed(8)
        self.num_wires = 4

    def test_cvqnn(self):
        """An execution test for the CVNeuralNet"""
        dev = qml.device('default.gaussian', wires=self.num_wires)

        weights = [[
            np.random.uniform(0, 2*np.pi, int(self.num_wires*(self.num_wires-1)/2)), #transmittivity angles
            np.random.uniform(0, 2*np.pi, int(self.num_wires*(self.num_wires-1)/2)), #phase angles
            np.random.uniform(0.1, 0.7, self.num_wires), #squeezing amounts
            np.random.uniform(0, 2*np.pi, int(self.num_wires*(self.num_wires-1)/2)), #squeezing angles
            np.random.uniform(0, 2*np.pi, int(self.num_wires*(self.num_wires-1)/2)), #transmittivity angles
            np.random.uniform(0, 2*np.pi, int(self.num_wires*(self.num_wires-1)/2)), #phase angles
            np.random.uniform(0.1, 2.0, self.num_wires), #displacement magnitudes
            np.random.uniform(0.1, 2*np.pi, self.num_wires), #displacement angles
            np.random.uniform(0.1, 0.3, self.num_wires) #kerr parameters
        ] for l in range(2)]

        with patch.object(Kerr, '__init__', return_value=None) as _: #Kerr() does not work on any core device, so we have to mock it here with a trivial class
            @qml.qnode(dev)
            def circuit(weights):
                qml.template.CVNeuralNet(weights, wires=range(self.num_wires))
                return qml.expval.X(wires=0)

            circuit(weights)

        # No assert because values are anyway not correct because Kerr had to be mocked


class TestStronglyEntanglingCircuit(BaseTest):
    """Tests for the StronglyEntanglingCircuit from the pennylane.template module."""

    def setUp(self):
        super().setUp()
        np.random.seed(0)

    def test_execution(self):
        """Tests the StronglyEntanglingCircuit for various parameters."""
        outcomes = []
        for num_wires in range(2,4):
            for num_layers in range(1,3):

                dev = qml.device('default.qubit', wires=num_wires)

                @qml.qnode(dev)
                def circuit(weights, x=None):
                    qml.BasisState(x, wires=range(num_wires))
                    qml.template.StronglyEntanglingCircuit(weights, True, wires=range(num_wires))
                    return qml.expval.PauliZ(0)

                weights=np.random.randn(num_layers, num_wires, 3)
                outcomes.append(circuit(weights, x=np.array(np.random.randint(0,1,num_wires))))
        self.assertAllAlmostEqual(np.array(outcomes), np.array([-0.29242496,  0.22129055,  0.07540091, -0.77626557]), delta=self.tol)




if __name__ == '__main__':
    print('Testing PennyLane version ' + qml.version() + ', pennylane.template.')
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (TestInterferometer, TestCVNeuralNet, TestStronglyEntanglingCircuit):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)
    unittest.TextTestRunner().run(suite)

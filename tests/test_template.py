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
from unittest.mock import patch
import logging as log
import itertools as it
from types import MethodType

from pennylane import numpy as np
from pennylane.ops import Kerr
from defaults import pennylane as qml, BaseTest

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
        varphi = np.random.uniform(0, 2*np.pi, self.num_subsystems)

        test = self
        def new_execute(_, queue, *__):
            """to test whether the correct circuit is produced we inspect the
            Device._op_queue by patching Device.execute() with this function"""
            test.assertAllEqual([qml.Beamsplitter]*len(theta)+[qml.Rotation]*len(varphi), [type(op) for op in queue])
            #test.assertAllEqual([elem for t, p in zip(theta, phi) for elem in [[t, p], [p]]], [op.parameters for op in queue])
            test.assertAllEqual([[t, p] for t, p in zip(theta, phi)]+[ [p] for p in varphi], [op.parameters for op in queue])
            return np.array([0.5])

        dev.execute = MethodType(new_execute, dev)

        @qml.qnode(dev)
        def circuit(theta, phi, varphi):
            qml.template.Interferometer(theta=theta, phi=phi, varphi=varphi, wires=range(self.num_subsystems))
            return tuple(qml.expval.MeanPhoton(wires=wires) for wires in range(self.num_subsystems))

        circuit(theta, phi, varphi)

    def test_execution(self):
        """execution test for the Interferomerter."""
        dev = qml.device('default.gaussian', wires=self.num_subsystems)
        np.random.seed(8)
        squeezings = np.random.rand(self.num_subsystems, 2)
        num_params = int(self.num_subsystems*(self.num_subsystems-1)/2)
        theta = np.random.uniform(0, 2*np.pi, num_params)
        phi = np.random.uniform(0, 2*np.pi, num_params)
        varphi = np.random.uniform(0, 2*np.pi, self.num_subsystems)

        @qml.qnode(dev)
        def circuit(theta, phi, varphi):
            for wire in range(self.num_subsystems):
                qml.Squeezing(squeezings[wire][0], squeezings[wire][1], wires=wire)

            qml.template.Interferometer(theta=theta, phi=phi, varphi=varphi, wires=range(self.num_subsystems))
            return tuple(qml.expval.MeanPhoton(wires=wires) for wires in range(self.num_subsystems))

        self.assertAllAlmostEqual(circuit(theta, phi, varphi), np.array([0.96852694, 0.23878521, 0.82310606, 0.16547786]), delta=self.tol)
        #self.assertAllAlmostEqual(circuit(theta, phi, varphi[:-1]), np.array([0.96852694, 0.23878521, 0.82310606, 0.16547786]), delta=self.tol)

        print("jacobean="+str(qml.jacobian(circuit, 0)(theta, phi, varphi)))
        self.assertAllAlmostEqual(qml.jacobian(circuit, 0)(theta, phi, varphi),
                                  np.array(
                                      [[-6.18547621e-03, -3.20488416e-04, -4.20274088e-02, -6.21819677e-02,
                                        0.00000000e+00,  0.00000000e+00],
                                       [ 3.55439469e-04,  3.89820231e-02, -3.35281297e-03,  7.93009278e-04,
                                         8.30347900e-02, -3.45150712e-01],
                                       [ 5.44893709e-03,  9.30878023e-03, -5.33374090e-01,  6.13889584e-02,
                                         -1.16931386e-01,  3.45150712e-01],
                                       [ 3.81099656e-04, -4.79703149e-02,  5.78754312e-01,  0.00000000e+00,
                                         3.38965962e-02,  0.00000000e+00]]
                                  ), delta=self.tol)

        # self.assertAllAlmostEqual(qml.jacobian(circuit, 0)(theta, phi, varphi[:-1]),
        #                           np.array(
        #                               [[-5.03137764e-03, -3.20488416e-04, -4.20886887e-02, -6.16246214e-02, 0.00000000e+00, 0.00000000e+00],
        #                                [3.09578135e-04, 2.34020030e-02, -1.02412671e-02, 1.40942058e-03, 1.08395609e-01, -3.34712643e-01],
        #                                [4.34069985e-03, 1.40617349e-03, -5.27609524e-01, 6.02152008e-02, -9.48406160e-02, 3.34712643e-01],
        #                                [3.81099656e-04, -2.44876881e-02, 5.79939479e-01, 0.00000000e+00, -1.35549930e-02, 0.00000000e+00]]
        #                           ), delta=self.tol)


class TestCVNeuralNet(BaseTest):
    """Tests for the CVNeuralNet from the pennylane.template module."""

    num_subsystems = 4

    def setUp(self):
        super().setUp()
        np.random.seed(8)
        self.num_wires = 4
        self.num_layers = 2
        self.weights = [[
            np.random.uniform(0, 2*np.pi, int(self.num_wires*(self.num_wires-1)/2)), #transmittivity angles
            np.random.uniform(0, 2*np.pi, int(self.num_wires*(self.num_wires-1)/2)), #phase angles
            np.random.uniform(0, 2*np.pi, self.num_wires), #rotation angles
            np.random.uniform(0.1, 0.7, self.num_wires), #squeezing amounts
            np.random.uniform(0, 2*np.pi, int(self.num_wires*(self.num_wires-1)/2)), #squeezing angles
            np.random.uniform(0, 2*np.pi, int(self.num_wires*(self.num_wires-1)/2)), #transmittivity angles
            np.random.uniform(0, 2*np.pi, int(self.num_wires*(self.num_wires-1)/2)), #phase angles
            np.random.uniform(0, 2*np.pi, self.num_wires), #rotation angles
            np.random.uniform(0.1, 2.0, self.num_wires), #displacement magnitudes
            np.random.uniform(0.1, 2*np.pi, self.num_wires), #displacement angles
            np.random.uniform(0.1, 0.3, self.num_wires) #kerr parameters
        ] for l in range(self.num_layers)]

    def test_integration(self):
        """integration test for the CVNeuralNet."""
        dev = qml.device('default.gaussian', wires=self.num_subsystems)

        # to test whether the correct circuit is produced we inspect the
        # Device._op_queue by patching Device.execute()
        test = self
        test_weights = self.weights
        test_num_layers = self.num_layers
        def new_execute(_, queue, *__):
            """to test whether the correct circuit is produced we inspect the
            Device._op_queue by patching Device.execute() with this function"""
            test.assertAllEqual([type(op) for op in queue],
                                list(it.chain.from_iterable([[qml.Beamsplitter]*6+[qml.Rotation]*4+[qml.Squeezing]*4+[qml.Beamsplitter]*6+[qml.Rotation]*4+[qml.Displacement]*4+[qml.Kerr]*4]*test_num_layers)),
                                "Did not find the expected operations in the right order in the queue")

            split_queue = np.delete(np.array(np.split(queue, np.cumsum([6, 4, 4, 6, 4, 4, 4]*test_num_layers))), -1)
            split_queue = np.split(split_queue, test_num_layers)

            for l in range(test_num_layers):
                test.assertAllEqual([op.parameters for op in split_queue[l][0]], [[theta_1, phi_1] for theta_1, phi_1 in zip(test_weights[l][0], test_weights[l][1])])
                test.assertAllEqual([op.parameters for op in split_queue[l][1]], [varphi_1 for varphi_1 in test_weights[l][2]])
                test.assertAllEqual([op.parameters for op in split_queue[l][2]], [[r, phi_r] for r, phi_r in zip(test_weights[l][3], test_weights[l][4])])
                test.assertAllEqual([op.parameters for op in split_queue[l][3]], [[theta_2, phi_2] for theta_2, phi_2 in zip(test_weights[l][5], test_weights[l][6])])
                test.assertAllEqual([op.parameters for op in split_queue[l][4]], [varphi_2 for varphi_2 in test_weights[l][7]])
                test.assertAllEqual([op.parameters for op in split_queue[l][5]], [[a, phi_a] for a, phi_a in zip(test_weights[l][8], test_weights[l][9])])
                test.assertAllEqual([op.parameters for op in split_queue[l][6]], [[k] for k in test_weights[l][10]])

            return np.array([0.5])

        dev.execute = MethodType(new_execute, dev)

        @qml.qnode(dev)
        def circuit(weights):
            qml.template.CVNeuralNet(weights, wires=range(self.num_wires))
            return qml.expval.X(wires=0)

        circuit(self.weights)


    def test_execution(self):
        """An execution test for the CVNeuralNet"""
        dev = qml.device('default.gaussian', wires=self.num_wires)

        with patch.object(Kerr, '__init__', return_value=None) as _: #Kerr() does not work on any core device, so we have to mock it here with a trivial class
            @qml.qnode(dev)
            def circuit(weights):
                qml.template.CVNeuralNet(weights, wires=range(self.num_wires))
                return qml.expval.X(wires=0)

            circuit(self.weights)

        # No assert because values are anyway not correct because Kerr had to be mocked


class TestStronglyEntanglingCircuit(BaseTest):
    """Tests for the StronglyEntanglingCircuit from the pennylane.template module."""

    def test_integration(self):
        """integration test for the StronglyEntanglingCircuit."""
        np.random.seed(12)
        num_layers = 2
        for num_wires in range(2, 4):

            dev = qml.device('default.qubit', wires=num_wires)
            weights = np.random.randn(num_layers, num_wires, 3)

            # to test whether the correct circuit is produces we inspect the
            # Device._op_queue by patching Device.execute()
            test = self
            test_weights = weights
            def new_execute(_, queue, *__):
                """to test whether the correct circuit is produced we inspect the
                Device._op_queue by patching Device.execute() with this function"""
                test.assertAllEqual([type(op) for op in queue], ([qml.Rot]*num_wires+[qml.CNOT]*num_wires)*num_layers)

                split_queue = np.split(queue, np.cumsum([num_wires, num_wires]*num_layers))
                test.assertAllEqual(test_weights[0], np.array([op.parameters for op in split_queue[0]]))
                test.assertAllEqual(test_weights[1], np.array([op.parameters for op in split_queue[2]]))

                return np.array([0.5])

            dev.execute = MethodType(new_execute, dev)

            @qml.qnode(dev)
            def circuit(weights):
                qml.template.StronglyEntanglingCircuit(weights, True, wires=range(num_wires))
                return qml.expval.PauliZ(0)

            circuit(weights)

    def test_execution(self):
        """Tests the StronglyEntanglingCircuit for various parameters."""
        np.random.seed(0)
        outcomes = []
        for num_wires in range(2, 4):
            for num_layers in range(1, 3):

                dev = qml.device('default.qubit', wires=num_wires)
                weights = np.random.randn(num_layers, num_wires, 3)

                @qml.qnode(dev)
                def circuit(weights, x=None):
                    qml.BasisState(x, wires=range(num_wires))
                    qml.template.StronglyEntanglingCircuit(weights, True, wires=range(num_wires))
                    return qml.expval.PauliZ(0)

                outcomes.append(circuit(weights, x=np.array(np.random.randint(0, 1, num_wires))))

        self.assertAllAlmostEqual(np.array(outcomes), np.array([-0.29242496, 0.22129055, 0.07540091, -0.77626557]), delta=self.tol)




if __name__ == '__main__':
    print('Testing PennyLane version ' + qml.version() + ', pennylane.template.')
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (TestInterferometer, TestCVNeuralNet, TestStronglyEntanglingCircuit):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)
    unittest.TextTestRunner().run(suite)

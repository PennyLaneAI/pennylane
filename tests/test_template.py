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

from pennylane import numpy as np
from pennylane.qnode import QuantumFunctionError
from pennylane.plugins import DefaultGaussian
from defaults import pennylane as qml, BaseTest

log.getLogger('defaults')


class DummyDevice(DefaultGaussian):
    """Dummy device to allow Kerr operations"""
    _operation_map = DefaultGaussian._operation_map.copy()
    _operation_map['Kerr'] = lambda *x, **y: np.identity(2)


class TestInterferometer(BaseTest):
    """Tests for the Interferometer from the pennylane.template module."""

    def test_exceptions(self):
        """test that exceptions are correctly raised"""
        dev = qml.device('default.gaussian', wires=1)
        varphi = [0.42342]

        @qml.qnode(dev)
        def circuit(varphi, mesh):
            qml.template.Interferometer(theta=None, phi=None, varphi=varphi, mesh=mesh, wires=0)
            return qml.expval.MeanPhoton(0)

        with self.assertRaisesRegex(QuantumFunctionError, "The mesh parameter influences the "
        "circuit architecture and can not be passed as a QNode parameter."):
            circuit(varphi, 'rectangular')

        @qml.qnode(dev)
        def circuit(varphi, bs):
            qml.template.Interferometer(theta=None, phi=None, varphi=varphi, beamsplitter=bs, wires=0)
            return qml.expval.MeanPhoton(0)

        with self.assertRaisesRegex(QuantumFunctionError, "The beamsplitter parameter influences the "
        "circuit architecture and can not be passed as a QNode parameter."):
            circuit(varphi, 'clements')

    def test_clements_beamsplitter_convention(self):
        """test the beamsplitter convention"""
        N = 2
        wires = range(N)
        dev = qml.device('default.gaussian', wires=N)

        theta = [0.321]
        phi = [0.234]
        varphi = [0.42342]

        def circuit_rect(varphi):
            qml.template.Interferometer(theta, phi, varphi, mesh='rectangular', beamsplitter='clements', wires=wires)
            return [qml.expval.MeanPhoton(w) for w in wires]

        def circuit_tria(varphi):
            qml.template.Interferometer(theta, phi, varphi, mesh='triangular', beamsplitter='clements', wires=wires)
            return [qml.expval.MeanPhoton(w) for w in wires]

        for c in [circuit_rect, circuit_tria]:
            qnode = qml.QNode(c, dev)
            self.assertAllAlmostEqual(qnode(varphi), [0, 0], delta=self.tol)

            queue = qnode.queue
            self.assertEqual(len(queue), 3)

            self.assertTrue(isinstance(qnode.queue[0], qml.Rotation))
            self.assertAllEqual(qnode.queue[0].parameters, phi)

            self.assertTrue(isinstance(qnode.queue[1], qml.Beamsplitter))
            self.assertAllEqual(qnode.queue[1].parameters, [theta[0], 0])

            self.assertTrue(isinstance(qnode.queue[2], qml.Rotation))
            self.assertAllEqual(qnode.queue[2].parameters, varphi)

    def test_one_mode(self):
        """Test that a one mode interferometer correctly gives a rotation gate"""
        dev = qml.device('default.gaussian', wires=1)
        varphi = [0.42342]

        def circuit(varphi):
            qml.template.Interferometer(theta=None, phi=None, varphi=varphi, wires=0)
            return qml.expval.MeanPhoton(0)

        qnode = qml.QNode(circuit, dev)
        self.assertAllAlmostEqual(qnode(varphi), 0, delta=self.tol)

        queue = qnode.queue
        self.assertEqual(len(queue), 1)
        self.assertTrue(isinstance(qnode.queue[0], qml.Rotation))
        self.assertAllEqual(qnode.queue[0].parameters, varphi)

    def test_two_mode_rect(self):
        """Test that a two mode interferometer using the rectangular mesh
        correctly gives a beamsplitter+rotation gate"""
        N = 2
        wires = range(N)
        dev = qml.device('default.gaussian', wires=N)

        theta = [0.321]
        phi = [0.234]
        varphi = [0.42342]

        def circuit(varphi):
            qml.template.Interferometer(theta, phi, varphi, wires=wires)
            return [qml.expval.MeanPhoton(w) for w in wires]

        qnode = qml.QNode(circuit, dev)
        self.assertAllAlmostEqual(qnode(varphi), [0, 0], delta=self.tol)

        queue = qnode.queue
        self.assertEqual(len(queue), 2)

        self.assertTrue(isinstance(qnode.queue[0], qml.Beamsplitter))
        self.assertAllEqual(qnode.queue[0].parameters, theta+phi)

        self.assertTrue(isinstance(qnode.queue[1], qml.Rotation))
        self.assertAllEqual(qnode.queue[1].parameters, varphi)

    def test_two_mode_triangular(self):
        """Test that a two mode interferometer using the triangular mesh
        correctly gives a beamsplitter+rotation gate"""
        N = 2
        wires = range(N)
        dev = qml.device('default.gaussian', wires=N)

        theta = [0.321]
        phi = [0.234]
        varphi = [0.42342]

        def circuit(varphi):
            qml.template.Interferometer(theta, phi, varphi, mesh='triangular', wires=wires)
            return [qml.expval.MeanPhoton(w) for w in wires]

        qnode = qml.QNode(circuit, dev)
        self.assertAllAlmostEqual(qnode(varphi), [0, 0], delta=self.tol)

        queue = qnode.queue
        self.assertEqual(len(queue), 2)

        self.assertTrue(isinstance(qnode.queue[0], qml.Beamsplitter))
        self.assertAllEqual(qnode.queue[0].parameters, theta+phi)

        self.assertTrue(isinstance(qnode.queue[1], qml.Rotation))
        self.assertAllEqual(qnode.queue[1].parameters, varphi)

    def test_two_mode_rect_overparameterised(self):
        """Test that a two mode interferometer using the rectangular mesh
        correctly gives a beamsplitter+2 rotation gates when requested"""
        N = 2
        wires = range(N)
        dev = qml.device('default.gaussian', wires=N)

        theta = [0.321]
        phi = [0.234]
        varphi = [0.42342, 0.543]

        def circuit(varphi):
            qml.template.Interferometer(theta, phi, varphi, wires=wires)
            return [qml.expval.MeanPhoton(w) for w in wires]

        qnode = qml.QNode(circuit, dev)
        self.assertAllAlmostEqual(qnode(varphi), [0, 0], delta=self.tol)

        queue = qnode.queue
        self.assertEqual(len(queue), 3)

        self.assertTrue(isinstance(qnode.queue[0], qml.Beamsplitter))
        self.assertAllEqual(qnode.queue[0].parameters, theta+phi)

        self.assertTrue(isinstance(qnode.queue[1], qml.Rotation))
        self.assertAllEqual(qnode.queue[1].parameters, [varphi[0]])

        self.assertTrue(isinstance(qnode.queue[2], qml.Rotation))
        self.assertAllEqual(qnode.queue[2].parameters, [varphi[1]])

    def test_three_mode(self):
        """Test that a three mode interferometer using either mesh gives the correct gates"""
        N = 3
        wires = range(N)
        dev = qml.device('default.gaussian', wires=N)

        theta = [0.321, 0.4523, 0.21321]
        phi = [0.234, 0.324, 0.234]
        varphi = [0.42342, 0.234]

        def circuit_rect(varphi):
            qml.template.Interferometer(theta, phi, varphi, wires=wires)
            return [qml.expval.MeanPhoton(w) for w in wires]

        def circuit_tria(varphi):
            qml.template.Interferometer(theta, phi, varphi, wires=wires)
            return [qml.expval.MeanPhoton(w) for w in wires]

        for c in [circuit_rect, circuit_tria]:
            # test both meshes (both give identical results for the 3 mode case).
            qnode = qml.QNode(c, dev)
            self.assertAllAlmostEqual(qnode(varphi), [0]*N, delta=self.tol)

            queue = qnode.queue
            self.assertEqual(len(queue), 5)

            expected_bs_wires = [[0, 1], [1, 2], [0, 1]]

            for idx, op in enumerate(qnode.queue[:3]):
                self.assertTrue(isinstance(op, qml.Beamsplitter))
                self.assertAllEqual(op.parameters, [theta[idx], phi[idx]])
                self.assertAllEqual(op.wires, expected_bs_wires[idx])

            for idx, op in enumerate(qnode.queue[3:]):
                self.assertTrue(isinstance(op, qml.Rotation))
                self.assertAllEqual(op.parameters, [varphi[idx]])
                self.assertAllEqual(op.wires, [idx])

    def test_four_mode_rect(self):
        """Test that a 4 mode interferometer using rectangular mesh gives the correct gates"""
        N = 4
        wires = range(N)
        dev = qml.device('default.gaussian', wires=N)

        theta = [0.321, 0.4523, 0.21321, 0.123, 0.5234, 1.23]
        phi = [0.234, 0.324, 0.234, 1.453, 1.42341, -0.534]
        varphi = [0.42342, 0.234, 0.4523]

        def circuit_rect(varphi):
            qml.template.Interferometer(theta, phi, varphi, wires=wires)
            return [qml.expval.MeanPhoton(w) for w in wires]

        qnode = qml.QNode(circuit_rect, dev)
        self.assertAllAlmostEqual(qnode(varphi), [0]*N, delta=self.tol)

        queue = qnode.queue
        self.assertEqual(len(queue), 9)

        expected_bs_wires = [[0, 1], [2, 3], [1, 2], [0, 1], [2, 3], [1, 2]]

        for idx, op in enumerate(qnode.queue[:6]):
            self.assertTrue(isinstance(op, qml.Beamsplitter))
            self.assertAllEqual(op.parameters, [theta[idx], phi[idx]])
            self.assertAllEqual(op.wires, expected_bs_wires[idx])

        for idx, op in enumerate(qnode.queue[6:]):
            self.assertTrue(isinstance(op, qml.Rotation))
            self.assertAllEqual(op.parameters, [varphi[idx]])
            self.assertAllEqual(op.wires, [idx])

    def test_four_mode_triangular(self):
        """Test that a 4 mode interferometer using triangular mesh gives the correct gates"""
        N = 4
        wires = range(N)
        dev = qml.device('default.gaussian', wires=N)

        theta = [0.321, 0.4523, 0.21321, 0.123, 0.5234, 1.23]
        phi = [0.234, 0.324, 0.234, 1.453, 1.42341, -0.534]
        varphi = [0.42342, 0.234, 0.4523]

        def circuit_tria(varphi):
            qml.template.Interferometer(theta, phi, varphi, mesh='triangular', wires=wires)
            return [qml.expval.MeanPhoton(w) for w in wires]

        qnode = qml.QNode(circuit_tria, dev)
        self.assertAllAlmostEqual(qnode(varphi), [0]*N, delta=self.tol)

        queue = qnode.queue
        self.assertEqual(len(queue), 9)

        expected_bs_wires = [[2, 3], [1, 2], [0, 1], [2, 3], [1, 2], [2, 3]]

        for idx, op in enumerate(qnode.queue[:6]):
            self.assertTrue(isinstance(op, qml.Beamsplitter))
            self.assertAllEqual(op.parameters, [theta[idx], phi[idx]])
            self.assertAllEqual(op.wires, expected_bs_wires[idx])

        for idx, op in enumerate(qnode.queue[6:]):
            self.assertTrue(isinstance(op, qml.Rotation))
            self.assertAllEqual(op.parameters, [varphi[idx]])
            self.assertAllEqual(op.wires, [idx])

    def test_integration(self):
        """test integration with PennyLane and gradient calculations"""
        N = 4
        wires = range(N)
        dev = qml.device('default.gaussian', wires=N)

        sq = np.array([[0.8734294 , 0.96854066],
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

            qml.template.Interferometer(theta, phi, varphi, wires=wires)
            return [qml.expval.MeanPhoton(w) for w in wires]

        res = circuit(theta, phi, varphi)
        expected = np.array([0.96852694, 0.23878521, 0.82310606, 0.16547786])
        self.assertAllAlmostEqual(res, expected, delta=self.tol)

        res = qml.jacobian(circuit, 0)(theta, phi, varphi)
        expected = np.array([[-6.18547621e-03, -3.20488416e-04, -4.20274088e-02, -6.21819677e-02, 0.00000000e+00, 0.00000000e+00],
                            [3.55439469e-04, 3.89820231e-02, -3.35281297e-03, 7.93009278e-04, 8.30347900e-02, -3.45150712e-01],
                            [5.44893709e-03, 9.30878023e-03, -5.33374090e-01, 6.13889584e-02, -1.16931386e-01, 3.45150712e-01],
                            [3.81099656e-04, -4.79703149e-02, 5.78754312e-01, 0.00000000e+00, 3.38965962e-02, 0.00000000e+00]])
        self.assertAllAlmostEqual(res, expected, delta=self.tol)


class TestCVNeuralNet(BaseTest):
    """Tests for the CVNeuralNet from the pennylane.template module."""
    num_subsystems = 4

    def setUp(self):
        super().setUp()
        self.N = 4
        self.depth = 2
        self.weights = [
            [
                np.array([ 5.48791879, 6.08552046, 5.46131036, 3.33546468, 1.46227521, 0.0716208 ]),
                np.array([ 2.70471535, 2.52804815, 3.28406182, 3.0058243 , 3.48940764, 3.41419504]),
                np.array([ 4.7808479 , 4.47598146, 3.89357744, 2.67721355]),
                np.array([ 0.27344502, 0.68431314, 0.30026443, 0.23128064]),
                np.array([ 0.4134863 , 6.17555778, 0.80334114, 2.02400747, 0.44574704, 1.41227118]),
                np.array([ 2.47328111, 5.63064513, 2.17059932, 6.1873632 , 0.18052879, 2.20970037]),
                np.array([ 2.3936353 , 4.80135971, 5.89867895, 2.00867023, 2.71732643, 1.69737575]),
                np.array([ 5.03318258, 4.01017269, 0.43159284, 3.7928101 ]),
                np.array([ 1.61159166, 0.1608155 , 0.96535086, 1.60132783]),
                np.array([ 6.21267547, 3.71076099, 0.34060195, 2.86031556]),
                np.array([ 0.1376345 , 0.22541113, 0.14306356, 0.13019402])
            ],
            [
                np.array([ 3.36869403, 0.63074883, 4.59400392, 5.9040016 , 5.92704296, 2.35455147]),
                np.array([ 3.74320919, 4.15936005, 3.20807161, 2.95870535, 0.05574621, 0.42660569]),
                np.array([ 2.73203094, 2.71115444, 1.16794164, 3.32823666]),
                np.array([ 0.45945175, 0.53255468, 0.28383751, 0.34263728]),
                np.array([ 5.16969442, 3.6890488 , 4.43916808, 3.20808287, 5.21543123, 4.52815349]),
                np.array([ 5.44288268, 1.27806129, 1.87574979, 2.98956484, 3.10140853, 3.81814174]),
                np.array([ 5.14552399, 3.31578667, 5.90119363, 4.54515204, 1.12316345, 3.89384963]),
                np.array([ 3.5329307 , 4.79661266, 5.0683084 , 1.87631749]),
                np.array([ 0.36293094, 1.30725604, 0.11578591, 1.5983082 ]),
                np.array([ 3.20443756, 6.26536946, 6.18450567, 1.50406923]),
                np.array([ 0.26999146, 0.26256351, 0.14722687, 0.23137066])]
            ]

    def test_CVNeuralNet_integration(self):
        """integration test for the CVNeuralNet."""
        dev = DummyDevice(wires=self.num_subsystems)

        def circuit(weights):
            qml.template.CVNeuralNet(weights, wires=range(self.N))
            return qml.expval.X(wires=0)

        qnode = qml.QNode(circuit, dev)

        # execution test
        qnode(self.weights)
        queue = qnode.queue

        # Test that gates appear in the right order for each layer:
        # BS-R-S-BS-R-D-K
        for l in range(self.depth):
            gates = [qml.Beamsplitter, qml.Rotation, qml.Squeezing,
                     qml.Beamsplitter, qml.Rotation, qml.Displacement]

            # count the position of each group of gates in the layer
            num_gates_per_type = [0, 6, 4, 4, 6, 4, 4, 4]
            s = np.cumsum(num_gates_per_type)
            gc = l*sum(num_gates_per_type)+np.array(list(zip(s[:-1], s[1:])))

            # loop through expected gates
            for idx, g in enumerate(gates):
                # loop through where these gates should be in the queue
                for opidx, op in enumerate(queue[gc[idx, 0]:gc[idx, 1]]):
                    # check that op in queue is correct gate
                    self.assertTrue(isinstance(op, g))

                    # test that the parameters are correct
                    res_params = op.parameters

                    if idx == 0:
                        # first BS
                        exp_params = [self.weights[l][0][opidx], self.weights[l][1][opidx]]
                    elif idx == 1:
                        # first rot
                        exp_params = [self.weights[l][2][opidx]]
                    elif idx == 2:
                        # squeezing
                        exp_params = [self.weights[l][3][opidx], self.weights[l][4][opidx]]
                    elif idx == 3:
                        # second BS
                        exp_params = [self.weights[l][5][opidx], self.weights[l][6][opidx]]
                    elif idx == 4:
                        # second rot
                        exp_params = [self.weights[l][7][opidx]]
                    elif idx == 5:
                        # displacement
                        exp_params = [self.weights[l][8][opidx], self.weights[l][9][opidx]]

                    self.assertEqual(res_params, exp_params)


class TestStronglyEntanglingCircuit(BaseTest):
    """Tests for the StronglyEntanglingCircuit from the pennylane.template module."""

    def test_integration(self):
        """integration test for the StronglyEntanglingCircuit."""
        np.random.seed(12)
        num_layers = 2

        for num_wires in range(2, 4):
            dev = qml.device('default.qubit', wires=num_wires)
            weights = np.random.randn(num_layers, num_wires, 3)
            def circuit(weights):
                qml.template.StronglyEntanglingCircuit(weights, True, wires=range(num_wires))
                return qml.expval.PauliZ(0)

            qnode = qml.QNode(circuit, dev)
            qnode(weights)
            queue = qnode.queue

            # Test that gates appear in the right order
            exp_gates = [qml.Rot]*num_wires + [qml.CNOT]*num_wires
            exp_gates *= num_layers
            res_gates = [op for op in queue]

            for op1, op2 in zip(res_gates, exp_gates):
                self.assertTrue(isinstance(op1, op2))

            # test the device parameters
            for l in range(num_layers):
                layer_ops = queue[2*l*num_wires:2*(l+1)*num_wires]

                # check each rotation gate parameter
                for n in range(num_wires):
                    res_params = layer_ops[n].parameters
                    exp_params = weights[l, n, :]
                    self.assertAllEqual(res_params, exp_params)

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

        res = np.array(outcomes)
        expected = np.array([-0.29242496, 0.22129055, 0.07540091, -0.77626557])
        self.assertAllAlmostEqual(res, expected, delta=self.tol)


if __name__ == '__main__':
    print('Testing PennyLane version ' + qml.version() + ', pennylane.template.')
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (TestInterferometer, TestCVNeuralNet, TestStronglyEntanglingCircuit):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)
    unittest.TextTestRunner().run(suite)

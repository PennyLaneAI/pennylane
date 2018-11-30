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
Unit tests for the :mod:`pennylane.plugin.DefaultGaussian` device.
"""
# pylint: disable=protected-access,cell-var-from-loop
import unittest
import inspect
import logging as log

from pennylane import numpy as np

from defaults import pennylane as qml, BaseTest


log.getLogger('defaults')

class TestModel(BaseTest):
    """Tests for the pennylane.model module."""

    def test_variational_classifyer(self):
        """Tests the VariationalClassifyer for various parameters."""
        outcomes = []
        for num_wires in range(2,4):
            for num_layers in range(1,3):

                dev = qml.device('default.qubit', wires=num_wires)

                @qml.qnode(dev)
                def circuit(weights, x=None):
                    qml.BasisState(x, wires=range(num_wires))
                    qml.model.CircuitCentricClassifier(weights, True, wires=range(num_wires))
                    return qml.expval.PauliZ(0)

                np.random.seed(0)
                weights=np.random.randn(num_layers, num_wires, 3)
                outcomes.append(circuit(weights, x=np.array(np.random.randint(0,1,num_wires))))
        self.assertAllAlmostEqual(np.array(outcomes), np.array([-0.2924249613746693, -0.3005638739775734, -0.28908176874810054, 0.1339692149350618]), delta=self.tol)

if __name__ == '__main__':
    print('Testing PennyLane version ' + qml.version() + ', pennylane.model.')
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (TestModel):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)
    unittest.TextTestRunner().run(suite)

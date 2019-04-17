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
from unittest.mock import patch
import inspect
import logging as log

import pennylane as qml

from pennylane import numpy as np
from scipy.linalg import block_diag

from defaults import pennylane as qml, BaseTest
from pennylane.expval import Identity
from pennylane.qnode import QuantumFunctionError
from pennylane.plugins import DefaultQubit

log.getLogger('defaults')

class TestExpval(BaseTest):
    """Tests that the Expectations in expval work propperly."""

    def test_identiy_raises_exception_if_outside_qnode(self):
        """Tests that proper exceptions are raised if we try to call Idenity
        outside a QNode."""
        self.logTestName()

        with self.assertRaisesRegex(QuantumFunctionError, 'can only be used inside a qfunc'):
            Identity(wires=0)

    def test_identiy_raises_exception_if_cannot_guess_device_type(self):
        """Tests that proper exceptions are raised if Identity fails to guess
        whether on a device is CV or qubit."""
        self.logTestName()

        dev = qml.device('default.qubit', wires=1)
        dev._expectation_map = {}

        @qml.qnode(dev)
        def circuit():
            return qml.expval.Identity(wires=0)

        with self.assertRaisesRegex(QuantumFunctionError, 'Unable to determine whether this device supports CV or qubit'):
            circuit()

    @patch.object(DefaultQubit, 'pre_expval', lambda self: log.info(self.expval_queue))
    def test_pass_positional_wires_to_expval(self):
        """Tests whether the ability to pass wires as positional argument
        is retained"""
        self.logTestName()

        dev = qml.device('default.qubit', wires=1)

        @qml.qnode(dev)
        def circuit():
            return qml.expval.Identity(0)

        with self.assertLogs(level='INFO') as l:
            circuit()
            self.assertEqual(len(l.output), 1)
            self.assertEqual(len(l.records), 1)
            self.assertIn('pennylane.expval.qubit.Identity object', l.output[0])


if __name__ == '__main__':
    print('Testing PennyLane version ' + qml.version() + ', expval.')
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in ([TestExpval]):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)
    unittest.TextTestRunner().run(suite)

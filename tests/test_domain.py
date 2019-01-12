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
Unit tests for the :mod:`pennylane.domain` module.
"""
# pylint: disable=protected-access,cell-var-from-loop
import unittest
import inspect
import logging as log
import scipy as sp
from scipy.linalg import qr

from pennylane import numpy as np

from defaults import pennylane as qml, BaseTest
import pennylane.domain as dom

log.getLogger('defaults')

class TestConainsOperation(BaseTest):
    """Tests that the contains operations work as intended."""

    def test_foo(self):
        """Tests the CircuitCentric for various parameters."""
        assert(dom.Complex() in dom.Scalars())
        assert(dom.Complex() not in dom.Reals())

if __name__ == '__main__':
    print('Testing PennyLane version ' + qml.version() + ', pennylane.domain.')
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (TestConainsOperation, ):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)
    unittest.TextTestRunner().run(suite)

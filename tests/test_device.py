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
Unit tests for the :mod:`openqml` :class:`Device` class.
"""

import unittest
import logging as log
log.getLogger()

import autograd
from autograd import numpy as np

from defaults import openqml as qm, BaseTest

class DeviceTest(BaseTest):
    """Device tests.
    """
    def setUp(self):
        self.default_devices = ['default.qubit',
                                'strawberryfields.gaussian',
                                'strawberryfields.fock']

    def test_default_devices(self):
        "Tests that all of the default devices can be properly instantiated."
        log.info('...')

        for device_name in self.default_devices:
            dev = qm.device(device_name, wires=0, cutoff_dim=5)
            self.assertEqual(dev.short_name, device_name)



if __name__ == '__main__':
    print('Testing OpenQML version ' + qm.version() + ', Device class.')
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (DeviceTest,):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)

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
Default parameters, commandline arguments and common routines for the unit tests.
"""
import unittest
import os
import sys
import logging

import numpy as np

# Make sure pennylane is always imported from the same source distribution where the tests reside, not e.g., from site-packages.
# See https://docs.python-guide.org/en/latest/writing/structure/#test-suite
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pennylane

# defaults
TOLERANCE = os.environ.get("TOL", 1e-5)


# set up logging
if "LOGGING" in os.environ:
    logLevel = os.environ["LOGGING"]
    print('Logging:', logLevel)
    numeric_level = getattr(logging, logLevel.upper(), 10)
else:
    numeric_level = 100

logging.getLogger().setLevel(numeric_level)
logging.captureWarnings(True)


class BaseTest(unittest.TestCase):
    """ABC for tests.
    Encapsulates the user-given commandline parameters for the test run as class attributes.
    """
    tol = TOLERANCE

    def logTestName(self):
        logging.info('{}'.format(self.id()))

    def assertAllAlmostEqual(self, first, second, delta, msg=None):
        """
        Like assertAlmostEqual, but works with arrays. All the corresponding elements have to be almost equal.
        """
        if isinstance(first, tuple):
            # check each element of the tuple separately (needed for when the tuple elements are themselves batches)
            if np.all([np.all(first[idx] == second[idx]) for idx, _ in enumerate(first)]):
                return
            if np.all([np.all(np.abs(first[idx] - second[idx]) <= delta) for idx, _ in enumerate(first)]):
                return
        else:
            if np.all(first == second):
                return
            if np.all(np.abs(np.array(first) - np.array(second)) <= delta):
                return
        standardMsg = '{} != {} within {} delta'.format(first, second, delta)
        msg = self._formatMessage(msg, standardMsg)
        raise self.failureException(msg)

    def assertAllEqual(self, first, second, msg=None):
        """
        Like assertEqual, but works with arrays. All the corresponding elements have to be equal.
        """
        return self.assertAllAlmostEqual(first, second, delta=0.0, msg=msg)

    def assertAllTrue(self, value, msg=None):
        """
        Like assertTrue, but works with arrays. All the corresponding elements have to be True.
        """
        return self.assertTrue(np.all(value))

    def assertAlmostLess(self, first, second, delta, msg=None):
        """
        Like assertLess, but with a tolerance.
        """
        return self.assertLess(first, second+delta, msg=msg)

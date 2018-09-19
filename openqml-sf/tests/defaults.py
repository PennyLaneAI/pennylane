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

import argparse
import unittest
import os
import sys
import logging as log

import numpy as np

# Make sure openqml is always imported from the same source distribution where the tests reside, not e.g. from site-packages.
# See https://docs.python-guide.org/en/latest/writing/structure/#test-suite
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import openqml_sf


# defaults
TOLERANCE = 1e-5


def get_commandline_args():
    """Parse the commandline arguments for the unit tests.
    If none are given (e.g. when the test is run as a module instead of a script), the defaults are used.

    Returns:
      argparse.Namespace: parsed arguments in a namespace container
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tolerance', type=float, default=TOLERANCE, help='Numerical tolerance for equality tests.')
    parser.add_argument('--log', action='store_true', default=False, help='Turn verbose logging on.')

    # HACK: We only parse known args to enable unittest test discovery without parsing errors.
    args, _ = parser.parse_known_args()

    # set up logging for tests
    log.basicConfig(format='%(levelname)s: %(message)s', level=log.INFO if args.log else log.WARN)
    return args


# parse any possible commandline arguments
args = get_commandline_args()


class BaseTest(unittest.TestCase):
    """ABC for tests.
    Encapsulates the user-given commandline parameters for the test run as class attributes.
    """
    args = args
    tol = args.tolerance

    def assertAllAlmostEqual(self, first, second, delta, msg=None):
        """
        Like assertAlmostEqual, but works with arrays. All the corresponding elements have to be almost equal.
        """
        if isinstance(first, tuple):
            # check each element of the tuple separately (needed for when the tuple elements are themselves batches)
            if np.all([np.all(first[idx] == second[idx]) for idx, _ in enumerate(first)]):
                return
            if np.all([np.all(np.abs(first[idx] - second[idx])) <= delta for idx, _ in enumerate(first)]):
                return
        else:
            if np.all(first == second):
                return
            if np.all(np.abs(first - second) <= delta):
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

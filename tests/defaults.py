"""
Default parameters, commandline arguments and common routines for the unit tests.
"""

import argparse
import unittest
import os
import sys

import numpy as np

# Make sure openqml is always imported from the same source distribution where the tests reside, not e.g. from site-packages.
# See https://docs.python-guide.org/en/latest/writing/structure/#test-suite
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import openqml


# defaults
TOLERANCE = 1e-6


def get_commandline_args():
    """Parse the commandline arguments for the unit tests.
    If none are given (e.g. when the test is run as a module instead of a script), the defaults are used.

    Returns:
      argparse.Namespace: parsed arguments in a namespace container
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tolerance', type=float, default=TOLERANCE, help='Numerical tolerance for equality tests.')

    # HACK: We only parse known args to enable unittest test discovery without parsing errors.
    args, _ = parser.parse_known_args()
    return args


# parse any possible commandline arguments
args = get_commandline_args()


class BaseTest(unittest.TestCase):
    """ABC for tests.
    Encapsulates the user-given commandline parameters for the test run.
    """
    def setUp(self):
        self.args = args
        self.tol = args.tolerance

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

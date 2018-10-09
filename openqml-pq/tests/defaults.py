"""
Default parameters, commandline arguments and common routines for the unit tests.
"""

import argparse
import unittest
import os
import sys
#import numpy as np

import openqml as qm
from openqml import numpy as np

# Make sure openqml_pq is always imported from the same source distribution where the tests reside, not e.g. from site-packages.
# See https://docs.python-guide.org/en/latest/writing/structure/#test-suite
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import openqml_pq



BACKEND = "simulator"
OPTIMIZER = "SGD"
TOLERANCE = 1e-3
BATCH_SIZE = 2
BATCHED = False
MIXED = False

if "BACKEND" in os.environ:
    BACKEND = os.environ["BACKEND"]
    print('Backend:', BACKEND)

if "BATCHED" in os.environ:
    BATCHED = bool(int(os.environ["BATCHED"]))
    print('Batched:', BATCHED)

if "MIXED" in os.environ:
    MIXED = bool(int(os.environ["MIXED"]))
    print('Mixed:', MIXED)

def get_commandline_args():
    """Parse the commandline arguments for the unit tests.
    If none are given (e.g. when the test is run as a module instead of a script), the defaults are used.

    Returns:
      argparse.Namespace: parsed arguments in a namespace container
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--backend',   type=str,   default=BACKEND,   help='Backend to use for tests.', choices=['simulator', 'ibm'])
    parser.add_argument('-t', '--tolerance', type=float, default=TOLERANCE, help='Numerical tolerance for equality tests.')
    parser.add_argument('--batch_size',      type=int,   default=BATCH_SIZE,         help='Batch size.')
    parser.add_argument("--user", help="IBM Quantum Experience user name")
    parser.add_argument("--password", help="IBM Quantum Experience password")
    parser.add_argument("--optimizer", default=OPTIMIZER, choices=qm.optimizer.OPTIMIZER_NAMES, help="optimizer to use")

    batch_parser = parser.add_mutually_exclusive_group(required=False)
    batch_parser.add_argument('--batched', dest='batched', action='store_true')
    batch_parser.add_argument('--no-batched', dest='batched', action='store_false')
    parser.set_defaults(batched=BATCHED)

    mixed_parser = parser.add_mutually_exclusive_group(required=False)
    mixed_parser.add_argument('--mixed', dest='mixed', action='store_true')
    mixed_parser.add_argument('--pure', dest='mixed', action='store_false')
    parser.set_defaults(mixed=MIXED)

    # HACK: We only parse known args to enable unittest test discovery without parsing errors.
    args, _ = parser.parse_known_args()
    setup_plugin(args)
    return args

def setup_plugin(args):
    pass

# parse any possible commandline arguments
args = get_commandline_args()

class BaseTest(unittest.TestCase):
    """ABC for tests.
    Encapsulates the user-given commandline parameters for the test run.
    """
    num_subsystems = None  #: int: number of wires for the backend, must be overridden by child classes

    def setUp(self):
        self.args = args
        self.backend = args.backend
        self.tol = args.tolerance
        self.batched = args.batched

        # keyword arguments for the backend
        self.kwargs = dict(pure=not args.mixed)
        if args.batched:
            self.kwargs["batch_size"] = args.batch_size
            self.bsize = args.batch_size
        else:
            self.bsize = 1

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

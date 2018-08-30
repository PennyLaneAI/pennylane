"""
Unit tests for the :mod:`openqml` utility classes :class:`ParRef`, :class:`Command`.
"""

import unittest
import logging as log

import numpy as np
import numpy.random as nr

from defaults import openqml, BaseTest
from openqml.circuit import (ParRef, Command)


class BasicTest(BaseTest):
    """Utility class tests."""

    def test_parref(self):
        "Parameter reference tests."

        n = 10
        m = nr.randn(n)  # parameter multipliers
        par_fixed = nr.randn(n)  # fixed parameter values
        par_free  = nr.randn(n)  # free parameter values

        # test __str__()
        p = ParRef(0)
        log.info(1.2 * p * 0.4)
        log.info(-p)
        log.info(p)

        def check(par, res):
            "Apply the parameter mapping, compare with the expected result."
            self.assertAllAlmostEqual(ParRef.map(par, par_free), res, self.tol)

        # mapping function must yield the correct parameter values
        par = [m[k] * ParRef(k) for k in range(n)]
        check(par, par_free * m)

        par = [ParRef(k) * m[k] for k in range(n)]
        check(par, par_free * m)

        par = [-ParRef(k) for k in range(n)]
        check(par, -par_free)

        par = [m[k] * -ParRef(k) * m[k] for k in range(n)]
        check(par, -par_free * m**2)

        # fixed values remain constant
        check(par_fixed, par_fixed)


    def test_command(self):
        "Command class tests."
        pass


if __name__ == '__main__':
    print('Testing OpenQML version ' + openqml.version() + ', utility classes.')
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (BasicTest,):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)

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
Unit tests for the :mod:`pennylane` utility classes :class:`ParRef`, :class:`Command`.
"""
import unittest
import logging as log
from string import ascii_lowercase

import numpy as np
import numpy.random as nr

from defaults import pennylane, BaseTest
from pennylane.variable import Variable

log.getLogger('defaults')

class BasicTest(BaseTest):
    """Variable class tests."""
    def test_variable(self):
        """Positional Variable reference tests."""
        self.logTestName()

        n = 10
        m = nr.randn(n)  # parameter multipliers
        par_fixed = nr.randn(n)  # fixed parameter values
        par_free  = nr.randn(n)  # free parameter values
        Variable.free_param_values = par_free

        # test __str__()
        p = Variable(0)
        self.assertEqual(str(p), "Variable 0: name = None, ")
        self.assertEqual(str(-p), "Variable 0: name = None,  * -1")
        self.assertEqual(str(1.2*p*0.4), "Variable 0: name = None,  * 0.48")

        def check(par, res):
            "Apply the parameter mapping, compare with the expected result."
            temp = np.array([p.val if isinstance(p, Variable) else p for p in par])
            self.assertAllAlmostEqual(temp, res, self.tol)

        # mapping function must yield the correct parameter values
        par = [m[k] * Variable(k) for k in range(n)]
        check(par, par_free * m)
        par = [Variable(k) * m[k] for k in range(n)]
        check(par, par_free * m)
        par = [-Variable(k) for k in range(n)]
        check(par, -par_free)
        par = [m[k] * -Variable(k) * m[k] for k in range(n)]
        check(par, -par_free * m**2)

        # fixed values remain constant
        check(par_fixed, par_fixed)

    def test_keyword_variable(self):
        """Keyword Variable reference tests."""
        self.logTestName()

        n = 10
        m = nr.randn(n)  # parameter multipliers
        par_fixed = nr.randn(n)  # fixed parameter values
        par_free  = nr.randn(n)  # free parameter values
        Variable.kwarg_values = {k: v for k, v in zip(ascii_lowercase, par_free)}

        # test __str__()
        p = Variable(0, name='kw1')
        self.assertEqual(str(p), "Variable 0: name = kw1, ")
        self.assertEqual(str(-p), "Variable 0: name = kw1,  * -1")
        self.assertEqual(str(1.2*p*0.4), "Variable 0: name = kw1,  * 0.48")

        def check(par, res):
            "Apply the parameter mapping, compare with the expected result."
            temp = np.array([p.val if isinstance(p, Variable) else p for p in par]).flatten()
            self.assertAllAlmostEqual(temp, res, self.tol)

        # mapping function must yield the correct parameter values
        par = [m[k] * Variable(name=n) for k,n in zip(range(n), ascii_lowercase)]
        check(par, par_free * m)
        par = [Variable(name=n) * m[k] for k,n in zip(range(n), ascii_lowercase)]
        check(par, par_free * m)
        par = [-Variable(name=n) for k,n in zip(range(n), ascii_lowercase)]
        check(par, -par_free)
        par = [m[k] * -Variable(name=n) * m[k] for k,n in zip(range(n), ascii_lowercase)]
        check(par, -par_free * m**2)

        # fixed values remain constant
        check(par_fixed, par_fixed)

if __name__ == '__main__':
    print('Testing PennyLane version ' + pennylane.version() + ', Variable class.')
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (BasicTest,):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)
    unittest.TextTestRunner().run(suite)

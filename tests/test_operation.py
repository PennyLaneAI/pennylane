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
Unit tests for :mod:`openqml.operation`.
"""
import unittest
import logging as log
log.getLogger('defaults')

import numpy as np
import numpy.random as nr

from defaults import openqml, BaseTest
import openqml.qnode as oq
import openqml.operation as oo
import openqml.variable as ov


dev = openqml.device('default.qubit', wires=2)


class BasicTest(BaseTest):
    """Utility class tests."""
    def test_heisenberg(self):
        "Heisenberg picture adjoint actions of CV Operations."
        self.logTestName()

        def h_test(cls):
            "Test a gaussian CV operation."
            log.debug('\tTesting: cls.{}'.format(cls.__name__))
            # fixed parameter values
            if cls.par_domain == 'A':
                par = [nr.randn(1,1)] * cls.num_params
            else:
                par = list(nr.randn(cls.num_params))
            ww = list(range(cls.num_wires))
            op = cls(*par, wires=ww, do_queue=False)

            if issubclass(cls, oo.Expectation):
                Q = op.heisenberg_obs(0)
                # ev_order equals the number of dimensions of the H-rep array
                self.assertEqual(Q.ndim, cls.ev_order)
                return

            # not an Expectation
            # all gaussian ops use the 'A' method
            self.assertEqual(cls.grad_method, 'A')
            U = op.heisenberg_tr(2)
            I = np.eye(*U.shape)
            # first row is always (1,0,0...)
            self.assertAllEqual(U[0,:], I[:,0])

            # check the inverse transform
            V = op.heisenberg_tr(2, inverse=True)
            self.assertAlmostEqual(np.linalg.norm(U @ V -I), 0, delta=self.tol)
            self.assertAlmostEqual(np.linalg.norm(V @ U -I), 0, delta=self.tol)

            # compare gradient recipe to numerical gradient
            h = 1e-7
            U = op.heisenberg_tr(0)
            for k in range(cls.num_params):
                D = op.heisenberg_pd(k)  # using the recipe
                # using finite difference
                op.params[k] += h
                Up = op.heisenberg_tr(0)
                op.params = par
                G = (Up-U) / h
                self.assertAllAlmostEqual(D, G, delta=self.tol)

            # make sure that `heisenberg_expand` method receives enough wires to actually expand
            # when supplied `wires` value is zero, returns unexpanded matrix instead of raising Error
            # so only check multimode ops
            if len(op.wires) > 1:
                with self.assertRaisesRegex(ValueError, 'is too small to fit Heisenberg matrix'):
                    op.heisenberg_expand(U, len(op.wires) - 1)

            # validate size of input for `heisenberg_expand` method
            with self.assertRaisesRegex(ValueError, 'Heisenberg matrix is the wrong size'):
                U = U[1:,1:]
                op.heisenberg_expand(U, len(op.wires))


        for cls in openqml.ops.cv.all_ops + openqml.expval.cv.all_ops:
            if cls.supports_analytic:  # only test gaussian operations
                h_test(cls)


    def test_ops(self):
        "Operation initialization."
        self.logTestName()

        def op_test(cls):
            "Test the Operation subclass."
            log.debug('\tTesting: cls.{}'.format(cls.__name__))
            n = cls.num_params
            w = cls.num_wires
            ww = list(range(w))
            # valid pars
            if cls.par_domain == 'A':
                pars = [np.eye(2)] * n
            elif cls.par_domain == 'N':
                pars = [0] * n
            else:
                pars = [0.0] * n

            # valid call
            cls(*pars, wires=ww, do_queue=False)

            # too many parameters
            with self.assertRaisesRegex(ValueError, 'wrong number of parameters'):
                cls(*(n+1)*[0], wires=ww, do_queue=False)

            # too few parameters
            if n > 0:
                with self.assertRaisesRegex(ValueError, 'wrong number of parameters'):
                    cls(*(n-1)*[0], wires=ww, do_queue=False)

            if w > 0:
                # too many or too few wires
                with self.assertRaisesRegex(ValueError, 'wrong number of wires'):
                    cls(*pars, wires=list(range(w+1)), do_queue=False)
                with self.assertRaisesRegex(ValueError, 'wrong number of wires'):
                    cls(*pars, wires=list(range(w-1)), do_queue=False)
                # repeated wires
                if w > 1:
                    with self.assertRaisesRegex(ValueError, 'wires must be unique'):
                        cls(*pars, wires=w*[0], do_queue=False)

            if n == 0:
                return

            # wrong parameter types
            if cls.par_domain == 'A':
                # params must be arrays
                with self.assertRaisesRegex(TypeError, 'Array parameter expected'):
                    cls(*n*[0.0], wires=ww, do_queue=False)
            elif cls.par_domain == 'N':
                # params must be natural numbers
                with self.assertRaisesRegex(TypeError, 'Natural number'):
                    cls(*n*[0.7], wires=ww, do_queue=False)
                with self.assertRaisesRegex(TypeError, 'Natural number'):
                    cls(*n*[-1], wires=ww, do_queue=False)
            elif cls.par_domain == 'R':
                # params must be real numbers
                with self.assertRaisesRegex(TypeError, 'Real scalar parameter expected'):
                    cls(*n*[1j], wires=ww, do_queue=False)

            # if par_domain ever gets overridden to an unsupported value, should raise exception
            tmp = cls.par_domain
            with self.assertRaisesRegex(TypeError, 'Unknown parameter domain'):
                cls.par_domain = 'junk'
                cls(*n*[0.0], wires=ww, do_queue=False)
                cls.par_domain = 7
                cls(*n*[0.0], wires=ww, do_queue=False)

            cls.par_domain = tmp


        for cls in openqml.ops.qubit.all_ops:
            op_test(cls)

        for cls in openqml.ops.cv.all_ops:
            op_test(cls)

        for cls in openqml.expval.qubit.all_ops:
            op_test(cls)

        for cls in openqml.expval.cv.all_ops:
            op_test(cls)


if __name__ == '__main__':
    print('Testing OpenQML version ' + openqml.version() + ', Operation class.')
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (BasicTest,):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)
    unittest.TextTestRunner().run(suite)

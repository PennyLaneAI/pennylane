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
Unit tests for :mod:`pennylane.operation`.
"""
import unittest
import logging as log
log.getLogger('defaults')

import numpy as np
import numpy.random as nr

from defaults import pennylane, BaseTest
import pennylane.operation as oo
import pennylane.variable as ov


dev = pennylane.device('default.qubit', wires=2)


class BasicTest(BaseTest):
    """Utility class tests."""
    def test_heisenberg(self):
        "Heisenberg picture adjoint actions of CV Operations."
        self.logTestName()

        def h_test(cls):
            "Test a gaussian CV operation."
            log.debug('\tTesting: cls.{}'.format(cls.__name__))

            ww = list(range(cls.num_wires))

            # fixed parameter values
            if cls.par_domain == 'A':
                if cls.__name__ == "Interferometer":
                    ww = list(range(2))
                    par = [np.array([[0.83645892-0.40533293j, -0.20215326+0.30850569j],
                                     [-0.23889780-0.28101519j, -0.88031770-0.29832709j]])]
                else:
                    par = [np.array([[-1.82624687]])] * cls.num_params
            else:
                par = [-0.069125, 0.51778, 0.91133, 0.95904][:cls.num_params]

            op = cls(*par, wires=ww, do_queue=False)

            if issubclass(cls, oo.Expectation):
                Q = op.heisenberg_obs(0)
                # ev_order equals the number of dimensions of the H-rep array
                self.assertEqual(Q.ndim, cls.ev_order)
                return

            # not an Expectation

            U = op.heisenberg_tr(num_wires=2)
            I = np.eye(*U.shape)
            # first row is always (1,0,0...)
            self.assertAllEqual(U[0,:], I[:,0])

            # check the inverse transform
            V = op.heisenberg_tr(num_wires=2, inverse=True)
            self.assertAlmostEqual(np.linalg.norm(U @ V -I), 0, delta=self.tol)
            self.assertAlmostEqual(np.linalg.norm(V @ U -I), 0, delta=self.tol)

            if op.grad_recipe is not None:
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
                U_wrong_size = U[1:,1:]
                op.heisenberg_expand(U_wrong_size, len(op.wires))

            # ensure that `heisenberg_expand` raises exception if it receives an array with order > 2
            with self.assertRaisesRegex(ValueError, 'Only order-1 and order-2 arrays supported'):
                U_high_order = np.array([U] * 3)
                op.heisenberg_expand(U_high_order, len(op.wires))

        for cls in pennylane.ops.cv.all_ops + pennylane.expval.cv.all_ops:
            if cls.supports_heisenberg:  # only test gaussian operations
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
                # params must not be Variables
                with self.assertRaisesRegex(TypeError, 'Array parameter expected'):
                    cls(*n*[ov.Variable(0)], wires=ww, do_queue=False)
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
            with self.assertRaisesRegex(ValueError, 'Unknown parameter domain'):
                cls.par_domain = 'junk'
                cls(*n*[0.0], wires=ww, do_queue=False)
                cls.par_domain = 7
                cls(*n*[0.0], wires=ww, do_queue=False)

            cls.par_domain = tmp


        for cls in pennylane.ops.qubit.all_ops:
            op_test(cls)

        for cls in pennylane.ops.cv.all_ops:
            op_test(cls)

        for cls in pennylane.expval.qubit.all_ops:
            op_test(cls)

        for cls in pennylane.expval.cv.all_ops:
            op_test(cls)

    def test_operation_outside_queue(self):
        """Test that an error is raised if an operation is called
        outside of a QNode context."""
        self.logTestName()

        with self.assertRaisesRegex(pennylane.QuantumFunctionError, "can only be used inside a qfunc"):
            pennylane.qubit.Hadamard(wires=0)

    def test_operation_no_queue(self):
        """Test that an operation can be called outside a QNode with the do_queue flag"""
        self.logTestName()

        try:
            pennylane.qubit.Hadamard(wires=0, do_queue=False)
        except pennylane.QuantumFunctionError:
            self.fail("Operation failed to instantiate outside of QNode with do_queue=False.")


class DeveloperTests(BaseTest):
    """Test custom operations construction."""

    def test_incorrect_num_wires(self):
        """Test that an exception is raised if called with wrong number of wires"""
        self.logTestName()

        class DummyOp(oo.Operation):
            r"""Dummy custom operation"""
            num_wires = 1
            num_params = 1
            par_domain = 'R'
            grad_method = 'A'

        with self.assertRaisesRegex(ValueError, "wrong number of wires"):
            DummyOp(0.5, wires=[0, 1], do_queue=False)

    def test_incorrect_num_params(self):
        """Test that an exception is raised if called with wrong number of parameters"""
        self.logTestName()

        class DummyOp(oo.Operation):
            r"""Dummy custom operation"""
            num_wires = 1
            num_params = 1
            par_domain = 'R'
            grad_method = 'A'

        with self.assertRaisesRegex(ValueError, "wrong number of parameters"):
            DummyOp(0.5, 0.6, wires=0, do_queue=False)

    def test_incorrect_param_domain(self):
        """Test that an exception is raised if an incorrect parameter domain is requested"""
        self.logTestName()

        class DummyOp(oo.Operation):
            r"""Dummy custom operation"""
            num_wires = 1
            num_params = 1
            par_domain = 'J'
            grad_method = 'A'

        with self.assertRaisesRegex(ValueError, "Unknown parameter domain"):
            DummyOp(0.5, wires=0, do_queue=False)

    def test_incorrect_grad_recipe_length(self):
        """Test that an exception is raised if len(grad_recipe)!=len(num_params)"""
        self.logTestName()

        class DummyOp(oo.CVOperation):
            r"""Dummy custom operation"""
            num_wires = 1
            num_params = 1
            par_domain = 'R'
            grad_method = 'A'
            grad_recipe = [(0.5, 0.1), (0.43, 0.1)]

        with self.assertRaisesRegex(AssertionError, "Gradient recipe must have one entry for each parameter"):
            DummyOp(0.5, wires=[0, 1], do_queue=False)

    def test_grad_method_with_integer_params(self):
        """Test that an exception is raised if a non-None grad-method is provided for natural number params"""
        self.logTestName()

        class DummyOp(oo.Operation):
            r"""Dummy custom operation"""
            num_wires = 1
            num_params = 1
            par_domain = 'N'
            grad_method = 'A'

        with self.assertRaisesRegex(AssertionError, "An operation may only be differentiated with respect to real scalar parameters"):
            DummyOp(5, wires=[0, 1], do_queue=False)

    def test_analytic_grad_with_array_param(self):
        """Test that an exception is raised if an analytic gradient is requested with an array param"""
        self.logTestName()

        class DummyOp(oo.Operation):
            r"""Dummy custom operation"""
            num_wires = 1
            num_params = 1
            par_domain = 'A'
            grad_method = 'A'

        with self.assertRaisesRegex(AssertionError, "Operations that depend on arrays containing free variables may only be differentiated using the F method"):
            DummyOp(np.array([1.]), wires=[0, 1], do_queue=False)

    def test_numerical_grad_with_grad_recipe(self):
        """Test that an exception is raised if a numerical gradient is requested with a grad recipe"""
        self.logTestName()

        class DummyOp(oo.Operation):
            r"""Dummy custom operation"""
            num_wires = 1
            num_params = 1
            par_domain = 'R'
            grad_method = 'F'
            grad_recipe = [(0.5, 0.1)]

        with self.assertRaisesRegex(AssertionError, "Gradient recipe is only used by the A method"):
            DummyOp(0.5, wires=[0, 1], do_queue=False)

    def test_variable_instead_of_array(self):
        """Test that an exception is raised if an array is expected but a variable is passed"""
        self.logTestName()

        class DummyOp(oo.Operation):
            r"""Dummy custom operation"""
            num_wires = 1
            num_params = 1
            par_domain = 'A'
            grad_method = 'A'

        with self.assertRaisesRegex(TypeError, "Array parameter expected, got a Variable"):
            DummyOp(ov.Variable(0), wires=[0], do_queue=False)

    def test_array_instead_of_flattened_array(self):
        """Test that an exception is raised if an array is expected, but an array is passed
        to check_domain when flattened=True. In the initial release of the library, this is not
        accessible by the developer or the user, but is kept in case it will be used in the future."""
        self.logTestName()

        class DummyOp(oo.Operation):
            r"""Dummy custom operation"""
            num_wires = 1
            num_params = 1
            par_domain = 'A'
            grad_method = 'F'

        with self.assertRaisesRegex(TypeError, "Flattened array parameter expected"):
            op = DummyOp(np.array([1]), wires=[0], do_queue=False)
            op.check_domain(np.array([1]), True)

    def test_scalar_instead_of_array(self):
        """Test that an exception is raised if an array is expected but a scalar is passed"""
        self.logTestName()

        class DummyOp(oo.Operation):
            r"""Dummy custom operation"""
            num_wires = 1
            num_params = 1
            par_domain = 'A'
            grad_method = 'F'

        with self.assertRaisesRegex(TypeError, "Array parameter expected, got"):
            op = DummyOp(0.5, wires=[0], do_queue=False)

    def test_array_instead_of_real(self):
        """Test that an exception is raised if a real number is expected but an array is passed"""
        self.logTestName()

        class DummyOp(oo.Operation):
            r"""Dummy custom operation"""
            num_wires = 1
            num_params = 1
            par_domain = 'R'
            grad_method = 'F'

        with self.assertRaisesRegex(TypeError, "Real scalar parameter expected, got"):
            op = DummyOp(np.array([1.]), wires=[0], do_queue=False)

    def test_not_natural_param(self):
        """Test that an exception is raised if a natural number is expected but not passed"""
        self.logTestName()

        class DummyOp(oo.Operation):
            r"""Dummy custom operation"""
            num_wires = 1
            num_params = 1
            par_domain = 'N'
            grad_method = None

        with self.assertRaisesRegex(TypeError, "Natural number parameter expected, got"):
            op = DummyOp(0.5, wires=[0], do_queue=False)

        with self.assertRaisesRegex(TypeError, "Natural number parameter expected, got"):
            op = DummyOp(-2, wires=[0], do_queue=False)

if __name__ == '__main__':
    print('Testing PennyLane version ' + pennylane.version() + ', Operation class.')
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (BasicTest, DeveloperTests):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)
    unittest.TextTestRunner().run(suite)

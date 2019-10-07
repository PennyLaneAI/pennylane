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
import pytest
import numpy as np

import pennylane as qml
import pennylane.ops
import pennylane.operation as oo
import pennylane.variable as ov


# Operation subclasses to test
op_classes = [getattr(qml.ops, cls) for cls in qml.ops.__all__]
op_classes_cv = [getattr(qml.ops, cls) for cls in qml.ops._cv__all__]
op_classes_gaussian = [cls for cls in op_classes_cv if cls.supports_heisenberg]


class TestOperation:
    """Operation class tests."""

    @pytest.mark.parametrize("cls", op_classes_gaussian)
    def test_heisenberg(self, cls, tol):
        "Heisenberg picture adjoint actions of CV Operations."

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

        if issubclass(cls, qml.operation.Observable):
            Q = op.heisenberg_obs(0)
            # ev_order equals the number of dimensions of the H-rep array
            assert Q.ndim == cls.ev_order
            return

        # not an Expectation

        U = op.heisenberg_tr(num_wires=2)
        I = np.eye(*U.shape)
        # first row is always (1,0,0...)
        assert np.all(U[0, :] == I[:, 0])

        # check the inverse transform
        V = op.heisenberg_tr(num_wires=2, inverse=True)
        assert np.linalg.norm(U @ V -I) == pytest.approx(0, abs=tol)
        assert np.linalg.norm(V @ U -I) == pytest.approx(0, abs=tol)

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
                assert D == pytest.approx(G, abs=tol)

        # make sure that `heisenberg_expand` method receives enough wires to actually expand
        # when supplied `wires` value is zero, returns unexpanded matrix instead of raising Error
        # so only check multimode ops
        if len(op.wires) > 1:
            with pytest.raises(ValueError, match='is too small to fit Heisenberg matrix'):
                op.heisenberg_expand(U, len(op.wires) - 1)

        # validate size of input for `heisenberg_expand` method
        with pytest.raises(ValueError, match='Heisenberg matrix is the wrong size'):
            U_wrong_size = U[1:, 1:]
            op.heisenberg_expand(U_wrong_size, len(op.wires))

        # ensure that `heisenberg_expand` raises exception if it receives an array with order > 2
        with pytest.raises(ValueError, match='Only order-1 and order-2 arrays supported'):
            U_high_order = np.array([U] * 3)
            op.heisenberg_expand(U_high_order, len(op.wires))



    @pytest.mark.parametrize("cls", op_classes)
    def test_operation_init(self, cls, monkeypatch):
        "Operation subclass initialization."

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
        op = cls(*pars, wires=ww, do_queue=False)
        assert op.name == cls.__name__
        assert op.params == pars
        assert op._wires == ww

        # too many parameters
        with pytest.raises(ValueError, match='wrong number of parameters'):
            cls(*(n+1)*[0], wires=ww, do_queue=False)

        # too few parameters
        if n > 0:
            with pytest.raises(ValueError, match='wrong number of parameters'):
                cls(*(n-1)*[0], wires=ww, do_queue=False)

        if w > 0:
            # too many or too few wires
            with pytest.raises(ValueError, match='wrong number of wires'):
                cls(*pars, wires=list(range(w+1)), do_queue=False)
            with pytest.raises(ValueError, match='wrong number of wires'):
                cls(*pars, wires=list(range(w-1)), do_queue=False)
            # repeated wires
            if w > 1:
                with pytest.raises(ValueError, match='wires must be unique'):
                    cls(*pars, wires=w*[0], do_queue=False)

        if n == 0:
            return

        # wrong parameter types
        if cls.par_domain == 'A':
            # params must be arrays
            with pytest.raises(TypeError, match='Array parameter expected'):
                cls(*n*[0.0], wires=ww, do_queue=False)
            # params must not be Variables
            with pytest.raises(TypeError, match='Array parameter expected'):
                cls(*n*[ov.Variable(0)], wires=ww, do_queue=False)
        elif cls.par_domain == 'N':
            # params must be natural numbers
            with pytest.raises(TypeError, match='Natural number'):
                cls(*n*[0.7], wires=ww, do_queue=False)
            with pytest.raises(TypeError, match='Natural number'):
                cls(*n*[-1], wires=ww, do_queue=False)
        elif cls.par_domain == 'R':
            # params must be real numbers
            with pytest.raises(TypeError, match='Real scalar parameter expected'):
                cls(*n*[1j], wires=ww, do_queue=False)

        # if par_domain ever gets overridden to an unsupported value, should raise exception
        monkeypatch.setattr(cls, 'par_domain', 'junk')
        with pytest.raises(ValueError, match='Unknown parameter domain'):
            cls(*pars, wires=ww, do_queue=False)

        monkeypatch.setattr(cls, 'par_domain', 7)
        with pytest.raises(ValueError, match='Unknown parameter domain'):
            cls(*pars, wires=ww, do_queue=False)



    def test_operation_outside_queue(self):
        """Test that an error is raised if an operation is called
        outside of a QNode context."""

        with pytest.raises(qml.QuantumFunctionError, match="can only be used inside a qfunc"):
            qml.ops.Hadamard(wires=0)

    def test_operation_no_queue(self):
        """Test that an operation can be called outside a QNode with the do_queue flag"""
        try:
            qml.ops.Hadamard(wires=0, do_queue=False)
        except qml.QuantumFunctionError:
            pytest.fail("Operation failed to instantiate outside of QNode with do_queue=False.")


class TestOperationConstruction:
    """Test custom operations construction."""

    def test_incorrect_num_wires(self):
        """Test that an exception is raised if called with wrong number of wires"""

        class DummyOp(qml.operation.Operation):
            r"""Dummy custom operation"""
            num_wires = 1
            num_params = 1
            par_domain = 'R'
            grad_method = 'A'

        with pytest.raises(ValueError, match="wrong number of wires"):
            DummyOp(0.5, wires=[0, 1], do_queue=False)

    def test_incorrect_num_params(self):
        """Test that an exception is raised if called with wrong number of parameters"""

        class DummyOp(qml.operation.Operation):
            r"""Dummy custom operation"""
            num_wires = 1
            num_params = 1
            par_domain = 'R'
            grad_method = 'A'

        with pytest.raises(ValueError, match="wrong number of parameters"):
            DummyOp(0.5, 0.6, wires=0, do_queue=False)

    def test_incorrect_param_domain(self):
        """Test that an exception is raised if an incorrect parameter domain is requested"""

        class DummyOp(qml.operation.Operation):
            r"""Dummy custom operation"""
            num_wires = 1
            num_params = 1
            par_domain = 'J'
            grad_method = 'A'

        with pytest.raises(ValueError, match="Unknown parameter domain"):
            DummyOp(0.5, wires=0, do_queue=False)

    def test_incorrect_grad_recipe_length(self):
        """Test that an exception is raised if len(grad_recipe)!=len(num_params)"""

        class DummyOp(qml.operation.CVOperation):
            r"""Dummy custom operation"""
            num_wires = 1
            num_params = 1
            par_domain = 'R'
            grad_method = 'A'
            grad_recipe = [(0.5, 0.1), (0.43, 0.1)]

        with pytest.raises(AssertionError, match="Gradient recipe must have one entry for each parameter"):
            DummyOp(0.5, wires=[0, 1], do_queue=False)

    def test_grad_method_with_integer_params(self):
        """Test that an exception is raised if a non-None grad-method is provided for natural number params"""

        class DummyOp(qml.operation.Operation):
            r"""Dummy custom operation"""
            num_wires = 1
            num_params = 1
            par_domain = 'N'
            grad_method = 'A'

        with pytest.raises(AssertionError, match="An operation may only be differentiated with respect to real scalar parameters"):
            DummyOp(5, wires=[0, 1], do_queue=False)

    def test_analytic_grad_with_array_param(self):
        """Test that an exception is raised if an analytic gradient is requested with an array param"""

        class DummyOp(qml.operation.Operation):
            r"""Dummy custom operation"""
            num_wires = 1
            num_params = 1
            par_domain = 'A'
            grad_method = 'A'

        with pytest.raises(AssertionError, match="Operations that depend on arrays containing free variables may only be differentiated using the F method"):
            DummyOp(np.array([1.]), wires=[0, 1], do_queue=False)

    def test_numerical_grad_with_grad_recipe(self):
        """Test that an exception is raised if a numerical gradient is requested with a grad recipe"""

        class DummyOp(qml.operation.Operation):
            r"""Dummy custom operation"""
            num_wires = 1
            num_params = 1
            par_domain = 'R'
            grad_method = 'F'
            grad_recipe = [(0.5, 0.1)]

        with pytest.raises(AssertionError, match="Gradient recipe is only used by the A method"):
            DummyOp(0.5, wires=[0, 1], do_queue=False)

    def test_variable_instead_of_array(self):
        """Test that an exception is raised if an array is expected but a variable is passed"""

        class DummyOp(qml.operation.Operation):
            r"""Dummy custom operation"""
            num_wires = 1
            num_params = 1
            par_domain = 'A'
            grad_method = 'A'

        with pytest.raises(TypeError, match="Array parameter expected, got a Variable"):
            DummyOp(ov.Variable(0), wires=[0], do_queue=False)

    def test_array_instead_of_flattened_array(self):
        """Test that an exception is raised if an array is expected, but an array is passed
        to check_domain when flattened=True. In the initial release of the library, this is not
        accessible by the developer or the user, but is kept in case it will be used in the future."""

        class DummyOp(qml.operation.Operation):
            r"""Dummy custom operation"""
            num_wires = 1
            num_params = 1
            par_domain = 'A'
            grad_method = 'F'

        with pytest.raises(TypeError, match="Flattened array parameter expected"):
            op = DummyOp(np.array([1]), wires=[0], do_queue=False)
            op.check_domain(np.array([1]), True)

    def test_scalar_instead_of_array(self):
        """Test that an exception is raised if an array is expected but a scalar is passed"""

        class DummyOp(qml.operation.Operation):
            r"""Dummy custom operation"""
            num_wires = 1
            num_params = 1
            par_domain = 'A'
            grad_method = 'F'

        with pytest.raises(TypeError, match="Array parameter expected, got"):
            DummyOp(0.5, wires=[0], do_queue=False)

    def test_array_instead_of_real(self):
        """Test that an exception is raised if a real number is expected but an array is passed"""

        class DummyOp(qml.operation.Operation):
            r"""Dummy custom operation"""
            num_wires = 1
            num_params = 1
            par_domain = 'R'
            grad_method = 'F'

        with pytest.raises(TypeError, match="Real scalar parameter expected, got"):
            DummyOp(np.array([1.]), wires=[0], do_queue=False)

    def test_not_natural_param(self):
        """Test that an exception is raised if a natural number is expected but not passed"""

        class DummyOp(qml.operation.Operation):
            r"""Dummy custom operation"""
            num_wires = 1
            num_params = 1
            par_domain = 'N'
            grad_method = None

        with pytest.raises(TypeError, match="Natural number parameter expected, got"):
            DummyOp(0.5, wires=[0], do_queue=False)

        with pytest.raises(TypeError, match="Natural number parameter expected, got"):
            DummyOp(-2, wires=[0], do_queue=False)

    def test_no_wires_passed(self):
        """Test exception raised if no wires are passed"""

        class DummyOp(qml.operation.Operation):
            r"""Dummy custom operation"""
            num_wires = 1
            num_params = 1
            par_domain = 'N'
            grad_method = None

        with pytest.raises(ValueError, match="Must specify the wires"):
            DummyOp(0.54, do_queue=False)

    def test_wire_passed_positionally(self):
        """Test exception raised if wire is passed as a positional arg"""

        class DummyOp(qml.operation.Operation):
            r"""Dummy custom operation"""
            num_wires = 1
            num_params = 1
            par_domain = 'N'
            grad_method = None

        with pytest.raises(ValueError, match="Must specify the wires"):
            DummyOp(0.54, 0, do_queue=False)

    def test_observable_return_type_none(self):
        """Check that the return_type of an observable is initially None"""

        class DummyObserv(qml.operation.Observable):
            r"""Dummy custom observable"""
            num_wires = 1
            num_params = 1
            par_domain = 'N'
            grad_method = None

        assert DummyObserv(0, wires=[1], do_queue=False).return_type is None

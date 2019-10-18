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
from pennylane.operation import Tensor

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

        op = cls(*par, wires=ww)

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
        op = cls(*pars, wires=ww)
        assert op.name == cls.__name__
        assert op.params == pars
        assert op._wires == ww

        # too many parameters
        with pytest.raises(ValueError, match='wrong number of parameters'):
            cls(*(n+1)*[0], wires=ww)

        # too few parameters
        if n > 0:
            with pytest.raises(ValueError, match='wrong number of parameters'):
                cls(*(n-1)*[0], wires=ww)

        if w > 0:
            # too many or too few wires
            with pytest.raises(ValueError, match='wrong number of wires'):
                cls(*pars, wires=list(range(w+1)))
            with pytest.raises(ValueError, match='wrong number of wires'):
                cls(*pars, wires=list(range(w-1)))
            # repeated wires
            if w > 1:
                with pytest.raises(ValueError, match='wires must be unique'):
                    cls(*pars, wires=w*[0])

        if n == 0:
            return

        # wrong parameter types
        if cls.par_domain == 'A':
            # params must be arrays
            with pytest.raises(TypeError, match='Array parameter expected'):
                cls(*n*[0.0], wires=ww)
            # params must not be Variables
            with pytest.raises(TypeError, match='Array parameter expected'):
                cls(*n*[qml.variable.Variable(0)], wires=ww)
        elif cls.par_domain == 'N':
            # params must be natural numbers
            with pytest.raises(TypeError, match='Natural number'):
                cls(*n*[0.7], wires=ww)
            with pytest.raises(TypeError, match='Natural number'):
                cls(*n*[-1], wires=ww)
        elif cls.par_domain == 'R':
            # params must be real numbers
            with pytest.raises(TypeError, match='Real scalar parameter expected'):
                cls(*n*[1j], wires=ww)

        # if par_domain ever gets overridden to an unsupported value, should raise exception
        monkeypatch.setattr(cls, 'par_domain', 'junk')
        with pytest.raises(ValueError, match='Unknown parameter domain'):
            cls(*pars, wires=ww)

        monkeypatch.setattr(cls, 'par_domain', 7)
        with pytest.raises(ValueError, match='Unknown parameter domain'):
            cls(*pars, wires=ww)


    @pytest.fixture(scope="function")
    def qnode(self, mock_device):
        """Provides a QNode for the subsequent tests of do_queue"""

        def circuit(x):
            qml.RX(x, wires=[0])
            qml.CNOT(wires=[0, 1], do_queue=False)
            qml.RY(0.4, wires=[0])
            qml.RZ(-0.2, wires=[1], do_queue=False)
            return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliZ(1))

        node = qml.QNode(circuit, mock_device)
        node.construct([1.0])

        return node

    def test_operation_inside_context_do_queue_false(self, qnode):
        """Test that an operation does not get added to the QNode queue when do_queue=False"""
        assert len(qnode.ops) == 4
        assert qnode.ops[0].name == "RX"
        assert qnode.ops[1].name == "RY"
        assert qnode.ops[2].name == "PauliX"
        assert qnode.ops[3].name == "PauliZ"

    def test_operation_outside_context(self):
        """Test that an operation can be instantiated outside a QNode context, and that do_queue is ignored"""
        op = qml.ops.CNOT(wires=[0, 1], do_queue=False)
        assert isinstance(op, qml.operation.Operation)

        op = qml.ops.RX(0.5, wires=0, do_queue=True)
        assert isinstance(op, qml.operation.Operation)

        op = qml.ops.Hadamard(wires=0)
        assert isinstance(op, qml.operation.Operation)


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
            DummyOp(0.5, wires=[0, 1])

    def test_incorrect_num_params(self):
        """Test that an exception is raised if called with wrong number of parameters"""

        class DummyOp(qml.operation.Operation):
            r"""Dummy custom operation"""
            num_wires = 1
            num_params = 1
            par_domain = 'R'
            grad_method = 'A'

        with pytest.raises(ValueError, match="wrong number of parameters"):
            DummyOp(0.5, 0.6, wires=0)

    def test_incorrect_param_domain(self):
        """Test that an exception is raised if an incorrect parameter domain is requested"""

        class DummyOp(qml.operation.Operation):
            r"""Dummy custom operation"""
            num_wires = 1
            num_params = 1
            par_domain = 'J'
            grad_method = 'A'

        with pytest.raises(ValueError, match="Unknown parameter domain"):
            DummyOp(0.5, wires=0)

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
            DummyOp(0.5, wires=[0, 1])

    def test_grad_method_with_integer_params(self):
        """Test that an exception is raised if a non-None grad-method is provided for natural number params"""

        class DummyOp(qml.operation.Operation):
            r"""Dummy custom operation"""
            num_wires = 1
            num_params = 1
            par_domain = 'N'
            grad_method = 'A'

        with pytest.raises(AssertionError, match="An operation may only be differentiated with respect to real scalar parameters"):
            DummyOp(5, wires=[0, 1])

    def test_analytic_grad_with_array_param(self):
        """Test that an exception is raised if an analytic gradient is requested with an array param"""

        class DummyOp(qml.operation.Operation):
            r"""Dummy custom operation"""
            num_wires = 1
            num_params = 1
            par_domain = 'A'
            grad_method = 'A'

        with pytest.raises(AssertionError, match="Operations that depend on arrays containing free variables may only be differentiated using the F method"):
            DummyOp(np.array([1.]), wires=[0, 1])

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
            DummyOp(0.5, wires=[0, 1])

    def test_variable_instead_of_array(self):
        """Test that an exception is raised if an array is expected but a variable is passed"""

        class DummyOp(qml.operation.Operation):
            r"""Dummy custom operation"""
            num_wires = 1
            num_params = 1
            par_domain = 'A'
            grad_method = 'A'

        with pytest.raises(TypeError, match="Array parameter expected, got a Variable"):
            DummyOp(qml.variable.Variable(0), wires=[0])

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
            op = DummyOp(np.array([1]), wires=[0])
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
            DummyOp(0.5, wires=[0])

    def test_array_instead_of_real(self):
        """Test that an exception is raised if a real number is expected but an array is passed"""

        class DummyOp(qml.operation.Operation):
            r"""Dummy custom operation"""
            num_wires = 1
            num_params = 1
            par_domain = 'R'
            grad_method = 'F'

        with pytest.raises(TypeError, match="Real scalar parameter expected, got"):
            DummyOp(np.array([1.]), wires=[0])

    def test_not_natural_param(self):
        """Test that an exception is raised if a natural number is expected but not passed"""

        class DummyOp(qml.operation.Operation):
            r"""Dummy custom operation"""
            num_wires = 1
            num_params = 1
            par_domain = 'N'
            grad_method = None

        with pytest.raises(TypeError, match="Natural number parameter expected, got"):
            DummyOp(0.5, wires=[0])

        with pytest.raises(TypeError, match="Natural number parameter expected, got"):
            DummyOp(-2, wires=[0])

    def test_no_wires_passed(self):
        """Test exception raised if no wires are passed"""

        class DummyOp(qml.operation.Operation):
            r"""Dummy custom operation"""
            num_wires = 1
            num_params = 1
            par_domain = 'N'
            grad_method = None

        with pytest.raises(ValueError, match="Must specify the wires"):
            DummyOp(0.54)

    def test_wire_passed_positionally(self):
        """Test exception raised if wire is passed as a positional arg"""

        class DummyOp(qml.operation.Operation):
            r"""Dummy custom operation"""
            num_wires = 1
            num_params = 1
            par_domain = 'N'
            grad_method = None

        with pytest.raises(ValueError, match="Must specify the wires"):
            DummyOp(0.54, 0)

    def test_observable_return_type_none(self):
        """Check that the return_type of an observable is initially None"""

        class DummyObserv(qml.operation.Observable):
            r"""Dummy custom observable"""
            num_wires = 1
            num_params = 1
            par_domain = 'N'
            grad_method = None

        assert DummyObserv(0, wires=[1]).return_type is None


class TestTensor:
    """Unit tests for the Tensor class"""

    def test_construct(self):
        """Test construction of a tensor product"""
        X = qml.PauliX(0)
        Y = qml.PauliY(2)
        T = Tensor(X, Y)
        assert T.obs == [X, Y]

        T = Tensor(T, Y)
        assert T.obs == [X, Y, Y]

        with pytest.raises(ValueError, match="Can only perform tensor products between observables"):
            Tensor(T, qml.CNOT(wires=[0, 1]))

    def test_name(self):
        """Test that the names of the observables are
        returned as expected"""
        X = qml.PauliX(0)
        Y = qml.PauliY(2)
        t = Tensor(X, Y)
        assert t.name == [X.name, Y.name]

    def test_num_wires(self):
        """Test that the correct number of wires is returned"""
        p = np.array([0.5])
        X = qml.PauliX(0)
        Y = qml.Hermitian(p, wires=[1, 2])
        t = Tensor(X, Y)
        assert t.num_wires == 3

    def test_wires(self):
        """Test that the correct nested list of wires is returned"""
        p = np.array([0.5])
        X = qml.PauliX(0)
        Y = qml.Hermitian(p, wires=[1, 2])
        t = Tensor(X, Y)
        assert t.wires == [[0], [1, 2]]

    def test_params(self):
        """Test that the correct flattened list of parameters is returned"""
        p = np.array([0.5])
        X = qml.PauliX(0)
        Y = qml.Hermitian(p, wires=[1, 2])
        t = Tensor(X, Y)
        assert t.params == [p]

    def test_num_params(self):
        """Test that the correct number of parameters is returned"""
        p = np.array([0.5])
        X = qml.PauliX(0)
        Y = qml.Hermitian(p, wires=[1, 2])
        Z = qml.Hermitian(p, wires=[1, 2])
        t = Tensor(X, Y, Z)
        assert t.num_params == 2

    def test_parameters(self):
        """Test that the correct nested list of parameters is returned"""
        p = np.array([0.5])
        X = qml.PauliX(0)
        Y = qml.Hermitian(p, wires=[1, 2])
        t = Tensor(X, Y)
        assert t.parameters == [[], [p]]

    def test_multiply_obs(self):
        """Test that multiplying two observables
        produces a tensor"""
        X = qml.PauliX(0)
        Y = qml.Hadamard(2)
        t = X @ Y
        assert isinstance(t, Tensor)
        assert t.obs == [X, Y]

    def test_multiply_obs_tensor(self):
        """Test that multiplying an observable by a tensor
        produces a tensor"""
        X = qml.PauliX(0)
        Y = qml.Hadamard(2)
        Z = qml.PauliZ(1)

        t = X @ Y
        t = Z @ t

        assert isinstance(t, Tensor)
        assert t.obs == [Z, X, Y]

    def test_multiply_tensor_obs(self):
        """Test that multiplying a tensor by an observable
        produces a tensor"""
        X = qml.PauliX(0)
        Y = qml.Hadamard(2)
        Z = qml.PauliZ(1)

        t = X @ Y
        t = t @ Z

        assert isinstance(t, Tensor)
        assert t.obs == [X, Y, Z]

    def test_multiply_tensor_tensor(self):
        """Test that multiplying a tensor by a tensor
        produces a tensor"""
        X = qml.PauliX(0)
        Y = qml.PauliY(2)
        Z = qml.PauliZ(1)
        H = qml.Hadamard(3)

        t1 = X @ Y
        t2 = Z @ H
        t = t2 @ t1

        assert isinstance(t, Tensor)
        assert t.obs == [Z, H, X, Y]

    def test_multiply_tensor_in_place(self):
        """Test that multiplying a tensor in-place
        produces a tensor"""
        X = qml.PauliX(0)
        Y = qml.PauliY(2)
        Z = qml.PauliZ(1)
        H = qml.Hadamard(3)

        t = X
        t @= Y
        t @= Z @ H

        assert isinstance(t, Tensor)
        assert t.obs == [X, Y, Z, H]

    def test_operation_multiply_invalid(self):
        """Test that an exception is raised if an observable
        is multiplied by an operation"""
        X = qml.PauliX(0)
        Y = qml.CNOT(wires=[0, 1])
        Z = qml.PauliZ(0)

        with pytest.raises(ValueError, match="Can only perform tensor products between observables"):
            X @ Y

        with pytest.raises(ValueError, match="Can only perform tensor products between observables"):
            T = X @ Z
            T @ Y

        with pytest.raises(ValueError, match="Can only perform tensor products between observables"):
            T = X @ Z
            Y @ T

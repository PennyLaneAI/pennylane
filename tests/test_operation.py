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
from unittest.mock import patch

import pennylane as qml
from pennylane.plugins.default_qubit import I, X, Y, Rotx, Roty, Rotz, CRotx, CRoty, CRotz, CNOT, Rot3, Rphi
from pennylane.operation import Tensor

# pylint: disable=no-self-use, no-member, protected-access, pointless-statement

# Operation subclasses to test
op_classes = [getattr(qml.ops, cls) for cls in qml.ops.__all__]
op_classes_cv = [getattr(qml.ops, cls) for cls in qml.ops._cv__all__]
op_classes_gaussian = [cls for cls in op_classes_cv if cls.supports_heisenberg]

def U3(theta, phi, lam):
    return Rphi(phi) @ Rphi(lam) @ Rot3(lam, theta, -lam)


class TestOperation:
    """Operation class tests."""

    @pytest.mark.parametrize("test_class", op_classes_gaussian)
    def test_heisenberg(self, test_class, tol):
        "Heisenberg picture adjoint actions of CV Operations."

        ww = list(range(test_class.num_wires))

        # fixed parameter values
        if test_class.par_domain == 'A':
            if test_class.__name__ == "Interferometer":
                ww = list(range(2))
                par = [np.array([[0.83645892-0.40533293j, -0.20215326+0.30850569j],
                                 [-0.23889780-0.28101519j, -0.88031770-0.29832709j]])]
            else:
                par = [np.array([[-1.82624687]])] * test_class.num_params
        else:
            par = [-0.069125, 0.51778, 0.91133, 0.95904][:test_class.num_params]

        op = test_class(*par, wires=ww)

        if issubclass(test_class, qml.operation.Observable):
            Q = op.heisenberg_obs(0)
            # ev_order equals the number of dimensions of the H-rep array
            assert Q.ndim == test_class.ev_order
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
            for k in range(test_class.num_params):
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

    @pytest.mark.parametrize("test_class", op_classes)
    def test_operation_init(self, test_class, monkeypatch):
        "Operation subclass initialization."

        n = test_class.num_params
        w = test_class.num_wires
        ww = list(range(w))
        # valid pars
        if test_class.par_domain == 'A':
            pars = [np.eye(2)] * n
        elif test_class.par_domain == 'N':
            pars = [0] * n
        else:
            pars = [0.0] * n

        # valid call
        op = test_class(*pars, wires=ww)
        assert op.name == test_class.__name__
        assert op.params == pars
        assert op._wires == ww

        # too many parameters
        with pytest.raises(ValueError, match='wrong number of parameters'):
            test_class(*(n+1)*[0], wires=ww)

        # too few parameters
        if n > 0:
            with pytest.raises(ValueError, match='wrong number of parameters'):
                test_class(*(n-1)*[0], wires=ww)

        if w > 0:
            # too many or too few wires
            with pytest.raises(ValueError, match='wrong number of wires'):
                test_class(*pars, wires=list(range(w+1)))
            with pytest.raises(ValueError, match='wrong number of wires'):
                test_class(*pars, wires=list(range(w-1)))
            # repeated wires
            if w > 1:
                with pytest.raises(ValueError, match='wires must be unique'):
                    test_class(*pars, wires=w*[0])

        if n == 0:
            return

        # wrong parameter types
        if test_class.par_domain == 'A':
            # params must be arrays
            with pytest.raises(TypeError, match='Array parameter expected'):
                test_class(*n*[0.0], wires=ww)
            # params must not be Variables
            with pytest.raises(TypeError, match='Array parameter expected'):
                test_class(*n*[qml.variable.Variable(0)], wires=ww)
        elif test_class.par_domain == 'N':
            # params must be natural numbers
            with pytest.raises(TypeError, match='Natural number'):
                test_class(*n*[0.7], wires=ww)
            with pytest.raises(TypeError, match='Natural number'):
                test_class(*n*[-1], wires=ww)
        elif test_class.par_domain == 'R':
            # params must be real numbers
            with pytest.raises(TypeError, match='Real scalar parameter expected'):
                test_class(*n*[1j], wires=ww)

        # if par_domain ever gets overridden to an unsupported value, should raise exception
        monkeypatch.setattr(test_class, 'par_domain', 'junk')
        with pytest.raises(ValueError, match='Unknown parameter domain'):
            test_class(*pars, wires=ww)

        monkeypatch.setattr(test_class, 'par_domain', 7)
        with pytest.raises(ValueError, match='Unknown parameter domain'):
            test_class(*pars, wires=ww)

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
        node._construct([1.0], {})

        return node

    def test_operation_inside_context_do_queue_false(self, qnode):
        """Test that an operation does not get added to the QNode queue when do_queue=False"""
        assert len(qnode.ops) == 4
        assert qnode.ops[0].name == "RX"
        assert qnode.ops[1].name == "RY"
        assert qnode.ops[2].name == "PauliX"
        assert qnode.ops[3].name == "PauliZ"

    @pytest.fixture(scope="function")
    def qnode_for_inverse(self, mock_device):
        """Provides a QNode for the subsequent tests of inv"""

        def circuit(x):
            qml.RZ(x, wires=[1]).inv()
            qml.RZ(x, wires=[1]).inv().inv()
            return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliZ(1))

        node = qml.QNode(circuit, mock_device)
        node._construct([1.0], {})

        return node

    def test_operation_inverse_defined(self, qnode_for_inverse):
        """Test that the inverse of an operation is added to the QNode queue and the operation is an instance
        of the original class"""
        assert qnode_for_inverse.ops[0].name == "RZ.inv"
        assert qnode_for_inverse.ops[0].inverse
        assert issubclass(qnode_for_inverse.ops[0].__class__, qml.operation.Operation)
        assert qnode_for_inverse.ops[1].name == "RZ"
        assert not qnode_for_inverse.ops[1].inverse
        assert issubclass(qnode_for_inverse.ops[1].__class__, qml.operation.Operation)

    def test_operation_inverse_using_dummy_operation(self):

        some_param = 0.5

        class DummyOp(qml.operation.Operation):
            r"""Dummy custom Operation"""
            num_wires = 1
            num_params = 1
            par_domain = 'R'

        # Check that the name of the Operation is initialized fine
        dummy_op = DummyOp(some_param, wires=[1])

        assert not dummy_op.inverse

        dummy_op_class_name = dummy_op.name

        # Check that the name of the Operation was modified when applying the inverse
        assert dummy_op.inv().name == dummy_op_class_name + ".inv"
        assert dummy_op.inverse

        # Check that the name of the Operation is the original again, once applying the inverse a second time
        assert dummy_op.inv().name == dummy_op_class_name
        assert not dummy_op.inverse

    def test_operation_outside_context(self):
        """Test that an operation can be instantiated outside a QNode context, and that do_queue is ignored"""
        op = qml.ops.CNOT(wires=[0, 1], do_queue=False)
        assert isinstance(op, qml.operation.Operation)

        op = qml.ops.RX(0.5, wires=0, do_queue=True)
        assert isinstance(op, qml.operation.Operation)

        op = qml.ops.Hadamard(wires=0)
        assert isinstance(op, qml.operation.Operation)


class TestOperatorConstruction:
    """Test custom operators construction."""

    def test_incorrect_num_wires(self):
        """Test that an exception is raised if called with wrong number of wires"""

        class DummyOp(qml.operation.Operator):
            r"""Dummy custom operator"""
            num_wires = 1
            num_params = 1
            par_domain = 'R'

        with pytest.raises(ValueError, match="wrong number of wires"):
            DummyOp(0.5, wires=[0, 1])

    def test_non_unique_wires(self):
        """Test that an exception is raised if called with identical wires"""

        class DummyOp(qml.operation.Operator):
            r"""Dummy custom operator"""
            num_wires = 2
            num_params = 1
            par_domain = 'R'

        with pytest.raises(ValueError, match="wires must be unique"):
            DummyOp(0.5, wires=[1, 1], do_queue=False)

    def test_incorrect_num_params(self):
        """Test that an exception is raised if called with wrong number of parameters"""

        class DummyOp(qml.operation.Operator):
            r"""Dummy custom operator"""
            num_wires = 1
            num_params = 1
            par_domain = 'R'
            grad_method = 'A'

        with pytest.raises(ValueError, match="wrong number of parameters"):
            DummyOp(0.5, 0.6, wires=0)

    def test_incorrect_param_domain(self):
        """Test that an exception is raised if an incorrect parameter domain is requested"""

        class DummyOp(qml.operation.Operator):
            r"""Dummy custom operator"""
            num_wires = 1
            num_params = 1
            par_domain = 'J'
            grad_method = 'A'

        with pytest.raises(ValueError, match="Unknown parameter domain"):
            DummyOp(0.5, wires=0)


class TestOperationConstruction:
    """Test custom operations construction."""

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
            grad_method = 'F'

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


class TestObservableConstruction:
    """Test custom observables construction."""

    def test_observable_return_type_none(self):
        """Check that the return_type of an observable is initially None"""

        class DummyObserv(qml.operation.Observable):
            r"""Dummy custom observable"""
            num_wires = 1
            num_params = 1
            par_domain = 'N'
            grad_method = None

        assert DummyObserv(0, wires=[1]).return_type is None

    def test_observable_is_not_operation_but_operator(self):
        """Check that the Observable class inherits from an Operator, not from an Operation"""

        assert issubclass(qml.operation.Observable, qml.operation.Operator)
        assert not issubclass(qml.operation.Observable, qml.operation.Operation)

    def test_observable_is_operation_as_well(self):
        """Check that the Observable class inherits from an Operator class as well"""

        class DummyObserv(qml.operation.Observable, qml.operation.Operation):
            r"""Dummy custom observable"""
            num_wires = 1
            num_params = 1
            par_domain = 'N'
            grad_method = None

        assert issubclass(DummyObserv, qml.operation.Operator)
        assert issubclass(DummyObserv, qml.operation.Observable)
        assert issubclass(DummyObserv, qml.operation.Operation)


class TestOperatorIntegration:
    """ Integration tests for the Operator class"""

    def test_all_wires_defined_but_init_with_one(self):
        """Test that an exception is raised if the class is defined with ALL wires,
        but then instantiated with only one"""

        dev1 = qml.device("default.qubit", wires=2)

        class DummyOp(qml.operation.Operator):
            r"""Dummy custom operator"""
            num_wires = qml.operation.Wires.All
            num_params = 0
            par_domain = 'R'

        @qml.qnode(dev1)
        def circuit():
            DummyOp(wires=[0])
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(qml.QuantumFunctionError, match="Operator {} must act on all wires".format(DummyOp.__name__)):
            circuit()


class TestOperationIntegration:
    """ Integration tests for the Operation class"""

    def test_inverse_of_operation(self):
        """Test the inverse of an operation"""

        dev1 = qml.device("default.qubit", wires=2)

        @qml.qnode(dev1)
        def circuit():
            qml.PauliZ(wires=[0])
            qml.PauliZ(wires=[0]).inv()
            return qml.expval(qml.PauliZ(0))

        assert circuit() == 1

    def test_inverse_operations_not_supported(self):
        """Test that the inverse of operations is not currently
        supported on the default gaussian device"""

        dev1 = qml.device("default.gaussian", wires=2)

        @qml.qnode(dev1)
        def mean_photon_gaussian(mag_alpha, phase_alpha, phi):
            qml.Displacement(mag_alpha, phase_alpha, wires=0)
            qml.Rotation(phi, wires=0).inv()
            return qml.expval(qml.NumberOperator(0))

        with pytest.raises(qml.DeviceError, match="Gate Rotation.inv not supported on device {}"
                .format(dev1.short_name)):
            mean_photon_gaussian(0.015, 0.02, 0.005)

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

    def test_eigvals(self):
        """Test that the correct eigenvalues are returned for the Tensor"""
        X = qml.PauliX(0)
        Y = qml.PauliY(2)
        t = Tensor(X, Y)
        assert np.array_equal(t.eigvals, np.kron(qml.PauliX.eigvals, qml.PauliY.eigvals))

    @pytest.mark.usefixtures("tear_down_hermitian")
    def test_eigvals_hermitian(self, tol):
        """Test that the correct eigenvalues are returned for the Tensor"""
        X = qml.PauliX(0)
        hamiltonian = np.array([[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]])
        Herm = qml.Hermitian(hamiltonian, wires=[1, 2])
        t = Tensor(X, Herm)
        d = np.kron(np.array([1., -1.]), np.array([-1.,  1.,  1.,  1.]))
        t = t.eigvals
        assert np.allclose(t, d, atol=tol, rtol = 0)


class TestDecomposition:
    """Test for operation decomposition"""

    def test_U1_decomposition(self):
        """Test the decomposition of the U1 gate provides the equivalent phase shift gate"""
        phi = 0.432
        res = qml.U1.decomposition(phi, wires=0)

        assert len(res) == 1
        assert res[0].name == "PhaseShift"
        assert res[0].parameters == [phi]

    def test_rotation_decomposition(self):
        """Test the decomposition of the abritrary single
        qubit rotation"""
        phi = 0.432
        theta = 0.654
        omega = -5.43

        with qml.utils.OperationRecorder() as rec:
            qml.Rot.decomposition(phi, theta, omega, wires=0)

        assert len(rec.queue) == 3

        assert rec.queue[0].name == "RZ"
        assert rec.queue[0].parameters == [phi]

        assert rec.queue[1].name == "RY"
        assert rec.queue[1].parameters == [theta]

        assert rec.queue[2].name == "RZ"
        assert rec.queue[2].parameters == [omega]

    def test_crx_decomposition(self):
        """Test the decomposition of the controlled X
        qubit rotation"""
        phi = 0.432

        with qml.utils.OperationRecorder() as rec:
            qml.CRX.decomposition(phi, wires=[0, 1])

        assert len(rec.queue) == 6

        assert rec.queue[0].name == "RZ"
        assert rec.queue[0].parameters == [np.pi/2]
        assert rec.queue[0].wires == [1]

        assert rec.queue[1].name == "RY"
        assert rec.queue[1].parameters == [phi/2]
        assert rec.queue[1].wires == [1]

        assert rec.queue[2].name == "CNOT"
        assert rec.queue[2].parameters == []
        assert rec.queue[2].wires == [0, 1]

        assert rec.queue[3].name == "RY"
        assert rec.queue[3].parameters == [-phi/2]
        assert rec.queue[3].wires == [1]

        assert rec.queue[4].name == "CNOT"
        assert rec.queue[4].parameters == []
        assert rec.queue[4].wires == [0, 1]

        assert rec.queue[5].name == "RZ"
        assert rec.queue[5].parameters == [-np.pi/2]
        assert rec.queue[5].wires == [1]

    @pytest.mark.parametrize("phi", [0.03236*i for i in range(5)])
    def test_crx_decomposition_correctness(self, phi, tol):
        """Test that the decomposition of the controlled X
        qubit rotation is correct"""

        expected = CRotx(phi)

        obtained = np.kron(I, Rotz(-np.pi/2)) @ CNOT @ np.kron(I, Roty(-phi/2)) @ CNOT @ np.kron(I, Roty(phi/2)) @ np.kron(I, Rotz(np.pi/2))
        assert np.allclose(expected, obtained, atol=tol, rtol=0)


    def test_cry_decomposition(self):
        """Test the decomposition of the controlled Y
        qubit rotation"""
        phi = 0.432

        operation_wires = [0, 1]

        with qml.utils.OperationRecorder() as rec:
            qml.CRY.decomposition(phi, wires=operation_wires)

        assert len(rec.queue) == 4

        assert rec.queue[0].name == "U3"
        assert rec.queue[0].parameters == [phi/2, 0, 0]
        assert rec.queue[0].wires == [1]

        assert rec.queue[1].name == "CNOT"
        assert rec.queue[1].parameters == []
        assert rec.queue[1].wires == operation_wires

        assert rec.queue[2].name == "U3"
        assert rec.queue[2].parameters == [-phi/2, 0, 0]
        assert rec.queue[2].wires == [1]

        assert rec.queue[3].name == "CNOT"
        assert rec.queue[3].parameters == []
        assert rec.queue[3].wires == operation_wires

    @pytest.mark.parametrize("phi", [0.03236*i for i in range(5)])
    def test_cry_decomposition_correctness(self, phi, tol):
        """Test that the decomposition of the controlled Y
        qubit rotation is correct"""

        expected = CRoty(phi)

        obtained = CNOT @ np.kron(I, U3(-phi / 2, 0, 0)) @ CNOT @ np.kron(I, U3(phi / 2, 0, 0))
        assert np.allclose(expected, obtained, atol=tol, rtol=0)


    def test_crz_decomposition(self):
        """Test the decomposition of the controlled Z
        qubit rotation"""
        phi = 0.432

        operation_wires = [0, 1]

        with qml.utils.OperationRecorder() as rec:
            qml.CRZ.decomposition(phi, wires=operation_wires)

        assert len(rec.queue) == 4

        assert rec.queue[0].name == "PhaseShift"
        assert rec.queue[0].parameters == [phi/2]
        assert rec.queue[0].wires == [1]

        assert rec.queue[1].name == "CNOT"
        assert rec.queue[1].parameters == []
        assert rec.queue[1].wires == operation_wires

        assert rec.queue[2].name == "PhaseShift"
        assert rec.queue[2].parameters == [-phi/2]
        assert rec.queue[2].wires == [1]

        assert rec.queue[3].name == "CNOT"
        assert rec.queue[3].parameters == []
        assert rec.queue[3].wires == operation_wires

    @pytest.mark.parametrize("phi", [0.03236*i for i in range(5)])
    def test_crz_decomposition_correctness(self, phi, tol):
        """Test that the decomposition of the controlled Z
        qubit rotation is correct"""

        expected = CRotz(phi)

        obtained = CNOT @ np.kron(I, Rphi(-phi / 2)) @ CNOT @ np.kron(I, Rphi(phi / 2))
        assert np.allclose(expected, obtained, atol=tol, rtol=0)

    def test_U2_decomposition(self):
        """Test the U2 decomposition is correct"""
        phi = 0.432
        lam = 0.654

        with qml.utils.OperationRecorder() as rec:
            qml.U2.decomposition(phi, lam, wires=0)

        assert len(rec.queue) == 3

        assert rec.queue[0].name == "Rot"
        assert rec.queue[0].parameters == [lam, np.pi/2, -lam]

        assert rec.queue[1].name == "PhaseShift"
        assert rec.queue[1].parameters == [lam]

        assert rec.queue[2].name == "PhaseShift"
        assert rec.queue[2].parameters == [phi]

    def test_U3_decomposition(self):
        """Test the U3 decomposition is correct"""
        theta = 0.654
        phi = 0.432
        lam = 0.654

        with qml.utils.OperationRecorder() as rec:
            qml.U3.decomposition(theta, phi, lam, wires=0)

        assert len(rec.queue) == 3

        assert rec.queue[0].name == "Rot"
        assert rec.queue[0].parameters == [lam, theta, -lam]

        assert rec.queue[1].name == "PhaseShift"
        assert rec.queue[1].parameters == [lam]

        assert rec.queue[2].name == "PhaseShift"
        assert rec.queue[2].parameters == [phi]

    def test_basis_state_decomposition(self, monkeypatch):
        """Test the decomposition of BasisState calls the
        BasisStatePreparation template"""
        n = np.array([1, 0, 1, 1])
        wires=[0, 1, 2, 3]
        call_args = []

        # We have to patch BasisStatePreparation where it is loaded
        monkeypatch.setattr(qml.ops.qubit, "BasisStatePreparation", lambda *args: call_args.append(args))
        qml.BasisState.decomposition(n, wires=wires)

        assert len(call_args) == 1
        assert np.array_equal(call_args[0][0], n)
        assert np.array_equal(call_args[0][1], wires)

    def test_qubit_state_vector_decomposition(self, monkeypatch):
        """Test the decomposition of QubitStateVector calls the
        MottonenStatePreparation template"""
        state = np.array([1/2, 1j/np.sqrt(2), 0, -1/2])
        wires = [0, 1]
        call_args = []

        # We have to patch MottonenStatePreparation where it is loaded
        monkeypatch.setattr(qml.ops.qubit, "MottonenStatePreparation", lambda *args: call_args.append(args))
        qml.QubitStateVector.decomposition(state, wires=wires)

        assert len(call_args) == 1
        assert np.array_equal(call_args[0][0], state)
        assert np.array_equal(call_args[0][1], wires)

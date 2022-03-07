# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
import itertools
from functools import reduce
import warnings

import pytest
import numpy as np
from pennylane import numpy as pnp
from numpy.linalg import multi_dot

import pennylane as qml
from pennylane.operation import Tensor, operation_derivative, Operator, Operation

from gate_data import I, X, CNOT
from pennylane.wires import Wires


# pylint: disable=no-self-use, no-member, protected-access, pointless-statement


class TestOperatorConstruction:
    """Test custom operators construction."""

    def test_operation_outside_context(self):
        """Test that an operation can be instantiated outside a QNode context, and that do_queue is ignored"""
        op = qml.ops.CNOT(wires=[0, 1], do_queue=False)
        assert isinstance(op, qml.operation.Operation)

        op = qml.ops.RX(0.5, wires=0, do_queue=True)
        assert isinstance(op, qml.operation.Operation)

        op = qml.ops.Hadamard(wires=0)
        assert isinstance(op, qml.operation.Operation)

    def test_incorrect_num_wires(self):
        """Test that an exception is raised if called with wrong number of wires"""

        class DummyOp(qml.operation.Operator):
            r"""Dummy custom operator"""
            num_wires = 1

        with pytest.raises(ValueError, match="wrong number of wires"):
            DummyOp(0.5, wires=[0, 1])

    def test_non_unique_wires(self):
        """Test that an exception is raised if called with identical wires"""

        class DummyOp(qml.operation.Operator):
            r"""Dummy custom operator"""
            num_wires = 1

        with pytest.raises(qml.wires.WireError, match="Wires must be unique"):
            DummyOp(0.5, wires=[1, 1], do_queue=False)

    def test_incorrect_num_params(self):
        """Test that an exception is raised if called with wrong number of parameters"""

        class DummyOp(qml.operation.Operator):
            r"""Dummy custom operator that declares num_params as a instance property"""
            num_wires = 1
            grad_method = "A"

            @property
            def num_params(self):
                return 1

        with pytest.raises(ValueError, match="wrong number of parameters"):
            DummyOp(0.5, 0.6, wires=0)

        op = DummyOp(0.5, wires=0)
        assert op.num_params == 1

        class DummyOp2(qml.operation.Operator):
            r"""Dummy custom operator that declared num_params as a class property"""
            num_params = 4
            num_wires = 1
            grad_method = "A"

        with pytest.raises(ValueError, match="wrong number of parameters"):
            DummyOp2(0.5, 0.6, wires=0)

        op2 = DummyOp2(0.5, 0.3, 0.1, 0.2, wires=0)
        assert op2.num_params == 4
        assert DummyOp2.num_params == 4

        class DummyOp3(qml.operation.Operator):
            r"""Dummy custom operator that does not declare num_params at all"""
            num_wires = 1
            grad_method = "A"

        op3 = DummyOp3(0.5, 0.6, wires=0)

        assert op3.num_params == 2

    def test_name_setter(self):
        """Tests that we can set the name of an operator"""

        class DummyOp(qml.operation.Operator):
            r"""Dummy custom operator"""
            num_wires = 1

        op = DummyOp(wires=0)
        op.name = "MyOp"
        assert op.name == "MyOp"

    def test_default_hyperparams(self):
        """Tests that the hyperparams attribute is defined for all operations."""

        class MyOp(qml.operation.Operation):
            num_wires = 1

        class MyOpOverwriteInit(qml.operation.Operation):
            num_wires = 1

            def __init__(self, wires):
                pass

        op = MyOp(wires=0)
        assert op.hyperparameters == {}

        op = MyOpOverwriteInit(wires=0)
        assert op.hyperparameters == {}

    def test_custom_hyperparams(self):
        """Tests that an operation can add custom hyperparams."""

        class MyOp(qml.operation.Operation):
            num_wires = 1

            def __init__(self, wires, basis_state=None):
                self._hyperparameters = {"basis_state": basis_state}

        state = [0, 1, 0]
        assert MyOp(wires=1, basis_state=state).hyperparameters["basis_state"] == state


class TestOperationConstruction:
    """Test custom operations construction."""

    def test_grad_recipe_parameter_dependent(self):
        """Test that an operation with a gradient recipe that depends on
        its instantiated parameter values works correctly"""

        class DummyOp(qml.operation.Operation):
            r"""Dummy custom operation"""
            num_wires = 1
            grad_method = "A"

            @property
            def grad_recipe(self):
                x = self.data[0]
                return ([[1.0, 1.0, x], [1.0, 0.0, -x]],)

        x = 0.654
        op = DummyOp(x, wires=0)
        assert op.grad_recipe == ([[1.0, 1.0, x], [1.0, 0.0, -x]],)

    def test_warning_get_parameter_shift(self):
        """Test that ``get_parameter_shift`` issues a deprecation
        warning."""

        class DummyOp(qml.operation.Operation):
            r"""Dummy custom operation"""
            num_wires = 1
            num_params = 1
            grad_recipe = ("Dummy recipe",)

        op = DummyOp(0.1, wires=0)
        with pytest.warns(UserWarning, match="get_parameter_shift is deprecated"):
            assert op.get_parameter_shift(0) == "Dummy recipe"

    def test_error_get_parameter_shift_no_recipe(self):
        """Test that ``get_parameter_shift`` raises an Error if no grad_recipe
        is available, as we no longer assume the two-term rule by default."""

        class DummyOp(qml.operation.Operation):
            r"""Dummy custom operation"""
            num_wires = 1
            num_params = 1
            grad_recipe = (None,)

        op = DummyOp(0.1, wires=0)
        with pytest.raises(
            qml.operation.OperatorPropertyUndefined,
            match="The operation DummyOp does not have a parameter-shift recipe",
        ):
            op.get_parameter_shift(0)

    def test_default_grad_method_with_frequencies(self):
        """Test that the correct ``grad_method`` is returned by default
        if ``parameter_frequencies`` are present.
        """

        class DummyOp(qml.operation.Operation):
            r"""Dummy custom operation"""
            num_wires = 1

            @property
            def parameter_frequencies(self):
                return [(0.4, 1.2)]

        x = 0.654
        op = DummyOp(x, wires=0)
        assert op.grad_method == "A"

    def test_default_grad_method_with_generator(self):
        """Test that the correct ``grad_method`` is returned by default
        if a generator is present to determine parameter_frequencies from.
        """

        class DummyOp(qml.operation.Operation):
            r"""Dummy custom operation"""
            num_wires = 1

            def generator(self):
                return -0.2 * qml.PauliX(wires=self.wires)

        x = 0.654
        op = DummyOp(x, wires=0)
        assert op.grad_method == "A"

    def test_default_grad_method_numeric(self):
        """Test that the correct ``grad_method`` is returned by default
        if no information is present to deduce an analytic gradient method.
        """

        class DummyOp(qml.operation.Operation):
            r"""Dummy custom operation"""
            num_wires = 1

        x = 0.654
        op = DummyOp(x, wires=0)
        assert op.grad_method == "F"

    def test_default_grad_method_with_grad_recipe(self):
        """Test that the correct ``grad_method`` is returned by default
        if a grad_recipe is present.
        """

        class DummyOp(qml.operation.Operation):
            r"""Dummy custom operation"""
            num_wires = 1
            grad_recipe = ["not a recipe"]

        x = 0.654
        op = DummyOp(x, wires=0)
        assert op.grad_method == "A"

    def test_default_grad_no_param(self):
        """Test that the correct ``grad_method`` is returned by default
        if an operation does not have a parameter.
        """

        class DummyOp(qml.operation.Operation):
            r"""Dummy custom operation"""
            num_wires = 1

        op = DummyOp(wires=0)
        assert op.grad_method is None

    def test_frequencies_default_single_param(self):
        """Test that an operation with default parameter frequencies
        and a single parameter works correctly."""

        class DummyOp(qml.operation.Operation):
            r"""Dummy custom operation"""
            num_wires = 1
            grad_method = "A"

            def generator(self):
                return -0.2 * qml.PauliX(wires=self.wires)

        x = 0.654
        op = DummyOp(x, wires=0)
        assert op.parameter_frequencies == (0.4,)

    def test_frequencies_default_multi_param(self):
        """Test that an operation with default parameter frequencies and multiple
        parameters raises an error when calling parameter_frequencies."""

        class DummyOp(qml.operation.Operation):
            r"""Dummy custom operation"""
            num_params = 3
            num_wires = 1
            grad_method = "A"

        x = [0.654, 2.31, 0.1]
        op = DummyOp(*x, wires=0)
        with pytest.raises(
            qml.operation.OperatorPropertyUndefined, match="DummyOp does not have parameter"
        ):
            op.parameter_frequencies

    @pytest.mark.parametrize("num_param", [1, 2])
    def test_frequencies_parameter_dependent(self, num_param):
        """Test that an operation with parameter frequencies that depend on
        its instantiated parameter values works correctly"""

        class DummyOp(qml.operation.Operation):
            r"""Dummy custom operation"""
            num_params = num_param
            num_wires = 1
            grad_method = "A"

            @property
            def parameter_frequencies(self):
                x = self.data
                return [(0.2, _x) for _x in x]

        x = [0.654, 2.31][:num_param]
        op = DummyOp(*x, wires=0)
        f = op.parameter_frequencies
        for i in range(num_param):
            assert f[i] == (0.2, x[i])

    def test_no_wires_passed(self):
        """Test exception raised if no wires are passed"""

        class DummyOp(qml.operation.Operation):
            r"""Dummy custom operation"""
            num_wires = 1
            grad_method = None

        with pytest.raises(ValueError, match="Must specify the wires"):
            DummyOp(0.54)

    def test_wire_passed_positionally(self):
        """Test exception raised if wire is passed as a positional arg"""

        class DummyOp(qml.operation.Operation):
            r"""Dummy custom operation"""
            num_wires = 1
            grad_method = None

            @property
            def num_params(self):
                return 1

        with pytest.raises(ValueError, match="Must specify the wires"):
            DummyOp(0.54, 0)

    def test_id(self):
        """Test that the id attribute of an operator can be set."""

        class DummyOp(qml.operation.Operation):
            r"""Dummy custom operation"""
            num_wires = 1
            grad_method = None

        op = DummyOp(1.0, wires=0, id="test")
        assert op.id == "test"

    def test_control_wires(self):
        """Test that control_wires defaults to an empty Wires object."""

        class DummyOp(qml.operation.Operation):
            r"""Dummy custom operation"""
            num_wires = 1
            grad_method = None

        op = DummyOp(1.0, wires=0, id="test")
        assert op.control_wires == qml.wires.Wires([])


class TestObservableConstruction:
    """Test custom observables construction."""

    def test_observable_return_type_none(self):
        """Check that the return_type of an observable is initially None"""

        class DummyObserv(qml.operation.Observable):
            r"""Dummy custom observable"""
            num_wires = 1
            grad_method = None

        assert DummyObserv(0, wires=[1]).return_type is None

    def test_construction_with_wires_pos_arg(self):
        """Test that the wires can be given as a positional argument"""

        class DummyObserv(qml.operation.Observable):
            r"""Dummy custom observable"""
            num_wires = 1
            grad_method = None

        ob = DummyObserv([1])
        assert ob.wires == qml.wires.Wires(1)

    def test_observable_is_not_operation_but_operator(self):
        """Check that the Observable class inherits from an Operator, not from an Operation"""

        assert issubclass(qml.operation.Observable, qml.operation.Operator)
        assert not issubclass(qml.operation.Observable, qml.operation.Operation)

    def test_observable_is_operation_as_well(self):
        """Check that the Observable class inherits from an Operator class as well"""

        class DummyObserv(qml.operation.Observable, qml.operation.Operation):
            r"""Dummy custom observable"""
            num_wires = 1
            grad_method = None

        assert issubclass(DummyObserv, qml.operation.Operator)
        assert issubclass(DummyObserv, qml.operation.Observable)
        assert issubclass(DummyObserv, qml.operation.Operation)

    def test_tensor_n_multiple_modes(self):
        """Checks that the TensorN operator was constructed correctly when
        multiple modes were specified."""
        cv_obs = qml.TensorN(wires=[0, 1])

        assert isinstance(cv_obs, qml.TensorN)
        assert cv_obs.wires == Wires([0, 1])
        assert cv_obs.ev_order is None

    def test_tensor_n_single_mode_wires_explicit(self):
        """Checks that instantiating a TensorN when passing a single mode as a
        keyword argument returns a NumberOperator."""
        cv_obs = qml.TensorN(wires=[0])

        assert isinstance(cv_obs, qml.NumberOperator)
        assert cv_obs.wires == Wires([0])
        assert cv_obs.ev_order == 2

    def test_tensor_n_single_mode_wires_implicit(self):
        """Checks that instantiating TensorN when passing a single mode as a
        positional argument returns a NumberOperator."""
        cv_obs = qml.TensorN(1)

        assert isinstance(cv_obs, qml.NumberOperator)
        assert cv_obs.wires == Wires([1])
        assert cv_obs.ev_order == 2

    def test_repr(self):
        """Test the string representation of an observable with and without a return type."""

        m = qml.expval(qml.PauliZ(wires=["a"]) @ qml.PauliZ(wires=["b"]))
        expected = "expval(PauliZ(wires=['a']) @ PauliZ(wires=['b']))"
        assert str(m) == expected

        m = qml.probs(wires=["a"])
        expected = "probs(wires=['a'])"
        assert str(m) == expected

        m = qml.probs(op=qml.PauliZ(wires=["a"]))
        expected = "probs(PauliZ(wires=['a']))"
        assert str(m) == expected

        m = qml.PauliZ(wires=["a"]) @ qml.PauliZ(wires=["b"])
        expected = "PauliZ(wires=['a']) @ PauliZ(wires=['b'])"
        assert str(m) == expected

        m = qml.PauliZ(wires=["a"])
        expected = "PauliZ(wires=['a'])"
        assert str(m) == expected

    def test_id(self):
        """Test that the id attribute of an observable can be set."""

        class DummyObserv(qml.operation.Observable):
            r"""Dummy custom observable"""
            num_wires = 1
            grad_method = None

        op = DummyObserv(1.0, wires=0, id="test")
        assert op.id == "test"

    def test_wire_is_given_in_argument(self):
        class DummyObservable(qml.operation.Observable):
            num_wires = 1

        with pytest.raises(Exception, match="Must specify the wires *"):
            DummyObservable()


class TestOperatorIntegration:
    """Integration tests for the Operator class"""

    def test_all_wires_defined_but_init_with_one(self):
        """Test that an exception is raised if the class is defined with ALL wires,
        but then instantiated with only one"""

        dev1 = qml.device("default.qubit", wires=2)

        class DummyOp(qml.operation.Operation):
            r"""Dummy custom operator"""
            num_wires = qml.operation.WiresEnum.AllWires

        @qml.qnode(dev1)
        def circuit():
            DummyOp(wires=[0])
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(
            qml.QuantumFunctionError,
            match=f"Operator {DummyOp.__name__} must act on all wires",
        ):
            circuit()


class TestInverse:
    """Test inverse of operations"""

    def test_operation_inverse_using_dummy_operation(self):
        some_param = 0.5

        class DummyOp(qml.operation.Operation):
            r"""Dummy custom Operation"""
            num_wires = 1

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

        with pytest.raises(
            qml.DeviceError,
            match=r"inverse of gates are not supported on device default\.gaussian",
        ):
            mean_photon_gaussian(0.015, 0.02, 0.005)

    @pytest.fixture(scope="function")
    def qnode_for_inverse(self, mock_device):
        """Provides a QNode for the subsequent tests of inv"""

        def circuit(x):
            qml.RZ(x, wires=[1]).inv()
            qml.RZ(x, wires=[1]).inv().inv()
            return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliZ(1))

        node = qml.QNode(circuit, mock_device)
        node.construct([1.0], {})

        return node

    def test_operation_inverse_defined(self, qnode_for_inverse):
        """Test that the inverse of an operation is added to the QNode queue and the operation is an instance
        of the original class"""
        assert qnode_for_inverse.qtape.operations[0].name == "RZ.inv"
        assert qnode_for_inverse.qtape.operations[0].inverse
        assert issubclass(qnode_for_inverse.qtape.operations[0].__class__, qml.operation.Operation)
        assert qnode_for_inverse.qtape.operations[1].name == "RZ"
        assert not qnode_for_inverse.qtape.operations[1].inverse
        assert issubclass(qnode_for_inverse.qtape.operations[1].__class__, qml.operation.Operation)


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

        with pytest.raises(
            ValueError, match="Can only perform tensor products between observables"
        ):
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
        assert t.wires == Wires([0, 1, 2])

    def test_params(self):
        """Test that the correct flattened list of parameters is returned"""
        p = np.array([0.5])
        X = qml.PauliX(0)
        Y = qml.Hermitian(p, wires=[1, 2])
        t = Tensor(X, Y)
        assert t.data == [p]

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

    def test_label(self):
        """Test that Tensors are labelled as expected"""

        x = qml.PauliX(0)
        y = qml.PauliZ(2)
        T = Tensor(x, y)

        assert T.label() == "X@Z"
        assert T.label(decimals=2) == "X@Z"
        assert T.label(base_label=["X0", "Z2"]) == "X0@Z2"

        with pytest.raises(ValueError, match=r"Tensor label requires"):
            T.label(base_label="nope")

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

        with pytest.raises(
            ValueError, match="Can only perform tensor products between observables"
        ):
            X @ Y

        with pytest.raises(
            ValueError, match="Can only perform tensor products between observables"
        ):
            T = X @ Z
            T @ Y

        with pytest.raises(
            ValueError, match="Can only perform tensor products between observables"
        ):
            T = X @ Z
            Y @ T

    def test_eigvals(self):
        """Test that the correct eigenvalues are returned for the Tensor"""
        X = qml.PauliX(0)
        Y = qml.PauliY(2)
        t = Tensor(X, Y)
        assert np.array_equal(t.get_eigvals(), np.kron([1, -1], [1, -1]))

        # test that the eigvals are now cached and not recalculated
        assert np.array_equal(t._eigvals_cache, t.get_eigvals())

    @pytest.mark.usefixtures("tear_down_hermitian")
    def test_eigvals_hermitian(self, tol):
        """Test that the correct eigenvalues are returned for the Tensor containing an Hermitian observable"""
        X = qml.PauliX(0)
        hamiltonian = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
        Herm = qml.Hermitian(hamiltonian, wires=[1, 2])
        t = Tensor(X, Herm)
        d = np.kron(np.array([1.0, -1.0]), np.array([-1.0, 1.0, 1.0, 1.0]))
        t = t.get_eigvals()
        assert np.allclose(t, d, atol=tol, rtol=0)

    def test_eigvals_identity(self, tol):
        """Test that the correct eigenvalues are returned for the Tensor containing an Identity"""
        X = qml.PauliX(0)
        Iden = qml.Identity(1)
        t = Tensor(X, Iden)
        d = np.kron(np.array([1.0, -1.0]), np.array([1.0, 1.0]))
        t = t.get_eigvals()
        assert np.allclose(t, d, atol=tol, rtol=0)

    def test_eigvals_identity_and_hermitian(self, tol):
        """Test that the correct eigenvalues are returned for the Tensor containing
        multiple types of observables"""
        H = np.diag([1, 2, 3, 4])
        O = qml.PauliX(0) @ qml.Identity(2) @ qml.Hermitian(H, wires=[4, 5])
        res = O.get_eigvals()
        expected = np.kron(np.array([1.0, -1.0]), np.kron(np.array([1.0, 1.0]), np.arange(1, 5)))
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_diagonalizing_gates(self, tol):
        """Test that the correct diagonalizing gate set is returned for a Tensor of observables"""
        H = np.diag([1, 2, 3, 4])
        O = qml.PauliX(0) @ qml.Identity(2) @ qml.PauliY(1) @ qml.Hermitian(H, [5, 6])

        res = O.diagonalizing_gates()

        # diagonalize the PauliX on wire 0 (H.X.H = Z)
        assert isinstance(res[0], qml.Hadamard)
        assert res[0].wires == Wires([0])

        # diagonalize the PauliY on wire 1 (U.Y.U^\dagger = Z
        # where U = HSZ).
        assert isinstance(res[1], qml.PauliZ)
        assert res[1].wires == Wires([1])
        assert isinstance(res[2], qml.S)
        assert res[2].wires == Wires([1])
        assert isinstance(res[3], qml.Hadamard)
        assert res[3].wires == Wires([1])

        # diagonalize the Hermitian observable on wires 5, 6
        assert isinstance(res[4], qml.QubitUnitary)
        assert res[4].wires == Wires([5, 6])

        O = O @ qml.Hadamard(4)
        res = O.diagonalizing_gates()

        # diagonalize the Hadamard observable on wire 4
        # (RY(-pi/4).H.RY(pi/4) = Z)
        assert isinstance(res[-1], qml.RY)
        assert res[-1].wires == Wires([4])
        assert np.allclose(res[-1].parameters, -np.pi / 4, atol=tol, rtol=0)

    def test_diagonalizing_gates_numerically_diagonalizes(self, tol):
        """Test that the diagonalizing gate set numerically
        diagonalizes the tensor observable"""

        # create a tensor observable acting on consecutive wires
        H = np.diag([1, 2, 3, 4])
        O = qml.PauliX(0) @ qml.PauliY(1) @ qml.Hermitian(H, [2, 3])

        O_mat = O.get_matrix()
        diag_gates = O.diagonalizing_gates()

        # group the diagonalizing gates based on what wires they act on
        U_list = []
        for _, g in itertools.groupby(diag_gates, lambda x: x.wires.tolist()):
            # extract the matrices of each diagonalizing gate
            mats = [i.get_matrix() for i in g]

            # Need to revert the order in which the matrices are applied such that they adhere to the order
            # of matrix multiplication
            # E.g. for PauliY: [PauliZ(wires=self.wires), S(wires=self.wires), Hadamard(wires=self.wires)]
            # becomes Hadamard @ S @ PauliZ, where @ stands for matrix multiplication
            mats = mats[::-1]

            if len(mats) > 1:
                # multiply all unitaries together before appending
                mats = [multi_dot(mats)]

            # append diagonalizing unitary for specific wire to U_list
            U_list.append(mats[0])

        # since the test is assuming consecutive wires for each observable
        # in the tensor product, it is sufficient to Kronecker product
        # the entire list.
        U = reduce(np.kron, U_list)

        res = U @ O_mat @ U.conj().T
        expected = np.diag(O.get_eigvals())

        # once diagonalized by U, the result should be a diagonal
        # matrix of the eigenvalues.
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_tensor_matrix(self, tol):
        """Test that the tensor product matrix method returns
        the correct result"""
        H = np.diag([1, 2, 3, 4])
        O = qml.PauliX(0) @ qml.PauliY(1) @ qml.Hermitian(H, [2, 3])

        res = O.get_matrix()
        expected = reduce(np.kron, [qml.PauliX.compute_matrix(), qml.PauliY.compute_matrix(), H])

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_matrix_wire_order_not_implemented(self):
        """Test that an exception is raised if a wire_order is passed to the matrix method"""
        O = qml.PauliX(0) @ qml.PauliY(1)
        with pytest.raises(NotImplementedError, match="wire_order"):
            O.get_matrix(wire_order=[1, 0])

    def test_tensor_matrix_partial_wires_overlap_warning(self, tol):
        """Tests that a warning is raised if the wires the factors in
        the tensor product act on have partial overlaps."""
        H = np.diag([1, 2, 3, 4])
        O1 = qml.PauliX(0) @ qml.Hermitian(H, [0, 1])
        O2 = qml.Hermitian(H, [0, 1]) @ qml.PauliY(1)

        for O in (O1, O2):
            with pytest.warns(UserWarning, match="partially overlapping"):
                O.get_matrix()

    def test_tensor_matrix_too_large_warning(self, tol):
        """Tests that a warning is raised if wires occur in multiple of the
        factors in the tensor product, leading to a wrongly-sized matrix."""
        O = qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliX(0)
        with pytest.warns(UserWarning, match="The size of the returned matrix"):
            O.get_matrix()

    @pytest.mark.parametrize("classes", [(qml.PauliX, qml.PauliX), (qml.PauliZ, qml.PauliX)])
    def test_multiplication_matrix(self, tol, classes):
        """If using the ``@`` operator on two observables acting on the
        same wire, the tensor class should treat this as matrix multiplication."""
        c1, c2 = classes
        O = c1(0) @ c2(0)

        res = O.get_matrix()
        expected = c1.compute_matrix() @ c2.compute_matrix()

        assert np.allclose(res, expected, atol=tol, rtol=0)

    herm_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    tensor_obs = [
        (qml.PauliZ(0) @ qml.Identity(1) @ qml.PauliZ(2), [qml.PauliZ(0), qml.PauliZ(2)]),
        (
            qml.Identity(0)
            @ qml.PauliX(1)
            @ qml.Identity(2)
            @ qml.PauliZ(3)
            @ qml.PauliZ(4)
            @ qml.Identity(5),
            [qml.PauliX(1), qml.PauliZ(3), qml.PauliZ(4)],
        ),
        # List containing single observable is returned
        (qml.PauliZ(0) @ qml.Identity(1), [qml.PauliZ(0)]),
        (qml.Identity(0) @ qml.PauliX(1) @ qml.Identity(2), [qml.PauliX(1)]),
        (qml.Identity(0) @ qml.Identity(1), [qml.Identity(0)]),
        (
            qml.Identity(0) @ qml.Identity(1) @ qml.Hermitian(herm_matrix, wires=[2, 3]),
            [qml.Hermitian(herm_matrix, wires=[2, 3])],
        ),
    ]

    @pytest.mark.parametrize("tensor_observable, expected", tensor_obs)
    def test_non_identity_obs(self, tensor_observable, expected):
        """Tests that the non_identity_obs property returns a list that contains no Identity instances."""

        O = tensor_observable
        for idx, obs in enumerate(O.non_identity_obs):
            assert type(obs) == type(expected[idx])
            assert obs.wires == expected[idx].wires

    tensor_obs_pruning = [
        (qml.PauliZ(0) @ qml.Identity(1) @ qml.PauliZ(2), qml.PauliZ(0) @ qml.PauliZ(2)),
        (
            qml.Identity(0)
            @ qml.PauliX(1)
            @ qml.Identity(2)
            @ qml.PauliZ(3)
            @ qml.PauliZ(4)
            @ qml.Identity(5),
            qml.PauliX(1) @ qml.PauliZ(3) @ qml.PauliZ(4),
        ),
        # Single observable is returned
        (qml.PauliZ(0) @ qml.Identity(1), qml.PauliZ(0)),
        (qml.Identity(0) @ qml.PauliX(1) @ qml.Identity(2), qml.PauliX(1)),
        (qml.Identity(0) @ qml.Identity(1), qml.Identity(0)),
        (qml.Identity(0) @ qml.Identity(1), qml.Identity(0)),
        (
            qml.Identity(0) @ qml.Identity(1) @ qml.Hermitian(herm_matrix, wires=[2, 3]),
            qml.Hermitian(herm_matrix, wires=[2, 3]),
        ),
    ]

    @pytest.mark.parametrize("tensor_observable, expected", tensor_obs_pruning)
    def test_prune(self, tensor_observable, expected):
        """Tests that the prune method returns the expected Tensor or single non-Tensor Observable."""
        O = tensor_observable
        O_expected = expected

        O_pruned = O.prune()
        assert type(O_pruned) == type(expected)
        assert O_pruned.wires == expected.wires

    def test_prune_while_queueing_return_tensor(self):
        """Tests that pruning a tensor to a tensor in a tape context registers
        the pruned tensor as owned by the measurement,
        and turns the original tensor into an orphan without an owner."""

        with qml.tape.QuantumTape() as tape:
            # we assign operations to variables here so we can compare them below
            a = qml.PauliX(wires=0)
            b = qml.PauliY(wires=1)
            c = qml.Identity(wires=2)
            T = qml.operation.Tensor(a, b, c)
            T_pruned = T.prune()
            m = qml.expval(T_pruned)

        ann_queue = tape._queue

        # the pruned tensor became the owner of Paulis
        assert ann_queue[a]["owner"] == T_pruned
        assert ann_queue[b]["owner"] == T_pruned

        # the Identity is still owned by the original Tensor
        assert ann_queue[c]["owner"] == T
        # the original tensor still owns all three observables
        # but is not owned by a measurement
        assert ann_queue[T]["owns"] == (a, b, c)
        assert not hasattr(ann_queue[T], "owner")

        # the pruned tensor is owned by the measurement
        # and owns the two Paulis
        assert ann_queue[T_pruned]["owner"] == m
        assert ann_queue[T_pruned]["owns"] == (a, b)
        assert ann_queue[m]["owns"] == T_pruned

    def test_prune_while_queueing_return_obs(self):
        """Tests that pruning a tensor to an observable in a tape context registers
        the pruned observable as owned by the measurement,
        and turns the original tensor into an orphan without an owner."""

        with qml.tape.QuantumTape() as tape:
            a = qml.PauliX(wires=0)
            c = qml.Identity(wires=2)
            T = qml.operation.Tensor(a, c)
            T_pruned = T.prune()
            m = qml.expval(T_pruned)

        ann_queue = tape._queue

        # the pruned tensor is the Pauli observable
        assert T_pruned == a
        # pruned tensor/Pauli is owned by the measurement
        # since the entry in the dictionary got updated
        # when the pruned tensor's owner was memorized
        assert ann_queue[a]["owner"] == m
        # the Identity is still owned by the original Tensor
        assert ann_queue[c]["owner"] == T

        # the original tensor still owns both observables
        # but is not owned by a measurement
        assert ann_queue[T]["owns"] == (a, c)
        assert not hasattr(ann_queue[T], "owner")

        # the measurement owns the Pauli/pruned tensor
        assert ann_queue[m]["owns"] == T_pruned

    def test_sparse_matrix_no_wires(self):
        """Tests that the correct sparse matrix representation is used."""

        t = qml.PauliX(0) @ qml.PauliZ(1)
        s = t.sparse_matrix()

        assert np.allclose(s.row, [0, 1, 2, 3])
        assert np.allclose(s.col, [2, 3, 0, 1])
        assert np.allclose(s.data, [1, -1, 1, -1])

    def test_sparse_matrix_swapped_wires(self):
        """Tests that the correct sparse matrix representation is used
        when the custom wires swap the order."""

        t = qml.PauliX(0) @ qml.PauliZ(1)
        s = t.sparse_matrix(wires=[1, 0])

        assert np.allclose(s.row, [0, 1, 2, 3])
        assert np.allclose(s.col, [1, 0, 3, 2])
        assert np.allclose(s.data, [1, 1, -1, -1])

    def test_sparse_matrix_extra_wire(self):
        """Tests that the correct sparse matrix representation is used
        when the custom wires add an extra wire with an implied identity operation."""

        t = qml.PauliX(0) @ qml.PauliZ(1)
        s = t.sparse_matrix(wires=[0, 1, 2])

        assert s.shape == (8, 8)
        assert np.allclose(s.row, [0, 1, 2, 3, 4, 5, 6, 7])
        assert np.allclose(s.col, [4, 5, 6, 7, 0, 1, 2, 3])
        assert np.allclose(s.data, [1, 1, -1, -1, 1, 1, -1, -1])

    def test_sparse_matrix_error(self):
        """Tests that an error is raised if the sparse matrix is computed for
        a tensor whose constituent operations are not all single-qubit gates."""

        t = qml.PauliX(0) @ qml.Hermitian(np.eye(4), wires=[1, 2])
        with pytest.raises(ValueError, match="Can only compute"):
            t.sparse_matrix()


equal_obs = [
    (qml.PauliZ(0), qml.PauliZ(0), True),
    (qml.PauliZ(0) @ qml.PauliX(1), qml.PauliZ(0) @ qml.PauliX(1) @ qml.Identity(2), True),
    (qml.PauliZ("b"), qml.PauliZ("b") @ qml.Identity(1.3), True),
    (qml.PauliZ(0) @ qml.Identity(1), qml.PauliZ(0), True),
    (qml.PauliZ(0), qml.PauliZ(1) @ qml.Identity(0), False),
    (
        qml.Hermitian(np.array([[0, 1], [1, 0]]), 0),
        qml.Identity(1) @ qml.Hermitian(np.array([[0, 1], [1, 0]]), 0),
        True,
    ),
    (qml.PauliZ("a") @ qml.PauliX(1), qml.PauliX(1) @ qml.PauliZ("a"), True),
    (qml.PauliZ("a"), qml.Hamiltonian([1], [qml.PauliZ("a")]), True),
]

add_obs = [
    (qml.PauliZ(0) @ qml.Identity(1), qml.PauliZ(0), qml.Hamiltonian([2], [qml.PauliZ(0)])),
    (
        qml.PauliZ(0),
        qml.PauliZ(0) @ qml.PauliX(1),
        qml.Hamiltonian([1, 1], [qml.PauliZ(0), qml.PauliZ(0) @ qml.PauliX(1)]),
    ),
    (
        qml.PauliZ("b") @ qml.Identity(1),
        qml.Hamiltonian([3], [qml.PauliZ("b")]),
        qml.Hamiltonian([4], [qml.PauliZ("b")]),
    ),
    (
        qml.PauliX(0) @ qml.PauliZ(1),
        qml.PauliZ(1) @ qml.Identity(2) @ qml.PauliX(0),
        qml.Hamiltonian([2], [qml.PauliX(0) @ qml.PauliZ(1)]),
    ),
    (
        qml.Hermitian(np.array([[1, 0], [0, -1]]), 1.2),
        qml.Hamiltonian([3], [qml.Hermitian(np.array([[1, 0], [0, -1]]), 1.2)]),
        qml.Hamiltonian([4], [qml.Hermitian(np.array([[1, 0], [0, -1]]), 1.2)]),
    ),
]

mul_obs = [
    (qml.PauliZ(0), 3, qml.Hamiltonian([3], [qml.PauliZ(0)])),
    (qml.PauliZ(0) @ qml.Identity(1), 3, qml.Hamiltonian([3], [qml.PauliZ(0)])),
    (qml.PauliZ(0) @ qml.PauliX(1), 4.5, qml.Hamiltonian([4.5], [qml.PauliZ(0) @ qml.PauliX(1)])),
    (
        qml.Hermitian(np.array([[1, 0], [0, -1]]), "c"),
        3,
        qml.Hamiltonian([3], [qml.Hermitian(np.array([[1, 0], [0, -1]]), "c")]),
    ),
]

sub_obs = [
    (qml.PauliZ(0) @ qml.Identity(1), qml.PauliZ(0), qml.Hamiltonian([], [])),
    (
        qml.PauliZ(0),
        qml.PauliZ(0) @ qml.PauliX(1),
        qml.Hamiltonian([1, -1], [qml.PauliZ(0), qml.PauliZ(0) @ qml.PauliX(1)]),
    ),
    (
        qml.PauliZ(0) @ qml.Identity(1),
        qml.Hamiltonian([3], [qml.PauliZ(0)]),
        qml.Hamiltonian([-2], [qml.PauliZ(0)]),
    ),
    (
        qml.PauliX(0) @ qml.PauliZ(1),
        qml.PauliZ(3) @ qml.Identity(2) @ qml.PauliX(0),
        qml.Hamiltonian([1, -1], [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(3) @ qml.PauliX(0)]),
    ),
    (
        qml.Hermitian(np.array([[1, 0], [0, -1]]), 1.2),
        qml.Hamiltonian([3], [qml.Hermitian(np.array([[1, 0], [0, -1]]), 1.2)]),
        qml.Hamiltonian([-2], [qml.Hermitian(np.array([[1, 0], [0, -1]]), 1.2)]),
    ),
]


class TestTensorObservableOperations:
    """Tests arithmetic operations between observables/tensors"""

    def test_data(self):
        """Tests the data() method for Tensors and Observables"""

        obs = qml.PauliZ(0)
        data = obs._obs_data()

        assert data == {("PauliZ", Wires(0), ())}

        obs = qml.PauliZ(0) @ qml.PauliX(1)
        data = obs._obs_data()

        assert data == {("PauliZ", Wires(0), ()), ("PauliX", Wires(1), ())}

        obs = qml.Hermitian(np.array([[1, 0], [0, -1]]), 0)
        data = obs._obs_data()

        assert data == {
            (
                "Hermitian",
                Wires(0),
                (
                    b"\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xff\xff\xff\xff\xff\xff\xff",
                ),
            )
        }

    def test_equality_error(self):
        """Tests that the correct error is raised when compare() is called on invalid type"""

        obs = qml.PauliZ(0)
        tensor = qml.PauliZ(0) @ qml.PauliX(1)
        A = [[1, 0], [0, -1]]
        with pytest.raises(
            ValueError,
            match=r"Can only compare an Observable/Tensor, and a Hamiltonian/Observable/Tensor.",
        ):
            obs.compare(A)
            tensor.compare(A)

    @pytest.mark.parametrize(("obs1", "obs2", "res"), equal_obs)
    def test_equality(self, obs1, obs2, res):
        """Tests the compare() method for Tensors and Observables"""
        assert obs1.compare(obs2) == res

    @pytest.mark.parametrize(("obs1", "obs2", "obs"), add_obs)
    def test_addition(self, obs1, obs2, obs):
        """Tests addition between Tensors and Observables"""
        assert obs.compare(obs1 + obs2)

    @pytest.mark.parametrize(("coeff", "obs", "res_obs"), mul_obs)
    def test_scalar_multiplication(self, coeff, obs, res_obs):
        """Tests scalar multiplication of Tensors and Observables"""
        assert res_obs.compare(coeff * obs)
        assert res_obs.compare(obs * coeff)

    @pytest.mark.parametrize(("obs1", "obs2", "obs"), sub_obs)
    def test_subtraction(self, obs1, obs2, obs):
        """Tests subtraction between Tensors and Observables"""
        assert obs.compare(obs1 - obs2)

    def test_arithmetic_errors(self):
        """Tests that the arithmetic operations throw the correct errors"""
        obs = qml.PauliZ(0)
        tensor = qml.PauliZ(0) @ qml.PauliX(1)
        A = [[1, 0], [0, -1]]
        with pytest.raises(ValueError, match="Cannot add Observable"):
            obs + A
            tensor + A
        with pytest.raises(ValueError, match="Cannot multiply Observable"):
            obs * A
            A * tensor
        with pytest.raises(ValueError, match="Cannot subtract"):
            obs - A
            tensor - A


# Dummy class inheriting from Operator
class MyOp(Operator):
    num_wires = 1


# Dummy class inheriting from Operation
class MyGate(Operation):
    num_wires = 1


op = MyOp(wires=1)
gate = MyGate(wires=1)


class TestDefaultRepresentations:
    """Tests that the default representations raise custom errors"""

    def test_decomposition_undefined(self):
        """Tests that custom error is raised in the default decomposition representation."""
        with pytest.raises(qml.operation.DecompositionUndefinedError):
            MyOp.compute_decomposition(wires=[1])
        with pytest.raises(qml.operation.DecompositionUndefinedError):
            op.decomposition()

    def test_matrix_undefined(self):
        """Tests that custom error is raised in the default matrix representation."""
        with pytest.raises(qml.operation.MatrixUndefinedError):
            MyOp.compute_matrix()
        with pytest.raises(qml.operation.MatrixUndefinedError):
            op.get_matrix()

    def test_terms_undefined(self):
        """Tests that custom error is raised in the default terms representation."""
        with pytest.raises(qml.operation.TermsUndefinedError):
            MyOp.compute_terms(wires=[1])
        with pytest.raises(qml.operation.TermsUndefinedError):
            op.terms()

    def test_sparse_matrix_undefined(self):
        """Tests that custom error is raised in the default sparse matrix representation."""
        with pytest.raises(NotImplementedError):
            MyOp(wires="a").sparse_matrix(wire_order=["a", "b"])
        with pytest.raises(qml.operation.SparseMatrixUndefinedError):
            MyOp.compute_sparse_matrix()
        with pytest.raises(qml.operation.SparseMatrixUndefinedError):
            op.sparse_matrix()

    def test_eigvals_undefined(self):
        """Tests that custom error is raised in the default eigenvalue representation."""
        with pytest.raises(qml.operation.EigvalsUndefinedError):
            MyOp.compute_eigvals()
        with pytest.raises(qml.operation.EigvalsUndefinedError):
            op.get_eigvals()

    def test_diaggates_undefined(self):
        """Tests that custom error is raised in the default diagonalizing gates representation."""
        with pytest.raises(qml.operation.DiagGatesUndefinedError):
            MyOp.compute_diagonalizing_gates(wires=[1])
        with pytest.raises(qml.operation.DiagGatesUndefinedError):
            op.diagonalizing_gates()

    def test_adjoint_undefined(self):
        """Tests that custom error is raised in the default adjoint representation."""
        with pytest.raises(qml.operation.AdjointUndefinedError):
            gate.adjoint()

    def test_generator_undefined(self):
        """Tests that custom error is raised in the default generator representation."""
        with pytest.raises(qml.operation.GeneratorUndefinedError):
            gate.generator()


class TestChannel:
    """Unit tests for the Channel class"""

    def test_instance_made_correctly(self):
        """Test that instance of channel class is initialized correctly"""

        class DummyOp(qml.operation.Channel):
            r"""Dummy custom channel"""
            num_wires = 1
            grad_method = "F"

            @staticmethod
            def compute_kraus_matrices(p):
                K1 = np.sqrt(p) * X
                K2 = np.sqrt(1 - p) * I
                return [K1, K2]

        expected = np.array([[0, np.sqrt(0.1)], [np.sqrt(0.1), 0]])
        op = DummyOp(0.1, wires=0)
        assert np.all(op.kraus_matrices()[0] == expected)


class TestOperationDerivative:
    """Tests for operation_derivative function"""

    def test_no_generator_raise(self):
        """Tests if the function raises an exception if the input operation has no generator"""

        class CustomOp(qml.operation.Operation):
            num_wires = 1
            num_params = 1

        op = CustomOp(0.5, wires=0)

        with pytest.raises(
            qml.operation.GeneratorUndefinedError,
            match="Operation CustomOp does not have a generator",
        ):
            operation_derivative(op)

    def test_multiparam_raise(self):
        """Test if the function raises a ValueError if the input operation is composed of multiple
        parameters"""

        class RotWithGen(qml.Rot):
            def generator(self):
                return qml.Hermitian(np.zeros((2, 2)), wires=self.wires)

        op = RotWithGen(0.1, 0.2, 0.3, wires=0)

        with pytest.raises(ValueError, match="Operation RotWithGen is not written in terms of"):
            operation_derivative(op)

    def test_rx(self):
        """Test if the function correctly returns the derivative of RX"""
        p = 0.3
        op = qml.RX(p, wires=0)

        derivative = operation_derivative(op)

        expected_derivative = 0.5 * np.array(
            [[-np.sin(p / 2), -1j * np.cos(p / 2)], [-1j * np.cos(p / 2), -np.sin(p / 2)]]
        )

        assert np.allclose(derivative, expected_derivative)

        op.inv()
        derivative_inv = operation_derivative(op)
        expected_derivative_inv = 0.5 * np.array(
            [[-np.sin(p / 2), 1j * np.cos(p / 2)], [1j * np.cos(p / 2), -np.sin(p / 2)]]
        )

        assert not np.allclose(derivative, derivative_inv)
        assert np.allclose(derivative_inv, expected_derivative_inv)

    def test_phase(self):
        """Test if the function correctly returns the derivative of PhaseShift"""
        p = 0.3
        op = qml.PhaseShift(p, wires=0)

        derivative = operation_derivative(op)
        expected_derivative = np.array([[0, 0], [0, 1j * np.exp(1j * p)]])
        assert np.allclose(derivative, expected_derivative)

    def test_cry(self):
        """Test if the function correctly returns the derivative of CRY"""
        p = 0.3
        op = qml.CRY(p, wires=[0, 1])

        derivative = operation_derivative(op)
        expected_derivative = 0.5 * np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, -np.sin(p / 2), -np.cos(p / 2)],
                [0, 0, np.cos(p / 2), -np.sin(p / 2)],
            ]
        )
        assert np.allclose(derivative, expected_derivative)

    def test_cry_non_consecutive(self):
        """Test if the function correctly returns the derivative of CRY
        if the wires are not consecutive. This is expected behaviour, since
        without any other context, the operation derivative should make no
        assumption about the wire ordering."""
        p = 0.3
        op = qml.CRY(p, wires=[1, 0])

        derivative = operation_derivative(op)
        expected_derivative = 0.5 * np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, -np.sin(p / 2), -np.cos(p / 2)],
                [0, 0, np.cos(p / 2), -np.sin(p / 2)],
            ]
        )
        assert np.allclose(derivative, expected_derivative)


class TestCVOperation:
    """Test the CVOperation class"""

    def test_wires_not_found(self):
        """Make sure that `heisenberg_expand` method receives enough wires to actually expand"""

        class DummyOp(qml.operation.CVOperation):
            num_wires = 1

        op = DummyOp(wires=1)

        with pytest.raises(ValueError, match="do not exist on this device with wires"):
            op.heisenberg_expand(np.eye(3), Wires(["a", "b"]))

    def test_input_validation(self):
        """Make sure that size of input for `heisenberg_expand` method is validated"""

        class DummyOp(qml.operation.CVOperation):
            num_wires = 1

        op = DummyOp(wires=1)

        with pytest.raises(ValueError, match="Heisenberg matrix is the wrong size"):
            U_wrong_size = np.eye(1)
            op.heisenberg_expand(U_wrong_size, op.wires)

    def test_wrong_input_shape(self):
        """Ensure that `heisenberg_expand` raises exception if it receives an array with order > 2"""

        class DummyOp(qml.operation.CVOperation):
            num_wires = 1

        op = DummyOp(wires=1)

        with pytest.raises(ValueError, match="Only order-1 and order-2 arrays supported"):
            U_high_order = np.array([np.eye(3)] * 3)
            op.heisenberg_expand(U_high_order, op.wires)


class TestCriteria:
    rx = qml.RX(qml.numpy.array(0.3, requires_grad=True), wires=1)
    stiff_rx = qml.RX(0.3, wires=1)
    cnot = qml.CNOT(wires=[1, 0])
    rot = qml.Rot(*qml.numpy.array([0.1, -0.7, 0.2], requires_grad=True), wires=0)
    stiff_rot = qml.Rot(0.1, -0.7, 0.2, wires=0)
    exp = qml.expval(qml.PauliZ(0))

    def test_docstring(self):
        expected = "Returns ``True`` if an operator has a generator defined."
        assert qml.operation.has_gen.__doc__ == expected

    def test_has_gen(self):
        """Test has_gen criterion."""
        assert qml.operation.has_gen(self.rx)
        assert not qml.operation.has_gen(self.cnot)
        assert not qml.operation.has_gen(self.rot)
        assert not qml.operation.has_gen(self.exp)

    def test_has_grad_method(self):
        """Test has_grad_method criterion."""
        assert qml.operation.has_grad_method(self.rx)
        assert qml.operation.has_grad_method(self.rot)
        assert not qml.operation.has_grad_method(self.cnot)

    def test_has_multipar(self):
        """Test has_multipar criterion."""
        assert not qml.operation.has_multipar(self.rx)
        assert qml.operation.has_multipar(self.rot)
        assert not qml.operation.has_multipar(self.cnot)

    def test_has_nopar(self):
        """Test has_nopar criterion."""
        assert not qml.operation.has_nopar(self.rx)
        assert not qml.operation.has_nopar(self.rot)
        assert qml.operation.has_nopar(self.cnot)

    def test_has_unitary_gen(self):
        """Test has_unitary_gen criterion."""
        assert qml.operation.has_unitary_gen(self.rx)
        assert not qml.operation.has_unitary_gen(self.rot)
        assert not qml.operation.has_unitary_gen(self.cnot)

    def test_is_measurement(self):
        """Test is_measurement criterion."""
        assert not qml.operation.is_measurement(self.rx)
        assert not qml.operation.is_measurement(self.rot)
        assert not qml.operation.is_measurement(self.cnot)
        assert qml.operation.is_measurement(self.exp)

    def test_is_trainable(self):
        """Test is_trainable criterion."""
        assert qml.operation.is_trainable(self.rx)
        assert not qml.operation.is_trainable(self.stiff_rx)
        assert qml.operation.is_trainable(self.rot)
        assert not qml.operation.is_trainable(self.stiff_rot)
        assert not qml.operation.is_trainable(self.cnot)

    def test_composed(self):
        """Test has_gen criterion."""
        both = qml.operation.has_gen & qml.operation.is_trainable
        assert both(self.rx)
        assert not both(self.cnot)
        assert not both(self.rot)
        assert not both(self.exp)


class TestExpandMatrix:
    """Tests for the expand_matrix helper function."""

    def test_no_expansion(self):
        """Tests the case where the original matrix is not changed"""
        base_matrix = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
        res = qml.operation.expand_matrix(base_matrix, wires=[0, 2], wire_order=[0, 2])
        assert np.allclose(base_matrix, res)

    def test_permutation(self):
        """Tests the case where the original matrix is permuted"""
        base_matrix = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
        res = qml.operation.expand_matrix(base_matrix, wires=[0, 2], wire_order=[2, 0])

        expected = np.array([[1, 3, 2, 4], [9, 11, 10, 12], [5, 7, 6, 8], [13, 15, 14, 16]])
        assert np.allclose(expected, res)

    def test_expansion(self):
        """Tests the case where the original matrix is expanded"""
        base_matrix = np.array([[0, 1], [1, 0]])
        res = qml.operation.expand_matrix(base_matrix, wires=[2], wire_order=[0, 2])
        expected = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
        assert np.allclose(expected, res)

        res = qml.operation.expand_matrix(base_matrix, wires=[2], wire_order=[2, 0])
        expected = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]])
        assert np.allclose(expected, res)

    def test_autograd(self, tol):
        """Tests differentiation in autograd by checking how a specific element of the expanded matrix depends on the
        canonical matrix."""

        def func(mat):
            res = qml.operation.expand_matrix(mat, wires=[2], wire_order=[0, 2])
            return res[0, 1]

        base_matrix = pnp.array([[0.0, 1.0], [1.0, 0.0]], requires_grad=True)
        grad_fn = qml.grad(func)
        gradient = grad_fn(base_matrix)

        # the entry should propagate from position (0, 1) in the original tensor
        expected = np.array([[0.0, 1.0], [0.0, 0.0]])
        assert np.allclose(gradient, expected, atol=tol)

    def test_torch(self, tol):
        """Tests differentiation in torch by checking how a specific element of the expanded matrix depends on the
        canonical matrix."""
        torch = pytest.importorskip("torch")

        base_matrix = torch.tensor([[0.0, 1.0], [1.0, 0.0]], requires_grad=True)
        res = qml.operation.expand_matrix(base_matrix, wires=[2], wire_order=[0, 2])
        element = res[0, 1]
        element.backward()
        gradient = base_matrix.grad

        # the entry should propagate from position (0, 1) in the original tensor
        expected = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
        assert np.allclose(gradient, expected, atol=tol)

    def test_jax(self, tol):
        """Tests differentiation in jax by checking how a specific element of the expanded matrix depends on the
        canonical matrix."""
        jax = pytest.importorskip("jax")
        from jax import numpy as jnp

        def func(mat):
            res = qml.operation.expand_matrix(mat, wires=[2], wire_order=[0, 2])
            return res[0, 1]

        base_matrix = jnp.array([[0.0, 1.0], [1.0, 0.0]])
        grad_fn = jax.grad(func)
        gradient = grad_fn(base_matrix)

        # the entry should propagate from position (0, 1) in the original tensor
        expected = np.array([[0.0, 1.0], [0.0, 0.0]])
        assert np.allclose(gradient, expected, atol=tol)

    def test_tf(self, tol):
        """Tests differentiation in TensorFlow by checking how a specific element of the expanded matrix depends on the
        canonical matrix."""
        tf = pytest.importorskip("tensorflow")

        base_matrix = tf.Variable([[0.0, 1.0], [1.0, 0.0]])
        with tf.GradientTape() as tape:
            res = qml.operation.expand_matrix(base_matrix, wires=[2], wire_order=[0, 2])
            element = res[0, 1]

        gradient = tape.gradient(element, base_matrix)

        # the entry should propagate from position (0, 1) in the original tensor
        expected = tf.constant([[0.0, 1.0], [0.0, 0.0]])
        assert np.allclose(gradient, expected, atol=tol)

    def test_expand_one(self, tol):
        """Test that a 1 qubit gate correctly expands to 3 qubits."""
        U = np.array(
            [
                [0.83645892 - 0.40533293j, -0.20215326 + 0.30850569j],
                [-0.23889780 - 0.28101519j, -0.88031770 - 0.29832709j],
            ]
        )
        # test applied to wire 0
        res = qml.operation.expand_matrix(U, [0], [0, 4, 9])
        expected = np.kron(np.kron(U, I), I)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # test applied to wire 4
        res = qml.operation.expand_matrix(U, [4], [0, 4, 9])
        expected = np.kron(np.kron(I, U), I)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # test applied to wire 9
        res = qml.operation.expand_matrix(U, [9], [0, 4, 9])
        expected = np.kron(np.kron(I, I), U)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_expand_two_consecutive_wires(self, tol):
        """Test that a 2 qubit gate on consecutive wires correctly
        expands to 4 qubits."""
        U2 = np.array([[0, 1, 1, 1], [1, 0, 1, -1], [1, -1, 0, 1], [1, 1, -1, 0]]) / np.sqrt(3)

        # test applied to wire 0+1
        res = qml.operation.expand_matrix(U2, [0, 1], [0, 1, 2, 3])
        expected = np.kron(np.kron(U2, I), I)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # test applied to wire 1+2
        res = qml.operation.expand_matrix(U2, [1, 2], [0, 1, 2, 3])
        expected = np.kron(np.kron(I, U2), I)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # test applied to wire 2+3
        res = qml.operation.expand_matrix(U2, [2, 3], [0, 1, 2, 3])
        expected = np.kron(np.kron(I, I), U2)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_expand_two_reversed_wires(self, tol):
        """Test that a 2 qubit gate on reversed consecutive wires correctly
        expands to 4 qubits."""
        # CNOT with target on wire 1
        res = qml.operation.expand_matrix(CNOT, [1, 0], [0, 1, 2, 3])
        rows = np.array([0, 2, 1, 3])
        expected = np.kron(np.kron(CNOT[:, rows][rows], I), I)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_expand_three_consecutive_wires(self, tol):
        """Test that a 3 qubit gate on consecutive
        wires correctly expands to 4 qubits."""
        U_toffoli = np.diag([1 for i in range(8)])
        U_toffoli[6:8, 6:8] = np.array([[0, 1], [1, 0]])
        # test applied to wire 0,1,2
        res = qml.operation.expand_matrix(U_toffoli, [0, 1, 2], [0, 1, 2, 3])
        expected = np.kron(U_toffoli, I)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # test applied to wire 1,2,3
        res = qml.operation.expand_matrix(U_toffoli, [1, 2, 3], [0, 1, 2, 3])
        expected = np.kron(I, U_toffoli)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_expand_three_nonconsecutive_ascending_wires(self, tol):
        """Test that a 3 qubit gate on non-consecutive but ascending
        wires correctly expands to 4 qubits."""
        U_toffoli = np.diag([1 for i in range(8)])
        U_toffoli[6:8, 6:8] = np.array([[0, 1], [1, 0]])
        # test applied to wire 0,2,3
        res = qml.operation.expand_matrix(U_toffoli, [0, 2, 3], [0, 1, 2, 3])
        expected = (
            np.kron(qml.SWAP.compute_matrix(), np.kron(I, I))
            @ np.kron(I, U_toffoli)
            @ np.kron(qml.SWAP.compute_matrix(), np.kron(I, I))
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # test applied to wire 0,1,3
        res = qml.operation.expand_matrix(U_toffoli, [0, 1, 3], [0, 1, 2, 3])
        expected = (
            np.kron(np.kron(I, I), qml.SWAP.compute_matrix())
            @ np.kron(U_toffoli, I)
            @ np.kron(np.kron(I, I), qml.SWAP.compute_matrix())
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_expand_three_nonconsecutive_nonascending_wires(self, tol):
        """Test that a 3 qubit gate on non-consecutive non-ascending
        wires correctly expands to 4 qubits"""
        U_toffoli = np.diag([1 for i in range(8)])
        U_toffoli[6:8, 6:8] = np.array([[0, 1], [1, 0]])
        # test applied to wire 3, 1, 2
        res = qml.operation.expand_matrix(U_toffoli, [3, 1, 2], [0, 1, 2, 3])
        # change the control qubit on the Toffoli gate
        rows = np.array([0, 4, 1, 5, 2, 6, 3, 7])
        expected = np.kron(I, U_toffoli[:, rows][rows])
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # test applied to wire 3, 0, 2
        res = qml.operation.expand_matrix(U_toffoli, [3, 0, 2], [0, 1, 2, 3])
        # change the control qubit on the Toffoli gate
        rows = np.array([0, 4, 1, 5, 2, 6, 3, 7])
        expected = (
            np.kron(qml.SWAP.compute_matrix(), np.kron(I, I))
            @ np.kron(I, U_toffoli[:, rows][rows])
            @ np.kron(qml.SWAP.compute_matrix(), np.kron(I, I))
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_expand_matrix_usage_in_operator_class(self, tol):
        """Tests that the method is used correctly by defining a dummy operator and
        checking the permutation/expansion."""

        base_matrix = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])

        permuted_matrix = np.array([[1, 3, 2, 4], [9, 11, 10, 12], [5, 7, 6, 8], [13, 15, 14, 16]])

        expanded_matrix = np.array(
            [
                [1, 2, 0, 0, 3, 4, 0, 0],
                [5, 6, 0, 0, 7, 8, 0, 0],
                [0, 0, 1, 2, 0, 0, 3, 4],
                [0, 0, 5, 6, 0, 0, 7, 8],
                [9, 10, 0, 0, 11, 12, 0, 0],
                [13, 14, 0, 0, 15, 16, 0, 0],
                [0, 0, 9, 10, 0, 0, 11, 12],
                [0, 0, 13, 14, 0, 0, 15, 16],
            ]
        )

        class DummyOp(qml.operation.Operator):
            num_wires = 2

            def compute_matrix(*params, **hyperparams):
                return base_matrix

        op = DummyOp(wires=[0, 2])
        assert np.allclose(op.get_matrix(), base_matrix, atol=tol)
        assert np.allclose(op.get_matrix(wire_order=[2, 0]), permuted_matrix, atol=tol)
        assert np.allclose(op.get_matrix(wire_order=[0, 1, 2]), expanded_matrix, atol=tol)


def test_docstring_example_of_operator_class(tol):
    """Tests an example of how to create an operator which is used in the
    Operator class docstring, as well as in the 'adding_operators'
    page in the developer guide."""

    import pennylane as qml

    class FlipAndRotate(qml.operation.Operation):

        num_wires = qml.operation.AnyWires
        grad_method = "A"

        def __init__(self, angle, wire_rot, wire_flip=None, do_flip=False, do_queue=True, id=None):

            if do_flip and wire_flip is None:
                raise ValueError("Expected a wire to flip; got None.")

            shape = qml.math.shape(angle)
            if len(shape) > 1:
                raise ValueError(f"Expected a scalar angle; got angle of shape {shape}.")

            self._hyperparameters = {"do_flip": do_flip}

            all_wires = qml.wires.Wires(wire_rot) + qml.wires.Wires(wire_flip)
            super().__init__(angle, wires=all_wires, do_queue=do_queue, id=id)

        @property
        def num_params(self):
            return 1

        @staticmethod
        def compute_decomposition(angle, wires, do_flip):  # pylint: disable=arguments-differ
            op_list = []
            if do_flip:
                op_list.append(qml.PauliX(wires=wires[1]))
            op_list.append(qml.RX(angle, wires=wires[0]))
            return op_list

        def adjoint(self):
            return FlipAndRotate(
                -self.parameters[0],
                self.wires[0],
                self.wires[1],
                do_flip=self.hyperparameters["do_flip"],
            )

    dev = qml.device("default.qubit", wires=["q1", "q2", "q3"])

    @qml.qnode(dev)
    def circuit(angle):
        FlipAndRotate(angle, wire_rot="q1", wire_flip="q1")
        return qml.expval(qml.PauliZ("q1"))

    a = np.array(3.14)
    res = circuit(a)
    expected = -0.9999987318946099
    assert np.allclose(res, expected, atol=tol)


class TestDeprecationWarnings:
    """Tests for various deprecation warnings"""

    def test_matrix_deprecation(self):
        """Test that the Operator.matrix property raises a deprecation warning"""
        op = qml.RX(0.5, wires=0).inv()

        with pytest.warns(UserWarning, match="The 'matrix' property is deprecated"):
            m1 = op.matrix

        assert np.allclose(m1, op.get_matrix())
        assert np.allclose(m1, qml.RX.compute_matrix(-0.5))

    def test_matrix_tensor_deprecation(self):
        """Test that the Tensor.matrix property raises a deprecation warning"""
        op = qml.PauliZ(0) @ qml.PauliY(1)

        with pytest.warns(UserWarning, match="The 'matrix' property is deprecated"):
            m1 = op.matrix

        assert np.allclose(m1, op.get_matrix())

    def test_eigvals_deprecation(self):
        """Test that the Operator.eigvals property raises a deprecation warning"""
        op = qml.RX(0.5, wires=0)

        with pytest.warns(UserWarning, match="The 'eigvals' property is deprecated"):
            m1 = op.eigvals

        assert np.allclose(m1, op.get_eigvals())

    def test_eigvals_observable_deprecation(self):
        """Test that the Observable.eigvals property raises a deprecation warning"""
        op = qml.PauliX(0)

        with pytest.warns(UserWarning, match="The 'eigvals' property is deprecated"):
            m1 = op.eigvals

        assert np.allclose(m1, op.get_eigvals())

    def test_eigvals_measurement_deprecation(self):
        """Test that the MeasurementProcess.eigvals property raises a deprecation warning"""
        op = qml.expval(qml.PauliX(0))

        with pytest.warns(UserWarning, match="The 'eigvals' property is deprecated"):
            m1 = op.eigvals

        assert np.allclose(m1, op.get_eigvals())

    def test_eigvals_tensor_deprecation(self):
        """Test that the Tensor.matrix property raises a deprecation warning"""
        op = qml.PauliZ(0) @ qml.PauliY(1)

        with pytest.warns(UserWarning, match="The 'eigvals' property is deprecated"):
            m1 = op.eigvals

        assert np.allclose(m1, op.get_eigvals())

    def test_decomposition_deprecation_no_parameters(self):
        """Test that old-style staticmethod decompositions for an operation
        with no parameters raises a warning"""
        dev = qml.device("default.qubit", wires=1)

        class MyOp(Operation):
            num_wires = 1
            num_params = 0

            @staticmethod
            def decomposition(wires):
                qml.RY(0.5, wires=wires[0])

        @qml.qnode(dev)
        def qnode():
            MyOp(wires=0)
            return qml.state()

        with pytest.warns(UserWarning, match="is now an instance method"):
            result1 = qnode()

        # using an instance method will not raise a deprecation warning

        class MyOp(Operation):
            num_wires = 1
            num_params = 0

            def decomposition(self):
                qml.RY(0.5, wires=self.wires[0])

        @qml.qnode(dev)
        def qnode():
            MyOp(wires=0)
            return qml.state()

        with warnings.catch_warnings():
            # any warnings emitted will be raised as errors
            warnings.simplefilter("error")
            result2 = qnode()

        assert np.allclose(result1, result2)

    def test_decomposition_deprecation_parameters(self):
        """Test that old-style staticmethod decompositions for an operation
        with parameters raises a warning"""
        dev = qml.device("default.qubit", wires=1)

        class MyOp(Operation):
            num_wires = 1
            num_params = 2

            @staticmethod
            def decomposition(*params, wires):
                qml.RY(params[0], wires=wires[0])
                qml.PauliZ(wires=wires[0])
                qml.RX(params[1], wires=wires[0])

        @qml.qnode(dev)
        def qnode(*params):
            MyOp(*params, wires=0)
            return qml.state()

        with pytest.warns(UserWarning, match="is now an instance method"):
            result1 = qnode(0.1, 0.2)

        # using an instance method will not raise a deprecation warning

        class MyOp(Operation):
            num_wires = 1
            num_params = 2

            def decomposition(self):
                qml.RY(self.parameters[0], wires=self.wires[0])
                qml.PauliZ(wires=self.wires[0])
                qml.RX(self.parameters[1], wires=self.wires[0])

        @qml.qnode(dev)
        def qnode(*params):
            MyOp(*params, wires=0)
            return qml.state()

        with warnings.catch_warnings():
            # any warnings emitted will be raised as errors
            warnings.simplefilter("error")
            result2 = qnode(0.1, 0.2)

        assert np.allclose(result1, result2)

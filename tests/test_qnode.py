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
Unit tests for the :mod:`pennylane` :class:`QNode` class.
"""
import math
from unittest.mock import Mock, PropertyMock, patch

import pytest
from autograd import numpy as np

import pennylane as qml
from pennylane._device import Device
from pennylane.qnode import QNode, QuantumFunctionError, _flatten, unflatten

flat_dummy_array = np.linspace(-1, 1, 64)
test_shapes = [
    (64,),
    (64, 1),
    (32, 2),
    (16, 4),
    (8, 8),
    (16, 2, 2),
    (8, 2, 2, 2),
    (4, 2, 2, 2, 2),
    (2, 2, 2, 2, 2, 2),
]


class TestHelperMethods:
    """Tests the internal helper methods of QNode"""

    @pytest.mark.parametrize("shape", test_shapes)
    def test_flatten(self, shape):
        """Tests that _flatten successfully flattens multidimensional arrays."""

        reshaped = np.reshape(flat_dummy_array, shape)
        flattened = np.array([x for x in _flatten(reshaped)])

        assert flattened.shape == flat_dummy_array.shape
        assert np.array_equal(flattened, flat_dummy_array)

    @pytest.mark.parametrize("shape", test_shapes)
    def test_unflatten(self, shape):
        """Tests that _unflatten successfully unflattens multidimensional arrays."""

        reshaped = np.reshape(flat_dummy_array, shape)
        unflattened = np.array([x for x in unflatten(flat_dummy_array, reshaped)])

        assert unflattened.shape == reshaped.shape
        assert np.array_equal(unflattened, reshaped)

    def test_unflatten_error_unsupported_model(self):
        """Tests that unflatten raises an error if the given model is not supported"""

        with pytest.raises(TypeError, match="Unsupported type in the model"):
            model = lambda x: x  # not a valid model for unflatten
            unflatten(flat_dummy_array, model)

    def test_unflatten_error_too_many_elements(self):
        """Tests that unflatten raises an error if the given iterable has
           more elements than the model"""

        reshaped = np.reshape(flat_dummy_array, (16, 2, 2))

        with pytest.raises(ValueError, match="Flattened iterable has more elements than the model"):
            unflatten(np.concatenate([flat_dummy_array, flat_dummy_array]), reshaped)


class TestQNodeOperationQueue:
    """Tests that the QNode operation queue is properly filled and interacted with"""

    @pytest.fixture(scope="function")
    def opqueue_test_node(self, mock_device):
        """Provides a circuit for the subsequent tests of the operation queue"""

        def circuit(x):
            qml.RX(x, wires=[0])
            qml.CNOT(wires=[0, 1])
            qml.RY(0.4, wires=[0])
            qml.RZ(-0.2, wires=[1])
            return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliZ(1))

        node = qml.QNode(circuit, mock_device)
        node.construct([1.0])

        return node

    def test_operation_ordering(self, opqueue_test_node):
        """Tests that the ordering of the operations is correct"""

        assert opqueue_test_node.ops[0].name == "RX"
        assert opqueue_test_node.ops[1].name == "CNOT"
        assert opqueue_test_node.ops[2].name == "RY"
        assert opqueue_test_node.ops[3].name == "RZ"
        assert opqueue_test_node.ops[4].name == "PauliX"
        assert opqueue_test_node.ops[5].name == "PauliZ"

    def test_op_successors_operations_only(self, opqueue_test_node):
        """Tests that _op_successors properly extracts the successors that are operations"""

        operation_successors = opqueue_test_node._op_successors(0, only="G")

        assert opqueue_test_node.ops[0] not in operation_successors
        assert opqueue_test_node.ops[1] in operation_successors
        assert opqueue_test_node.ops[4] not in operation_successors

    def test_op_successors_observables_only(self, opqueue_test_node):
        """Tests that _op_successors properly extracts the successors that are observables"""

        observable_successors = opqueue_test_node._op_successors(0, only="E")

        assert opqueue_test_node.ops[0] not in observable_successors
        assert opqueue_test_node.ops[1] not in observable_successors
        assert opqueue_test_node.ops[4] in observable_successors

    def test_op_successors_both_operations_and_observables(self, opqueue_test_node):
        """Tests that _op_successors properly extracts all successors"""

        successors = opqueue_test_node._op_successors(0, only=None)

        assert opqueue_test_node.ops[0] not in successors
        assert opqueue_test_node.ops[1] in successors
        assert opqueue_test_node.ops[4] in successors

    # TODO
    # once _op_successors has been upgraded to return only strict successors using a DAG
    # add a test that checks that the strict ordering is used
    # successors = q._op_successors(2, only=None)
    # assert q.ops[4] in successors
    # assert q.ops[5] not in successors


@pytest.fixture(scope="function")
def operable_mock_device_2_wires():
    """A mock instance of the abstract Device class that can support
       qfuncs."""

    with patch.multiple(
        Device,
        __abstractmethods__=set(),
        operations=PropertyMock(return_value=["RX", "RY", "CNOT"]),
        observables=PropertyMock(return_value=["PauliX", "PauliY", "PauliZ"]),
        reset=Mock(),
        apply=Mock(),
        expval=Mock(return_value=1),
    ):
        yield Device(wires=2)


class TestQNodeExceptions:
    """Tests that QNode raises proper errors"""

    def test_current_context_modified_outside_construct(self, mock_device, monkeypatch):
        """Tests that the QNode properly raises an error if the _current_context
           was modified outside of construct"""

        def circuit(x):
            qml.RX(x, wires=[0])
            return qml.expval(qml.PauliZ(wires=0))

        node = qml.QNode(circuit, mock_device)

        monkeypatch.setattr(QNode, "_current_context", node)

        with pytest.raises(
            QuantumFunctionError,
            match="QNode._current_context must not be modified outside this method.",
        ):
            node.construct([0.0])

    def test_return_of_non_observable(self, operable_mock_device_2_wires):
        """Tests that the QNode properly raises an error if the qfunc returns something
           besides observables."""

        def circuit(x):
            qml.RX(x, wires=[0])
            return qml.expval(qml.PauliZ(wires=0)), 0.3

        node = qml.QNode(circuit, operable_mock_device_2_wires)

        with pytest.raises(QuantumFunctionError, match="must return either"):
            node(0.5)

    def test_observable_not_returned(self, operable_mock_device_2_wires):
        """Tests that the QNode properly raises an error if the qfunc does not
           return all observables."""

        def circuit(x):
            qml.RX(x, wires=[0])
            ex = qml.expval(qml.PauliZ(wires=1))
            return qml.expval(qml.PauliZ(wires=0))

        node = qml.QNode(circuit, operable_mock_device_2_wires)

        with pytest.raises(QuantumFunctionError, match="All measured observables"):
            node(0.5)

    def test_observable_order_violated(self, operable_mock_device_2_wires):
        """Tests that the QNode properly raises an error if the qfunc does not
           return all observables in the correct order."""

        def circuit(x):
            qml.RX(x, wires=[0])
            ex = qml.expval(qml.PauliZ(wires=1))
            return qml.expval(qml.PauliZ(wires=0)), ex

        node = qml.QNode(circuit, operable_mock_device_2_wires)

        with pytest.raises(QuantumFunctionError, match="All measured observables"):
            node(0.5)

    def test_operations_after_observables(self, operable_mock_device_2_wires):
        """Tests that the QNode properly raises an error if the qfunc contains
           operations after observables."""

        def circuit(x):
            qml.RX(x, wires=[0])
            ex = qml.expval(qml.PauliZ(wires=1))
            qml.RY(0.5, wires=[0])
            return qml.expval(qml.PauliZ(wires=0))

        node = qml.QNode(circuit, operable_mock_device_2_wires)

        with pytest.raises(QuantumFunctionError, match="gates must precede"):
            node(0.5)

    def test_multiple_measurements_on_same_wire(self, operable_mock_device_2_wires):
        """Tests that the QNode properly raises an error if the same wire
           is measured multiple times."""

        def circuit(x):
            qml.RX(x, wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliX(0))

        node = qml.QNode(circuit, operable_mock_device_2_wires)

        with pytest.raises(QuantumFunctionError, match="can only be measured once"):
            node(0.5)

    def test_operation_on_nonexistant_wire(self, operable_mock_device_2_wires):
        """Tests that the QNode properly raises an error if an operation
           is applied to a non-existant wire."""

        operable_mock_device_2_wires.num_wires = 2

        def circuit(x):
            qml.RX(x, wires=[0])
            qml.CNOT(wires=[0, 2])
            return qml.expval(qml.PauliZ(0))

        node = qml.QNode(circuit, operable_mock_device_2_wires)

        with pytest.raises(QuantumFunctionError, match="applied to invalid wire"):
            node(0.5)

    def test_observable_on_nonexistant_wire(self, operable_mock_device_2_wires):
        """Tests that the QNode properly raises an error if an observable
           is measured on a non-existant wire."""

        operable_mock_device_2_wires.num_wires = 2

        def circuit(x):
            qml.RX(x, wires=[0])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(2))

        node = qml.QNode(circuit, operable_mock_device_2_wires)

        with pytest.raises(QuantumFunctionError, match="applied to invalid wire"):
            node(0.5)

    def test_mixing_of_cv_and_qubit_operations(self, operable_mock_device_2_wires):
        """Tests that the QNode properly raises an error if qubit and
           CV operations are mixed in the same qfunc."""

        def circuit(x):
            qml.RX(x, wires=[0])
            qml.Displacement(0.5, 0, wires=[0])
            return qml.expval(qml.PauliZ(0))

        node = qml.QNode(circuit, operable_mock_device_2_wires)

        with pytest.raises(QuantumFunctionError, match="Continuous and discrete"):
            node(0.5)


class TestQNodeJacobianExceptions:
    """Tests that QNode.jacobian raises proper errors"""

    def test_undifferentiable_operation(self, operable_mock_device_2_wires):
        """Tests that QNode.jacobian properly raises an error if the
           qfunc contains an operation that is not differentiable."""

        def circuit(x):
            qml.BasisState(np.array([x, 0]), wires=[0, 1])
            qml.RX(x, wires=[0])
            return qml.expval(qml.PauliZ(0))

        node = qml.QNode(circuit, operable_mock_device_2_wires)

        with pytest.raises(ValueError, match="Cannot differentiate wrt parameter"):
            node.jacobian(0.5)

    def test_operation_not_supporting_analytic_gradient(self, operable_mock_device_2_wires):
        """Tests that QNode.jacobian properly raises an error if the
           qfunc contains an operation that does not support analytic gradients."""

        def circuit(x):
            qml.RX(x, wires=[0])
            return qml.expval(qml.Hermitian(np.diag([x, 0]), 0))

        node = qml.QNode(circuit, operable_mock_device_2_wires)

        with pytest.raises(ValueError, match="analytic gradient method cannot be used with"):
            node.jacobian(0.5, method="A")

    def test_bogus_gradient_method_set(self, operable_mock_device_2_wires):
        """Tests that QNode.jacobian properly raises an error if the
           gradient method set is bogus."""

        def circuit(x):
            qml.RX(x, wires=[0])
            return qml.expval(qml.PauliZ(0))

        # in non-cached mode, the grad method would be
        # recomputed and overwritten from the
        # bogus value 'J'. Caching stops this from happening.
        node = qml.QNode(circuit, operable_mock_device_2_wires, cache=True)

        node.evaluate([0.0])
        keys = node.grad_method_for_par.keys()
        if keys:
            k0 = [k for k in keys][0]

        node.grad_method_for_par[k0] = "J"

        with pytest.raises(ValueError, match="Unknown gradient method"):
            node.jacobian(0.5)

    def test_indices_not_unique(self, operable_mock_device_2_wires):
        """Tests that QNode.jacobian properly raises an error if the
           jacobian is requested for non-unique indices."""

        def circuit(x):
            qml.Rot(0.3, x, -0.2, wires=[0])
            return qml.expval(qml.PauliZ(0))

        node = qml.QNode(circuit, operable_mock_device_2_wires)

        with pytest.raises(ValueError, match="Parameter indices must be unique."):
            node.jacobian(0.5, which=[0, 0])

    def test_indices_nonexistant(self, operable_mock_device_2_wires):
        """Tests that QNode.jacobian properly raises an error if the
           jacobian is requested for non-existant parameters."""

        def circuit(x):
            qml.Rot(0.3, x, -0.2, wires=[0])
            return qml.expval(qml.PauliZ(0))

        node = qml.QNode(circuit, operable_mock_device_2_wires)

        with pytest.raises(ValueError, match="Tried to compute the gradient wrt"):
            node.jacobian(0.5, which=[0, 6])

        with pytest.raises(ValueError, match="Tried to compute the gradient wrt"):
            node.jacobian(0.5, which=[1, -1])

    def test_unknown_method(self, operable_mock_device_2_wires):
        """Tests that QNode.jacobian properly raises an error if the
           gradient method is unknown."""

        def circuit(x):
            qml.Rot(0.3, x, -0.2, wires=[0])
            return qml.expval(qml.PauliZ(0))

        node = qml.QNode(circuit, operable_mock_device_2_wires)

        with pytest.raises(ValueError, match="Unknown gradient method"):
            node.jacobian(0.5, method="unknown")

    def test_wrong_order_in_finite_difference(self, operable_mock_device_2_wires):
        """Tests that QNode.jacobian properly raises an error if finite
           differences are attempted with wrong order."""

        def circuit(x):
            qml.Rot(0.3, x, -0.2, wires=[0])
            return qml.expval(qml.PauliZ(0))

        node = qml.QNode(circuit, operable_mock_device_2_wires)

        with pytest.raises(ValueError, match="Order must be 1 or 2"):
            node.jacobian(0.5, method="F", order=3)


class TestQNodeParameters:
    """Tests the handling of parameters in the QNode"""

    @pytest.mark.parametrize(
        "x,y",
        zip(np.linspace(-2 * np.pi, 2 * np.pi, 7), np.linspace(-2 * np.pi, 2 * np.pi, 7) ** 2 / 11),
    )
    def test_fanout(self, qubit_device_1_wire, tol, x, y):
        """Tests that qnodes can compute the correct function when the
           same parameter is used in multiple gates."""

        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RZ(y, wires=[0])
            qml.RX(x, wires=[0])
            return qml.expval(qml.PauliZ(0))

        def analytic_expval(x, y):
            return math.cos(x) ** 2 - math.cos(y) * math.sin(x) ** 2

        node = qml.QNode(circuit, qubit_device_1_wire)

        assert np.isclose(node(x, y), analytic_expval(x, y), atol=tol, rtol=0)

    def test_array_parameters_scalar_return(self, qubit_device_1_wire, tol):
        """Test that QNode can take arrays as input arguments, and that they interact properly with Autograd.
           Test case for a circuit that returns a scalar."""

        def circuit(dummy1, array, dummy2):
            qml.RY(0.5 * array[0, 1], wires=0)
            qml.RY(-0.5 * array[1, 1], wires=0)
            return qml.expval(qml.PauliX(0))

        node = qml.QNode(circuit, qubit_device_1_wire)

        args = (0.46, np.array([[2.0, 3.0, 0.3], [7.0, 4.0, 2.1]]), -0.13)
        grad_target = (
            np.array(1.0),
            np.array([[0.5, 0.43879, 0], [0, -0.43879, 0]]),
            np.array(-0.4),
        )
        cost_target = 1.03257

        def cost(x, array, y):
            c = node(0.111, array, 4.5)
            return c + 0.5 * array[0, 0] + x - 0.4 * y

        cost_grad = qml.grad(cost, argnum=[0, 1, 2])
        computed_grad = cost_grad(*args)

        assert np.isclose(cost(*args), cost_target, atol=tol, rtol=0)

        assert np.allclose(computed_grad[0], grad_target[0], atol=tol, rtol=0)
        assert np.allclose(computed_grad[1], grad_target[1], atol=tol, rtol=0)
        assert np.allclose(computed_grad[2], grad_target[2], atol=tol, rtol=0)

    def test_qnode_array_parameters_1_vector_return(self, qubit_device_1_wire, tol):
        """Test that QNode can take arrays as input arguments, and that they interact properly with Autograd.
           Test case for a circuit that returns a 1-vector."""

        def circuit(dummy1, array, dummy2):
            qml.RY(0.5 * array[0, 1], wires=0)
            qml.RY(-0.5 * array[1, 1], wires=0)
            return (qml.expval(qml.PauliX(0)),)

        node = qml.QNode(circuit, qubit_device_1_wire)

        args = (0.46, np.array([[2.0, 3.0, 0.3], [7.0, 4.0, 2.1]]), -0.13)
        grad_target = (
            np.array(1.0),
            np.array([[0.5, 0.43879, 0], [0, -0.43879, 0]]),
            np.array(-0.4),
        )
        cost_target = 1.03257

        def cost(x, array, y):
            c = node(0.111, array, 4.5)[0]
            return c + 0.5 * array[0, 0] + x - 0.4 * y

        cost_grad = qml.grad(cost, argnum=[0, 1, 2])
        computed_grad = cost_grad(*args)

        assert np.isclose(cost(*args), cost_target, atol=tol, rtol=0)

        assert np.allclose(computed_grad[0], grad_target[0], atol=tol, rtol=0)
        assert np.allclose(computed_grad[1], grad_target[1], atol=tol, rtol=0)
        assert np.allclose(computed_grad[2], grad_target[2], atol=tol, rtol=0)

    def test_qnode_array_parameters_2_vector_return(self, qubit_device_2_wires, tol):
        """Test that QNode can take arrays as input arguments, and that they interact properly with Autograd.
           Test case for a circuit that returns a 2-vector."""

        def circuit(dummy1, array, dummy2):
            qml.RY(0.5 * array[0, 1], wires=0)
            qml.RY(-0.5 * array[1, 1], wires=0)
            qml.RY(array[1, 0], wires=1)
            return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliX(1))

        node = qml.QNode(circuit, qubit_device_2_wires)

        args = (0.46, np.array([[2.0, 3.0, 0.3], [7.0, 4.0, 2.1]]), -0.13)
        grad_target = (
            np.array(1.0),
            np.array([[0.5, 0.43879, 0], [0, -0.43879, 0]]),
            np.array(-0.4),
        )
        cost_target = 1.03257

        def cost(x, array, y):
            c = node(0.111, array, 4.5)[0]
            return c + 0.5 * array[0, 0] + x - 0.4 * y

        cost_grad = qml.grad(cost, argnum=[0, 1, 2])
        computed_grad = cost_grad(*args)

        assert np.isclose(cost(*args), cost_target, atol=tol, rtol=0)

        assert np.allclose(computed_grad[0], grad_target[0], atol=tol, rtol=0)
        assert np.allclose(computed_grad[1], grad_target[1], atol=tol, rtol=0)
        assert np.allclose(computed_grad[2], grad_target[2], atol=tol, rtol=0)

    def test_array_parameters_evaluate(self, qubit_device_2_wires, tol):
        """Tests that array parameters gives same result as positional arguments."""
        a, b, c = 0.5, 0.54, 0.3

        def ansatz(x, y, z):
            qml.QubitStateVector(np.array([1, 0, 1, 1]) / np.sqrt(3), wires=[0, 1])
            qml.Rot(x, y, z, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))

        @qml.qnode(qubit_device_2_wires)
        def circuit1(x, y, z):
            return ansatz(x, y, z)

        @qml.qnode(qubit_device_2_wires)
        def circuit2(x, array):
            return ansatz(x, array[0], array[1])

        @qml.qnode(qubit_device_2_wires)
        def circuit3(array):
            return ansatz(*array)

        positional_res = circuit1(a, b, c)
        positional_grad = circuit1.jacobian([a, b, c])

        array_res = circuit2(a, np.array([b, c]))
        array_grad = circuit2.jacobian([a, np.array([b, c])])

        assert np.allclose(positional_res, array_res, atol=tol, rtol=0)
        assert np.allclose(positional_grad, array_grad, atol=tol, rtol=0)

        list_res = circuit2(a, [b, c])
        list_grad = circuit2.jacobian([a, [b, c]])

        assert np.allclose(positional_res, list_res, atol=tol, rtol=0)
        assert np.allclose(positional_grad, list_grad, atol=tol, rtol=0)

        array_res = circuit3(np.array([a, b, c]))
        array_grad = circuit3.jacobian([np.array([a, b, c])])

        list_res = circuit3([a, b, c])
        list_grad = circuit3.jacobian([[a, b, c]])

        assert np.allclose(positional_res, array_res, atol=tol, rtol=0)
        assert np.allclose(positional_grad, array_grad, atol=tol, rtol=0)

    def test_multiple_expectation_different_wires(self, qubit_device_2_wires, tol):
        """Tests that qnodes return multiple expectation values."""

        a, b, c = 0.5, 0.54, 0.3

        @qml.qnode(qubit_device_2_wires)
        def circuit(x, y, z):
            qml.RX(x, wires=[0])
            qml.RZ(y, wires=[0])
            qml.CNOT(wires=[0, 1])
            qml.RY(y, wires=[0])
            qml.RX(z, wires=[0])
            return qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(1))

        def analytic_expval(a, b, c):
            return [-1 * math.cos(a) * math.cos(b) * math.sin(c), math.cos(a)]

        res = circuit(a, b, c)
        analytic_res = analytic_expval(a, b, c)

        assert np.allclose(res, analytic_res, atol=tol, rtol=0)


class TestQNodeKeywordArguments:
    """Tests that the qnode properly handles keyword arguments."""

    def test_multiple_keywordargs_used(self, qubit_device_2_wires, tol):
        """Tests that qnodes use multiple keyword arguments."""

        def circuit(w, x=None, y=None):
            qml.RX(x, wires=[0])
            qml.RX(y, wires=[1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        node = qml.QNode(circuit, qubit_device_2_wires)

        c = node(1.0, x=np.pi, y=np.pi)

        assert np.allclose(c, [-1.0, -1.0], atol=tol, rtol=0)

    def test_multidimensional_keywordargs_used(self, qubit_device_2_wires, tol):
        """Tests that qnodes use multi-dimensional keyword arguments."""

        def circuit(w, x=None):
            qml.RX(x[0], wires=[0])
            qml.RX(x[1], wires=[1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        node = qml.QNode(circuit, qubit_device_2_wires)

        c = node(1.0, x=[np.pi, np.pi])

        assert np.allclose(c, [-1.0, -1.0], atol=tol, rtol=0)

    def test_keywordargs_for_wires(self, qubit_device_2_wires, tol):
        """Tests that wires can be passed as keyword arguments."""

        default_q = 0

        def circuit(x, q=default_q):
            qml.RX(x, wires=[q])
            return qml.expval(qml.PauliZ(q))

        node = qml.QNode(circuit, qubit_device_2_wires)

        c = node(np.pi, q=1)

        assert node.queue[0].wires == [1]
        assert np.isclose(c, -1.0, atol=tol, rtol=0)

        c = node(np.pi)

        assert node.queue[0].wires == [default_q]
        assert np.isclose(c, -1.0, atol=tol, rtol=0)

    def test_keywordargs_used(self, qubit_device_1_wire, tol):
        """Tests that qnodes use keyword arguments."""

        def circuit(w, x=None):
            qml.RX(x, wires=[0])
            return qml.expval(qml.PauliZ(0))

        node = qml.QNode(circuit, qubit_device_1_wire)

        c = node(1.0, x=np.pi)

        assert np.isclose(c, -1.0, atol=tol, rtol=0)

    def test_keywordarg_updated_in_multiple_calls(self, qubit_device_2_wires, tol):
        """Tests that qnodes update keyword arguments in consecutive calls."""

        def circuit(w, x=None):
            qml.RX(w, wires=[0])
            qml.RX(x, wires=[1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        node = qml.QNode(circuit, qubit_device_2_wires)

        c1 = node(0.1, x=0.0)
        c2 = node(0.1, x=np.pi)

        assert c1[1] != c2[1]

    def test_keywordarg_passes_through_classicalnode(self, qubit_device_2_wires, tol):
        """Tests that qnodes' keyword arguments pass through classical nodes."""

        def circuit(w, x=None):
            qml.RX(w, wires=[0])
            qml.RX(x, wires=[1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        node = qml.QNode(circuit, qubit_device_2_wires)

        def classical_node(w, x=None):
            return node(w, x=x)

        c = classical_node(0.0, x=np.pi)

        assert np.allclose(c, [1.0, -1.0], atol=tol, rtol=0)


class TestQNodeGradients:
    """Qnode gradient tests."""

    @pytest.mark.parametrize("shape", [(8,), (8, 1), (4, 2), (2, 2, 2), (2, 1, 2, 1, 2)])
    def test_multidim_array(self, shape, tol):
        """Tests that arguments which are multidimensional arrays are
        properly evaluated and differentiated in QNodes."""

        base_array = np.linspace(-1.0, 1.0, 8)
        multidim_array = np.reshape(base_array, shape)

        def circuit(w):
            qml.RX(w[np.unravel_index(0, shape)], wires=0)  # base_array[0]
            qml.RX(w[np.unravel_index(1, shape)], wires=1)  # base_array[1]
            qml.RX(w[np.unravel_index(2, shape)], wires=2)  # ...
            qml.RX(w[np.unravel_index(3, shape)], wires=3)
            qml.RX(w[np.unravel_index(4, shape)], wires=4)
            qml.RX(w[np.unravel_index(5, shape)], wires=5)
            qml.RX(w[np.unravel_index(6, shape)], wires=6)
            qml.RX(w[np.unravel_index(7, shape)], wires=7)
            return tuple(qml.expval(qml.PauliZ(idx)) for idx in range(len(base_array)))

        dev = qml.device("default.qubit", wires=8)
        circuit = qml.QNode(circuit, dev)

        # circuit evaluations
        circuit_output = circuit(multidim_array)
        expected_output = np.cos(base_array)
        assert np.allclose(circuit_output, expected_output, atol=tol, rtol=0)

        # circuit jacobians
        circuit_jacobian = circuit.jacobian([multidim_array])
        expected_jacobian = -np.diag(np.sin(base_array))
        assert np.allclose(circuit_jacobian, expected_jacobian, atol=tol, rtol=0)

    def test_qnode_cv_gradient_methods(self):
        """Tests the gradient computation methods on CV circuits."""
        # we can only use the 'A' method on parameters which only affect gaussian operations
        # that are not succeeded by nongaussian operations

        par = [0.4, -2.3]
        dev = qml.device("default.qubit", wires=2)

        def check_methods(qf, d):
            q = qml.QNode(qf, dev)
            # NOTE: the default plugin is a discrete (qubit) simulator, it cannot
            # execute CV gates, but the QNode can be constructed
            q.construct(par)
            assert q.grad_method_for_par == d

        def qf(x, y):
            qml.Displacement(x, 0, wires=[0])
            qml.CubicPhase(0.2, wires=[0])
            qml.Squeezing(0.3, y, wires=[1])
            qml.Rotation(1.3, wires=[1])
            # nongaussian succeeding x but not y
            # TODO when QNode uses a DAG to describe the circuit, uncomment this line
            # qml.Kerr(0.4, [0])
            return qml.expval(qml.X(0)), qml.expval(qml.X(1))

        check_methods(qf, {0: "F", 1: "A"})

        def qf(x, y):
            qml.Displacement(x, 0, wires=[0])
            qml.CubicPhase(0.2, wires=[0])  # nongaussian succeeding x
            qml.Squeezing(0.3, x, wires=[1])  # x affects gates on both wires, y unused
            qml.Rotation(1.3, wires=[1])
            return qml.expval(qml.X(0)), qml.expval(qml.X(1))

        check_methods(qf, {0: "F"})

        def qf(x, y):
            qml.Displacement(x, 0, wires=[0])
            qml.Displacement(1.2, y, wires=[0])
            qml.Beamsplitter(0.2, 1.7, wires=[0, 1])
            qml.Rotation(1.9, wires=[0])
            qml.Kerr(0.3, wires=[1])  # nongaussian succeeding both x and y due to the beamsplitter
            return qml.expval(qml.X(0)), qml.expval(qml.X(1))

        check_methods(qf, {0: "F", 1: "F"})

        def qf(x, y):
            qml.Kerr(y, wires=[1])
            qml.Displacement(x, 0, wires=[0])
            qml.Beamsplitter(0.2, 1.7, wires=[0, 1])
            return qml.expval(qml.X(0)), qml.expval(qml.X(1))

        check_methods(qf, {0: "A", 1: "F"})

    def test_qnode_gradient_multiple_gate_parameters(self, tol):
        """Tests that gates with multiple free parameters yield correct gradients."""
        par = [0.5, 0.3, -0.7]

        def qf(x, y, z):
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[0])
            qml.RY(-0.2, wires=[0])
            return qml.expval(qml.PauliZ(0))

        dev = qml.device("default.qubit", wires=1)
        q = qml.QNode(qf, dev)
        value = q(*par)
        grad_A = q.jacobian(par, method="A")
        grad_F = q.jacobian(par, method="F")

        # analytic method works for every parameter
        assert q.grad_method_for_par == {0: "A", 1: "A", 2: "A"}
        # gradient has the correct shape and every element is nonzero
        assert grad_A.shape == (1, 3)
        assert np.count_nonzero(grad_A) == 3
        # the different methods agree
        assert np.allclose(grad_A, grad_F, atol=tol, rtol=0)

    def test_qnode_gradient_repeated_gate_parameters(self, tol):
        """Tests that repeated use of a free parameter in a
        multi-parameter gate yield correct gradients."""
        par = [0.8, 1.3]

        def qf(x, y):
            qml.RX(np.pi / 4, wires=[0])
            qml.Rot(y, x, 2 * x, wires=[0])
            return qml.expval(qml.PauliX(0))

        dev = qml.device("default.qubit", wires=1)
        q = qml.QNode(qf, dev)
        grad_A = q.jacobian(par, method="A")
        grad_F = q.jacobian(par, method="F")

        # the different methods agree
        assert np.allclose(grad_A, grad_F, atol=tol, rtol=0)

    def test_qnode_gradient_parameters_inside_array(self, tol):
        """Tests that free parameters inside an array passed to
        an Operation yield correct gradients."""
        par = [0.8, 1.3]

        def qf(x, y):
            qml.RX(x, wires=[0])
            qml.RY(x, wires=[0])
            return qml.expval(qml.Hermitian(np.diag([y, 1]), 0))

        dev = qml.device("default.qubit", wires=1)
        q = qml.QNode(qf, dev)
        grad = q.jacobian(par)
        grad_F = q.jacobian(par, method="F")

        # par[0] can use the 'A' method, par[1] cannot
        assert q.grad_method_for_par == {0: "A", 1: "F"}
        # the different methods agree
        assert np.allclose(grad, grad_F, atol=tol, rtol=0)

    def test_array_parameters_autograd(self, tol):
        """Test that gradients of array parameters give
        same results as positional arguments."""

        a, b, c = 0.5, 0.54, 0.3

        def ansatz(x, y, z):
            qml.QubitStateVector(np.array([1, 0, 1, 1]) / np.sqrt(3), wires=[0, 1])
            qml.Rot(x, y, z, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        def circuit1(x, y, z):
            return ansatz(x, y, z)

        def circuit2(x, array):
            return ansatz(x, array[0], array[1])

        def circuit3(array):
            return ansatz(*array)

        dev = qml.device("default.qubit", wires=2)
        circuit1 = qml.QNode(circuit1, dev)
        grad1 = qml.grad(circuit1, argnum=[0, 1, 2])

        positional_grad = circuit1.jacobian([a, b, c])
        positional_autograd = grad1(a, b, c)
        assert np.allclose(positional_grad, positional_autograd, atol=tol, rtol=0)

        circuit2 = qml.QNode(circuit2, dev)
        grad2 = qml.grad(circuit2, argnum=[0, 1])

        circuit3 = qml.QNode(circuit3, dev)
        grad3 = qml.grad(circuit3, argnum=0)

        array_grad = circuit3.jacobian([np.array([a, b, c])])
        array_autograd = grad3(np.array([a, b, c]))
        assert np.allclose(array_grad, array_autograd, atol=tol, rtol=0)

    @staticmethod
    def expected_jacobian(x, y, z):
        dw0dx = 2 / 3 * np.sin(x) * np.sin(y)
        dw0dy = 1 / 3 * (np.sin(y) - 2 * np.cos(x) * np.cos(y))
        dw0dz = 0

        dw1dx = -2 / 3 * np.cos(x) * np.sin(y)
        dw1dy = -2 / 3 * np.cos(y) * np.sin(x)
        dw1dz = 0

        return np.array([[dw0dx, dw0dy, dw0dz], [dw1dx, dw1dy, dw1dz]])

    def test_multiple_expectation_jacobian_positional(self, tol):
        """Tests that qnodes using positional arguments return
        correct gradients for multiple expectation values."""
        a, b, c = 0.5, 0.54, 0.3

        def circuit(x, y, z):
            qml.QubitStateVector(np.array([1, 0, 1, 1]) / np.sqrt(3), wires=[0, 1])
            qml.Rot(x, y, z, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))

        dev = qml.device("default.qubit", wires=2)
        circuit = qml.QNode(circuit, dev)

        # compare our manual Jacobian computation to theoretical result
        # Note: circuit.jacobian actually returns a full jacobian in this case
        res = circuit.jacobian(np.array([a, b, c]))
        assert np.allclose(self.expected_jacobian(a, b, c), res, atol=tol, rtol=0)

        # compare our manual Jacobian computation to autograd
        # not sure if this is the intended usage of jacobian
        jac0 = qml.jacobian(circuit, 0)
        jac1 = qml.jacobian(circuit, 1)
        jac2 = qml.jacobian(circuit, 2)
        res = np.stack([jac0(a, b, c), jac1(a, b, c), jac2(a, b, c)]).T

        assert np.allclose(self.expected_jacobian(a, b, c), res, atol=tol, rtol=0)

    def test_multiple_expectation_jacobian_array(self, tol):
        """Tests that qnodes using an array argument return correct gradients
        for multiple expectation values."""
        a, b, c = 0.5, 0.54, 0.3

        def circuit(weights):
            qml.QubitStateVector(np.array([1, 0, 1, 1]) / np.sqrt(3), wires=[0, 1])
            qml.Rot(weights[0], weights[1], weights[2], wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))

        dev = qml.device("default.qubit", wires=2)
        circuit = qml.QNode(circuit, dev)

        res = circuit.jacobian([np.array([a, b, c])])
        assert np.allclose(self.expected_jacobian(a, b, c), res, atol=tol, rtol=0)

        jac = qml.jacobian(circuit, 0)
        res = jac(np.array([a, b, c]))
        assert np.allclose(self.expected_jacobian(a, b, c), res, atol=tol, rtol=0)

    def test_keywordarg_not_differentiated(self, tol):
        """Tests that qnodes do not differentiate w.r.t. keyword arguments."""
        a, b = 0.5, 0.54

        def circuit1(weights, x=0.3):
            qml.QubitStateVector(np.array([1, 0, 1, 1]) / np.sqrt(3), wires=[0, 1])
            qml.Rot(weights[0], weights[1], x, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))

        dev = qml.device("default.qubit", wires=2)
        circuit1 = qml.QNode(circuit1, dev)

        def circuit2(weights):
            qml.QubitStateVector(np.array([1, 0, 1, 1]) / np.sqrt(3), wires=[0, 1])
            qml.Rot(weights[0], weights[1], 0.3, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))

        circuit2 = qml.QNode(circuit2, dev)

        res1 = circuit1.jacobian([np.array([a, b])])
        res2 = circuit2.jacobian([np.array([a, b])])

        assert np.allclose(res1, res2, atol=tol, rtol=0)

    def test_differentiate_all_positional(self, tol):
        """Tests that all positional arguments are differentiated."""

        def circuit1(a, b, c):
            qml.RX(a, wires=0)
            qml.RX(b, wires=1)
            qml.RX(c, wires=2)
            return tuple(qml.expval(qml.PauliZ(idx)) for idx in range(3))

        dev = qml.device("default.qubit", wires=3)
        circuit1 = qml.QNode(circuit1, dev)

        vals = np.array([np.pi, np.pi / 2, np.pi / 3])
        circuit_output = circuit1(*vals)
        expected_output = np.cos(vals)
        assert np.allclose(circuit_output, expected_output, atol=tol, rtol=0)

        # circuit jacobians
        circuit_jacobian = circuit1.jacobian(vals)
        expected_jacobian = -np.diag(np.sin(vals))
        assert np.allclose(circuit_jacobian, expected_jacobian, atol=tol, rtol=0)

    def test_differentiate_first_positional(self, tol):
        """Tests that the first positional arguments are differentiated."""

        def circuit2(a, b):
            qml.RX(a, wires=0)
            return qml.expval(qml.PauliZ(0))

        dev = qml.device("default.qubit", wires=2)
        circuit2 = qml.QNode(circuit2, dev)

        a = 0.7418
        b = -5.0
        circuit_output = circuit2(a, b)
        expected_output = np.cos(a)
        assert np.allclose(circuit_output, expected_output, atol=tol, rtol=0)

        # circuit jacobians
        circuit_jacobian = circuit2.jacobian([a, b])
        expected_jacobian = np.array([[-np.sin(a), 0]])
        assert np.allclose(circuit_jacobian, expected_jacobian, atol=tol, rtol=0)

    def test_differentiate_second_positional(self, tol):
        """Tests that the second positional arguments are differentiated."""

        def circuit3(a, b):
            qml.RX(b, wires=0)
            return qml.expval(qml.PauliZ(0))

        dev = qml.device("default.qubit", wires=2)
        circuit3 = qml.QNode(circuit3, dev)

        a = 0.7418
        b = -5.0
        circuit_output = circuit3(a, b)
        expected_output = np.cos(b)
        assert np.allclose(circuit_output, expected_output, atol=tol, rtol=0)

        # circuit jacobians
        circuit_jacobian = circuit3.jacobian([a, b])
        expected_jacobian = np.array([[0, -np.sin(b)]])
        assert np.allclose(circuit_jacobian, expected_jacobian, atol=tol, rtol=0)

    def test_differentiate_second_third_positional(self, tol):
        """Tests that the second and third positional arguments are differentiated."""

        def circuit4(a, b, c):
            qml.RX(b, wires=0)
            qml.RX(c, wires=1)
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        dev = qml.device("default.qubit", wires=2)
        circuit4 = qml.QNode(circuit4, dev)

        a = 0.7418
        b = -5.0
        c = np.pi / 7
        circuit_output = circuit4(a, b, c)
        expected_output = np.array([[np.cos(b), np.cos(c)]])
        assert np.allclose(circuit_output, expected_output, atol=tol, rtol=0)

        # circuit jacobians
        circuit_jacobian = circuit4.jacobian([a, b, c])
        expected_jacobian = np.array([[0.0, -np.sin(b), 0.0], [0.0, 0.0, -np.sin(c)]])
        assert np.allclose(circuit_jacobian, expected_jacobian, atol=tol, rtol=0)

    def test_differentiate_positional_multidim(self, tol):
        """Tests that all positional arguments are differentiated
        when they are multidimensional."""

        def circuit(a, b):
            qml.RX(a[0], wires=0)
            qml.RX(a[1], wires=1)
            qml.RX(b[2, 1], wires=2)
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2))

        dev = qml.device("default.qubit", wires=3)
        circuit = qml.QNode(circuit, dev)

        a = np.array([-np.sqrt(2), -0.54])
        b = np.array([np.pi / 7] * 6).reshape([3, 2])
        circuit_output = circuit(a, b)
        expected_output = np.cos(np.array([[a[0], a[1], b[-1, 0]]]))
        assert np.allclose(circuit_output, expected_output, atol=tol, rtol=0)

        # circuit jacobians
        circuit_jacobian = circuit.jacobian([a, b])
        expected_jacobian = np.array(
            [
                [-np.sin(a[0])] + [0.0] * 7,  # expval 0
                [0.0, -np.sin(a[1])] + [0.0] * 6,  # expval 1
                [0.0] * 2 + [0.0] * 5 + [-np.sin(b[2, 1])],
            ]
        )  # expval 2
        assert np.allclose(circuit_jacobian, expected_jacobian, atol=tol, rtol=0)

    def test_controlled_RX_gradient(self, tol):
        """Test gradient of controlled RX gate"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x):
            qml.PauliX(wires=0)
            qml.CRX(x, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        a = 0.542  # any value of a should give zero gradient

        # get the analytic gradient
        gradA = circuit.jacobian([a], method="A")
        # get the finite difference gradient
        gradF = circuit.jacobian([a], method="F")

        # the expected gradient
        expected = 0

        assert np.allclose(gradF, expected, atol=tol, rtol=0)
        assert np.allclose(gradA, expected, atol=tol, rtol=0)

        @qml.qnode(dev)
        def circuit1(x):
            qml.RX(x, wires=0)
            qml.CRX(x, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        b = 0.123  # gradient is -sin(x)

        # get the analytic gradient
        gradA = circuit1.jacobian([b], method="A")
        # get the finite difference gradient
        gradF = circuit1.jacobian([b], method="F")

        # the expected gradient
        expected = -np.sin(b)

        assert np.allclose(gradF, expected, atol=tol, rtol=0)
        assert np.allclose(gradA, expected, atol=tol, rtol=0)

    def test_controlled_RY_gradient(self, tol):
        """Test gradient of controlled RY gate"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x):
            qml.PauliX(wires=0)
            qml.CRY(x, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        a = 0.542  # any value of a should give zero gradient

        # get the analytic gradient
        gradA = circuit.jacobian([a], method="A")
        # get the finite difference gradient
        gradF = circuit.jacobian([a], method="F")

        # the expected gradient
        expected = 0

        assert np.allclose(gradF, expected, atol=tol, rtol=0)
        assert np.allclose(gradA, expected, atol=tol, rtol=0)

        @qml.qnode(dev)
        def circuit1(x):
            qml.RX(x, wires=0)
            qml.CRY(x, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        b = 0.123  # gradient is -sin(x)

        # get the analytic gradient
        gradA = circuit1.jacobian([b], method="A")
        # get the finite difference gradient
        gradF = circuit1.jacobian([b], method="F")

        # the expected gradient
        expected = -np.sin(b)

        assert np.allclose(gradF, expected, atol=tol, rtol=0)
        assert np.allclose(gradA, expected, atol=tol, rtol=0)

    def test_controlled_RZ_gradient(self, tol):
        """Test gradient of controlled RZ gate"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x):
            qml.PauliX(wires=0)
            qml.CRZ(x, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        a = 0.542  # any value of a should give zero gradient

        # get the analytic gradient
        gradA = circuit.jacobian([a], method="A")
        # get the finite difference gradient
        gradF = circuit.jacobian([a], method="F")

        # the expected gradient
        expected = 0

        assert np.allclose(gradF, expected, atol=tol, rtol=0)
        assert np.allclose(gradA, expected, atol=tol, rtol=0)

        @qml.qnode(dev)
        def circuit1(x):
            qml.RX(x, wires=0)
            qml.CRZ(x, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        b = 0.123  # gradient is -sin(x)

        # get the analytic gradient
        gradA = circuit1.jacobian([b], method="A")
        # get the finite difference gradient
        gradF = circuit1.jacobian([b], method="F")

        # the expected gradient
        expected = -np.sin(b)

        assert np.allclose(gradF, expected, atol=tol, rtol=0)
        assert np.allclose(gradA, expected, atol=tol, rtol=0)


class TestQNodeVariance:
    """Qnode variance tests."""

    def test_involutory_variance(self, tol):
        """Tests qubit observable that are involutory"""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(a):
            qml.RX(a, wires=0)
            return qml.var(qml.PauliZ(0))

        a = 0.54
        var = circuit(a)
        expected = 1 - np.cos(a) ** 2
        assert np.allclose(var, expected, atol=tol, rtol=0)

        # circuit jacobians
        gradA = circuit.jacobian([a], method="A")
        gradF = circuit.jacobian([a], method="F")
        expected = 2 * np.sin(a) * np.cos(a)
        assert np.allclose(gradF, expected, atol=tol, rtol=0)
        assert np.allclose(gradA, expected, atol=tol, rtol=0)

    def test_non_involutory_variance(self, tol):
        """Tests a qubit Hermitian observable that is not involutory"""
        dev = qml.device("default.qubit", wires=1)

        A = np.array([[4, -1 + 6j], [-1 - 6j, 2]])

        @qml.qnode(dev)
        def circuit(a):
            qml.RX(a, wires=0)
            return qml.var(qml.Hermitian(A, 0))

        a = 0.54
        var = circuit(a)
        expected = (39 / 2) - 6 * np.sin(2 * a) + (35 / 2) * np.cos(2 * a)
        assert np.allclose(var, expected, atol=tol, rtol=0)

        # circuit jacobians
        gradA = circuit.jacobian([a], method="A")
        gradF = circuit.jacobian([a], method="F")
        expected = -35 * np.sin(2 * a) - 12 * np.cos(2 * a)
        assert np.allclose(gradA, expected, atol=tol, rtol=0)
        assert np.allclose(gradF, expected, atol=tol, rtol=0)

    def test_fanout(self, tol):
        """Tests qubit observable with repeated parameters"""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(a):
            qml.RX(a, wires=0)
            qml.RY(a, wires=0)
            return qml.var(qml.PauliZ(0))

        a = 0.54
        var = circuit(a)
        expected = 0.5 * np.sin(a) ** 2 * (np.cos(2 * a) + 3)
        assert np.allclose(var, expected, atol=tol, rtol=0)

        # circuit jacobians
        gradA = circuit.jacobian([a], method="A")
        gradF = circuit.jacobian([a], method="F")
        expected = 4 * np.sin(a) * np.cos(a) ** 3
        assert np.allclose(gradA, expected, atol=tol, rtol=0)
        assert np.allclose(gradF, expected, atol=tol, rtol=0)

    def test_expval_and_variance(self, tol):
        """Test that the qnode works for a combination of expectation
        values and variances"""
        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def circuit(a, b, c):
            qml.RX(a, wires=0)
            qml.RY(b, wires=1)
            qml.CNOT(wires=[1, 2])
            qml.RX(c, wires=2)
            qml.CNOT(wires=[0, 1])
            qml.RZ(c, wires=2)
            return qml.var(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.var(qml.PauliZ(2))

        a = 0.54
        b = -0.423
        c = 0.123

        var = circuit(a, b, c)
        expected = np.array(
            [
                np.sin(a) ** 2,
                np.cos(a) * np.cos(b),
                0.25 * (3 - 2 * np.cos(b) ** 2 * np.cos(2 * c) - np.cos(2 * b)),
            ]
        )
        assert np.allclose(var, expected, atol=tol, rtol=0)

        # # circuit jacobians
        gradA = circuit.jacobian([a, b, c], method="A")
        gradF = circuit.jacobian([a, b, c], method="F")
        expected = np.array(
            [
                [2 * np.cos(a) * np.sin(a), -np.cos(b) * np.sin(a), 0],
                [
                    0,
                    -np.cos(a) * np.sin(b),
                    0.5 * (2 * np.cos(b) * np.cos(2 * c) * np.sin(b) + np.sin(2 * b)),
                ],
                [0, 0, np.cos(b) ** 2 * np.sin(2 * c)],
            ]
        ).T
        assert np.allclose(gradF, expected, atol=tol, rtol=0)
        assert np.allclose(gradA, expected, atol=tol, rtol=0)

    def test_first_order_cv(self, tol):
        """Test variance of a first order CV expectation value"""
        dev = qml.device("default.gaussian", wires=1)

        @qml.qnode(dev)
        def circuit(r, phi):
            qml.Squeezing(r, 0, wires=0)
            qml.Rotation(phi, wires=0)
            return qml.var(qml.X(0))

        r = 0.543
        phi = -0.654

        var = circuit(r, phi)
        expected = np.exp(2 * r) * np.sin(phi) ** 2 + np.exp(-2 * r) * np.cos(phi) ** 2
        assert np.allclose(var, expected, atol=tol, rtol=0)

        # circuit jacobians
        gradA = circuit.jacobian([r, phi], method="A")
        gradF = circuit.jacobian([r, phi], method="F")
        expected = np.array(
            [
                2 * np.exp(2 * r) * np.sin(phi) ** 2 - 2 * np.exp(-2 * r) * np.cos(phi) ** 2,
                2 * np.sinh(2 * r) * np.sin(2 * phi),
            ]
        )
        assert np.allclose(gradA, expected, atol=tol, rtol=0)
        assert np.allclose(gradF, expected, atol=tol, rtol=0)

    def test_second_order_cv(self, tol):
        """Test variance of a second order CV expectation value"""
        dev = qml.device("default.gaussian", wires=1)

        @qml.qnode(dev)
        def circuit(n, a):
            qml.ThermalState(n, wires=0)
            qml.Displacement(a, 0, wires=0)
            return qml.var(qml.NumberOperator(0))

        n = 0.12
        a = 0.765

        var = circuit(n, a)
        expected = n ** 2 + n + np.abs(a) ** 2 * (1 + 2 * n)
        assert np.allclose(var, expected, atol=tol, rtol=0)

        # circuit jacobians
        gradF = circuit.jacobian([n, a], method="F")
        expected = np.array([2 * a ** 2 + 2 * n + 1, 2 * a * (2 * n + 1)])
        assert np.allclose(gradF, expected, atol=tol, rtol=0)

    def test_error_analytic_second_order_cv(self):
        """Test exception raised if attempting to use a second
        order observable to compute the variance derivative analytically"""
        dev = qml.device("default.gaussian", wires=1)

        @qml.qnode(dev)
        def circuit(a):
            qml.Displacement(a, 0, wires=0)
            return qml.var(qml.NumberOperator(0))

        with pytest.raises(ValueError, match=r"cannot be used with the parameter\(s\) \{0\}"):
            circuit.jacobian([1.0], method="A")


class TestQNodeCacheing:
    """Tests for the QNode construction caching"""

    def test_no_caching(self):
        """Test that the circuit structure changes on
        subsequent evalutions with caching turned off
        """
        dev = qml.device("default.qubit", wires=2)

        def circuit(x, c=None):
            qml.RX(x, wires=0)

            for i in range(c):
                qml.RX(x, wires=i)

            return qml.expval(qml.PauliZ(0))

        circuit = qml.QNode(circuit, dev, cache=False)

        # first evaluation
        circuit(0, c=0)
        # check structure
        assert len(circuit.queue) == 1

        # second evaluation
        circuit(0, c=1)
        # check structure
        assert len(circuit.queue) == 2

    def test_caching(self):
        """Test that the circuit structure does not change on
        subsequent evalutions with caching turned on
        """
        dev = qml.device("default.qubit", wires=2)

        def circuit(x, c=None):
            qml.RX(x, wires=0)

            for i in range(c.val):
                qml.RX(x, wires=i)

            return qml.expval(qml.PauliZ(0))

        circuit = qml.QNode(circuit, dev, cache=True)

        # first evaluation
        circuit(0, c=0)
        # check structure
        assert len(circuit.queue) == 1

        # second evaluation
        circuit(0, c=1)
        # check structure
        assert len(circuit.queue) == 1

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
import contextlib
import io
import math
import textwrap

import pytest
import numpy as np
from scipy.linalg import block_diag

from pennylane.plugins.default_qubit import Y, Z

import pennylane as qml
from pennylane._device import Device
from pennylane.qnode import QNode, QuantumFunctionError


class TestQNodeOperationQueue:
    """Tests that the QNode operation queue is properly filled and interacted with"""

    @pytest.fixture(scope="function")
    def qnode(self, mock_device):
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

    def test_operation_ordering(self, qnode):
        """Tests that the ordering of the operations is correct"""

        assert qnode.ops[0].name == "RX"
        assert qnode.ops[1].name == "CNOT"
        assert qnode.ops[2].name == "RY"
        assert qnode.ops[3].name == "RZ"
        assert qnode.ops[4].name == "PauliX"
        assert qnode.ops[5].name == "PauliZ"

    def test_op_successors_operations_only(self, qnode):
        """Tests that _op_successors properly extracts the successors that are operations"""

        operation_successors = qnode._op_successors(qnode.ops[0], only="G")

        assert qnode.ops[0] not in operation_successors
        assert qnode.ops[1] in operation_successors
        assert qnode.ops[4] not in operation_successors

    def test_op_successors_observables_only(self, qnode):
        """Tests that _op_successors properly extracts the successors that are observables"""

        observable_successors = qnode._op_successors(qnode.ops[0], only="E")

        assert qnode.ops[0] not in observable_successors
        assert qnode.ops[1] not in observable_successors
        assert qnode.ops[4] in observable_successors

    def test_op_successors_both_operations_and_observables(self, qnode):
        """Tests that _op_successors properly extracts all successors"""

        successors = qnode._op_successors(qnode.ops[0], only=None)

        assert qnode.ops[0] not in successors
        assert qnode.ops[1] in successors
        assert qnode.ops[4] in successors

    def test_op_successors_both_operations_and_observables_nodes(self, qnode):
        """Tests that _op_successors properly extracts all successor nodes"""

        successors = qnode._op_successors(qnode.ops[0], only=None)

        assert qnode.circuit.operations[0] not in successors
        assert qnode.circuit.operations[1] in successors
        assert qnode.circuit.operations[2] in successors
        assert qnode.circuit.operations[3] in successors
        assert qnode.circuit.observables[0] in successors

    def test_op_successors_both_operations_and_observables_strict_ordering(self, qnode):
        """Tests that _op_successors properly extracts all successors"""

        successors = qnode._op_successors(qnode.ops[2], only=None)

        assert qnode.circuit.operations[0] not in successors
        assert qnode.circuit.operations[1] not in successors
        assert qnode.circuit.operations[2] not in successors
        assert qnode.circuit.operations[3] not in successors
        assert qnode.circuit.observables[0] in successors

    def test_op_successors_extracts_all_successors(self, qnode):
        """Tests that _op_successors properly extracts all successors"""
        successors = qnode._op_successors(qnode.ops[2], only=None)
        assert qnode.ops[4] in successors
        assert qnode.ops[5] not in successors

    def test_print_applied(self, mock_device):
        """Test that printing applied gates works correctly"""

        H = np.array([[0, 1], [1, 0]])

        def circuit(x):
            qml.RX(x, wires=[0])
            qml.CNOT(wires=[0, 1])
            qml.RY(0.4, wires=[0])
            qml.RZ(-0.2, wires=[1])
            return qml.expval(qml.PauliX(0)), qml.var(qml.Hermitian(H, wires=1))

        expected_qnode_print = textwrap.dedent("""\
            Operations
            ==========
            RX({x}, wires=[0])
            CNOT(wires=[0, 1])
            RY(0.4, wires=[0])
            RZ(-0.2, wires=[1])

            Observables
            ===========
            expval(PauliX(wires=[0]))
            var(Hermitian([[0 1]
             [1 0]], wires=[1]))""")

        node = qml.QNode(circuit, mock_device)

        # test before construction
        f = io.StringIO()

        with contextlib.redirect_stdout(f):
            node.print_applied()
            out = f.getvalue().strip()

        assert out == "QNode has not yet been executed."

        # construct QNode
        f = io.StringIO()
        node.construct([0.1])

        with contextlib.redirect_stdout(f):
            node.print_applied()
            out = f.getvalue().strip()

        assert out == expected_qnode_print.format(x=0.1)


@pytest.fixture(scope="function")
def operable_mock_device_2_wires(monkeypatch):
    """A mock instance of the abstract Device class that can support qfuncs."""

    dev = Device
    with monkeypatch.context() as m:
        m.setattr(dev, '__abstractmethods__', frozenset())
        m.setattr(dev, 'operations', ["RX", "RY", "CNOT"])
        m.setattr(dev, 'observables', ["PauliX", "PauliY", "PauliZ"])
        m.setattr(dev, 'reset', lambda self: None)
        m.setattr(dev, 'apply', lambda self, x, y, z: None)
        m.setattr(dev, 'expval', lambda self, x, y, z: 1)
        yield Device(wires=2)


class TestQNodeBestMethod:
    """
    Test different flows of _best_method
    """
    def test_best_method_with_non_gaussian_successors(self, tol, gaussian_device_2_wires):
        """Tests that the analytic differentiation method is allowed and matches numerical
        differentiation if a non-Gaussian gate is not succeeded by an observable."""

        @qml.qnode(gaussian_device_2_wires)
        def circuit(x):
            qml.Squeezing(x, 0, wires=[0])
            qml.Beamsplitter(np.pi/4, 0, wires=[0, 1])
            qml.Kerr(0.54, wires=[1])
            return qml.expval(qml.NumberOperator(0))

        res = circuit.jacobian([0.321], method='A')
        expected = circuit.jacobian([0.321], method='F')
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_best_method_with_gaussian_successors_fails(self, gaussian_device_2_wires):
        """Tests that the analytic differentiation method is not allowed
        if a non-Gaussian gate is succeeded by an observable."""

        @qml.qnode(gaussian_device_2_wires)
        def circuit(x):
            qml.Squeezing(x, 0, wires=[0])
            qml.Beamsplitter(np.pi/4, 0, wires=[0, 1])
            qml.Kerr(0.54, wires=[1])
            return qml.expval(qml.NumberOperator(1))

        with pytest.raises(ValueError, match="analytic gradient method cannot be used with"):
            circuit.jacobian([0.321], method='A')


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

    def test_transform_observable_incorrect_heisenberg_size(self):
        """Test that an exception is raised in the case that the
        dimensions of a CV observable Heisenberg representation does not match
        the ev_order attribute"""

        dev = qml.device("default.gaussian", wires=1)

        class P(qml.operation.CVObservable):
            """Dummy CV observable with incorrect ev_order"""
            num_wires = 1
            num_params = 0
            par_domain = None
            ev_order = 2

            @staticmethod
            def _heisenberg_rep(p):
                return np.array([0, 1, 0])

        def circuit(x):
            qml.Displacement(x, 0.1, wires=0)
            return qml.expval(P(0))

        node = qml.QNode(circuit, dev)

        with pytest.raises(QuantumFunctionError, match="Mismatch between polynomial order"):
            node.jacobian([0.5])


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
            qml.Kerr(0.4, wires=[0])
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

    def test_qnode_gradient_gate_with_two_parameters(self, tol):
        """Test that a gate with two parameters yields
        correct gradients"""
        def qf(r0, phi0, r1, phi1):
            qml.Squeezing(r0, phi0, wires=[0])
            qml.Squeezing(r1, phi1, wires=[0])
            return qml.expval(qml.NumberOperator(0))

        dev = qml.device('default.gaussian', wires=2)
        q = qml.QNode(qf, dev)

        par = [0.543, 0.123, 0.654, -0.629]

        grad_A = q.jacobian(par, method='A')
        grad_F = q.jacobian(par, method='F')

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

        #compare with what we get if argnum is a list
        res2 = qml.jacobian(circuit, argnum=[0, 1, 2])(a, b, c)
        assert np.allclose(res, res2, atol=tol, rtol=0)

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


gradient_test_data = [
    (0.5, -0.1),
    (0.0, np.pi),
    (-3.6, -3.6),
    (1.0, 2.5),
]


dev = qml.device("default.qubit", wires=2)


@qml.qnode(dev)
def f(x):
    qml.RX(x, wires=0)
    return qml.expval(qml.PauliZ(0))


@qml.qnode(dev)
def g(y):
    qml.RY(y, wires=0)
    return qml.expval(qml.PauliX(0))


class TestMultiQNodeGradients:
    """Multi Qnode gradient tests."""

    @pytest.mark.parametrize("x, y", gradient_test_data)
    def test_add_qnodes_gradient(self, x, y):
        """Test the gradient of addition of two QNode circuits"""

        def add(a, b):
            return a + b

        a = f(x)
        b = g(y)

        # addition
        assert qml.grad(add, argnum=0)(a, b) == 1.0
        assert qml.grad(add, argnum=1)(a, b) == 1.0

        # same value added to itself; autograd doesn't distinguish inputs
        assert qml.grad(add, argnum=0)(a, a) == 1.0

    @pytest.mark.parametrize("x, y", gradient_test_data)
    def test_subtract_qnodes_gradient(self, x, y):
        """Test the gradient of subtraction of two QNode circuits"""

        def subtract(a, b):
            return a - b

        a = f(x)
        b = g(y)

        # subtraction
        assert qml.grad(subtract, argnum=0)(a, b) == 1.0
        assert qml.grad(subtract, argnum=1)(a, b) == -1.0

    @pytest.mark.parametrize("x, y", gradient_test_data)
    def test_multiply_qnodes_gradient(self, x, y):
        """Test the gradient of multiplication of two QNode circuits"""

        def mult(a, b):
            return a * b

        a = f(x)
        b = g(y)

        # multipication
        assert qml.grad(mult, argnum=0)(a, b) == b
        assert qml.grad(mult, argnum=1)(a, b) == a

    @pytest.mark.parametrize("x, y", gradient_test_data)
    def test_division_qnodes_gradient(self, x, y):
        """Test the gradient of division of two QNode circuits"""

        def div(a, b):
            return a / b

        a = f(x)
        b = g(y)

        # division
        assert qml.grad(div, argnum=0)(a, b) == 1 / b
        assert qml.grad(div, argnum=1)(a, b) == -a / b ** 2

    @pytest.mark.parametrize("x, y", gradient_test_data)
    def test_composing_qnodes_gradient(self, x, y):
        """Test the gradient of composing of two QNode circuits"""

        def compose(f, x):
            return f(x)

        a = f(x)
        b = g(y)

        # composition
        assert qml.grad(compose, argnum=1)(f, x) == qml.grad(f, argnum=0)(x)
        assert qml.grad(compose, argnum=1)(f, a) == qml.grad(f, argnum=0)(a)
        assert qml.grad(compose, argnum=1)(f, b) == qml.grad(f, argnum=0)(b)

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


class TestMetricTensor:
    """Tests for metric tensor subcircuit construction and evaluation"""

    def test_no_generator(self):
        """Test exception is raised if subcircuit contains an
        operation with no generator"""
        dev = qml.device('default.qubit', wires=1)

        def circuit(a):
            qml.Rot(a, 0, 0, wires=0)
            return qml.expval(qml.PauliX(0))

        circuit = qml.QNode(circuit, dev)

        with pytest.raises(QuantumFunctionError, match="has no defined generator"):
            circuit.construct_metric_tensor([1])

    def test_generator_no_expval(self, monkeypatch):
        """Test exception is raised if subcircuit contains an
        operation with generator object that is not an observable"""
        dev = qml.device('default.qubit', wires=1)

        def circuit(a):
            qml.RX(a, wires=0)
            return qml.expval(qml.PauliX(0))

        circuit = qml.QNode(circuit, dev)

        with monkeypatch.context() as m:
            m.setattr('pennylane.RX.generator', [qml.RX, 1])

            with pytest.raises(QuantumFunctionError, match="no corresponding observable"):
                circuit.construct_metric_tensor([1])

    def test_construct_subcircuit(self):
        """Test correct subcircuits constructed"""
        dev = qml.device('default.qubit', wires=2)

        def circuit(a, b, c):
            qml.RX(a, wires=0)
            qml.RY(b, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PhaseShift(c, wires=1)
            return qml.expval(qml.PauliX(0))

        circuit = qml.QNode(circuit, dev)

        circuit.construct_metric_tensor([1, 1, 1])
        res = circuit._metric_tensor_subcircuits

        # first parameter subcircuit
        assert len(res[(0,)]['queue']) == 0
        assert res[(0,)]['scale'] == [-0.5]
        assert isinstance(res[(0,)]['observable'][0], qml.PauliX)

        # second parameter subcircuit
        assert len(res[(1,)]['queue']) == 1
        assert res[(1,)]['scale'] == [-0.5]
        assert isinstance(res[(1,)]['queue'][0], qml.RX)
        assert isinstance(res[(1,)]['observable'][0], qml.PauliY)

        # third parameter subcircuit
        assert len(res[(2,)]['queue']) == 3
        assert res[(2,)]['scale'] == [1]
        assert isinstance(res[(2,)]['queue'][0], qml.RX)
        assert isinstance(res[(2,)]['queue'][1], qml.RY)
        assert isinstance(res[(2,)]['queue'][2], qml.CNOT)
        assert isinstance(res[(2,)]['observable'][0], qml.Hermitian)
        assert np.all(res[(2,)]['observable'][0].params[0] == qml.PhaseShift.generator[0])

    def test_construct_subcircuit_layers(self):
        """Test correct subcircuits constructed
        when a layer structure exists"""
        dev = qml.device('default.qubit', wires=3)

        def circuit(params):
            # section 1
            qml.RX(params[0], wires=0)
            # section 2
            qml.RY(params[1], wires=0)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            # section 3
            qml.RX(params[2], wires=0)
            qml.RY(params[3], wires=1)
            qml.RZ(params[4], wires=2)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            # section 4
            qml.RX(params[5], wires=0)
            qml.RY(params[6], wires=1)
            qml.RZ(params[7], wires=2)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            return qml.expval(qml.PauliX(0))

        circuit = qml.QNode(circuit, dev)

        params = np.ones([8])
        circuit.construct_metric_tensor([params])
        res = circuit._metric_tensor_subcircuits

        # this circuit should split into 4 independent
        # sections or layers when constructing subcircuits
        assert len(res) == 4

        # first layer subcircuit
        layer = res[(0,)]
        assert len(layer['queue']) == 0
        assert len(layer['observable']) == 1
        assert isinstance(layer['observable'][0], qml.PauliX)

        # second layer subcircuit
        layer = res[(1,)]
        assert len(layer['queue']) == 1
        assert len(layer['observable']) == 1
        assert isinstance(layer['queue'][0], qml.RX)
        assert isinstance(layer['observable'][0], qml.PauliY)

        # third layer subcircuit
        layer = res[(2, 3, 4)]
        assert len(layer['queue']) == 4
        assert len(layer['observable']) == 3
        assert isinstance(layer['queue'][0], qml.RX)
        assert isinstance(layer['queue'][1], qml.RY)
        assert isinstance(layer['queue'][2], qml.CNOT)
        assert isinstance(layer['queue'][3], qml.CNOT)
        assert isinstance(layer['observable'][0], qml.PauliX)
        assert isinstance(layer['observable'][1], qml.PauliY)
        assert isinstance(layer['observable'][2], qml.PauliZ)

        # fourth layer subcircuit
        layer = res[(5, 6, 7)]
        assert len(layer['queue']) == 9
        assert len(layer['observable']) == 3
        assert isinstance(layer['queue'][0], qml.RX)
        assert isinstance(layer['queue'][1], qml.RY)
        assert isinstance(layer['queue'][2], qml.CNOT)
        assert isinstance(layer['queue'][3], qml.CNOT)
        assert isinstance(layer['queue'][4], qml.RX)
        assert isinstance(layer['queue'][5], qml.RY)
        assert isinstance(layer['queue'][6], qml.RZ)
        assert isinstance(layer['queue'][7], qml.CNOT)
        assert isinstance(layer['queue'][8], qml.CNOT)
        assert isinstance(layer['observable'][0], qml.PauliX)
        assert isinstance(layer['observable'][1], qml.PauliY)
        assert isinstance(layer['observable'][2], qml.PauliZ)

    def test_evaluate_subcircuits(self, tol):
        """Test subcircuits evaluate correctly"""
        dev = qml.device('default.qubit', wires=2)

        def circuit(a, b, c):
            qml.RX(a, wires=0)
            qml.RY(b, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PhaseShift(c, wires=1)
            return qml.expval(qml.PauliX(0))

        # construct subcircuits
        circuit = qml.QNode(circuit, dev)
        circuit.construct_metric_tensor([1, 1, 1])

        a = 0.432
        b = 0.12
        c = -0.432

        # evaluate subcircuits
        circuit.metric_tensor(a, b, c)

        # first parameter subcircuit
        res = circuit._metric_tensor_subcircuits[(0,)]['result']
        expected = 0.25
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # second parameter subcircuit
        res = circuit._metric_tensor_subcircuits[(1,)]['result']
        expected = np.cos(a)**2/4
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # third parameter subcircuit
        res = circuit._metric_tensor_subcircuits[(2,)]['result']
        expected = (3-2*np.cos(a)**2*np.cos(2*b)-np.cos(2*a))/16
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_evaluate_diag_metric_tensor(self, tol):
        """Test that a diagonal metric tensor evaluates correctly"""
        dev = qml.device('default.qubit', wires=2)

        @qml.qnode(dev)
        def circuit(a, b, c):
            qml.RX(a, wires=0)
            qml.RY(b, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PhaseShift(c, wires=1)
            return qml.expval(qml.PauliX(0))

        a = 0.432
        b = 0.12
        c = -0.432

        # evaluate metric tensor
        g = circuit.metric_tensor(a, b, c)

        # check that the metric tensor is correct
        expected = np.array([1, np.cos(a)**2, (3-2*np.cos(a)**2*np.cos(2*b)-np.cos(2*a))/4])/4
        assert np.allclose(g, np.diag(expected), atol=tol, rtol=0)

    @pytest.fixture
    def sample_circuit(self):
        """Sample variational circuit fixture used in the
        next couple of tests"""
        dev = qml.device('default.qubit', wires=3)

        def non_parametrized_layer(a, b, c):
            qml.RX(a, wires=0)
            qml.RX(b, wires=1)
            qml.RX(c, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            qml.RZ(a, wires=0)
            qml.Hadamard(wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RZ(b, wires=1)
            qml.Hadamard(wires=0)

        a = 0.5
        b = 0.1
        c = 0.5

        @qml.qnode(dev)
        def final(x, y, z, h, g, f):
            non_parametrized_layer(a, b, c)
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.RZ(z, wires=2)
            non_parametrized_layer(a, b, c)
            qml.RY(f, wires=1)
            qml.RZ(g, wires=2)
            qml.RX(h, wires=1)
            return qml.expval(qml.PauliX(0))

        return dev, final, non_parametrized_layer, a, b, c

    def test_evaluate_block_diag_metric_tensor(self, sample_circuit, tol):
        """Test that a block diagonal metric tensor evaluates correctly,
        by comparing it to a known analytic result as well as numerical
        computation."""
        dev, circuit, non_parametrized_layer, a, b, c = sample_circuit

        params = [-0.282203, 0.145554, 0.331624, -0.163907, 0.57662, 0.081272]
        x, y, z, h, g, f = params

        G = circuit.metric_tensor(x, y, z, h, g, f)

        # ============================================
        # Test block diag metric tensor of first layer is correct.
        # We do this by comparing against the known analytic result.
        # First layer includes the non_parametrized_layer,
        # followed by observables corresponding to generators of:
        #   qml.RX(x, wires=0)
        #   qml.RY(y, wires=1)
        #   qml.RZ(z, wires=2)

        G1 = np.zeros([3, 3])

        # diag elements
        G1[0, 0] = np.sin(a)**2/4
        G1[1, 1] = (
            16 * np.cos(a) ** 2 * np.sin(b) ** 3 * np.cos(b) * np.sin(2 * c)
            + np.cos(2 * b) * (2 - 8 * np.cos(a) ** 2 * np.sin(b) ** 2 * np.cos(2 * c))
            + np.cos(2 * (a - b))
            + np.cos(2 * (a + b))
            - 2 * np.cos(2 * a)
            + 14
        ) / 64
        G1[2, 2] = (3-np.cos(2*a)-2*np.cos(a)**2*np.cos(2*(b+c)))/16

        # off diag elements
        G1[0, 1] = np.sin(a)**2 * np.sin(b) * np.cos(b+c)/4
        G1[0, 2] = np.sin(a)**2 * np.cos(b+c)/4
        G1[1, 2] = -np.sin(b) * (
            np.cos(2 * (a - b - c))
            + np.cos(2 * (a + b + c))
            + 2 * np.cos(2 * a)
            + 2 * np.cos(2 * (b + c))
            - 6
        ) / 32

        G1[1, 0] = G1[0, 1]
        G1[2, 0] = G1[0, 2]
        G1[2, 1] = G1[1, 2]

        assert np.allclose(G[:3, :3], G1, atol=tol, rtol=0)

        # =============================================
        # Test block diag metric tensor of second layer is correct.
        # We do this by computing the required expectation values
        # numerically.
        # The second layer includes the non_parametrized_layer,
        # RX, RY, RZ gates (x, y, z params), a 2nd non_parametrized_layer,
        # followed by the qml.RY(f, wires=2) operation.
        #
        # Observable is simply generator of:
        #   qml.RY(f, wires=2)
        #
        # Note: since this layer only consists of a single parameter,
        # only need to compute a single diagonal element.

        @qml.qnode(dev)
        def layer2_diag(x, y, z, h, g, f):
            non_parametrized_layer(a, b, c)
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.RZ(z, wires=2)
            non_parametrized_layer(a, b, c)
            qml.RY(f, wires=2)
            return qml.var(qml.PauliX(1))

        G2 = layer2_diag(x, y, z, h, g, f)/4
        assert np.allclose(G[3:4, 3:4], G2, atol=tol, rtol=0)

        # =============================================
        # Test block diag metric tensor of third layer is correct.
        # We do this by computing the required expectation values
        # numerically using multiple circuits.
        # The second layer includes the non_parametrized_layer,
        # RX, RY, RZ gates (x, y, z params), and a 2nd non_parametrized_layer.
        #
        # Observables are the generators of:
        #   qml.RY(f, wires=1)
        #   qml.RZ(g, wires=2)
        G3 = np.zeros([2, 2])

        @qml.qnode(dev)
        def layer3_diag(x, y, z, h, g, f):
            non_parametrized_layer(a, b, c)
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.RZ(z, wires=2)
            non_parametrized_layer(a, b, c)
            return qml.var(qml.PauliZ(2)), qml.var(qml.PauliY(1))

        @qml.qnode(dev)
        def layer3_off_diag_first_order(x, y, z, h, g, f):
            non_parametrized_layer(a, b, c)
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.RZ(z, wires=2)
            non_parametrized_layer(a, b, c)
            return qml.expval(qml.PauliZ(2)), qml.expval(qml.PauliY(1))

        @qml.qnode(dev)
        def layer3_off_diag_second_order(x, y, z, h, g, f):
            non_parametrized_layer(a, b, c)
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.RZ(z, wires=2)
            non_parametrized_layer(a, b, c)
            return qml.expval(qml.Hermitian(np.kron(Z, Y), wires=[2, 1]))

        # calculate the diagonal terms
        varK0, varK1 = layer3_diag(x, y, z, h, g, f)
        G3[0, 0] = varK0/4
        G3[1, 1] = varK1/4

        # calculate the off-diagonal terms
        exK0, exK1 = layer3_off_diag_first_order(x, y, z, h, g, f)
        exK01 = layer3_off_diag_second_order(x, y, z, h, g, f)

        G3[0, 1] = (exK01 - exK0*exK1)/4
        G3[1, 0] = (exK01 - exK0*exK1)/4

        assert np.allclose(G[4:6, 4:6], G3, atol=tol, rtol=0)

        # ============================================
        # Finally, double check that the entire metric
        # tensor is as computed.

        G_expected = block_diag(G1, G2, G3)
        assert np.allclose(G, G_expected, atol=tol, rtol=0)

    def test_evaluate_diag_approx_metric_tensor(self, sample_circuit, tol):
        """Test that a metric tensor under the
        diagonal approximation evaluates correctly."""
        dev, circuit, non_parametrized_layer, a, b, c = sample_circuit
        params = [-0.282203, 0.145554, 0.331624, -0.163907, 0.57662, 0.081272]
        x, y, z, h, g, f = params

        G = circuit.metric_tensor(x, y, z, h, g, f, diag_approx=True)

        # ============================================
        # Test block diag metric tensor of first layer is correct.
        # We do this by comparing against the known analytic result.
        # First layer includes the non_parametrized_layer,
        # followed by observables corresponding to generators of:
        #   qml.RX(x, wires=0)
        #   qml.RY(y, wires=1)
        #   qml.RZ(z, wires=2)

        G1 = np.zeros([3, 3])

        # diag elements
        G1[0, 0] = np.sin(a)**2/4
        G1[1, 1] = (
            16 * np.cos(a) ** 2 * np.sin(b) ** 3 * np.cos(b) * np.sin(2 * c)
            + np.cos(2 * b) * (2 - 8 * np.cos(a) ** 2 * np.sin(b) ** 2 * np.cos(2 * c))
            + np.cos(2 * (a - b))
            + np.cos(2 * (a + b))
            - 2 * np.cos(2 * a)
            + 14
        ) / 64
        G1[2, 2] = (3-np.cos(2*a)-2*np.cos(a)**2*np.cos(2*(b+c)))/16

        assert np.allclose(G[:3, :3], G1, atol=tol, rtol=0)

        # =============================================
        # Test metric tensor of second layer is correct.
        # We do this by computing the required expectation values
        # numerically.
        # The second layer includes the non_parametrized_layer,
        # RX, RY, RZ gates (x, y, z params), a 2nd non_parametrized_layer,
        # followed by the qml.RY(f, wires=2) operation.
        #
        # Observable is simply generator of:
        #   qml.RY(f, wires=2)
        #
        # Note: since this layer only consists of a single parameter,
        # only need to compute a single diagonal element.

        @qml.qnode(dev)
        def layer2_diag(x, y, z, h, g, f):
            non_parametrized_layer(a, b, c)
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.RZ(z, wires=2)
            non_parametrized_layer(a, b, c)
            qml.RY(f, wires=2)
            return qml.var(qml.PauliX(1))

        G2 = layer2_diag(x, y, z, h, g, f)/4
        assert np.allclose(G[3:4, 3:4], G2, atol=tol, rtol=0)

        # =============================================
        # Test block diag metric tensor of third layer is correct.
        # We do this by computing the required expectation values
        # numerically using multiple circuits.
        # The second layer includes the non_parametrized_layer,
        # RX, RY, RZ gates (x, y, z params), and a 2nd non_parametrized_layer.
        #
        # Observables are the generators of:
        #   qml.RY(f, wires=1)
        #   qml.RZ(g, wires=2)
        G3 = np.zeros([2, 2])

        @qml.qnode(dev)
        def layer3_diag(x, y, z, h, g, f):
            non_parametrized_layer(a, b, c)
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.RZ(z, wires=2)
            non_parametrized_layer(a, b, c)
            return qml.var(qml.PauliZ(2)), qml.var(qml.PauliY(1))

        # calculate the diagonal terms
        varK0, varK1 = layer3_diag(x, y, z, h, g, f)
        G3[0, 0] = varK0/4
        G3[1, 1] = varK1/4

        assert np.allclose(G[4:6, 4:6], G3, atol=tol, rtol=0)

        # ============================================
        # Finally, double check that the entire metric
        # tensor is as computed.

        G_expected = block_diag(G1, G2, G3)
        assert np.allclose(G, G_expected, atol=tol, rtol=0)


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

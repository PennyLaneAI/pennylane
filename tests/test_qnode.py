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
import pytest
import unittest
import logging as log
log.getLogger('defaults')

import autograd
from autograd import numpy as np

from defaults import pennylane as qml, BaseTest

from pennylane.qnode import _flatten, unflatten, QNode, QuantumFunctionError
from pennylane.plugins.default_qubit import CNOT, Rotx, Roty, Rotz, I, Y, Z
from pennylane._device import DeviceError


def expZ(state):
    return np.abs(state[0]) ** 2 - np.abs(state[1]) ** 2


thetas = np.linspace(-2*np.pi, 2*np.pi, 7)

a = np.linspace(-1,1,64)
a_shapes = [(64,),
            (64,1),
            (32,2),
            (16,4),
            (8,8),
            (16,2,2),
            (8,2,2,2),
            (4,2,2,2,2),
            (2,2,2,2,2,2)]

b = np.linspace(-1., 1., 8)
b_shapes = [(8,), (8,1), (4,2), (2,2,2), (2,1,2,1,2)]


class BasicTest(BaseTest):
    """Qnode basic tests.
    """
    def setUp(self):
        self.dev1 = qml.device('default.qubit', wires=1)
        self.dev2 = qml.device('default.qubit', wires=2)

    def test_flatten(self):
        "Tests that _flatten successfully flattens multidimensional arrays."
        self.logTestName()
        flat = a
        for s in a_shapes:
            reshaped = np.reshape(flat, s)
            flattened = np.array([x for x in _flatten(reshaped)])

            self.assertEqual(flattened.shape, flat.shape)
            self.assertAllEqual(flattened, flat)


    def test_unflatten(self):
        "Tests that _unflatten successfully unflattens multidimensional arrays."
        self.logTestName()
        flat = a
        for s in a_shapes:
            reshaped = np.reshape(flat, s)
            unflattened = np.array([x for x in unflatten(flat, reshaped)])

            self.assertEqual(unflattened.shape, reshaped.shape)
            self.assertAllEqual(unflattened, reshaped)

        with self.assertRaisesRegex(TypeError, 'Unsupported type in the model'):
            model = lambda x: x # not a valid model for unflatten
            unflatten(flat, model)

        with self.assertRaisesRegex(ValueError, 'Flattened iterable has more elements than the model'):
            unflatten(np.concatenate([flat, flat]), reshaped)


    def test_op_successors(self):
        "Tests QNode._op_successors()."
        self.logTestName()

        def qf(x):
            qml.RX(x, wires=[0])
            qml.CNOT(wires=[0, 1])
            qml.RY(0.4, wires=[0])
            qml.RZ(-0.2, wires=[1])
            return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliZ(1))
        q = qml.QNode(qf, self.dev2)
        q.construct([1.0])

        # the six operations in qf should appear in q.ops in the same order they appear above
        self.assertTrue(q.ops[0].name == 'RX')
        self.assertTrue(q.ops[1].name == 'CNOT')
        self.assertTrue(q.ops[2].name == 'RY')
        self.assertTrue(q.ops[3].name == 'RZ')
        self.assertTrue(q.ops[4].name == 'PauliX')
        self.assertTrue(q.ops[5].name == 'PauliZ')
        # only gates
        gate_successors = q._op_successors(0, only='G')
        self.assertTrue(q.ops[0] not in gate_successors)
        self.assertTrue(q.ops[1] in gate_successors)
        self.assertTrue(q.ops[4] not in gate_successors)
        # only evs
        ev_sucessors = q._op_successors(0, only='E')
        self.assertTrue(q.ops[0] not in ev_sucessors)
        self.assertTrue(q.ops[1] not in ev_sucessors)
        self.assertTrue(q.ops[4] in ev_sucessors)
        # both
        successors = q._op_successors(0, only=None)
        self.assertTrue(q.ops[0] not in successors)
        self.assertTrue(q.ops[1] in successors)
        self.assertTrue(q.ops[4] in successors)
        # TODO once _op_successors has been upgraded to return only strict successors using a DAG
        #successors = q._op_successors(2, only=None)
        #self.assertTrue(q.ops[4] in successors)
        #self.assertTrue(q.ops[5] not in successors)


    def test_qnode_fail(self):
        "Tests that QNode initialization failures correctly raise exceptions."
        self.logTestName()
        par = 0.5

        #---------------------------------------------------------
        ## QNode internal issues

        # current context should not be set before `construct` is called
        def qf(x):
            return qml.expval(qml.PauliZ(wires=0))
        qnode = QNode(qf, self.dev1)
        QNode._current_context = qnode
        with self.assertRaisesRegex(QuantumFunctionError, 'QNode._current_context must not be modified outside this method.'):
            qnode.construct([0.0])
        QNode._current_context = None

        #---------------------------------------------------------
        ## faulty quantum functions

        # qfunc must return only Expectations
        @qml.qnode(self.dev2)
        def qf(x):
            qml.RX(x, wires=[0])
            return qml.expval(qml.PauliZ(wires=0)), 0.3
        with self.assertRaisesRegex(QuantumFunctionError, 'must return either'):
            qf(par)

        # all EVs must be returned...
        @qml.qnode(self.dev2)
        def qf(x):
            qml.RX(x, wires=[0])
            ex = qml.expval(qml.PauliZ(wires=1))
            return qml.expval(qml.PauliZ(0))
        with self.assertRaisesRegex(QuantumFunctionError, 'All measured observables'):
            qf(par)

        # ...in the correct order
        @qml.qnode(self.dev2)
        def qf(x):
            qml.RX(x, wires=[0])
            ex = qml.expval(qml.PauliZ(wires=1))
            return qml.expval(qml.PauliZ(0)), ex
        with self.assertRaisesRegex(QuantumFunctionError, 'All measured observables'):
            qf(par)

        # gates must precede EVs
        @qml.qnode(self.dev2)
        def qf(x):
            qml.RX(x, wires=[0])
            ev = qml.expval(qml.PauliZ(wires=1))
            qml.RY(0.5, wires=[0])
            return ev
        with self.assertRaisesRegex(QuantumFunctionError, 'gates must precede'):
            qf(par)

        # a wire cannot be measured more than once
        @qml.qnode(self.dev2)
        def qf(x):
            qml.RX(x, wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliX(0))
        with self.assertRaisesRegex(QuantumFunctionError, 'can only be measured once'):
            qf(par)

        # device must have enough wires for the qfunc
        @qml.qnode(self.dev2)
        def qf(x):
            qml.RX(x, wires=[0])
            qml.CNOT(wires=[0, 2])
            return qml.expval(qml.PauliZ(0))
        with self.assertRaisesRegex(QuantumFunctionError, 'applied to invalid wire'):
            qf(par)

        # CV and discrete ops must not be mixed
        @qml.qnode(self.dev1)
        def qf(x):
            qml.RX(x, wires=[0])
            qml.Displacement(0.5, 0, wires=[0])
            return qml.expval(qml.PauliZ(0))
        with self.assertRaisesRegex(QuantumFunctionError, 'Continuous and discrete'):
            qf(par)

        # default plugin cannot execute CV operations, neither gates...
        @qml.qnode(self.dev1)
        def qf(x):
            qml.Displacement(0.5, 0, wires=[0])
            return qml.expval(qml.X(0))
        with self.assertRaisesRegex(DeviceError, 'Gate [a-zA-Z]+ not supported on device'):
            qf(par)

        # ...nor observables
        @qml.qnode(self.dev1)
        def qf(x):
            return qml.expval(qml.X(wires=0))
        with self.assertRaisesRegex(DeviceError, 'Observable [a-zA-Z]+ not supported on device'):
            qf(par)


    def test_jacobian_fail(self):
        "Tests that QNode.jacobian failures correctly raise exceptions."
        self.logTestName()
        par = 0.5

        #---------------------------------------------------------
        ## bad circuit

        # undifferentiable operation
        def qf(x):
            qml.BasisState(np.array([x, 0]), wires=[0,1])
            qml.RX(x, wires=[0])
            return qml.expval(qml.PauliZ(0))
        q = qml.QNode(qf, self.dev2)
        with self.assertRaisesRegex(ValueError, 'Cannot differentiate wrt parameter'):
            q.jacobian(par)

        # operation that does not support the 'A' method
        def qf(x):
            qml.RX(x, wires=[0])
            return qml.expval(qml.Hermitian(np.diag([x, 0]), 0))
        q = qml.QNode(qf, self.dev2)
        with self.assertRaisesRegex(ValueError, 'analytic gradient method cannot be used with'):
            q.jacobian(par, method='A')

        # bogus gradient method set
        def qf(x):
            qml.RX(x, wires=[0])
            return qml.expval(qml.PauliZ(0))

        # in non-cached mode, the grad method would be
        # recomputed and overwritten from the
        # bogus value 'J'. Caching stops this from happening.
        q = qml.QNode(qf, self.dev2, cache=True)

        q.evaluate([0.0])
        keys = q.grad_method_for_par.keys()
        if len(keys) > 0:
            k0 = [k for k in keys][0]

        q.grad_method_for_par[k0] = 'J'
        with self.assertRaisesRegex(ValueError, 'Unknown gradient method'):
            q.jacobian(par)

        #---------------------------------------------------------
        ## bad input parameters

        def qf_ok(x):
            qml.Rot(0.3, x, -0.2, wires=[0])
            return qml.expval(qml.PauliZ(0))

        # if indices wrt. which the gradient is taken are specified they must be unique
        q = qml.QNode(qf_ok, self.dev2)
        with self.assertRaisesRegex(ValueError, 'indices must be unique'):
            q.jacobian(par, which=[0,0])

        # gradient wrt. nonexistent parameters
        q = qml.QNode(qf_ok, self.dev2)
        with self.assertRaisesRegex(ValueError, 'Tried to compute the gradient wrt'):
            q.jacobian(par, which=[0,6])
        with self.assertRaisesRegex(ValueError, 'Tried to compute the gradient wrt'):
            q.jacobian(par, which=[1,-1])

        # unknown grad method
        q = qml.QNode(qf_ok, self.dev1)
        with self.assertRaisesRegex(ValueError, 'Unknown gradient method'):
            q.jacobian(par, method='unknown')

        # only order-1 and order-2 finite diff methods are available
        q = qml.QNode(qf_ok, self.dev1)
        with self.assertRaisesRegex(ValueError, 'Order must be 1 or 2'):
            q.jacobian(par, method='F', order=3)


    def test_qnode_fanout(self):
        "Tests that qnodes can compute the correct function when the same parameter is used in multiple gates."
        self.logTestName()

        def circuit(reused_param, other_param):
            qml.RX(reused_param, wires=[0])
            qml.RZ(other_param, wires=[0])
            qml.RX(reused_param, wires=[0])
            return qml.expval(qml.PauliZ(0))

        f = qml.QNode(circuit, self.dev1)

        for reused_param in thetas:
            for theta in thetas:
                other_param = theta ** 2 / 11
                y_eval = f(reused_param, other_param)
                Rx = Rotx(reused_param)
                Rz = Rotz(other_param)
                zero_state = np.array([1.,0.])
                final_state = (Rx @ Rz @ Rx @ zero_state)
                y_true = expZ(final_state)
                self.assertAlmostEqual(y_eval, y_true, delta=self.tol)


    def test_qnode_array_parameters(self):
        "Test that QNode can take arrays as input arguments, and that they interact properly with autograd."
        self.logTestName()

        @qml.qnode(self.dev1)
        def circuit_n1s(dummy1, array, dummy2):
            qml.RY(0.5 * array[0,1], wires=0)
            qml.RY(-0.5 * array[1,1], wires=0)
            return qml.expval(qml.PauliX(0))  # returns a scalar

        @qml.qnode(self.dev1)
        def circuit_n1v(dummy1, array, dummy2):
            qml.RY(0.5 * array[0,1], wires=0)
            qml.RY(-0.5 * array[1,1], wires=0)
            return qml.expval(qml.PauliX(0)),  # note the comma, returns a 1-vector

        @qml.qnode(self.dev2)
        def circuit_nn(dummy1, array, dummy2):
            qml.RY(0.5 * array[0,1], wires=0)
            qml.RY(-0.5 * array[1,1], wires=0)
            qml.RY(array[1,0], wires=1)
            return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliX(1))  # returns a 2-vector

        args = (0.46, np.array([[2., 3., 0.3], [7., 4., 2.1]]), -0.13)
        grad_target = (np.array(1.), np.array([[0.5,  0.43879, 0], [0, -0.43879, 0]]), np.array(-0.4))
        cost_target = 1.03257
        for circuit in [circuit_n1s, circuit_n1v, circuit_nn]:
            def cost(x, array, y):
                c = circuit(0.111, array, 4.5)
                if not np.isscalar(c):
                    c = c[0]  # get a scalar
                return c +0.5*array[0,0] +x -0.4*y

            cost_grad = qml.grad(cost, argnum=[0, 1, 2])
            self.assertAllAlmostEqual(cost(*args), cost_target, delta=self.tol)
            self.assertAllAlmostEqual(cost_grad(*args), grad_target, delta=self.tol)


    def test_array_parameters_evaluate(self):
        "Test that array parameters gives same result as positional arguments."
        self.logTestName()

        a, b, c = 0.5, 0.54, 0.3

        def ansatz(x, y, z):
            qml.QubitStateVector(np.array([1, 0, 1, 1])/np.sqrt(3), wires=[0, 1])
            qml.Rot(x, y, z, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))

        @qml.qnode(self.dev2)
        def circuit1(x, y, z):
            return ansatz(x, y, z)

        @qml.qnode(self.dev2)
        def circuit2(x, array):
            return ansatz(x, array[0], array[1])

        @qml.qnode(self.dev2)
        def circuit3(array):
            return ansatz(*array)

        positional_res = circuit1(a, b, c)
        positional_grad = circuit1.jacobian([a, b, c])

        array_res = circuit2(a, np.array([b, c]))
        array_grad = circuit2.jacobian([a, np.array([b, c])])

        list_res = circuit2(a, [b, c])
        list_grad = circuit2.jacobian([a, [b, c]])

        self.assertAllAlmostEqual(positional_res, array_res, delta=self.tol)
        self.assertAllAlmostEqual(positional_grad, array_grad, delta=self.tol)

        array_res = circuit3(np.array([a, b, c]))
        array_grad = circuit3.jacobian([np.array([a, b, c])])

        list_res = circuit3([a, b, c])
        list_grad = circuit3.jacobian([[a, b, c]])

        self.assertAllAlmostEqual(positional_res, array_res, delta=self.tol)
        self.assertAllAlmostEqual(positional_grad, array_grad, delta=self.tol)


    def test_multiple_expectation_different_wires(self):
        "Tests that qnodes return multiple expectation values."
        self.logTestName()

        a, b, c = 0.5, 0.54, 0.3

        @qml.qnode(self.dev2)
        def circuit(x, y, z):
            qml.RX(x, wires=[0])
            qml.RZ(y, wires=[0])
            qml.CNOT(wires=[0, 1])
            qml.RY(y, wires=[0])
            qml.RX(z, wires=[0])
            return qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(1))

        res = circuit(a, b, c)

        out_state = np.kron(Rotx(c), I) @ np.kron(Roty(b), I) @ CNOT \
            @ np.kron(Rotz(b), I) @ np.kron(Rotx(a), I) @ np.array([1, 0, 0, 0])

        ex0 = np.vdot(out_state, np.kron(Y, I) @ out_state)
        ex1 = np.vdot(out_state, np.kron(I, Z) @ out_state)
        ex = np.array([ex0, ex1])
        self.assertAllAlmostEqual(ex, res, delta=self.tol)


    def test_multiple_keywordargs_used(self):
        "Tests that qnodes use multiple keyword arguments."
        self.logTestName()

        def circuit(w, x=None, y=None):
            qml.RX(x, wires=[0])
            qml.RX(y, wires=[1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        circuit = qml.QNode(circuit, self.dev2)

        c = circuit(1., x=np.pi, y=np.pi)
        self.assertAllAlmostEqual(c, [-1., -1.], delta=self.tol)


    def test_multidimensional_keywordargs_used(self):
        "Tests that qnodes use multi-dimensional keyword arguments."
        self.logTestName()

        def circuit(w, x=None):
            qml.RX(x[0], wires=[0])
            qml.RX(x[1], wires=[1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        circuit = qml.QNode(circuit, self.dev2)

        c = circuit(1., x=[np.pi, np.pi])
        self.assertAllAlmostEqual(c, [-1., -1.], delta=self.tol)


    def test_keywordargs_for_wires(self):
        "Tests that wires can be passed as keyword arguments."
        self.logTestName()

        default_q = 0

        def circuit(x, q=default_q):
            qml.RX(x, wires=[q])
            return qml.expval(qml.PauliZ(q))

        circuit = qml.QNode(circuit, self.dev2)

        c = circuit(np.pi, q=1)
        self.assertEqual(circuit.queue[0].wires, [1])
        self.assertAlmostEqual(c, -1., delta=self.tol)

        c = circuit(np.pi)
        self.assertEqual(circuit.queue[0].wires, [default_q])
        self.assertAlmostEqual(c, -1., delta=self.tol)


    def test_keywordargs_used(self):
        "Tests that qnodes use keyword arguments."
        self.logTestName()

        def circuit(w, x=None):
            qml.RX(x, wires=[0])
            return qml.expval(qml.PauliZ(0))

        circuit = qml.QNode(circuit, self.dev1)

        c = circuit(1., x=np.pi)
        self.assertAlmostEqual(c, -1., delta=self.tol)


    def test_keywordarg_updated_in_multiple_calls(self):
        "Tests that qnodes update keyword arguments in consecutive calls."
        self.logTestName()

        def circuit(w, x=None):
            qml.RX(w, wires=[0])
            qml.RX(x, wires=[1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        circuit = qml.QNode(circuit, self.dev2)

        c1 = circuit(0.1, x=0.)
        c2 = circuit(0.1, x=np.pi)
        self.assertTrue(c1[1] != c2[1])


    def test_keywordarg_passes_through_classicalnode(self):
        "Tests that qnodes' keyword arguments pass through classical nodes."
        self.logTestName()

        def circuit(w, x=None):
            qml.RX(w, wires=[0])
            qml.RX(x, wires=[1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        circuit = qml.QNode(circuit, self.dev2)

        def classnode(w, x=None):
            return circuit(w, x=x)

        c = classnode(0., x=np.pi)
        self.assertAllAlmostEqual(c, [1., -1.], delta=self.tol)


class TestQNodeGradients:
    """Qnode gradient tests."""

    @pytest.mark.parametrize('s', b_shapes)
    def test_multidim_array(self, s, tol):
        """Tests that arguments which are multidimensional arrays are
        properly evaluated and differentiated in QNodes."""

        multidim_array = np.reshape(b, s)
        def circuit(w):
            qml.RX(w[np.unravel_index(0,s)], wires=0) # b[0]
            qml.RX(w[np.unravel_index(1,s)], wires=1) # b[1]
            qml.RX(w[np.unravel_index(2,s)], wires=2) # ...
            qml.RX(w[np.unravel_index(3,s)], wires=3)
            qml.RX(w[np.unravel_index(4,s)], wires=4)
            qml.RX(w[np.unravel_index(5,s)], wires=5)
            qml.RX(w[np.unravel_index(6,s)], wires=6)
            qml.RX(w[np.unravel_index(7,s)], wires=7)
            return tuple(qml.expval(qml.PauliZ(idx)) for idx in range(len(b)))

        dev = qml.device('default.qubit', wires=8)
        circuit = qml.QNode(circuit, dev)

        # circuit evaluations
        circuit_output = circuit(multidim_array)
        expected_output = np.cos(b)
        assert np.allclose(circuit_output, expected_output, atol=tol, rtol=0)

        # circuit jacobians
        circuit_jacobian = circuit.jacobian([multidim_array])
        expected_jacobian = -np.diag(np.sin(b))
        assert np.allclose(circuit_jacobian, expected_jacobian, atol=tol, rtol=0)

    def test_qnode_cv_gradient_methods(self):
        """Tests the gradient computation methods on CV circuits."""
        # we can only use the 'A' method on parameters which only affect gaussian operations
        # that are not succeeded by nongaussian operations

        par = [0.4, -2.3]
        dev = qml.device('default.qubit', wires=2)

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
            #qml.Kerr(0.4, [0])
            return qml.expval(qml.X(0)), qml.expval(qml.X(1))

        check_methods(qf, {0:'F', 1:'A'})

        def qf(x, y):
            qml.Displacement(x, 0, wires=[0])
            qml.CubicPhase(0.2, wires=[0])  # nongaussian succeeding x
            qml.Squeezing(0.3, x, wires=[1])  # x affects gates on both wires, y unused
            qml.Rotation(1.3, wires=[1])
            return qml.expval(qml.X(0)), qml.expval(qml.X(1))

        check_methods(qf, {0:'F'})

        def qf(x, y):
            qml.Displacement(x, 0, wires=[0])
            qml.Displacement(1.2, y, wires=[0])
            qml.Beamsplitter(0.2, 1.7, wires=[0, 1])
            qml.Rotation(1.9, wires=[0])
            qml.Kerr(0.3, wires=[1])  # nongaussian succeeding both x and y due to the beamsplitter
            return qml.expval(qml.X(0)), qml.expval(qml.X(1))

        check_methods(qf, {0:'F', 1:'F'})

        def qf(x, y):
            qml.Kerr(y, wires=[1])
            qml.Displacement(x, 0, wires=[0])
            qml.Beamsplitter(0.2, 1.7, wires=[0, 1])
            return qml.expval(qml.X(0)), qml.expval(qml.X(1))

        check_methods(qf, {0:'A', 1:'F'})

    def test_qnode_gradient_multiple_gate_parameters(self, tol):
        """Tests that gates with multiple free parameters yield correct gradients."""
        par = [0.5, 0.3, -0.7]

        def qf(x, y, z):
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[0])
            qml.RY(-0.2, wires=[0])
            return qml.expval(qml.PauliZ(0))

        dev = qml.device('default.qubit', wires=1)
        q = qml.QNode(qf, dev)
        value = q(*par)
        grad_A = q.jacobian(par, method='A')
        grad_F = q.jacobian(par, method='F')

        # analytic method works for every parameter
        assert q.grad_method_for_par == {0:'A', 1:'A', 2:'A'}
        # gradient has the correct shape and every element is nonzero
        assert grad_A.shape == (1,3)
        assert np.count_nonzero(grad_A) == 3
        # the different methods agree
        assert np.allclose(grad_A, grad_F, atol=tol, rtol=0)


    def test_qnode_gradient_repeated_gate_parameters(self, tol):
        """Tests that repeated use of a free parameter in a
        multi-parameter gate yield correct gradients."""
        par = [0.8, 1.3]

        def qf(x, y):
            qml.RX(np.pi/4, wires=[0])
            qml.Rot(y, x, 2*x, wires=[0])
            return qml.expval(qml.PauliX(0))

        dev = qml.device('default.qubit', wires=1)
        q = qml.QNode(qf, dev)
        grad_A = q.jacobian(par, method='A')
        grad_F = q.jacobian(par, method='F')

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

        dev = qml.device('default.qubit', wires=1)
        q = qml.QNode(qf, dev)
        grad = q.jacobian(par)
        grad_F = q.jacobian(par, method='F')

        # par[0] can use the 'A' method, par[1] cannot
        assert q.grad_method_for_par == {0:'A', 1:'F'}
        # the different methods agree
        assert np.allclose(grad, grad_F, atol=tol, rtol=0)

    def test_array_parameters_autograd(self, tol):
        """Test that gradients of array parameters give
        same results as positional arguments."""

        a, b, c = 0.5, 0.54, 0.3

        def ansatz(x, y, z):
            qml.QubitStateVector(np.array([1, 0, 1, 1])/np.sqrt(3), wires=[0, 1])
            qml.Rot(x, y, z, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        def circuit1(x, y, z):
            return ansatz(x, y, z)

        def circuit2(x, array):
            return ansatz(x, array[0], array[1])

        def circuit3(array):
            return ansatz(*array)

        dev = qml.device('default.qubit', wires=2)
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
        dw0dx = 2/3*np.sin(x)*np.sin(y)
        dw0dy = 1/3*(np.sin(y)-2*np.cos(x)*np.cos(y))
        dw0dz = 0

        dw1dx = -2/3*np.cos(x)*np.sin(y)
        dw1dy = -2/3*np.cos(y)*np.sin(x)
        dw1dz = 0

        return np.array([[dw0dx, dw0dy, dw0dz],
                         [dw1dx, dw1dy, dw1dz]])

    def test_multiple_expectation_jacobian_positional(self, tol):
        """Tests that qnodes using positional arguments return
        correct gradients for multiple expectation values."""
        a, b, c = 0.5, 0.54, 0.3

        def circuit(x, y, z):
            qml.QubitStateVector(np.array([1, 0, 1, 1])/np.sqrt(3), wires=[0, 1])
            qml.Rot(x, y, z, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))

        dev = qml.device('default.qubit', wires=2)
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
        res = np.stack([jac0(a,b,c), jac1(a,b,c), jac2(a,b,c)]).T

        assert np.allclose(self.expected_jacobian(a, b, c), res, atol=tol, rtol=0)

    def test_multiple_expectation_jacobian_array(self, tol):
        """Tests that qnodes using an array argument return correct gradients
        for multiple expectation values."""
        a, b, c = 0.5, 0.54, 0.3

        def circuit(weights):
           qml.QubitStateVector(np.array([1, 0, 1, 1])/np.sqrt(3), wires=[0, 1])
           qml.Rot(weights[0], weights[1], weights[2], wires=0)
           qml.CNOT(wires=[0, 1])
           return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))

        dev = qml.device('default.qubit', wires=2)
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
           qml.QubitStateVector(np.array([1, 0, 1, 1])/np.sqrt(3), wires=[0, 1])
           qml.Rot(weights[0], weights[1], x, wires=0)
           qml.CNOT(wires=[0, 1])
           return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))

        dev = qml.device('default.qubit', wires=2)
        circuit1 = qml.QNode(circuit1, dev)

        def circuit2(weights):
           qml.QubitStateVector(np.array([1, 0, 1, 1])/np.sqrt(3), wires=[0, 1])
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

        dev = qml.device('default.qubit', wires=3)
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

        dev = qml.device('default.qubit', wires=2)
        circuit2 = qml.QNode(circuit2, dev)

        a = 0.7418
        b = -5.
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

        dev = qml.device('default.qubit', wires=2)
        circuit3 = qml.QNode(circuit3, dev)

        a = 0.7418
        b = -5.
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

        dev = qml.device('default.qubit', wires=2)
        circuit4 = qml.QNode(circuit4, dev)

        a = 0.7418
        b = -5.
        c = np.pi / 7
        circuit_output = circuit4(a, b, c)
        expected_output = np.array([[np.cos(b), np.cos(c)]])
        assert np.allclose(circuit_output, expected_output, atol=tol, rtol=0)

        # circuit jacobians
        circuit_jacobian = circuit4.jacobian([a, b, c])
        expected_jacobian = np.array([[0., -np.sin(b), 0.],
                                      [0., 0., -np.sin(c)]])
        assert np.allclose(circuit_jacobian, expected_jacobian, atol=tol, rtol=0)

    def test_differentiate_positional_multidim(self, tol):
        """Tests that all positional arguments are differentiated
        when they are multidimensional."""

        def circuit(a, b):
            qml.RX(a[0], wires=0)
            qml.RX(a[1], wires=1)
            qml.RX(b[2, 1], wires=2)
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2))

        dev = qml.device('default.qubit', wires=3)
        circuit = qml.QNode(circuit, dev)

        a = np.array([-np.sqrt(2), -0.54])
        b = np.array([np.pi / 7] * 6).reshape([3, 2])
        circuit_output = circuit(a, b)
        expected_output = np.cos(np.array([[a[0], a[1], b[-1, 0]]]))
        assert np.allclose(circuit_output, expected_output, atol=tol, rtol=0)

        # circuit jacobians
        circuit_jacobian = circuit.jacobian([a, b])
        expected_jacobian = np.array([[-np.sin(a[0])] + [0.] * 7,  # expval 0
                                      [0., -np.sin(a[1])] + [0.] * 6,  # expval 1
                                      [0.] * 2 + [0.] * 5 + [-np.sin(b[2, 1])]])  # expval 2
        assert np.allclose(circuit_jacobian, expected_jacobian, atol=tol, rtol=0)


class TestQNodeCacheing:
    """Tests for the QNode construction caching"""

    def test_no_caching(self):
        """Test that the circuit structure changes on
        subsequent evalutions with caching turned off
        """
        dev = qml.device('default.qubit', wires=2)

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
        dev = qml.device('default.qubit', wires=2)

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

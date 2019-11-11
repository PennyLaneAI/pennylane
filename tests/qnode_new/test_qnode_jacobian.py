# Copyright 2019 Xanadu Quantum Technologies Inc.

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
Unit tests for the :mod:`pennylane` :class:`JacobianQNode` class.
"""

import pytest
import numpy as np

import pennylane as qml
from pennylane._device import Device
from pennylane.operation import CVObservable
from pennylane.qnode_new.qnode import QuantumFunctionError
from pennylane.qnode_new.jacobian import JacobianQNode as JNode


thetas = np.linspace(-2*np.pi, 2*np.pi, 8)


@pytest.fixture(scope="function")
def operable_mock_device_2_wires(monkeypatch):
    """A mock instance of the abstract Device class that can support qfuncs."""

    dev = Device
    with monkeypatch.context() as m:
        m.setattr(dev, '__abstractmethods__', frozenset())
        m.setattr(dev, 'capabilities', lambda self: {"model": "qubit"})
        m.setattr(dev, 'operations', ["BasisState", "RX", "RY", "CNOT", "Rot", "PhaseShift"])
        m.setattr(dev, 'observables', ["PauliX", "PauliY", "PauliZ"])
        m.setattr(dev, 'reset', lambda self: None)
        m.setattr(dev, 'apply', lambda self, x, y, z: None)
        m.setattr(dev, 'expval', lambda self, x, y, z: 1)
        yield Device(wires=2)


@pytest.fixture(scope="function")
def operable_mock_CV_device_2_wires(monkeypatch):
    """A mock instance of the abstract Device class that can support qfuncs."""

    dev = Device
    with monkeypatch.context() as m:
        m.setattr(dev, '__abstractmethods__', frozenset())
        m.setattr(dev, 'capabilities', lambda self: {"model": "cv"})
        m.setattr(dev, 'operations', ["Displacement", "CubicPhase", "Squeezing", "Rotation", "Kerr", "Beamsplitter"])
        m.setattr(dev, 'observables', ["X", "NumberOperator", "PolyXP"])
        m.setattr(dev, 'reset', lambda self: None)
        m.setattr(dev, 'apply', lambda self, x, y, z: None)
        m.setattr(dev, 'expval', lambda self, x, y, z: 1)
        yield Device(wires=2)


@pytest.fixture(scope="module")
def gaussian_dev():
    return qml.device('default.gaussian', wires=2)


class TestAJacobianQNodeDetails:
    """Test configuration details of the autograd interface"""

    def test_interface_str(self, qubit_device_2_wires):
        """Test that the interface string is correctly identified
        as None"""
        def circuit(x, y, z):
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))

        circuit = JNode(circuit, qubit_device_2_wires)
        assert circuit.interface == None


class TestJNodeExceptions:
    """Tests that JNode.jacobian raises proper errors."""

    def test_gradient_of_sample(self, operable_mock_device_2_wires):
        """Differentiation of a sampled output."""

        def circuit(x):
            qml.RX(x, wires=[0])
            return qml.sample(qml.PauliZ(0)), qml.sample(qml.PauliX(1))

        node = JNode(circuit, operable_mock_device_2_wires)

        with pytest.raises(QuantumFunctionError,
                           match="Circuits that include sampling can not be differentiated."):
            node.jacobian(1.0)

    def test_nondifferentiable_operator(self, operable_mock_device_2_wires):
        """Differentiating wrt. a parameter
        that appears as an argument to a nondifferentiable operator."""

        def circuit(x):
            qml.BasisState(np.array([x, 0]), wires=[0, 1])  # not differentiable
            qml.RX(x, wires=[0])
            return qml.expval(qml.PauliZ(0))

        node = JNode(circuit, operable_mock_device_2_wires)

        with pytest.raises(ValueError, match="Cannot differentiate wrt. parameter"):
            node.jacobian(0.5)

    def test_operator_not_supporting_pd_analytic(self, operable_mock_device_2_wires):
        """Differentiating wrt. a parameter that appears
        as an argument to an operation that does not support parameter-shift derivatives."""

        def circuit(x):
            qml.RX(x, wires=[0])
            return qml.expval(qml.Hermitian(np.diag([x, 0]), 0))

        node = JNode(circuit, operable_mock_device_2_wires)

        with pytest.raises(ValueError, match="parameter-shift gradient method cannot be used with"):
            node.jacobian(0.5, method="A")

    def test_bogus_gradient_method_set(self, operable_mock_device_2_wires):
        """The gradient method set is bogus."""

        def circuit(x):
            qml.RX(x, wires=[0])
            return qml.expval(qml.PauliZ(0))

        # in mutable mode, the grad method would be
        # recomputed and overwritten from the
        # bogus value 'J'. Caching stops this from happening.
        node = JNode(circuit, operable_mock_device_2_wires, mutable=False)

        node.evaluate([0.0], {})
        node.par_to_grad_method[0] = "J"

        with pytest.raises(ValueError, match="Unknown gradient method"):
            node.jacobian(0.5)

    def test_indices_not_unique(self, operable_mock_device_2_wires):
        """The Jacobian is requested for non-unique indices."""

        def circuit(x):
            qml.Rot(0.3, x, -0.2, wires=[0])
            return qml.expval(qml.PauliZ(0))

        node = JNode(circuit, operable_mock_device_2_wires)

        with pytest.raises(ValueError, match="Parameter indices must be unique."):
            node.jacobian(0.5, wrt=[0, 0])

    def test_indices_nonexistant(self, operable_mock_device_2_wires):
        """ The Jacobian is requested for non-existant parameters."""

        def circuit(x):
            qml.Rot(0.3, x, -0.2, wires=[0])
            return qml.expval(qml.PauliZ(0))

        node = JNode(circuit, operable_mock_device_2_wires)

        with pytest.raises(ValueError, match="Tried to compute the gradient with respect to"):
            node.jacobian(0.5, wrt=[0, 6])

        with pytest.raises(ValueError, match="Tried to compute the gradient with respect to"):
            node.jacobian(0.5, wrt=[1, -1])

    def test_unknown_gradient_method(self, operable_mock_device_2_wires):
        """ The gradient method is unknown."""

        def circuit(x):
            qml.Rot(0.3, x, -0.2, wires=[0])
            return qml.expval(qml.PauliZ(0))

        node = JNode(circuit, operable_mock_device_2_wires)

        with pytest.raises(ValueError, match="Unknown gradient method"):
            node.jacobian(0.5, method="unknown")

    def test_wrong_order_in_finite_difference(self, operable_mock_device_2_wires):
        """Finite difference are attempted with wrong order."""

        def circuit(x):
            qml.Rot(0.3, x, -0.2, wires=[0])
            return qml.expval(qml.PauliZ(0))

        node = JNode(circuit, operable_mock_device_2_wires)

        with pytest.raises(ValueError, match="Order must be 1 or 2"):
            node.jacobian(0.5, method="F", options={'order': 3})

    def test_transform_observable_incorrect_heisenberg_size(self):
        """The number of dimensions of a CV observable Heisenberg representation does
        not match the ev_order attribute."""

        class P(CVObservable):
            """Dummy CV observable with incorrect ev_order"""
            num_wires = 1
            num_params = 0
            par_domain = None
            ev_order = 2

            @staticmethod
            def _heisenberg_rep(p):
                return np.array([0, 1, 0])

        dev = qml.device("default.gaussian", wires=1)
        def circuit(x):
            qml.Displacement(x, 0.1, wires=0)
            return qml.expval(P(0))

        node = JNode(circuit, dev)

        with pytest.raises(QuantumFunctionError, match="Mismatch between polynomial order"):
            node.jacobian([0.5])



class TestJNodeBestMethod:
    """
    Test different flows of _best_method TODO more
    """
    def test_best_method_with_non_gaussian_successors(self, tol, gaussian_device_2_wires):
        """Tests that the analytic differentiation method is allowed and matches numerical
        differentiation if a non-Gaussian gate is not succeeded by an observable."""

        def circuit(x):
            qml.Squeezing(x, 0, wires=[0])
            qml.Beamsplitter(np.pi/4, 0, wires=[0, 1])
            qml.Kerr(0.54, wires=[1])
            return qml.expval(qml.NumberOperator(0))

        node = JNode(circuit, gaussian_device_2_wires, properties={'vis_check': False})

        res = node.jacobian([0.321], wrt=[0], method="A")
        expected = node.jacobian([0.321], method="F")
        assert res == pytest.approx(expected, abs=tol)

    def test_best_method_with_gaussian_successors_fails(self, operable_mock_CV_device_2_wires):
        """Tests that the parameter-shift differentiation method is not allowed
        if a non-gaussian gate is between a differentiable gaussian gate and an observable."""

        def circuit(x):
            qml.Squeezing(x, 0, wires=[0])
            qml.Beamsplitter(np.pi/4, 0, wires=[0, 1])
            qml.Kerr(0.54, wires=[1])
            return qml.expval(qml.NumberOperator(1))

        node = JNode(circuit, operable_mock_CV_device_2_wires)

        with pytest.raises(ValueError, match="parameter-shift gradient method cannot be used with"):
            node.jacobian([0.321], method="A")

    def test_cv_gradient_methods(self, operable_mock_CV_device_2_wires):
        """Tests the gradient computation methods on CV circuits."""
        # we can only use the "A" method on parameters which only affect gaussian operations
        # that are not succeeded by nongaussian operations

        par = [0.4, -2.3]

        def check_methods(qf, d):
            q = JNode(qf, operable_mock_CV_device_2_wires)
            q._construct(par, {})
            assert q.par_to_grad_method == d

        def qf(x, y):
            qml.Displacement(x, 0, wires=[0])
            qml.CubicPhase(0.2, wires=[0])
            qml.Squeezing(0.3, y, wires=[1])
            qml.Rotation(1.3, wires=[1])
            # nongaussian succeeding x but not y
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
            qml.Displacement(1.2, y, wires=[1])
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


class PolyN(qml.ops.PolyXP):
    "Mimics NumberOperator using the arbitrary 2nd order observable interface. Results should be identical."
    def __init__(self, wires):
        hbar = 2
        q = np.diag([-0.5, 0.5/hbar, 0.5/hbar])
        super().__init__(q, wires=wires)
        self.name = 'PolyXP'

cv_ops = [getattr(qml.ops, name) for name in qml.ops._cv__ops__]
analytic_cv_ops = [cls for cls in cv_ops if cls.supports_analytic]


class TestJNodeJacobianCV:
    """JNode.jacobian tests for CV circuits."""

    def test_keywordarg_second_order_cv(self, tol):
        """Non-differentiable keyword arguments with a second order CV expectation value."""
        dev = qml.device("default.gaussian", wires=3)
        def circuit(x, *, k=0.0):
            qml.Displacement(x, 0, wires=0)
            qml.Rotation(k, wires=0)
            return qml.expval(qml.PolyXP(np.diag([0, 1, 0]), wires=0))  # X^2

        node = JNode(circuit, dev)
        par = [0.62]
        aux = {'k': 0.4}

        # circuit jacobians
        grad_A = node.jacobian(par, aux, method="A")
        grad_F = node.jacobian(par, aux, method="F")
        expected = np.array([[8 * par[0] * np.cos(aux['k']) ** 2]])
        assert grad_A == pytest.approx(grad_F, abs=tol)
        assert grad_A == pytest.approx(expected, abs=tol)

    def test_keywordarg_with_positional_arg_immutable_second_order_cv(self, tol):
        """Non-differentiable keyword arguments appear in the same op with differentiable arguments,
        qfunc is immutable so kwargs are passed as Variables."""
        dev = qml.device("default.gaussian", wires=1)
        def circuit(x, *, k=0.0):
            qml.Displacement(0.5, 0, wires=0)
            qml.Squeezing(x, k, wires=0)
            return qml.expval(qml.X(0))

        node = JNode(circuit, dev, mutable=False)
        par = [0.39]
        aux = {'k': -0.7}

        # circuit jacobians
        grad_A = node.jacobian(par, aux, method="A", options={'force_order2': True})
        grad_F = node.jacobian(par, aux, method="F")
        assert grad_A == pytest.approx(grad_F, abs=tol)

    @pytest.mark.parametrize('O', [qml.ops.X, qml.ops.NumberOperator, PolyN])
    @pytest.mark.parametrize('G', analytic_cv_ops)
    def test_cv_gradients_gaussian_circuit(self, G, O, gaussian_dev, tol):
        """Tests that the gradients of circuits of gaussian gates match between the finite difference and analytic methods."""

        tol = 1e-5
        par = [0.4]
        def circuit(x):
            args = [0.3] * G.num_params
            args[0] = x
            qml.Displacement(0.5, 0, wires=0)
            G(*args, wires=range(G.num_wires))
            qml.Beamsplitter(1.3, -2.3, wires=[0, 1])
            qml.Displacement(-0.5, 0.1, wires=0)
            qml.Squeezing(0.5, -1.5, wires=0)
            qml.Rotation(-1.1, wires=0)
            return qml.expval(O(wires=0))

        q = JNode(circuit, gaussian_dev)
        val = q.evaluate(par, {})

        grad_F  = q.jacobian(par, method="F")
        grad_A2 = q.jacobian(par, method="A", options={'force_order2': True})
        if O.ev_order == 1:
            grad_A = q.jacobian(par, method="A")
            # the different methods agree
            assert grad_A == pytest.approx(grad_F, abs=tol)

        # analytic method works for every parameter
        assert q.par_to_grad_method == {0:"A"}
        # the different methods agree
        assert grad_A2 == pytest.approx(grad_F, abs=tol)


    def test_cv_gradients_multiple_gate_parameters(self, gaussian_dev, tol):
        "Tests that gates with multiple free parameters yield correct gradients."
        par = [0.4, -0.3, -0.7, 0.2]

        def qf(r0, phi0, r1, phi1):
            qml.Squeezing(r0, phi0, wires=[0])
            qml.Squeezing(r1, phi1, wires=[0])
            return qml.expval(qml.NumberOperator(0))

        q = JNode(qf, gaussian_dev)
        grad_F = q.jacobian(par, method="F")
        grad_A = q.jacobian(par, method="A")
        grad_A2 = q.jacobian(par, method="A", options={'force_order2': True})

        # analytic method works for every parameter
        assert q.par_to_grad_method == {i:"A" for i in range(4)}
        # the different methods agree
        assert grad_A == pytest.approx(grad_F, abs=tol)
        assert grad_A2 == pytest.approx(grad_F, abs=tol)

        # check against the known analytic formula
        r0, phi0, r1, phi1 = par
        dn = np.zeros([4])
        dn[0] = np.cosh(2 * r1) * np.sinh(2 * r0) + np.cos(phi0 - phi1) * np.cosh(2 * r0) * np.sinh(2 * r1)
        dn[1] = -0.5 * np.sin(phi0 - phi1) * np.sinh(2 * r0) * np.sinh(2 * r1)
        dn[2] = np.cos(phi0 - phi1) * np.cosh(2 * r1) * np.sinh(2 * r0) + np.cosh(2 * r0) * np.sinh(2 * r1)
        dn[3] = 0.5 * np.sin(phi0 - phi1) * np.sinh(2 * r0) * np.sinh(2 * r1)

        assert dn[np.newaxis, :] == pytest.approx(grad_F, abs=tol)

    def test_cv_gradients_repeated_gate_parameters(self, gaussian_dev, tol):
        "Tests that repeated use of a free parameter in a multi-parameter gate yield correct gradients."
        par = [0.2, 0.3]

        def qf(x, y):
            qml.Displacement(x, 0, wires=[0])
            qml.Squeezing(y, -1.3*y, wires=[0])
            return qml.expval(qml.X(0))

        q = JNode(qf, gaussian_dev)
        grad_F = q.jacobian(par, method="F")
        grad_A = q.jacobian(par, method="A")
        grad_A2 = q.jacobian(par, method="A", options={'force_order2': True})

        # analytic method works for every parameter
        assert q.par_to_grad_method == {0:"A", 1:"A"}
        # the different methods agree
        assert grad_A == pytest.approx(grad_F, abs=tol)
        assert grad_A2 == pytest.approx(grad_F, abs=tol)


    def test_cv_gradients_parameters_inside_array(self, gaussian_dev, tol):
        "Tests that free parameters inside an array passed to an Operation yield correct gradients."
        par = [0.4, 1.3]

        def qf(x, y):
            qml.Displacement(0.5, 0, wires=[0])
            qml.Squeezing(x, 0, wires=[0])
            M = np.zeros((5, 5), dtype=object)
            M[1,1] = y
            M[1,2] = 1.0
            M[2,1] = 1.0
            return qml.expval(qml.PolyXP(M, [0, 1]))

        q = JNode(qf, gaussian_dev)
        grad = q.jacobian(par)
        grad_F = q.jacobian(par, method="F")
        grad_A = q.jacobian(par, method='B')
        grad_A2 = q.jacobian(par, method='B', options={'force_order2': True})

        # par[0] can use the "A" method, par[1] cannot
        assert q.par_to_grad_method == {0:"A", 1:"F"}
        # the different methods agree
        assert grad == pytest.approx(grad_F, abs=tol)


    def test_fanout_multiple_params(self, qubit_device_1_wire, tol):
        "Tests that the correct gradient is computed for qnodes which use the same parameter in multiple gates."

        from pennylane.plugins.default_qubit import Rotx as Rx, Roty as Ry, Rotz as Rz

        def expZ(state):
            return np.abs(state[0]) ** 2 - np.abs(state[1]) ** 2

        extra_param = 0.31
        def circuit(reused_param, other_param):
            qml.RX(extra_param, wires=[0])
            qml.RY(reused_param, wires=[0])
            qml.RZ(other_param, wires=[0])
            qml.RX(reused_param, wires=[0])
            return qml.expval(qml.PauliZ(0))

        f = JNode(circuit, qubit_device_1_wire)
        zero_state = np.array([1., 0.])

        for reused_p in thetas:
            reused_p = reused_p ** 3 / 19
            for other_p in thetas:
                other_p = other_p ** 2 / 11

                # analytic gradient
                grad_A = f.jacobian([reused_p, other_p])

                # manual gradient
                grad_true0 = (expZ(Rx(reused_p) @ Rz(other_p) @ Ry(reused_p + np.pi / 2) @ Rx(extra_param) @ zero_state) \
                             -expZ(Rx(reused_p) @ Rz(other_p) @ Ry(reused_p - np.pi / 2) @ Rx(extra_param) @ zero_state)) / 2
                grad_true1 = (expZ(Rx(reused_p + np.pi / 2) @ Rz(other_p) @ Ry(reused_p) @ Rx(extra_param) @ zero_state) \
                             -expZ(Rx(reused_p - np.pi / 2) @ Rz(other_p) @ Ry(reused_p) @ Rx(extra_param) @ zero_state)) / 2
                grad_true = grad_true0 + grad_true1 # product rule

                assert grad_A[0, 0] == pytest.approx(grad_true, abs=tol)


    def test_cv_gradient_fanout(self, gaussian_dev, tol):
        "Tests that CV qnodes can compute the correct gradient when the same parameter is used in multiple gates."
        par = [0.5, 1.3]

        def circuit(x, y):
            qml.Displacement(x, 0, wires=[0])
            qml.Rotation(y, wires=[0])
            qml.Displacement(0, x, wires=[0])
            return qml.expval(qml.X(0))

        q = JNode(circuit, gaussian_dev)
        grad_F = q.jacobian(par, method="F")
        grad_A = q.jacobian(par, method="A")
        grad_A2 = q.jacobian(par, method="A", options={'force_order2': True})

        # analytic method works for every parameter
        assert q.par_to_grad_method == {0:"A", 1:"A"}
        # the different methods agree
        assert grad_A == pytest.approx(grad_F, abs=tol)
        assert grad_A2 == pytest.approx(grad_F, abs=tol)


    @pytest.mark.parametrize('name', qml.ops._cv__ops__)
    def test_CVOperation_with_heisenberg_and_no_params(self, name, gaussian_dev, tol):
        """An integration test for CV gates that support analytic differentiation
        if succeeding the gate to be differentiated, but cannot be differentiated
        themselves (for example, they may be Gaussian but accept no parameters).

        This ensures that, assuming their _heisenberg_rep is defined, the quantum
        gradient analytic method can still be used, and returns the correct result.
        """

        cls = getattr(qml.ops, name)
        if cls.supports_heisenberg and (not cls.supports_analytic):
            U = np.array([[0.51310276+0.81702166j, 0.13649626+0.22487759j],
                          [0.26300233+0.00556194j, -0.96414101-0.03508489j]])

            if cls.num_wires <= 0:
                w = list(range(2))
            else:
                w = list(range(cls.num_wires))

            def circuit(x):
                qml.Displacement(x, 0, wires=0)

                if cls.par_domain == "A":
                    cls(U, wires=w)
                else:
                    cls(wires=w)
                return qml.expval(qml.X(0))

            qnode = JNode(circuit, gaussian_dev)
            grad_F = qnode.jacobian(0.5, method="F")
            grad_A = qnode.jacobian(0.5, method="A")
            grad_A2 = qnode.jacobian(0.5, method="A", options={'force_order2': True})

            # par[0] can use the "A" method
            assert qnode.par_to_grad_method == {0: "A"}

            # the different methods agree
            assert grad_A == pytest.approx(grad_F, abs=tol)
            assert grad_A2 == pytest.approx(grad_F, abs=tol)



class TestJNodeJacobianQubit:
    """JNode.jacobian tests for qubit circuits."""

    @pytest.mark.parametrize("shape", [(8,), (8, 1), (4, 2), (2, 2, 2), (2, 1, 2, 1, 2)])
    def test_multidim_array_parameter(self, shape, tol):
        """Tests that arguments which are multidimensional arrays are
        properly evaluated and differentiated in JNodes."""

        n = np.prod(shape)
        base_array = np.linspace(-1.0, 1.0, n)
        multidim_array = np.reshape(base_array, shape)

        def circuit(w):
            for k in range(n):
                qml.RX(w[np.unravel_index(k, shape)], wires=k)  # base_array[k]
            return tuple(qml.expval(qml.PauliZ(idx)) for idx in range(n))

        dev = qml.device("default.qubit", wires=n)
        circuit = JNode(circuit, dev)

        # circuit evaluations
        circuit_output = circuit(multidim_array)
        expected_output = np.cos(base_array)
        assert circuit_output == pytest.approx(expected_output, abs=tol)

        # circuit jacobians
        circuit_jacobian = circuit.jacobian([multidim_array])
        expected_jacobian = -np.diag(np.sin(base_array))
        assert circuit_jacobian == pytest.approx(expected_jacobian, abs=tol)

    def test_gradient_gate_with_multiple_parameters(self, tol, qubit_device_1_wire):
        """Tests that gates with multiple free parameters yield correct gradients."""
        par = [0.5, 0.3, -0.7]

        def qf(x, y, z):
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[0])
            qml.RY(-0.2, wires=[0])
            return qml.expval(qml.PauliZ(0))

        q = JNode(qf, qubit_device_1_wire)
        value = q(*par)
        grad_A = q.jacobian(par, method="A")
        grad_F = q.jacobian(par, method="F")

        # analytic method works for every parameter
        assert q.par_to_grad_method == {0: "A", 1: "A", 2: "A"}
        # gradient has the correct shape and every element is nonzero
        assert grad_A.shape == (1, 3)
        assert np.count_nonzero(grad_A) == 3
        # the different methods agree
        assert grad_A == pytest.approx(grad_F, abs=tol)

    def test_gradient_gate_with_two_parameters(self, tol, gaussian_dev):
        """Test that a gate with two parameters yields
        correct gradients"""
        def qf(r0, phi0, r1, phi1):
            qml.Squeezing(r0, phi0, wires=[0])
            qml.Squeezing(r1, phi1, wires=[0])
            return qml.expval(qml.NumberOperator(0))

        q = JNode(qf, gaussian_dev)

        par = [0.543, 0.123, 0.654, -0.629]

        grad_A = q.jacobian(par, method="A")
        grad_F = q.jacobian(par, method="F")

        # the different methods agree
        assert grad_A == pytest.approx(grad_F, abs=tol)

    def test_gradient_repeated_gate_parameters(self, tol, qubit_device_1_wire):
        """Tests that repeated use of a free parameter in a
        multi-parameter gate yield correct gradients."""
        par = [0.8, 1.3]

        def qf(x, y):
            qml.RX(np.pi / 4, wires=[0])
            qml.Rot(y, x, 2 * x, wires=[0])
            return qml.expval(qml.PauliX(0))

        q = JNode(qf, qubit_device_1_wire)
        grad_A = q.jacobian(par, method="A")
        grad_F = q.jacobian(par, method="F")

        # the different methods agree
        assert grad_A == pytest.approx(grad_F, abs=tol)

    def test_gradient_parameters_inside_array(self, tol, qubit_device_1_wire):
        """Tests that free parameters inside an array passed to
        an Operation yield correct gradients."""
        par = [0.8, 1.3]

        def qf(x, y):
            qml.RX(x, wires=[0])
            qml.RY(x, wires=[0])
            return qml.expval(qml.Hermitian(np.diag([y, 1]), 0))

        q = JNode(qf, qubit_device_1_wire)
        grad = q.jacobian(par)
        grad_F = q.jacobian(par, method="F")

        # par[0] can use the "A" method, par[1] cannot
        assert q.par_to_grad_method == {0: "A", 1: "F"}
        # the different methods agree
        assert grad == pytest.approx(grad_F, abs=tol)

    def test_keywordarg_not_differentiated(self, tol, qubit_device_2_wires):
        """Tests that qnodes do not differentiate w.r.t. keyword arguments."""
        par = np.array([0.5, 0.54])

        def circuit1(weights, x=0.3):
            qml.QubitStateVector(np.array([1, 0, 1, 1]) / np.sqrt(3), wires=[0, 1])
            qml.Rot(weights[0], weights[1], x, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))

        circuit1 = JNode(circuit1, qubit_device_2_wires)

        def circuit2(weights):
            qml.QubitStateVector(np.array([1, 0, 1, 1]) / np.sqrt(3), wires=[0, 1])
            qml.Rot(weights[0], weights[1], 0.3, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))

        circuit2 = JNode(circuit2, qubit_device_2_wires)

        res1 = circuit1.jacobian([par])
        res2 = circuit2.jacobian([par])
        assert res1 == pytest.approx(res2, abs=tol)

    def test_differentiate_all_positional(self, tol):
        """Tests that all positional arguments are differentiated."""

        def circuit1(a, b, c):
            qml.RX(a, wires=0)
            qml.RX(b, wires=1)
            qml.RX(c, wires=2)
            return tuple(qml.expval(qml.PauliZ(idx)) for idx in range(3))

        dev = qml.device("default.qubit", wires=3)
        circuit1 = JNode(circuit1, dev)

        vals = np.array([np.pi, np.pi / 2, np.pi / 3])
        circuit_output = circuit1(*vals)
        expected_output = np.cos(vals)
        assert circuit_output == pytest.approx(expected_output, abs=tol)

        # circuit jacobians
        circuit_jacobian = circuit1.jacobian(vals)
        expected_jacobian = -np.diag(np.sin(vals))
        assert circuit_jacobian == pytest.approx(expected_jacobian, abs=tol)

    def test_differentiate_first_positional(self, tol):
        """Tests that the first positional arguments are differentiated."""

        def circuit2(a, b):
            qml.RX(a, wires=0)
            return qml.expval(qml.PauliZ(0))

        dev = qml.device("default.qubit", wires=2)
        circuit2 = JNode(circuit2, dev)

        a = 0.7418
        b = -5.0
        circuit_output = circuit2(a, b)
        expected_output = np.cos(a)
        assert circuit_output == pytest.approx(expected_output, abs=tol)

        # circuit jacobians
        circuit_jacobian = circuit2.jacobian([a, b])
        expected_jacobian = np.array([[-np.sin(a), 0]])
        assert circuit_jacobian == pytest.approx(expected_jacobian, abs=tol)

    def test_differentiate_second_positional(self, tol):
        """Tests that the second positional arguments are differentiated."""

        def circuit3(a, b):
            qml.RX(b, wires=0)
            return qml.expval(qml.PauliZ(0))

        dev = qml.device("default.qubit", wires=2)
        circuit3 = JNode(circuit3, dev)

        a = 0.7418
        b = -5.0
        circuit_output = circuit3(a, b)
        expected_output = np.cos(b)
        assert circuit_output == pytest.approx(expected_output, abs=tol)

        # circuit jacobians
        circuit_jacobian = circuit3.jacobian([a, b])
        expected_jacobian = np.array([[0, -np.sin(b)]])
        assert circuit_jacobian == pytest.approx(expected_jacobian, abs=tol)

    def test_differentiate_second_third_positional(self, tol):
        """Tests that the second and third positional arguments are differentiated."""

        def circuit4(a, b, c):
            qml.RX(b, wires=0)
            qml.RX(c, wires=1)
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        dev = qml.device("default.qubit", wires=2)
        circuit4 = JNode(circuit4, dev)

        a = 0.7418
        b = -5.0
        c = np.pi / 7
        circuit_output = circuit4(a, b, c)
        expected_output = np.array([np.cos(b), np.cos(c)])
        assert circuit_output == pytest.approx(expected_output, abs=tol)

        # circuit jacobians
        circuit_jacobian = circuit4.jacobian([a, b, c])
        expected_jacobian = np.array([[0.0, -np.sin(b), 0.0], [0.0, 0.0, -np.sin(c)]])
        assert circuit_jacobian == pytest.approx(expected_jacobian, abs=tol)

    def test_differentiate_positional_multidim(self, tol):
        """Tests that all positional arguments are differentiated
        when they are multidimensional."""

        def circuit(a, b):
            qml.RX(a[0], wires=0)
            qml.RX(a[1], wires=1)
            qml.RX(b[2, 1], wires=2)
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2))

        dev = qml.device("default.qubit", wires=3)
        circuit = JNode(circuit, dev)

        a = np.array([-np.sqrt(2), -0.54])
        b = np.array([np.pi / 7] * 6).reshape([3, 2])
        circuit_output = circuit(a, b)
        expected_output = np.cos(np.array([a[0], a[1], b[-1, 0]]))
        assert circuit_output == pytest.approx(expected_output, abs=tol)

        # circuit jacobians
        circuit_jacobian = circuit.jacobian([a, b])
        expected_jacobian = np.array(
            [
                [-np.sin(a[0])] + [0.0] * 7,  # expval 0
                [0.0, -np.sin(a[1])] + [0.0] * 6,  # expval 1
                [0.0] * 2 + [0.0] * 5 + [-np.sin(b[2, 1])],
            ]
        )  # expval 2
        assert circuit_jacobian == pytest.approx(expected_jacobian, abs=tol)

    def test_array_parameters_evaluate(self, qubit_device_2_wires, tol):
        """Tests that array parameters gives same result as positional arguments."""
        a, b, c = 0.5, 0.54, 0.3

        def ansatz(x, y, z):
            qml.QubitStateVector(np.array([1, 0, 1, 1]) / np.sqrt(3), wires=[0, 1])
            qml.Rot(x, y, z, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))

        def circuit1(x, y, z):
            return ansatz(x, y, z)
        circuit1 = JNode(circuit1, qubit_device_2_wires)

        def circuit2(x, array):
            return ansatz(x, array[0], array[1])
        circuit2 = JNode(circuit2, qubit_device_2_wires)

        def circuit3(array):
            return ansatz(*array)
        circuit3 = JNode(circuit3, qubit_device_2_wires)

        positional_res = circuit1(a, b, c)
        positional_grad = circuit1.jacobian([a, b, c])

        array_res = circuit2(a, np.array([b, c]))
        array_grad = circuit2.jacobian([a, np.array([b, c])])

        assert positional_res == pytest.approx(array_res, abs=tol)
        assert positional_grad == pytest.approx(array_grad, abs=tol)

        list_res = circuit2(a, [b, c])
        list_grad = circuit2.jacobian([a, [b, c]])

        assert positional_res == pytest.approx(list_res, abs=tol)
        assert positional_grad == pytest.approx(list_grad, abs=tol)

        array_res = circuit3(np.array([a, b, c]))
        array_grad = circuit3.jacobian([np.array([a, b, c])])

        list_res = circuit3([a, b, c])
        list_grad = circuit3.jacobian([[a, b, c]])

        assert positional_res == pytest.approx(array_res, abs=tol)
        assert positional_grad == pytest.approx(array_grad, abs=tol)


    @pytest.mark.parametrize('G', [qml.ops.RX, qml.ops.RY, qml.ops.RZ])
    def test_pauli_rotation_gradient(self, G, qubit_device_1_wire, tol):
        "Tests that the automatic gradients of Pauli rotations are correct."

        def circuit(x):
            qml.RX(x, wires=[0])
            return qml.expval(qml.PauliZ(0))

        circuit = JNode(circuit, qubit_device_1_wire)
        for theta in thetas:
            autograd_val = circuit.jacobian([theta])
            manualgrad_val = (circuit(theta + np.pi / 2) - circuit(theta - np.pi / 2)) / 2
            assert autograd_val == pytest.approx(manualgrad_val, abs=tol)

    def test_Rot_gradient(self, qubit_device_1_wire, tol):
        "Tests that the automatic gradient of a arbitrary Euler-angle-parameterized gate is correct."

        def circuit(x,y,z):
            qml.Rot(x,y,z, wires=[0])
            return qml.expval(qml.PauliZ(0))

        circuit = JNode(circuit, qubit_device_1_wire)
        eye = np.eye(3)
        for theta in thetas:
            angle_inputs = np.array([theta, theta ** 3, np.sqrt(2) * theta])
            autograd_val = circuit.jacobian(angle_inputs)
            manualgrad_val = np.zeros((1,3))
            for idx in range(3):
                onehot_idx = eye[idx]
                param1 = angle_inputs + np.pi / 2 * onehot_idx
                param2 = angle_inputs - np.pi / 2 * onehot_idx
                manualgrad_val[0, idx] = (circuit(*param1) - circuit(*param2)) / 2
            assert autograd_val == pytest.approx(manualgrad_val, abs=tol)

    def test_controlled_RX_gradient(self, tol):
        """Test gradient of controlled RX gate"""
        dev = qml.device("default.qubit", wires=2)

        def circuit(x):
            qml.PauliX(wires=0)
            qml.CRX(x, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        circuit = JNode(circuit, dev)

        a = 0.542  # any value of a should give zero gradient

        # get the analytic gradient
        gradA = circuit.jacobian([a], method="A")
        # get the finite difference gradient
        gradF = circuit.jacobian([a], method="F")

        # the expected gradient
        expected = 0

        assert gradF == pytest.approx(expected, abs=tol)
        assert gradA == pytest.approx(expected, abs=tol)

        def circuit1(x):
            qml.RX(x, wires=0)
            qml.CRX(x, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        circuit1 = JNode(circuit1, dev)

        b = 0.123  # gradient is -sin(x)

        # get the analytic gradient
        gradA = circuit1.jacobian([b], method="A")
        # get the finite difference gradient
        gradF = circuit1.jacobian([b], method="F")

        # the expected gradient
        expected = -np.sin(b)

        assert gradF == pytest.approx(expected, abs=tol)
        assert gradA == pytest.approx(expected, abs=tol)

    def test_controlled_RY_gradient(self, tol):
        """Test gradient of controlled RY gate"""
        dev = qml.device("default.qubit", wires=2)

        def circuit(x):
            qml.PauliX(wires=0)
            qml.CRY(x, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        circuit = JNode(circuit, dev)

        a = 0.542  # any value of a should give zero gradient

        # get the analytic gradient
        gradA = circuit.jacobian([a], method="A")
        # get the finite difference gradient
        gradF = circuit.jacobian([a], method="F")

        # the expected gradient
        expected = 0

        assert gradF == pytest.approx(expected, abs=tol)
        assert gradA == pytest.approx(expected, abs=tol)

        def circuit1(x):
            qml.RX(x, wires=0)
            qml.CRY(x, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        circuit1 = JNode(circuit1, dev)

        b = 0.123  # gradient is -sin(x)

        # get the analytic gradient
        gradA = circuit1.jacobian([b], method="A")
        # get the finite difference gradient
        gradF = circuit1.jacobian([b], method="F")

        # the expected gradient
        expected = -np.sin(b)

        assert gradF == pytest.approx(expected, abs=tol)
        assert gradA == pytest.approx(expected, abs=tol)

    def test_controlled_RZ_gradient(self, tol):
        """Test gradient of controlled RZ gate"""
        dev = qml.device("default.qubit", wires=2)

        def circuit(x):
            qml.PauliX(wires=0)
            qml.CRZ(x, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        circuit = JNode(circuit, dev)

        a = 0.542  # any value of a should give zero gradient

        # get the analytic gradient
        gradA = circuit.jacobian([a], method="A")
        # get the finite difference gradient
        gradF = circuit.jacobian([a], method="F")

        # the expected gradient
        expected = 0

        assert gradF == pytest.approx(expected, abs=tol)
        assert gradA == pytest.approx(expected, abs=tol)

        def circuit1(x):
            qml.RX(x, wires=0)
            qml.CRZ(x, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        circuit1 = JNode(circuit1, dev)

        b = 0.123  # gradient is -sin(x)

        # get the analytic gradient
        gradA = circuit1.jacobian([b], method="A")
        # get the finite difference gradient
        gradF = circuit1.jacobian([b], method="F")

        # the expected gradient
        expected = -np.sin(b)

        assert gradF == pytest.approx(expected, abs=tol)
        assert gradA == pytest.approx(expected, abs=tol)



class TestJNodeVariance:
    """JNode variance tests."""

    def test_involutory_variance(self, tol, qubit_device_1_wire):
        """Tests qubit observable that are involutory"""
        def circuit(a):
            qml.RX(a, wires=0)
            return qml.var(qml.PauliZ(0))

        circuit = JNode(circuit, qubit_device_1_wire)

        a = 0.54
        var = circuit(a)
        expected = 1 - np.cos(a) ** 2
        assert var == pytest.approx(expected, abs=tol)

        # circuit jacobians
        gradA = circuit.jacobian([a], method="A")
        gradF = circuit.jacobian([a], method="F")
        expected = 2 * np.sin(a) * np.cos(a)
        assert gradF == pytest.approx(expected, abs=tol)
        assert gradA == pytest.approx(expected, abs=tol)

    def test_non_involutory_variance(self, tol, qubit_device_1_wire):
        """Tests a qubit Hermitian observable that is not involutory"""
        A = np.array([[4, -1 + 6j], [-1 - 6j, 2]])

        def circuit(a):
            qml.RX(a, wires=0)
            return qml.var(qml.Hermitian(A, 0))

        circuit = JNode(circuit, qubit_device_1_wire)

        a = 0.54
        var = circuit(a)
        expected = (39 / 2) - 6 * np.sin(2 * a) + (35 / 2) * np.cos(2 * a)
        assert var == pytest.approx(expected, abs=tol)

        # circuit jacobians
        gradA = circuit.jacobian([a], method="A")
        gradF = circuit.jacobian([a], method="F")
        expected = -35 * np.sin(2 * a) - 12 * np.cos(2 * a)
        assert gradA == pytest.approx(expected, abs=tol)
        assert gradF == pytest.approx(expected, abs=tol)

    def test_fanout(self, tol, qubit_device_1_wire):
        """Tests qubit observable with repeated parameters"""
        def circuit(a):
            qml.RX(a, wires=0)
            qml.RY(a, wires=0)
            return qml.var(qml.PauliZ(0))

        circuit = JNode(circuit, qubit_device_1_wire)

        a = 0.54
        var = circuit(a)
        expected = 0.5 * np.sin(a) ** 2 * (np.cos(2 * a) + 3)
        assert var == pytest.approx(expected, abs=tol)

        # circuit jacobians
        gradA = circuit.jacobian([a], method="A")
        gradF = circuit.jacobian([a], method="F")
        expected = 4 * np.sin(a) * np.cos(a) ** 3
        assert gradA == pytest.approx(expected, abs=tol)
        assert gradF == pytest.approx(expected, abs=tol)

    def test_first_order_cv(self, tol):
        """Test variance of a first order CV expectation value"""
        dev = qml.device("default.gaussian", wires=1)

        def circuit(r, phi):
            qml.Squeezing(r, 0, wires=0)
            qml.Rotation(phi, wires=0)
            return qml.var(qml.X(0))

        circuit = JNode(circuit, dev)

        r = 0.543
        phi = -0.654
        var = circuit(r, phi)
        expected = np.exp(2 * r) * np.sin(phi) ** 2 + np.exp(-2 * r) * np.cos(phi) ** 2
        assert var == pytest.approx(expected, abs=tol)

        # circuit jacobians
        gradA = circuit.jacobian([r, phi], method="A")
        gradF = circuit.jacobian([r, phi], method="F")
        expected = np.array(
            [[
                2 * np.exp(2 * r) * np.sin(phi) ** 2 - 2 * np.exp(-2 * r) * np.cos(phi) ** 2,
                2 * np.sinh(2 * r) * np.sin(2 * phi),
            ]]
        )
        assert gradA == pytest.approx(expected, abs=tol)
        assert gradF == pytest.approx(expected, abs=tol)

    def test_second_order_cv(self, tol):
        """Test variance of a second order CV expectation value"""
        dev = qml.device("default.gaussian", wires=1)

        def circuit(n, a):
            qml.ThermalState(n, wires=0)
            qml.Displacement(a, 0, wires=0)
            return qml.var(qml.NumberOperator(0))

        circuit = JNode(circuit, dev)

        n = 0.12
        a = 0.765
        var = circuit(n, a)
        expected = n ** 2 + n + np.abs(a) ** 2 * (1 + 2 * n)
        assert var == pytest.approx(expected, abs=tol)

        # circuit jacobians
        gradF = circuit.jacobian([n, a], method="F")
        expected = np.array([[2 * a ** 2 + 2 * n + 1, 2 * a * (2 * n + 1)]])
        assert gradF == pytest.approx(expected, abs=tol)

    def test_expval_and_variance(self, tol):
        """Test that the qnode works for a combination of expectation
        values and variances"""
        dev = qml.device("default.qubit", wires=3)

        def circuit(a, b, c):
            qml.RX(a, wires=0)
            qml.RY(b, wires=1)
            qml.CNOT(wires=[1, 2])
            qml.RX(c, wires=2)
            qml.CNOT(wires=[0, 1])
            qml.RZ(c, wires=2)
            return qml.var(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.var(qml.PauliZ(2))

        circuit = JNode(circuit, dev)

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
        assert var == pytest.approx(expected, abs=tol)

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
        assert gradF == pytest.approx(expected, abs=tol)
        assert gradA == pytest.approx(expected, abs=tol)

    def test_expval_and_variance_cv(self, tol):
        """Test that the qnode works for a combination of CV expectation
        values and variances"""
        dev = qml.device("default.gaussian", wires=3)

        def circuit(a, b):
            qml.Displacement(0.5, 0, wires=0)
            qml.Squeezing(a, 0, wires=0)
            qml.Squeezing(b, 0, wires=1)
            qml.Beamsplitter(0.6, -0.3, wires=[0, 1])
            qml.Squeezing(-0.3, 0, wires=2)
            qml.Beamsplitter(1.4, 0.5, wires=[1, 2])
            return qml.var(qml.X(0)), qml.expval(qml.X(1)), qml.var(qml.X(2))  # TODO can you return them in arbitrary order?

        node = JNode(circuit, dev)
        par = [0.54, -0.423]

        # jacobians must match
        gradA = node.jacobian(par, method="A")
        gradF = node.jacobian(par, method="F")
        assert gradA == pytest.approx(gradF, abs=tol)

    def test_error_analytic_second_order_cv(self):
        """Test exception raised if attempting to use a second
        order observable to compute the variance derivative analytically"""
        dev = qml.device("default.gaussian", wires=1)

        def circuit(a):
            qml.Displacement(a, 0, wires=0)
            return qml.var(qml.NumberOperator(0))

        circuit = JNode(circuit, dev)

        with pytest.raises(ValueError, match=r"cannot be used with the parameters \{0\}"):
            circuit.jacobian([1.0], method="A")

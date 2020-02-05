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
Unit tests for the PennyLane :class:`~.CVQNode` class.
"""
import pytest
import numpy as np

import pennylane as qml
from pennylane._device import Device
from pennylane.operation import CVObservable
from pennylane.qnodes.base import QuantumFunctionError
from pennylane.qnodes.cv import CVQNode


class PolyN(qml.ops.PolyXP):
    """Mimics NumberOperator using the arbitrary 2nd order observable interface.
    Results should be identical."""
    def __init__(self, wires):
        hbar = 2
        q = np.diag([-0.5, 0.5/hbar, 0.5/hbar])
        super().__init__(q, wires=wires)
        self.name = 'PolyXP'


cv_ops = [getattr(qml.ops, name) for name in qml.ops._cv__ops__]
analytic_cv_ops = [cls for cls in cv_ops if cls.supports_parameter_shift]


@pytest.fixture(scope="function")
def operable_mock_CV_device_2_wires(monkeypatch):
    """A mock instance of the abstract Device class that can support qfuncs."""

    dev = Device
    with monkeypatch.context() as m:
        m.setattr(dev, '__abstractmethods__', frozenset())
        m.setattr(dev, '_capabilities', {"model": "cv"})
        m.setattr(dev, 'operations', ["FockState", "Displacement", "CubicPhase", "Squeezing", "Rotation", "Kerr", "Beamsplitter"])
        m.setattr(dev, 'observables', ["X", "NumberOperator", "PolyXP"])
        m.setattr(dev, 'reset', lambda self: None)
        m.setattr(dev, 'apply', lambda self, x, y, z: None)
        m.setattr(dev, 'expval', lambda self, x, y, z: 1)
        yield Device(wires=2)


def test_transform_observable_incorrect_heisenberg_size():
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

    node = CVQNode(circuit, dev)

    with pytest.raises(QuantumFunctionError, match="Mismatch between the polynomial order"):
        node.jacobian([0.5])


class TestBestMethod:
    """
    Test different flows of _best_method using a mock device. TODO more
    """
    def test_gaussian_successors_fails(self, operable_mock_CV_device_2_wires):
        """Tests that the parameter-shift differentiation method is not allowed
        if a non-gaussian gate is between a differentiable gaussian gate and an observable."""

        def circuit(x):
            qml.Squeezing(x, 0, wires=[0])
            qml.Beamsplitter(np.pi/4, 0, wires=[0, 1])
            qml.Kerr(0.54, wires=[1])
            return qml.expval(qml.NumberOperator(1))

        node = CVQNode(circuit, operable_mock_CV_device_2_wires)

        with pytest.raises(ValueError, match="analytic gradient method cannot be used with"):
            node.jacobian([0.321], method="A")

        assert node.par_to_grad_method == {0: "F"}

    def test_correct_method_non_gaussian_successor_one_param(self, operable_mock_CV_device_2_wires):
        """Tests that a non-Gaussian succeeding a parameter fallsback to finite-diff"""
        par = [0.4, -2.3]

        def qf(x, y):
            qml.Displacement(x, 0, wires=[0])
            qml.CubicPhase(0.2, wires=[0])
            qml.Squeezing(0.3, y, wires=[1])
            qml.Rotation(1.3, wires=[1])
            # nongaussian succeeding x but not y
            return qml.expval(qml.X(0)), qml.expval(qml.X(1))

        q = CVQNode(qf, operable_mock_CV_device_2_wires)
        q._construct(par, {})
        assert q.par_to_grad_method == {0: "F", 1: "A"}

    def test_correct_method_non_gaussian_successor_unused_param(self, operable_mock_CV_device_2_wires):
        """Tests that a non-Gaussian succeeding a parameter fallsback to finite-diff
        alongside an unused parameter"""
        par = [0.4, -2.3]

        def qf(x, y):
            qml.Displacement(x, 0, wires=[0])
            qml.CubicPhase(0.2, wires=[0])  # nongaussian succeeding x
            qml.Squeezing(0.3, x, wires=[1])  # x affects gates on both wires, y unused
            qml.Rotation(1.3, wires=[1])
            return qml.expval(qml.X(0)), qml.expval(qml.X(1))

        q = CVQNode(qf, operable_mock_CV_device_2_wires)
        q._construct(par, {})
        assert q.par_to_grad_method == {0: "F", 1: "0"}

    def test_param_not_differentiable(self, operable_mock_CV_device_2_wires):
        """Tests that a parameter is not differentiable if used in an operation
        where grad_method=None"""
        par = [0.4]

        def qf(x):
            qml.FockState(x, wires=[0])
            qml.Rotation(1.3, wires=[0])
            return qml.expval(qml.X(0))

        q = CVQNode(qf, operable_mock_CV_device_2_wires)
        q._construct(par, {})
        assert q.par_to_grad_method == {0: None}

    def test_param_no_observables(self, operable_mock_CV_device_2_wires):
        """Tests that a parameter has 0 gradient if it is not followed by any observables"""
        par = [0.4]

        def qf(x):
            qml.Displacement(x, 0, wires=[0])
            qml.Squeezing(0.3, x, wires=[0])
            qml.Rotation(1.3, wires=[1])
            return qml.expval(qml.X(1))

        q = CVQNode(qf, operable_mock_CV_device_2_wires)
        q._construct(par, {})
        assert q.par_to_grad_method == {0: "0"}

    def test_correct_method_non_gaussian_successor_all_params(self, operable_mock_CV_device_2_wires):
        """Tests that a non-Gaussian succeeding all parameters fallsback to finite-diff"""
        par = [0.4, -2.3]

        def qf(x, y):
            qml.Displacement(x, 0, wires=[0])
            qml.Displacement(1.2, y, wires=[1])
            qml.Beamsplitter(0.2, 1.7, wires=[0, 1])
            qml.Rotation(1.9, wires=[0])
            qml.Kerr(0.3, wires=[1])  # nongaussian succeeding both x and y due to the beamsplitter
            return qml.expval(qml.X(0)), qml.expval(qml.X(1))

        q = CVQNode(qf, operable_mock_CV_device_2_wires)
        q._construct(par, {})
        assert q.par_to_grad_method == {0: "F", 1: "F"}

    def test_correct_method_non_gaussian_preceeding_one_param(self, operable_mock_CV_device_2_wires):
        """Tests that a non-Gaussian preceeding one parameter fallsback to finite-diff"""
        par = [0.4, -2.3]

        def qf(x, y):
            qml.Kerr(y, wires=[1])
            qml.Displacement(x, 0, wires=[0])
            qml.Beamsplitter(0.2, 1.7, wires=[0, 1])
            return qml.expval(qml.X(0)), qml.expval(qml.X(1))

        q = CVQNode(qf, operable_mock_CV_device_2_wires)
        q._construct(par, {})
        assert q.par_to_grad_method == {0: "A", 1: "F"}

    def test_correct_method_non_gaussian_observable(self, operable_mock_CV_device_2_wires):
        """Tests that a non-Gaussian observable one parameter fallsback to finite-diff"""
        par = [0.4, -2.3]

        def qf(x, y):
            qml.Displacement(x, 0, wires=[0])  # followed by nongaussian observable
            qml.Beamsplitter(0.2, 1.7, wires=[0, 1])
            qml.Displacement(y, 0, wires=[1])  # followed by order-2 observable
            return qml.expval(qml.FockStateProjector(np.array([2]), 0)), qml.expval(qml.NumberOperator(1))

        q = CVQNode(qf, operable_mock_CV_device_2_wires)
        q._construct(par, {})
        assert q.par_to_grad_method == {0: "F", 1: "A"}


class TestExpectationJacobian:
    """Jacobian integration tests for CV expectations."""

    def test_keywordarg_second_order_cv(self, tol):
        """Non-differentiable keyword arguments with a second order CV expectation value."""

        dev = qml.device("default.gaussian", wires=3)
        def circuit(x, *, k=0.0):
            qml.Displacement(x, 0, wires=0)
            qml.Rotation(k, wires=0)
            return qml.expval(qml.PolyXP(np.diag([0, 1, 0]), wires=0))  # X^2

        node = CVQNode(circuit, dev)
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

        node = CVQNode(circuit, dev, mutable=False)
        par = [0.39]
        aux = {'k': -0.7}

        # circuit jacobians
        grad_A = node.jacobian(par, aux, method="A", options={'force_order2': True})
        grad_F = node.jacobian(par, aux, method="F")
        assert grad_A == pytest.approx(grad_F, abs=tol)

    @pytest.mark.parametrize('O', [qml.ops.X, qml.ops.NumberOperator, PolyN, qml.ops.Identity])
    @pytest.mark.parametrize('G', analytic_cv_ops)
    def test_cv_gradients_gaussian_circuit(self, G, O, tol):
        """Tests that the gradients of circuits of gaussian gates match between the finite difference and analytic methods."""
        gaussian_dev = qml.device("default.gaussian", wires=2)

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

        q = CVQNode(circuit, gaussian_dev)
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

    def test_gradient_gate_with_two_parameters(self, tol):
        """Gates with two parameters yield the correct parshift gradient."""

        dev = qml.device("default.gaussian", wires=1)
        def qf(r0, phi0, r1, phi1):
            qml.Squeezing(r0, phi0, wires=[0])
            qml.Squeezing(r1, phi1, wires=[0])
            return qml.expval(qml.NumberOperator(0))

        q = CVQNode(qf, dev)
        par = [0.543, 0.123, 0.654, -0.629]

        grad_A = q.jacobian(par, method="A")
        grad_F = q.jacobian(par, method="F")
        # the different methods agree
        assert grad_A == pytest.approx(grad_F, abs=tol)

    def test_cv_gradients_multiple_gate_parameters(self, tol):
        """Tests that gates with multiple free parameters yield correct gradients."""

        gaussian_dev = qml.device("default.gaussian", wires=2)
        def qf(r0, phi0, r1, phi1):
            qml.Squeezing(r0, phi0, wires=[0])
            qml.Squeezing(r1, phi1, wires=[0])
            return qml.expval(qml.NumberOperator(0))

        q = CVQNode(qf, gaussian_dev)
        par = [0.4, -0.3, -0.7, 0.2]

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

    def test_cv_gradients_repeated_gate_parameters(self, tol):
        """Tests that repeated use of a free parameter in a multi-parameter gate yield correct gradients."""
        gaussian_dev = qml.device("default.gaussian", wires=2)
        par = [0.2, 0.3]

        def qf(x, y):
            qml.Displacement(x, 0, wires=[0])
            qml.Squeezing(y, -1.3*y, wires=[0])
            return qml.expval(qml.X(0))

        q = CVQNode(qf, gaussian_dev)
        grad_F = q.jacobian(par, method="F")
        grad_A = q.jacobian(par, method="A")
        grad_A2 = q.jacobian(par, method="A", options={'force_order2': True})

        # analytic method works for every parameter
        assert q.par_to_grad_method == {0:"A", 1:"A"}
        # the different methods agree
        assert grad_A == pytest.approx(grad_F, abs=tol)
        assert grad_A2 == pytest.approx(grad_F, abs=tol)

    def test_cv_gradients_parameters_inside_array(self, tol):
        """Tests that free parameters inside an array passed to an Operation yield correct gradients."""
        gaussian_dev = qml.device("default.gaussian", wires=2)
        par = [0.4, 1.3]

        def qf(x, y):
            qml.Displacement(0.5, 0, wires=[0])
            qml.Squeezing(x, 0, wires=[0])
            M = np.zeros((5, 5), dtype=object)
            M[1,1] = y
            M[1,2] = 1.0
            M[2,1] = 1.0
            return qml.expval(qml.PolyXP(M, [0, 1]))

        q = CVQNode(qf, gaussian_dev)

        grad_best = q.jacobian(par)
        grad_best2 = q.jacobian(par, options={"force_order2": True})
        grad_F = q.jacobian(par, method="F")

        # par[0] can use the "A" method, par[1] cannot
        assert q.par_to_grad_method == {0: "A", 1: "F"}
        # the different methods agree
        assert grad_best == pytest.approx(grad_F, abs=tol)
        assert grad_best2 == pytest.approx(grad_F, abs=tol)

    def test_cv_gradient_fanout(self, tol):
        """Tests that CV qnodes can compute the correct gradient when the same parameter is used
        in multiple gates."""
        gaussian_dev = qml.device("default.gaussian", wires=2)
        par = [0.5, 1.3]

        def circuit(x, y):
            qml.Displacement(x, 0, wires=[0])
            qml.Rotation(y, wires=[0])
            qml.Displacement(0, x, wires=[0])
            return qml.expval(qml.X(0))

        q = CVQNode(circuit, gaussian_dev)
        grad_F = q.jacobian(par, method="F")
        grad_A = q.jacobian(par, method="A")
        grad_A2 = q.jacobian(par, method="A", options={'force_order2': True})

        # analytic method works for every parameter
        assert q.par_to_grad_method == {0:"A", 1:"A"}
        # the different methods agree
        assert grad_A == pytest.approx(grad_F, abs=tol)
        assert grad_A2 == pytest.approx(grad_F, abs=tol)

    @pytest.mark.parametrize('name', qml.ops._cv__ops__)
    def test_CVOperation_with_heisenberg_and_no_parshift(self, name, tol):
        """An integration test for Gaussian CV gates that have a Heisenberg representation
        but cannot be differentiated using the parameter-shift method themselves
        (for example, they may accept no parameters, or have no gradient recipe).

        Tests that the parameter-shift method can still be used with other gates in the circuit.
        """
        gaussian_dev = qml.device("default.gaussian", wires=2)

        cls = getattr(qml.ops, name)
        if cls.supports_heisenberg and (not cls.supports_parameter_shift):
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

            qnode = CVQNode(circuit, gaussian_dev)
            grad_F = qnode.jacobian(0.5, method="F")
            grad_A = qnode.jacobian(0.5, method="A")
            grad_A2 = qnode.jacobian(0.5, method="A", options={'force_order2': True})

            # par[0] can use the "A" method
            assert qnode.par_to_grad_method == {0: "A"}

            # the different methods agree
            assert grad_A == pytest.approx(grad_F, abs=tol)
            assert grad_A2 == pytest.approx(grad_F, abs=tol)

    def test_non_gaussian_gate_successor(self, gaussian_device, tol):
        """Parshift differentiation method is allowed and matches finite diff
        if a non-Gaussian gate follows the parametrized gate but is not followed by an observable."""

        def circuit(x):
            qml.Squeezing(x, 0, wires=[0])
            qml.Beamsplitter(1.1, 0, wires=[0, 1])
            qml.Kerr(0.54, wires=[1])  # nongaussian
            return qml.expval(qml.NumberOperator(0))

        node = CVQNode(circuit, gaussian_device)
        par = [0.321]

        grad_A = node.jacobian(par, wrt=[0], method="A")
        grad_F = node.jacobian(par, method="F")
        assert grad_A == pytest.approx(grad_F, abs=tol)
        assert node.par_to_grad_method == {0: "A"}

    def test_non_gaussian_obs_predecessor(self, gaussian_device, tol):
        """Parshift differentiation method is allowed and matches finite diff
        if a non-Gaussian gate precedes an observable but is not preceded by the parametrized gate."""

        def circuit(x):
            qml.Squeezing(x, 0, wires=[0])
            qml.Kerr(0.54, wires=[1])  # nongaussian
            qml.Beamsplitter(1.1, 0, wires=[0, 1])
            return qml.expval(qml.NumberOperator(0))

        node = CVQNode(circuit, gaussian_device)
        par = [0.321]

        grad_A = node.jacobian(par, wrt=[0], method="A")
        grad_F = node.jacobian(par, method="F")
        assert grad_A == pytest.approx(grad_F, abs=tol)
        assert node.par_to_grad_method == {0: "A"}

    def test_second_order_obs_not_following_gate(self, tol):
        """Parshift differentiation method matches finite diff and analytical result
        when we have order-2 observables that do not follow the parametrized gate.
        """
        num_wires = 2
        dev = qml.device("default.gaussian", wires=2)
        def circuit(params):
            for i in range(num_wires):
                qml.Squeezing(params[i], 0, wires=i)
            return [qml.expval(qml.NumberOperator(wires=i)) for i in range(num_wires)]

        node = CVQNode(circuit, dev)
        par = [0.321, -0.184]

        res = node(par)
        res_true = np.sinh(np.abs(par)) ** 2  # analytical result
        assert res == pytest.approx(res_true, abs=tol)

        grad_A = node.jacobian([par], method="A")
        grad_F = node.jacobian([par], method="F")
        grad_true = np.diag(np.sinh(2 * np.abs(par)) * np.sign(par))  # analytical gradient
        assert grad_A == pytest.approx(grad_F, abs=tol)
        assert grad_A == pytest.approx(grad_true, abs=tol)

    @pytest.mark.xfail(reason="FIXME: 'A' method fails on QuadOperator (it has no gradient recipe)", raises=AttributeError, strict=True)
    def test_quadoperator(self, tol):
        """Test the differentiation of CV observables that depend on positional qfunc parameters."""

        def circuit(a):
            qml.Displacement(1.0, 0, wires=0)
            return qml.expval(qml.QuadOperator(a, 0))

        gaussian_dev = qml.device("default.gaussian", wires=1)
        qnode = CVQNode(circuit, gaussian_dev)

        par = [0.6]
        grad_F = qnode.jacobian(par, method='F')
        grad_A = qnode.jacobian(par, method='A')
        grad_A2 = qnode.jacobian(par, method='A', options={'force_order2': True})

        # par 0 can use the 'A' method
        assert qnode.par_to_grad_method == {0: 'A'}
        # the different methods agree
        assert grad_A == pytest.approx(grad_F, abs=tol)
        assert grad_A2 == pytest.approx(grad_F, abs=tol)


class TestVarianceJacobian:
    """Variance analytic jacobian integration tests."""

    def test_first_order_cv(self, tol):
        """Test variance of a first order CV expectation value"""
        dev = qml.device("default.gaussian", wires=1)

        def circuit(r, phi):
            qml.Squeezing(r, 0, wires=0)
            qml.Rotation(phi, wires=0)
            return qml.var(qml.X(0))

        circuit = CVQNode(circuit, dev)

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

        circuit = CVQNode(circuit, dev)

        n = 0.12
        a = 0.765
        var = circuit(n, a)
        expected = n ** 2 + n + np.abs(a) ** 2 * (1 + 2 * n)
        assert var == pytest.approx(expected, abs=tol)

        # circuit jacobians
        gradF = circuit.jacobian([n, a], method="F")
        expected = np.array([[2 * a ** 2 + 2 * n + 1, 2 * a * (2 * n + 1)]])
        assert gradF == pytest.approx(expected, abs=tol)


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
            return qml.var(qml.X(0)), qml.expval(qml.X(1)), qml.var(qml.X(2))

        node = CVQNode(circuit, dev)
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

        circuit = CVQNode(circuit, dev)

        with pytest.raises(ValueError, match=r"cannot be used with the parameters \{0\}"):
            circuit.jacobian([1.0], method="A")

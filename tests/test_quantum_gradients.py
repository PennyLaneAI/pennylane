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
Unit tests for the computing gradients of quantum functions.
"""
import pytest
import autograd
import autograd.numpy as np

import pennylane as qml
from pennylane.plugins.default_qubit import Rotx as Rx, Roty as Ry, Rotz as Rz


def expZ(state):
    return np.abs(state[0]) ** 2 - np.abs(state[1]) ** 2

hbar = 2
mag_alphas = np.linspace(0, 1.5, 5)
thetas = np.linspace(-2*np.pi, 2*np.pi, 8)
sqz_vals = np.linspace(0., 1., 5)


@pytest.fixture(scope="function")
def gaussian_dev():
    return qml.device('default.gaussian', wires=2)

@pytest.fixture(scope="function")
def qubit1_dev():
    return qml.device('default.qubit', wires=1)

@pytest.fixture(scope="function")
def qubit2_dev():
    return qml.device('default.qubit', wires=2)


class TestCVGradient:
    """Tests of the automatic gradient method for CV circuits.
    """
    def test_rotation_gradient(self, gaussian_dev, tol):
        "Tests that the automatic gradient of a phase space rotation is correct."

        alpha = 0.5

        @qml.qnode(gaussian_dev)
        def circuit(y):
            qml.Displacement(alpha, 0., wires=[0])
            qml.Rotation(y, wires=[0])
            return qml.expval(qml.X(0))

        grad_fn = autograd.grad(circuit)

        for theta in thetas:
            autograd_val = grad_fn(theta)
            # qfunc evalutes to hbar * alpha * cos(theta)
            manualgrad_val = - hbar * alpha * np.sin(theta)

            assert autograd_val == pytest.approx(manualgrad_val, abs=tol)

    def test_beamsplitter_gradient(self, gaussian_dev, tol):
        "Tests that the automatic gradient of a beamsplitter is correct."

        alpha = 0.5

        @qml.qnode(gaussian_dev)
        def circuit(y):
            qml.Displacement(alpha, 0., wires=[0])
            qml.Beamsplitter(y, 0, wires=[0, 1])
            return qml.expval(qml.X(0))

        grad_fn = autograd.grad(circuit)

        for theta in thetas:
            autograd_val = grad_fn(theta)
            # qfunc evalutes to hbar * alpha * cos(theta)
            manualgrad_val = - hbar * alpha * np.sin(theta)

            assert autograd_val == pytest.approx(manualgrad_val, abs=tol)

    def test_displacement_gradient(self, gaussian_dev, tol):
        "Tests that the automatic gradient of a phase space displacement is correct."

        @qml.qnode(gaussian_dev)
        def circuit(r, phi):
            qml.Displacement(r, phi, wires=[0])
            return qml.expval(qml.X(0))

        grad_fn = autograd.grad(circuit)

        for mag in mag_alphas:
            for theta in thetas:
                #alpha = mag * np.exp(1j * theta)
                autograd_val = grad_fn(mag, theta)
                # qfunc evalutes to hbar * Re(alpha)
                manualgrad_val = hbar * np.cos(theta)

                assert autograd_val == pytest.approx(manualgrad_val, abs=tol)

    def test_squeeze_gradient(self, gaussian_dev, tol):
        "Tests that the automatic gradient of a phase space squeezing is correct."

        alpha = 0.5

        @qml.qnode(gaussian_dev)
        def circuit(y, r=0.5):
            qml.Displacement(r, 0., wires=[0])
            qml.Squeezing(y, 0., wires=[0])
            return qml.expval(qml.X(0))

        grad_fn = autograd.grad(circuit, 0)

        for r in sqz_vals:
            autograd_val = grad_fn(r)
            # qfunc evaluates to -exp(-r) * hbar * Re(alpha)
            manualgrad_val = -np.exp(-r) * hbar * alpha
            assert autograd_val == pytest.approx(manualgrad_val, abs=tol)

    def test_number_state_gradient(self, gaussian_dev, tol):
        "Tests that the automatic gradient of a squeezed state with number state expectation is correct."

        @qml.qnode(gaussian_dev)
        def circuit(y):
            qml.Squeezing(y, 0., wires=[0])
            return qml.expval(qml.FockStateProjector(np.array([2, 0]), wires=[0, 1]))

        grad_fn = autograd.grad(circuit, 0)

        # (d/dr) |<2|S(r)>|^2 = 0.5 tanh(r)^3 (2 csch(r)^2 - 1) sech(r)
        for r in sqz_vals[1:]: # formula above is not valid for r=0
            autograd_val = grad_fn(r)
            manualgrad_val = 0.5*np.tanh(r)**3 * (2/(np.sinh(r)**2)-1) / np.cosh(r)
            assert autograd_val == pytest.approx(manualgrad_val, abs=tol)

    def test_cv_gradients_gaussian_circuit(self, gaussian_dev, tol):
        """Tests that the gradients of circuits of gaussian gates match between the finite difference and analytic methods."""

        class PolyN(qml.ops.PolyXP):
            "Mimics NumberOperator using the arbitrary 2nd order observable interface. Results should be identical."
            def __init__(self, wires):
                hbar = 2
                q = np.diag([-0.5, 0.5/hbar, 0.5/hbar])
                super().__init__(q, wires=wires)
                self.name = 'PolyXP'

        gates = []
        for name in qml.ops._cv__ops__:
            cls = getattr(qml.ops, name)

            if cls.supports_analytic:
                gates.append(cls)

        obs   = [qml.ops.X, qml.ops.NumberOperator, PolyN]
        par = [0.4]

        for G in reversed(gates):
            #log.debug('Testing gate %s...', G.__name__[0])
            for O in obs:
                #log.debug('Testing observable %s...', O.__name__[0])
                def circuit(x):
                    args = [0.3] * G.num_params
                    args[0] = x
                    qml.Displacement(0.5, 0, wires=0)
                    G(*args, wires=range(G.num_wires))
                    qml.Beamsplitter(1.3, -2.3, wires=[0, 1])
                    qml.Displacement(-0.5, 0, wires=0)
                    qml.Squeezing(0.5, -1.5, wires=0)
                    qml.Rotation(-1.1, wires=0)
                    return qml.expval(O(wires=0))

                q = qml.QNode(circuit, gaussian_dev)
                val = q.evaluate(par)
                # log.info('  value:', val)
                grad_F  = q.jacobian(par, method='F')
                grad_A2 = q.jacobian(par, method='A', force_order2=True)
                # log.info('  grad_F: ', grad_F)
                # log.info('  grad_A2: ', grad_A2)
                if O.ev_order == 1:
                    grad_A = q.jacobian(par, method='A')
                    # log.info('  grad_A: ', grad_A)
                    # the different methods agree
                    assert grad_A == pytest.approx(grad_F, abs=tol)

                # analytic method works for every parameter
                assert q.grad_method_for_par == {0:'A'}
                # the different methods agree
                assert grad_A2 == pytest.approx(grad_F, abs=tol)

    def test_cv_gradients_multiple_gate_parameters(self, gaussian_dev, tol):
        "Tests that gates with multiple free parameters yield correct gradients."
        par = [0.4, -0.3, -0.7, 0.2]

        def qf(r0, phi0, r1, phi1):
            qml.Squeezing(r0, phi0, wires=[0])
            qml.Squeezing(r1, phi1, wires=[0])
            return qml.expval(qml.NumberOperator(0))

        q = qml.QNode(qf, gaussian_dev)
        grad_F = q.jacobian(par, method='F')
        grad_A = q.jacobian(par, method='A')
        grad_A2 = q.jacobian(par, method='A', force_order2=True)

        # analytic method works for every parameter
        assert q.grad_method_for_par == {i:'A' for i in range(4)}
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

        q = qml.QNode(qf, gaussian_dev)
        grad_F = q.jacobian(par, method='F')
        grad_A = q.jacobian(par, method='A')
        grad_A2 = q.jacobian(par, method='A', force_order2=True)

        # analytic method works for every parameter
        assert q.grad_method_for_par == {0:'A', 1:'A'}
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

        q = qml.QNode(qf, gaussian_dev)
        grad = q.jacobian(par)
        grad_F = q.jacobian(par, method='F')
        grad_A = q.jacobian(par, method='B')
        grad_A2 = q.jacobian(par, method='B', force_order2=True)

        # par[0] can use the 'A' method, par[1] cannot
        assert q.grad_method_for_par == {0:'A', 1:'F'}
        # the different methods agree
        assert grad == pytest.approx(grad_F, abs=tol)


    def test_cv_gradient_fanout(self, gaussian_dev, tol):
        "Tests that qnodes can compute the correct gradient when the same parameter is used in multiple gates."
        par = [0.5, 1.3]

        def circuit(x, y):
            qml.Displacement(x, 0, wires=[0])
            qml.Rotation(y, wires=[0])
            qml.Displacement(0, x, wires=[0])
            return qml.expval(qml.X(0))

        q = qml.QNode(circuit, gaussian_dev)
        grad_F = q.jacobian(par, method='F')
        grad_A = q.jacobian(par, method='A')
        grad_A2 = q.jacobian(par, method='A', force_order2=True)

        # analytic method works for every parameter
        assert q.grad_method_for_par == {0:'A', 1:'A'}
        # the different methods agree
        assert grad_A == pytest.approx(grad_F, abs=tol)
        assert grad_A2 == pytest.approx(grad_F, abs=tol)

    def test_CVOperation_with_heisenberg_and_no_params(self, gaussian_dev, tol):
        """An integration test for CV gates that support analytic differentiation
        if succeeding the gate to be differentiated, but cannot be differentiated
        themselves (for example, they may be Gaussian but accept no parameters).

        This ensures that, assuming their _heisenberg_rep is defined, the quantum
        gradient analytic method can still be used, and returns the correct result."""

        for name in qml.ops._cv__ops__:
            cls = getattr(qml.ops, name)
            if cls.supports_heisenberg and (not cls.supports_analytic):
                dev = qml.device('default.gaussian', wires=2)

                U = np.array([[0.51310276+0.81702166j, 0.13649626+0.22487759j],
                              [0.26300233+0.00556194j, -0.96414101-0.03508489j]])

                if cls.num_wires <= 0:
                    w = list(range(2))
                else:
                    w = list(range(cls.num_wires))

                def circuit(x):
                    qml.Displacement(x, 0, wires=0)

                    if cls.par_domain == 'A':
                        cls(U, wires=w)
                    else:
                        cls(wires=w)
                    return qml.expval(qml.X(0))

                qnode = qml.QNode(circuit, dev)
                grad_F = qnode.jacobian(0.5, method='F')
                grad_A = qnode.jacobian(0.5, method='A')
                grad_A2 = qnode.jacobian(0.5, method='A', force_order2=True)

                # par[0] can use the 'A' method
                assert qnode.grad_method_for_par == {0: 'A'}

                # the different methods agree
                assert grad_A == pytest.approx(grad_F, abs=tol)
                assert grad_A2 == pytest.approx(grad_F, abs=tol)


class TestQubitGradient:
    """Tests of the automatic gradient method for qubit gates.
    """
    def test_RX_gradient(self, qubit1_dev, tol):
        "Tests that the automatic gradient of a Pauli X-rotation is correct."

        @qml.qnode(qubit1_dev)
        def circuit(x):
            qml.RX(x, wires=[0])
            return qml.expval(qml.PauliZ(0))

        grad_fn = autograd.grad(circuit)

        for theta in thetas:
            autograd_val = grad_fn(theta)
            manualgrad_val = (circuit(theta + np.pi / 2) - circuit(theta - np.pi / 2)) / 2
            assert autograd_val == pytest.approx(manualgrad_val, abs=tol)

    def test_RY_gradient(self, qubit1_dev, tol):
        "Tests that the automatic gradient of a Pauli Y-rotation is correct."

        @qml.qnode(qubit1_dev)
        def circuit(x):
            qml.RY(x, wires=[0])
            return qml.expval(qml.PauliZ(0))

        grad_fn = autograd.grad(circuit)

        for theta in thetas:
            autograd_val = grad_fn(theta)
            manualgrad_val = (circuit(theta + np.pi / 2) - circuit(theta - np.pi / 2)) / 2
            assert autograd_val == pytest.approx(manualgrad_val, abs=tol)

    def test_RZ_gradient(self, qubit1_dev, tol):
        "Tests that the automatic gradient of a Pauli Z-rotation is correct."

        @qml.qnode(qubit1_dev)
        def circuit(x):
            qml.RZ(x, wires=[0])
            return qml.expval(qml.PauliZ(0))

        grad_fn = autograd.grad(circuit)

        for theta in thetas:
            autograd_val = grad_fn(theta)
            manualgrad_val = (circuit(theta + np.pi / 2) - circuit(theta - np.pi / 2)) / 2
            assert autograd_val == pytest.approx(manualgrad_val, abs=tol)

    def test_Rot(self, qubit1_dev, tol):
        "Tests that the automatic gradient of a arbitrary Euler-angle-parameterized gate is correct."

        @qml.qnode(qubit1_dev)
        def circuit(x,y,z):
            qml.Rot(x,y,z, wires=[0])
            return qml.expval(qml.PauliZ(0))

        grad_fn = autograd.grad(circuit, argnum=[0,1,2])

        eye = np.eye(3)
        for theta in thetas:
            angle_inputs = np.array([theta, theta ** 3, np.sqrt(2) * theta])
            autograd_val = grad_fn(*angle_inputs)
            for idx in range(3):
                onehot_idx = eye[idx]
                param1 = angle_inputs + np.pi / 2 * onehot_idx
                param2 = angle_inputs - np.pi / 2 * onehot_idx
                manualgrad_val = (circuit(*param1) - circuit(*param2)) / 2
                assert autograd_val[idx] == pytest.approx(manualgrad_val, abs=tol)

    def test_qfunc_gradients(self, qubit2_dev, tol):
        "Tests that the various ways of computing the gradient of a qfunc all agree."

        def circuit(x, y, z):
            qml.RX(x, wires=[0])
            qml.CNOT(wires=[0, 1])
            qml.RY(-1.6, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[1, 0])
            qml.RX(z, wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        qnode = qml.QNode(circuit, qubit2_dev)
        params = np.array([0.1, -1.6, np.pi / 5])

        # manual gradients
        grad_fd1 = qnode.jacobian(params, method='F', order=1)
        grad_fd2 = qnode.jacobian(params, method='F', order=2)
        grad_angle = qnode.jacobian(params, method='A')

        # automatic gradient
        grad_fn = autograd.grad(qnode.evaluate)
        grad_auto = grad_fn(params)[np.newaxis, :]  # so shapes will match

        # gradients computed with different methods must agree
        assert grad_fd1 == pytest.approx(grad_fd2, abs=tol)
        assert grad_fd1 == pytest.approx(grad_angle, abs=tol)
        assert grad_fd1 == pytest.approx(grad_auto, abs=tol)

    def test_hybrid_gradients(self, qubit2_dev, tol):
        "Tests that the various ways of computing the gradient of a hybrid computation all agree."

        # input data is the first parameter
        def classifier_circuit(in_data, x):
            qml.RX(in_data, wires=[0])
            qml.CNOT(wires=[0, 1])
            qml.RY(-1.6, wires=[0])
            qml.RY(in_data, wires=[1])
            qml.CNOT(wires=[1, 0])
            qml.RX(x, wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        classifier = qml.QNode(classifier_circuit, qubit2_dev)

        param = -0.1259
        in_data = np.array([-0.1, -0.88, np.exp(0.5)])
        out_data = np.array([1.5, np.pi / 3, 0.0])

        def error(p):
            "Total square error of classifier predictions."
            ret = 0
            for d_in, d_out in zip(in_data, out_data):
                #args = np.array([d_in, p])
                square_diff = (classifier(d_in, p) - d_out) ** 2
                ret = ret + square_diff
            return ret

        def d_error(p, grad_method):
            "Gradient of error, computed manually."
            ret = 0
            for d_in, d_out in zip(in_data, out_data):
                args = np.array([d_in, p])
                diff = (classifier(*args) - d_out)
                ret = ret + 2 * diff * classifier.jacobian(args, which=[1], method=grad_method)
            return ret

        y0 = error(param)
        grad = autograd.grad(error)
        grad_auto = grad(param)

        grad_fd1 = d_error(param, 'F')
        grad_angle = d_error(param, 'A')

        # gradients computed with different methods must agree
        assert grad_fd1 == pytest.approx(grad_angle, abs=tol)
        assert grad_fd1 == pytest.approx(grad_auto, abs=tol)
        assert grad_angle == pytest.approx(grad_auto, abs=tol)

    def test_qnode_gradient_fanout(self, qubit1_dev, tol):
        "Tests that the correct gradient is computed for qnodes which use the same parameter in multiple gates."

        extra_param = 0.31
        def circuit(reused_param, other_param):
            qml.RX(extra_param, wires=[0])
            qml.RY(reused_param, wires=[0])
            qml.RZ(other_param, wires=[0])
            qml.RX(reused_param, wires=[0])
            return qml.expval(qml.PauliZ(0))

        f = qml.QNode(circuit, qubit1_dev)
        zero_state = np.array([1., 0.])

        for reused_p in thetas:
            reused_p = reused_p ** 3 / 19
            for other_p in thetas:
                other_p = other_p ** 2 / 11

                # autograd gradient
                grad = autograd.grad(f)
                grad_eval = grad(reused_p, other_p)

                # manual gradient
                grad_true0 = (expZ(Rx(reused_p) @ Rz(other_p) @ Ry(reused_p + np.pi / 2) @ Rx(extra_param) @ zero_state) \
                             -expZ(Rx(reused_p) @ Rz(other_p) @ Ry(reused_p - np.pi / 2) @ Rx(extra_param) @ zero_state)) / 2
                grad_true1 = (expZ(Rx(reused_p + np.pi / 2) @ Rz(other_p) @ Ry(reused_p) @ Rx(extra_param) @ zero_state) \
                             -expZ(Rx(reused_p - np.pi / 2) @ Rz(other_p) @ Ry(reused_p) @ Rx(extra_param) @ zero_state)) / 2
                grad_true = grad_true0 + grad_true1 # product rule

                assert grad_eval == pytest.approx(grad_true, abs=tol)

    def test_gradient_exception_on_sample(self, qubit2_dev):
        """Tests that the proper exception is raised if differentiation of sampling is attempted."""

        @qml.qnode(qubit2_dev)
        def circuit(x):
            qml.RX(x, wires=[0])
            return qml.sample(qml.PauliZ(0)), qml.sample(qml.PauliX(1))

        with pytest.raises(qml.QuantumFunctionError,
                           match="Circuits that include sampling can not be differentiated."):
            grad_fn = autograd.jacobian(circuit)
            grad_fn(1.0)

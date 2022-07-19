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
Unit tests for computing Autograd gradients of quantum functions.
"""

import pytest
import autograd
import pennylane.numpy as anp  # only to be used inside classical computational nodes
import numpy as np

import pennylane as qml
from gate_data import Rotx as Rx, Roty as Ry, Rotz as Rz


alpha = 0.5  # displacement in tests
hbar = 2
mag_alphas = np.linspace(0, 1.5, 5)
thetas = np.linspace(-2 * np.pi, 2 * np.pi, 8)
sqz_vals = np.linspace(0.0, 1.0, 5)


class PolyN(qml.ops.PolyXP):
    "Mimics NumberOperator using the arbitrary 2nd order observable interface. Results should be identical."

    def __init__(self, wires):
        hbar = 2
        q = np.diag([-0.5, 0.5 / hbar, 0.5 / hbar])
        super().__init__(q, wires=wires)
        self.name = "PolyXP"


@pytest.fixture(scope="module")
def gaussian_dev():
    return qml.device("default.gaussian", wires=2)


@pytest.fixture(scope="module")
def grad_fn_R(gaussian_dev):
    @qml.qnode(gaussian_dev)
    def circuit(y):
        qml.Displacement(alpha, 0.0, wires=[0])
        qml.Rotation(y, wires=[0])
        return qml.expval(qml.X(0))

    return autograd.grad(circuit)


@pytest.fixture(scope="module")
def grad_fn_BS(gaussian_dev):
    @qml.qnode(gaussian_dev)
    def circuit(y):
        qml.Displacement(alpha, 0.0, wires=[0])
        qml.Beamsplitter(y, 0, wires=[0, 1])
        return qml.expval(qml.X(0))

    return autograd.grad(circuit)


@pytest.fixture(scope="module")
def grad_fn_D(gaussian_dev):
    @qml.qnode(gaussian_dev)
    def circuit(r, phi):
        qml.Displacement(r, phi, wires=[0])
        return qml.expval(qml.X(0))

    return autograd.grad(circuit)


@pytest.fixture(scope="module")
def grad_fn_S(gaussian_dev):
    @qml.qnode(gaussian_dev)
    def circuit(y):
        qml.Displacement(alpha, 0.0, wires=[0])
        qml.Squeezing(y, 0.0, wires=[0])
        return qml.expval(qml.X(0))

    return autograd.grad(circuit)


@pytest.fixture(scope="module")
def grad_fn_S_Fock(gaussian_dev):
    @qml.qnode(gaussian_dev)
    def circuit(y):
        qml.Squeezing(y, 0.0, wires=[0])
        return qml.expval(qml.FockStateProjector(np.array([2, 0]), wires=[0, 1]))

    return autograd.grad(circuit)


class TestCVGradient:
    """Tests of the automatic gradient method for CV circuits."""

    @pytest.mark.parametrize("theta", thetas)
    def test_rotation_gradient(self, theta, grad_fn_R, tol):
        "Tests that the automatic gradient of a phase space rotation is correct."

        autograd_val = grad_fn_R(theta)
        # qfunc evalutes to hbar * alpha * cos(theta)
        manualgrad_val = -hbar * alpha * np.sin(theta)
        assert autograd_val == pytest.approx(manualgrad_val, abs=tol)

    @pytest.mark.parametrize("theta", thetas)
    def test_beamsplitter_gradient(self, theta, grad_fn_BS, tol):
        "Tests that the automatic gradient of a beamsplitter is correct."

        autograd_val = grad_fn_BS(theta)
        # qfunc evalutes to hbar * alpha * cos(theta)
        manualgrad_val = -hbar * alpha * np.sin(theta)
        assert autograd_val == pytest.approx(manualgrad_val, abs=tol)

    @pytest.mark.parametrize("mag", mag_alphas)
    @pytest.mark.parametrize("theta", thetas)
    def test_displacement_gradient(self, mag, theta, grad_fn_D, tol):
        "Tests that the automatic gradient of a phase space displacement is correct."

        # alpha = mag * np.exp(1j * theta)
        autograd_val = grad_fn_D(mag, theta)
        # qfunc evalutes to hbar * Re(alpha)
        manualgrad_val = hbar * np.cos(theta)
        assert autograd_val == pytest.approx(manualgrad_val, abs=tol)

    @pytest.mark.parametrize("r", sqz_vals)
    def test_squeeze_gradient(self, r, grad_fn_S, tol):
        "Tests that the automatic gradient of a phase space squeezing is correct."

        autograd_val = grad_fn_S(r)
        # qfunc evaluates to -exp(-r) * hbar * Re(alpha)
        manualgrad_val = -np.exp(-r) * hbar * alpha
        assert autograd_val == pytest.approx(manualgrad_val, abs=tol)

    @pytest.mark.parametrize("r", sqz_vals[1:])  # formula is not valid for r=0
    def test_number_state_gradient(self, r, grad_fn_S_Fock, tol):
        "Tests that the automatic gradient of a squeezed state with number state expectation is correct."

        # (d/dr) |<2|S(r)>|^2 = 0.5 tanh(r)^3 (2 csch(r)^2 - 1) sech(r)
        autograd_val = grad_fn_S_Fock(r)
        manualgrad_val = 0.5 * np.tanh(r) ** 3 * (2 / (np.sinh(r) ** 2) - 1) / np.cosh(r)
        assert autograd_val == pytest.approx(manualgrad_val, abs=tol)

    @pytest.mark.autograd
    @pytest.mark.parametrize("O", [qml.ops.X, qml.ops.NumberOperator])
    @pytest.mark.parametrize(
        "make_gate",
        [lambda x: qml.Rotation(x, wires=0), lambda x: qml.ControlledPhase(x, wires=[0, 1])],
    )
    def test_cv_gradients_gaussian_circuit(self, make_gate, O, gaussian_dev, tol):
        """Tests that the gradients of circuits of gaussian gates match
        between the finite difference and analytic methods."""

        tol = 1e-5
        par = anp.array(0.4, requires_grad=True)

        def circuit(x):
            qml.Displacement(0.5, 0, wires=0)
            make_gate(x)
            qml.Beamsplitter(1.3, -2.3, wires=[0, 1])
            qml.Displacement(-0.5, 0.1, wires=0)
            qml.Squeezing(0.5, -1.5, wires=0)
            qml.Rotation(-1.1, wires=0)
            return qml.expval(O(wires=0))

        q = qml.QNode(circuit, gaussian_dev)
        val = q(par)

        grad_F = qml.gradients.finite_diff(q)(par)
        grad_A2 = qml.gradients.param_shift_cv(q, dev=gaussian_dev, force_order2=True)(par)
        if O.ev_order == 1:
            grad_A = qml.gradients.param_shift_cv(q, dev=gaussian_dev)(par)
            # the different methods agree
            assert qml.math.shape(grad_A) == qml.math.shape(grad_F) == ()
            assert np.allclose(grad_A, grad_F, atol=tol)

        # the different methods agree
        assert qml.math.shape(grad_A2) == qml.math.shape(grad_F) == ()
        assert np.allclose(grad_A2, grad_F, atol=tol)

    @pytest.mark.autograd
    def test_cv_gradients_multiple_gate_parameters(self, gaussian_dev, tol):
        "Tests that gates with multiple free parameters yield correct gradients."
        par = anp.array([0.4, -0.3, -0.7, 0.2], requires_grad=True)

        def qf(r0, phi0, r1, phi1):
            qml.Squeezing(r0, phi0, wires=[0])
            qml.Squeezing(r1, phi1, wires=[0])
            return qml.expval(qml.NumberOperator(0))

        q = qml.QNode(qf, gaussian_dev)
        q(*par)
        grad_F = qml.gradients.finite_diff(q)(*par)
        grad_A = qml.gradients.param_shift_cv(q, dev=gaussian_dev)(*par)
        grad_A2 = qml.gradients.param_shift_cv(q, dev=gaussian_dev, force_order2=True)(*par)

        # the different methods agree
        assert qml.math.allclose(grad_A, grad_F, atol=tol, rtol=0)
        assert qml.math.allclose(grad_A2, grad_F, atol=tol, rtol=0)

        # check against the known analytic formula
        r0, phi0, r1, phi1 = par
        dn = np.zeros([4])
        dn[0] = np.cosh(2 * r1) * np.sinh(2 * r0) + np.cos(phi0 - phi1) * np.cosh(2 * r0) * np.sinh(
            2 * r1
        )
        dn[1] = -0.5 * np.sin(phi0 - phi1) * np.sinh(2 * r0) * np.sinh(2 * r1)
        dn[2] = np.cos(phi0 - phi1) * np.cosh(2 * r1) * np.sinh(2 * r0) + np.cosh(2 * r0) * np.sinh(
            2 * r1
        )
        dn[3] = 0.5 * np.sin(phi0 - phi1) * np.sinh(2 * r0) * np.sinh(2 * r1)

        assert all(qml.math.isclose(dn[i], grad_F[i], atol=tol, rtol=0) for i in range(4))

    @pytest.mark.autograd
    def test_cv_gradients_repeated_gate_parameters(self, gaussian_dev, tol):
        "Tests that repeated use of a free parameter in a multi-parameter gate yield correct gradients."
        par = anp.array([0.2, 0.3], requires_grad=True)

        def qf(x, y):
            qml.Displacement(x, 0, wires=[0])
            qml.Squeezing(y, -1.3 * y, wires=[0])
            return qml.expval(qml.X(0))

        q = qml.QNode(qf, gaussian_dev)
        q(*par)
        grad_F = qml.gradients.finite_diff(q)(*par)
        grad_A = qml.gradients.param_shift_cv(q, dev=gaussian_dev)(*par)
        grad_A2 = qml.gradients.param_shift_cv(q, dev=gaussian_dev, force_order2=True)(*par)

        # the different methods agree
        assert qml.math.allclose(grad_A, grad_F, atol=tol, rtol=0)
        assert qml.math.allclose(grad_A2, grad_F, atol=tol, rtol=0)

    @pytest.mark.autograd
    def test_cv_gradients_parameters_inside_array(self, gaussian_dev, tol):
        "Tests that free parameters inside an array passed to an Operation yield correct gradients."
        par = anp.array([0.4, 1.3], requires_grad=True)

        def qf(x, y):
            qml.Displacement(0.5, 0, wires=[0])
            qml.Squeezing(x, 0, wires=[0])
            M = np.zeros((5, 5))
            M[1, 1] = y
            M[1, 2] = 1.0
            M[2, 1] = 1.0
            return qml.expval(qml.PolyXP(M, [0, 1]))

        q = qml.QNode(qf, gaussian_dev)
        q(*par)
        grad_F = qml.gradients.finite_diff(q, hybrid=False)(*par)
        grad_A = qml.gradients.param_shift_cv(q, dev=gaussian_dev, hybrid=False)(*par)
        grad_A2 = qml.gradients.param_shift_cv(
            q, dev=gaussian_dev, force_order2=True, hybrid=False
        )(*par)

        # the different methods agree
        assert grad_A2 == pytest.approx(grad_F, abs=tol)

    @pytest.mark.autograd
    def test_cv_gradient_fanout(self, gaussian_dev, tol):
        "Tests that qnodes can compute the correct gradient when the same parameter is used in multiple gates."
        par = anp.array([0.5, 1.3], requires_grad=True)

        def circuit(x, y):
            qml.Displacement(x, 0, wires=[0])
            qml.Rotation(y, wires=[0])
            qml.Displacement(0, x, wires=[0])
            return qml.expval(qml.X(0))

        q = qml.QNode(circuit, gaussian_dev)
        q(*par)
        grad_F = qml.gradients.finite_diff(q)(*par)
        grad_A = qml.gradients.param_shift_cv(q, dev=gaussian_dev)(*par)
        grad_A2 = qml.gradients.param_shift_cv(q, dev=gaussian_dev, force_order2=True)(*par)

        # the different methods agree
        assert qml.math.allclose(grad_A, grad_F, atol=tol, rtol=0)
        assert qml.math.allclose(grad_A2, grad_F, atol=tol, rtol=0)

    @pytest.mark.autograd
    def test_CVOperation_with_heisenberg_and_no_params(self, gaussian_dev, tol):
        """An integration test for InterferometerUnitary, a gate that supports analytic
        differentiation if succeeding the gate to be differentiated, but cannot be
        differentiated itself.

        This ensures that, assuming the _heisenberg_rep is defined, the quantum
        gradient analytic method can still be used, and returns the correct result.
        """

        x = anp.array(0.5, requires_grad=True)

        U = np.array(
            [
                [0.51310276 + 0.81702166j, 0.13649626 + 0.22487759j],
                [0.26300233 + 0.00556194j, -0.96414101 - 0.03508489j],
            ]
        )

        def circuit(x):
            qml.Displacement(x, 0, wires=0)
            qml.InterferometerUnitary(U, wires=[0, 1])
            return qml.expval(qml.X(0))

        qnode = qml.QNode(circuit, gaussian_dev)
        qnode(x)
        grad_F = qml.gradients.finite_diff(qnode)(x)
        grad_A = qml.gradients.param_shift_cv(qnode, dev=gaussian_dev)(x)
        grad_A2 = qml.gradients.param_shift_cv(qnode, dev=gaussian_dev, force_order2=True)(x)

        # the different methods agree
        assert qml.math.shape(grad_A) == qml.math.shape(grad_A2) == qml.math.shape(grad_F) == ()
        assert np.allclose(grad_A, grad_F, atol=tol)
        assert np.allclose(grad_A2, grad_F, atol=tol)


class TestQubitGradient:
    """Tests of the automatic gradient method for qubit gates."""

    def test_RX_gradient(self, qubit_device_1_wire, tol):
        "Tests that the automatic gradient of a Pauli X-rotation is correct."

        @qml.qnode(qubit_device_1_wire)
        def circuit(x):
            qml.RX(x, wires=[0])
            return qml.expval(qml.PauliZ(0))

        grad_fn = autograd.grad(circuit)

        for theta in thetas:
            autograd_val = grad_fn(theta)
            manualgrad_val = (circuit(theta + np.pi / 2) - circuit(theta - np.pi / 2)) / 2
            assert autograd_val == pytest.approx(manualgrad_val, abs=tol)

    def test_RY_gradient(self, qubit_device_1_wire, tol):
        "Tests that the automatic gradient of a Pauli Y-rotation is correct."

        @qml.qnode(qubit_device_1_wire)
        def circuit(x):
            qml.RY(x, wires=[0])
            return qml.expval(qml.PauliZ(0))

        grad_fn = autograd.grad(circuit)

        for theta in thetas:
            autograd_val = grad_fn(theta)
            manualgrad_val = (circuit(theta + np.pi / 2) - circuit(theta - np.pi / 2)) / 2
            assert autograd_val == pytest.approx(manualgrad_val, abs=tol)

    def test_RZ_gradient(self, qubit_device_1_wire, tol):
        "Tests that the automatic gradient of a Pauli Z-rotation is correct."

        @qml.qnode(qubit_device_1_wire)
        def circuit(x):
            qml.RZ(x, wires=[0])
            return qml.expval(qml.PauliZ(0))

        grad_fn = autograd.grad(circuit)

        for theta in thetas:
            autograd_val = grad_fn(theta)
            manualgrad_val = (circuit(theta + np.pi / 2) - circuit(theta - np.pi / 2)) / 2
            assert autograd_val == pytest.approx(manualgrad_val, abs=tol)

    def test_Rot(self, qubit_device_1_wire, tol):
        "Tests that the automatic gradient of a arbitrary Euler-angle-parameterized gate is correct."

        @qml.qnode(qubit_device_1_wire)
        def circuit(x, y, z):
            qml.Rot(x, y, z, wires=[0])
            return qml.expval(qml.PauliZ(0))

        grad_fn = autograd.grad(circuit, argnum=[0, 1, 2])

        eye = np.eye(3)
        for theta in thetas:
            angle_inputs = np.array([theta, theta**3, np.sqrt(2) * theta])
            autograd_val = grad_fn(*angle_inputs)
            for idx in range(3):
                onehot_idx = eye[idx]
                param1 = angle_inputs + np.pi / 2 * onehot_idx
                param2 = angle_inputs - np.pi / 2 * onehot_idx
                manualgrad_val = (circuit(*param1) - circuit(*param2)) / 2
                assert autograd_val[idx] == pytest.approx(manualgrad_val, abs=tol)

    def test_U2(self, tol):
        """Tests that the gradient of an arbitrary U2 gate is correct"""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(x, y):
            qml.QubitStateVector(1j * np.array([1, -1]) / np.sqrt(2), wires=[0])
            qml.U2(x, y, wires=[0])
            return qml.expval(qml.PauliX(0))

        phi = -0.234
        lam = 0.654

        res = circuit(phi, lam)
        expected = np.sin(lam) * np.sin(phi)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        grad_fn = autograd.grad(circuit, argnum=[0, 1])
        res = grad_fn(phi, lam)
        expected = np.array([np.sin(lam) * np.cos(phi), np.cos(lam) * np.sin(phi)])
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_U3(self, tol):
        """Tests that the gradient of an arbitrary U3 gate is correct"""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(x, y, z):
            qml.QubitStateVector(1j * np.array([1, -1]) / np.sqrt(2), wires=[0])
            qml.U3(x, y, z, wires=[0])
            return qml.expval(qml.PauliX(0))

        theta = 0.543
        phi = -0.234
        lam = 0.654

        res = circuit(theta, phi, lam)
        expected = np.sin(lam) * np.sin(phi) - np.cos(theta) * np.cos(lam) * np.cos(phi)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        grad_fn = autograd.grad(circuit, argnum=[0, 1, 2])
        res = grad_fn(theta, phi, lam)
        expected = np.array(
            [
                np.sin(theta) * np.cos(lam) * np.cos(phi),
                np.cos(theta) * np.cos(lam) * np.sin(phi) + np.sin(lam) * np.cos(phi),
                np.cos(theta) * np.sin(lam) * np.cos(phi) + np.cos(lam) * np.sin(phi),
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.autograd
    def test_qfunc_gradients(self, qubit_device_2_wires, tol):
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

        qnode = qml.QNode(circuit, qubit_device_2_wires, diff_method="parameter-shift")
        params = anp.array([0.1, -1.6, np.pi / 5], requires_grad=True)

        h = 1e-4 if qubit_device_2_wires.R_DTYPE is np.float32 else 1e-7

        # manual gradients
        grad_fd1 = qml.gradients.finite_diff(qnode, h=h, approx_order=1)(*params)
        grad_fd2 = qml.gradients.finite_diff(qnode, h=h, approx_order=2)(*params)
        grad_angle = qml.gradients.param_shift(qnode)(*params)

        # automatic gradient
        grad_fn = qml.grad(qnode)
        grad_auto = grad_fn(*params)

        # gradients computed with different methods must agree
        assert qml.math.allclose(grad_fd1, grad_fd2, atol=tol, rtol=0)
        assert qml.math.allclose(grad_fd1, grad_angle, atol=tol, rtol=0)
        assert qml.math.allclose(grad_fd1, grad_auto, atol=tol, rtol=0)

    @pytest.mark.autograd
    def test_hybrid_gradients(self, qubit_device_2_wires, tol):
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

        classifier = qml.QNode(
            classifier_circuit, qubit_device_2_wires, diff_method="parameter-shift"
        )

        param = anp.array(-0.1259, requires_grad=True)
        in_data = qml.numpy.array([-0.1, -0.88, np.exp(0.5)], requires_grad=False)
        out_data = np.array([1.5, np.pi / 3, 0.0])

        def error(p):
            "Total square error of classifier predictions."
            ret = 0
            for d_in, d_out in zip(in_data, out_data):
                square_diff = (classifier(d_in, p) - d_out) ** 2
                ret = ret + square_diff
            return ret

        def d_error(p, grad_method):
            "Gradient of error, computed manually."
            ret = 0
            for d_in, d_out in zip(in_data, out_data):
                args = (d_in, p)
                diff = classifier(*args) - d_out
                ret = ret + 2 * diff * grad_method(classifier)(d_in, p)
            return ret

        y0 = error(param)
        grad = autograd.grad(error)
        grad_auto = grad(param)

        h = 1e-3 if qubit_device_2_wires.R_DTYPE is np.float32 else 1e-7

        grad_fd1 = d_error(param, lambda x: qml.gradients.finite_diff(x, h=h))
        grad_angle = d_error(param, qml.gradients.param_shift)

        # gradients computed with different methods must agree
        assert grad_fd1 == pytest.approx(grad_angle, abs=tol)
        assert grad_fd1 == pytest.approx(grad_auto, abs=tol)
        assert grad_angle == pytest.approx(grad_auto, abs=tol)

    @pytest.mark.autograd
    def test_hybrid_gradients_autograd_numpy(self, qubit_device_2_wires):
        "Test the gradient of a hybrid computation requiring autograd.numpy functions."

        def circuit(x, y):
            "Quantum node."
            qml.RX(x, wires=[0])
            qml.CNOT(wires=[0, 1])
            qml.RY(y, wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        quantum = qml.QNode(circuit, qubit_device_2_wires, diff_method="parameter-shift")

        def classical(p):
            "Classical node, requires autograd.numpy functions."
            return anp.exp(anp.sum(quantum(p[0], anp.log(p[1]))))

        def d_classical(a, b, grad_method):
            "Gradient of classical computed symbolically, can use normal numpy functions."
            val = classical((a, b))
            J = grad_method(quantum)(a, np.log(b))
            return val * np.array([J[0][0] + J[0][1], (J[1][0] + J[1][1]) / b])

        param = anp.array([-0.1259, 1.53], requires_grad=True)
        y0 = classical(param)
        grad_classical = autograd.jacobian(classical)
        grad_auto = grad_classical(param)

        h = 5e-4 if qubit_device_2_wires.R_DTYPE is np.float32 else 1e-7
        atol = 2e-3 if qubit_device_2_wires.R_DTYPE is np.float32 else 1e-5

        grad_fd1 = d_classical(*param, lambda x: qml.gradients.finite_diff(x, h=h))
        grad_angle = d_classical(*param, qml.gradients.param_shift)

        # gradients computed with different methods must agree
        assert qml.math.allclose(grad_fd1, grad_angle, atol=atol, rtol=0)
        assert qml.math.allclose(grad_fd1, grad_auto, atol=atol, rtol=0)
        assert qml.math.allclose(grad_angle, grad_auto, atol=atol, rtol=0)

    def test_qnode_gradient_fanout(self, qubit_device_1_wire, tol):
        "Tests that the correct gradient is computed for qnodes which use the same parameter in multiple gates."

        def expZ(state):
            return np.abs(state[0]) ** 2 - np.abs(state[1]) ** 2

        extra_param = 0.31

        def circuit(reused_param, other_param):
            qml.RX(extra_param, wires=[0])
            qml.RY(reused_param, wires=[0])
            qml.RZ(other_param, wires=[0])
            qml.RX(reused_param, wires=[0])
            return qml.expval(qml.PauliZ(0))

        f = qml.QNode(circuit, qubit_device_1_wire)
        zero_state = np.array([1.0, 0.0])

        for reused_p in thetas:
            reused_p = reused_p**3 / 19
            for other_p in thetas:
                other_p = other_p**2 / 11

                # autograd gradient
                grad = autograd.grad(f)
                grad_eval = grad(reused_p, other_p)

                # manual gradient
                grad_true0 = (
                    expZ(
                        Rx(reused_p)
                        @ Rz(other_p)
                        @ Ry(reused_p + np.pi / 2)
                        @ Rx(extra_param)
                        @ zero_state
                    )
                    - expZ(
                        Rx(reused_p)
                        @ Rz(other_p)
                        @ Ry(reused_p - np.pi / 2)
                        @ Rx(extra_param)
                        @ zero_state
                    )
                ) / 2
                grad_true1 = (
                    expZ(
                        Rx(reused_p + np.pi / 2)
                        @ Rz(other_p)
                        @ Ry(reused_p)
                        @ Rx(extra_param)
                        @ zero_state
                    )
                    - expZ(
                        Rx(reused_p - np.pi / 2)
                        @ Rz(other_p)
                        @ Ry(reused_p)
                        @ Rx(extra_param)
                        @ zero_state
                    )
                ) / 2
                grad_true = grad_true0 + grad_true1  # product rule

                assert grad_eval == pytest.approx(grad_true, abs=tol)

    @pytest.mark.autograd
    def test_autograd_trainable_no_warn_grad(self, recwarn):
        """Test that no warning is raised if positional arguments are marked as
        trainable using the requires_grad attribute."""
        dev = qml.device("default.qubit", wires=5)

        @qml.qnode(dev)
        def test(x):
            qml.RZ(x, wires=[0])
            return qml.expval(qml.PauliZ(0))

        par = anp.array(0.3, requires_grad=True)
        qml.grad(test)(par)
        assert len(recwarn) == 0

    @pytest.mark.autograd
    def test_autograd_trainable_argnum_no_warn_grad(self, recwarn):
        """Test that no warning is raised if positional arguments are marked as
        trainable using the argnum argument."""
        dev = qml.device("default.qubit", wires=5)

        @qml.qnode(dev)
        def test(x):
            qml.RZ(x, wires=[0])
            return qml.expval(qml.PauliZ(0))

        par = np.array(0.3)
        qml.grad(test, argnum=0)(par)
        assert len(recwarn) == 0

    @pytest.mark.autograd
    def test_autograd_trainable_no_warn_jacobian(self, recwarn):
        """Test that no warning is raised if positional arguments are marked as
        trainable using the requires_grad attribute."""
        dev = qml.device("default.qubit", wires=5)

        @qml.qnode(dev)
        def test(x):
            qml.RZ(x, wires=[0])
            return qml.probs(wires=[0])

        par = anp.array(0.3, requires_grad=True)
        qml.jacobian(test)(par)
        assert len(recwarn) == 0

    @pytest.mark.autograd
    def test_autograd_trainable_argnum_no_warn_jacobian(self, recwarn):
        """Test that no warning is raised if positional arguments are marked as
        trainable using the argnum argument."""
        dev = qml.device("default.qubit", wires=5)

        @qml.qnode(dev)
        def test(x):
            qml.RZ(x, wires=[0])
            return qml.probs(wires=[0])

        par = np.array(0.3)
        qml.jacobian(test, argnum=0)(par)
        assert len(recwarn) == 0


class TestFourTermParameterShifts:
    """Tests for quantum gradients that require a 4-term shift formula"""

    @pytest.mark.autograd
    @pytest.mark.parametrize("G", [qml.CRX, qml.CRY, qml.CRZ])
    def test_controlled_rotation_gradient(self, G, tol):
        """Test gradient of controlled RX gate"""
        dev = qml.device("default.qubit", wires=2)
        b = anp.array(0.123, requires_grad=True)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(b):
            qml.QubitStateVector(np.array([1.0, -1.0]) / np.sqrt(2), wires=0)
            G(b, wires=[0, 1])
            return qml.expval(qml.PauliX(0))

        res = circuit(b)
        assert np.allclose(res, -np.cos(b / 2), atol=tol, rtol=0)

        grad = qml.grad(circuit)(b)
        expected = np.sin(b / 2) / 2
        assert np.allclose(grad, expected, atol=tol, rtol=0)

    @pytest.mark.autograd
    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, np.pi, 7))
    def test_CRot_gradient(self, theta, tol):
        """Tests that the automatic gradient of a arbitrary controlled Euler-angle-parameterized
        gate is correct."""
        dev = qml.device("default.qubit", wires=2)
        a, b, c = anp.array([theta, theta**3, np.sqrt(2) * theta], requires_grad=True)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(a, b, c):
            qml.QubitStateVector(np.array([1.0, -1.0]) / np.sqrt(2), wires=0)
            qml.CRot(a, b, c, wires=[0, 1])
            return qml.expval(qml.PauliX(0))

        res = circuit(a, b, c)
        expected = -np.cos(b / 2) * np.cos(0.5 * (a + c))
        assert np.allclose(res, expected, atol=tol, rtol=0)

        grad = qml.grad(circuit)(a, b, c)
        expected = np.array(
            [
                [
                    0.5 * np.cos(b / 2) * np.sin(0.5 * (a + c)),
                    0.5 * np.sin(b / 2) * np.cos(0.5 * (a + c)),
                    0.5 * np.cos(b / 2) * np.sin(0.5 * (a + c)),
                ]
            ]
        )
        assert np.allclose(grad, expected, atol=tol, rtol=0)


@pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift", "finite-diff"])
@pytest.mark.parametrize("argnum", [0, 2, [1], [0, 2], [0, 1, 2]])
class TestArgnum:
    """Test various argnum scenarios for qml.grad and qml.jacobian and compare output
    to equivalent requires_grad configuration of the inputs."""

    @pytest.mark.autograd
    @pytest.mark.parametrize(
        "input",
        [
            ((0.1, 0.2), (0.1, 0.2)),
            [[0.1, 0.2], [0.1, 0.2]],
            np.array([[0.1, 0.2], [0.1, 0.2]]),
            qml.numpy.tensor([[0.1, 0.2], [0.1, 0.2]]),
        ],
    )
    def test_grad(self, diff_method, input, argnum):
        """Test qml.grad with various argnums"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(x, y, z):
            qml.RX(z[0][1], wires=0)
            qml.RZ(x[0][1], wires=0)
            qml.RY(y[0][1], wires=0)
            return qml.expval(qml.PauliZ(0))

        result = qml.grad(circuit, argnum=argnum)(input, input, input)

        params = [input, input, input]
        for i in range(3):
            if i in ([argnum] if isinstance(argnum, int) else argnum):
                params[i] = qml.numpy.tensor(params[i], requires_grad=True)
            else:
                params[i] = qml.numpy.tensor(params[i], requires_grad=False)

        expected = qml.grad(circuit)(*params)

        assert np.allclose(result, expected)

    @pytest.mark.autograd
    @pytest.mark.parametrize(
        "input",
        [np.array([[0.1, 0.2], [0.1, 0.2]]), qml.numpy.tensor([[0.1, 0.2], [0.1, 0.2]])],
    )
    def test_jacobian(self, diff_method, input, argnum):
        """Test qml.jacobian with various argnums (no support for lists/tuples)"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(x, y, z):
            qml.RX(z[0][1], wires=0)
            qml.RZ(x[0][1], wires=0)
            qml.RY(y[0][1], wires=0)
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        result = qml.jacobian(circuit, argnum=argnum)(input, input, input)

        params = [input, input, input]
        for i in range(3):
            if i in ([argnum] if isinstance(argnum, int) else argnum):
                params[i] = qml.numpy.tensor(params[i], requires_grad=True)
            else:
                params[i] = qml.numpy.tensor(params[i], requires_grad=False)

        expected = qml.jacobian(circuit)(*params)

        assert np.allclose(result, expected)

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
# pylint: disable=no-value-for-parameter

import autograd
import numpy as np
import pytest

import pennylane as qml
import pennylane.numpy as anp  # only to be used inside classical computational nodes

alpha = 0.5  # displacement in tests
hbar = 2
mag_alphas = np.linspace(0, 1.5, 5)
thetas = np.linspace(-2 * np.pi, 2 * np.pi, 8)
sqz_vals = np.linspace(0.0, 1.0, 5)


# pylint: disable=too-few-public-methods
class PolyN(qml.ops.PolyXP):
    "Mimics NumberOperator using the arbitrary 2nd order observable interface. Results should be identical."

    def __init__(self, wires):
        q = np.diag([-0.5, 0.5 / hbar, 0.5 / hbar])
        super().__init__(q, wires=wires)
        self.name = "PolyXP"


@pytest.fixture(scope="module", name="gaussian_dev")
def gaussian_dev_fixture():
    """Gaussian device."""
    return qml.device("default.gaussian", wires=2)


@pytest.fixture(scope="module", name="grad_fn_R")
def grad_fn_R_fixture(gaussian_dev):
    """Gradient with autograd."""

    @qml.qnode(gaussian_dev)
    def circuit(y):
        qml.Displacement(alpha, 0.0, wires=[0])
        qml.Rotation(y, wires=[0])
        return qml.expval(qml.QuadX(0))

    return autograd.grad(circuit)


@pytest.fixture(scope="module", name="grad_fn_BS")
def grad_fn_BS_fixture(gaussian_dev):
    """Gradient with autograd."""

    @qml.qnode(gaussian_dev)
    def circuit(y):
        qml.Displacement(alpha, 0.0, wires=[0])
        qml.Beamsplitter(y, 0, wires=[0, 1])
        return qml.expval(qml.QuadX(0))

    return autograd.grad(circuit)


@pytest.fixture(scope="module", name="grad_fn_D")
def grad_fn_D_fixture(gaussian_dev):
    """Gradient with autograd."""

    @qml.qnode(gaussian_dev)
    def circuit(r, phi):
        qml.Displacement(r, phi, wires=[0])
        return qml.expval(qml.QuadX(0))

    return autograd.grad(circuit)


@pytest.fixture(scope="module", name="grad_fn_S")
def grad_fn_S_fixture(gaussian_dev):
    """Gradient with autograd."""

    @qml.qnode(gaussian_dev)
    def circuit(y):
        qml.Displacement(alpha, 0.0, wires=[0])
        qml.Squeezing(y, 0.0, wires=[0])
        return qml.expval(qml.QuadX(0))

    return autograd.grad(circuit)


@pytest.fixture(scope="module", name="grad_fn_S_Fock")
def grad_fn_S_Fock_fixture(gaussian_dev):
    """Gradient with autograd."""

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
    @pytest.mark.parametrize("O", [qml.ops.QuadX, qml.ops.NumberOperator])
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
            return qml.expval(qml.QuadX(0))

        q = qml.QNode(qf, gaussian_dev)
        q(*par)
        grad_F = qml.gradients.finite_diff(q)(*par)
        grad_A = qml.gradients.param_shift_cv(q, dev=gaussian_dev)(*par)
        grad_A2 = qml.gradients.param_shift_cv(q, dev=gaussian_dev, force_order2=True)(*par)

        # the different methods agree
        assert qml.math.allclose(grad_A, grad_F, atol=tol, rtol=0)
        assert qml.math.allclose(grad_A2, grad_F, atol=tol, rtol=0)

    @pytest.mark.jax
    def test_cv_gradients_parameters_inside_array(self, gaussian_dev, tol):
        "Tests that free parameters inside an array passed to an Operation yield correct gradients."
        import jax

        par = jax.numpy.array([0.4, 1.3])

        @qml.qnode(device=gaussian_dev, diff_method="finite-diff")
        def qf(x, y):
            qml.Displacement(0.5, 0, wires=[0])
            qml.Squeezing(x, 0, wires=[0])
            M = np.zeros((5, 5))
            M[1, 1] = y
            M[1, 2] = 1.0
            M[2, 1] = 1.0
            return qml.expval(qml.PolyXP(M, [0, 1]))

        grad_F = jax.grad(qf)(*par)

        @qml.qnode(
            device=gaussian_dev,
            diff_method="parameter-shift",
            gradient_kwargs={"force_order2": True},
        )
        def qf2(x, y):
            qml.Displacement(0.5, 0, wires=[0])
            qml.Squeezing(x, 0, wires=[0])
            M = np.zeros((5, 5))
            M[1, 1] = y
            M[1, 2] = 1.0
            M[2, 1] = 1.0
            return qml.expval(qml.PolyXP(M, [0, 1]))

        grad_A2 = jax.grad(qf2)(*par)

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
            return qml.expval(qml.QuadX(0))

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
            return qml.expval(qml.QuadX(0))

        qnode = qml.QNode(circuit, gaussian_dev)
        qnode(x)
        grad_F = qml.gradients.finite_diff(qnode)(x)
        grad_A = qml.gradients.param_shift_cv(qnode, dev=gaussian_dev)(x)
        grad_A2 = qml.gradients.param_shift_cv(qnode, dev=gaussian_dev, force_order2=True)(x)

        # the different methods agree
        assert qml.math.shape(grad_A) == qml.math.shape(grad_A2) == qml.math.shape(grad_F) == ()
        assert np.allclose(grad_A, grad_F, atol=tol)
        assert np.allclose(grad_A2, grad_F, atol=tol)

    def test_cv_gradient_multiple_measurement_error(self, gaussian_dev):
        """Tests multiple measurements are not supported."""
        par = anp.array([0.5, 1.3], requires_grad=True)

        def circuit(x, y):
            qml.Displacement(x, 0, wires=[0])
            qml.Rotation(y, wires=[0])
            qml.Displacement(0, x, wires=[0])
            return qml.expval(qml.QuadX(0)), qml.expval(qml.NumberOperator(0))

        q = qml.QNode(circuit, gaussian_dev)

        with pytest.raises(
            ValueError,
            match="Computing the gradient of CV circuits that return more than one measurement is not possible.",
        ):
            qml.gradients.param_shift_cv(q, dev=gaussian_dev)(*par)

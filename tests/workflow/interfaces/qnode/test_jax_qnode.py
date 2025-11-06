# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Integration tests for using the JAX-Python interface with a QNode"""
# pylint: disable=no-member, too-many-arguments, unexpected-keyword-arg, use-implicit-booleaness-not-comparison

from itertools import product

import numpy as np
import pytest

import pennylane as qml
from pennylane import qnode
from pennylane.devices import DefaultQubit
from pennylane.exceptions import DeviceError


def get_device(device_name, wires, seed):
    if device_name == "lightning.qubit":
        return qml.device("lightning.qubit", wires=wires)
    return qml.device(device_name, seed=seed)


# device, diff_method, grad_on_execution, device_vjp
device_and_diff_method = [
    ["default.qubit", "backprop", True, False],
    ["default.qubit", "finite-diff", False, False],
    ["default.qubit", "parameter-shift", False, False],
    ["default.qubit", "adjoint", True, False],
    ["default.qubit", "adjoint", False, False],
    ["default.qubit", "adjoint", True, True],
    ["default.qubit", "adjoint", False, True],
    ["default.qubit", "spsa", False, False],
    ["default.qubit", "hadamard", False, False],
    ["lightning.qubit", "adjoint", False, True],
    ["lightning.qubit", "adjoint", True, True],
    ["lightning.qubit", "adjoint", False, False],
    ["lightning.qubit", "adjoint", True, False],
    ["reference.qubit", "parameter-shift", False, False],
]

pytestmark = pytest.mark.jax

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

TOL_FOR_SPSA = 1.0
H_FOR_SPSA = 0.05


@pytest.mark.parametrize("interface", ["auto", "jax"])
@pytest.mark.parametrize(
    "dev_name,diff_method,grad_on_execution,device_vjp", device_and_diff_method
)
class TestQNode:
    """Test that using the QNode with JAX integrates with the PennyLane
    stack"""

    def test_execution_with_interface(
        self, dev_name, diff_method, grad_on_execution, interface, device_vjp, seed
    ):
        """Test execution works with the interface"""
        if diff_method == "backprop":
            pytest.skip("Test does not support backprop")

        @qnode(
            get_device(dev_name, wires=1, seed=seed),
            interface=interface,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return qml.expval(qml.PauliZ(0))

        a = jax.numpy.array(0.1)
        circuit(a)

        assert circuit.interface == interface

        # gradients should work
        grad = jax.grad(circuit)(a)
        assert isinstance(grad, jax.Array)
        assert grad.shape == ()

    def test_changing_trainability(
        self, dev_name, diff_method, grad_on_execution, device_vjp, interface, tol, seed
    ):  # pylint:disable=unused-argument
        """Test changing the trainability of parameters changes the
        number of differentiation requests made"""
        if diff_method != "parameter-shift":
            pytest.skip("Test only supports parameter-shift")

        a = jax.numpy.array(0.1)
        b = jax.numpy.array(0.2)

        @qnode(
            get_device(dev_name, wires=2, seed=seed),
            interface=interface,
            diff_method="parameter-shift",
        )
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Hamiltonian([1, 1], [qml.PauliZ(0), qml.PauliY(1)]))

        grad_fn = jax.grad(circuit, argnums=[0, 1])
        res = grad_fn(a, b)

        expected = [-np.sin(a) + np.sin(a) * np.sin(b), -np.cos(a) * np.cos(b)]
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # make the second QNode argument a constant
        grad_fn = jax.grad(circuit, argnums=0)
        res = grad_fn(a, b)

        expected = [-np.sin(a) + np.sin(a) * np.sin(b)]
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_classical_processing(
        self, dev_name, diff_method, grad_on_execution, device_vjp, interface, seed
    ):
        """Test classical processing within the quantum tape"""
        a = jax.numpy.array(0.1)
        b = jax.numpy.array(0.2)
        c = jax.numpy.array(0.3)

        @qnode(
            get_device(dev_name, wires=1, seed=seed),
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(a, b, c):
            qml.RY(a * c, wires=0)
            qml.RZ(b, wires=0)
            qml.RX(c + c**2 + jax.numpy.sin(a), wires=0)
            return qml.expval(qml.PauliZ(0))

        res = jax.grad(circuit, argnums=[0, 2])(a, b, c)

        assert len(res) == 2

    def test_matrix_parameter(
        self, dev_name, diff_method, grad_on_execution, interface, device_vjp, tol, seed
    ):
        """Test that the jax interface works correctly
        with a matrix parameter"""
        U = jax.numpy.array([[0, 1], [1, 0]])
        a = jax.numpy.array(0.1)

        @qnode(
            get_device(dev_name, wires=1, seed=seed),
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(U, a):
            qml.QubitUnitary(U, wires=0)
            qml.RY(a, wires=0)
            return qml.expval(qml.PauliZ(0))

        res = jax.grad(circuit, argnums=1)(U, a)
        assert np.allclose(res, np.sin(a), atol=tol, rtol=0)

    def test_differentiable_expand(
        self, dev_name, diff_method, grad_on_execution, interface, device_vjp, tol, seed
    ):
        """Test that operation and nested tape expansion
        is differentiable"""
        kwargs = {
            "diff_method": diff_method,
            "interface": interface,
            "grad_on_execution": grad_on_execution,
            "device_vjp": device_vjp,
        }
        gradient_kwargs = {}
        if diff_method == "spsa":
            gradient_kwargs["sampler_rng"] = np.random.default_rng(seed)
            gradient_kwargs["num_directions"] = 10
            tol = TOL_FOR_SPSA

        class U3(qml.U3):  # pylint:disable=too-few-public-methods
            def decomposition(self):
                theta, phi, lam = self.data
                wires = self.wires
                return [
                    qml.Rot(lam, theta, -lam, wires=wires),
                    qml.PhaseShift(phi + lam, wires=wires),
                ]

        a = jax.numpy.array(0.1)
        p = jax.numpy.array([0.1, 0.2, 0.3])

        @qnode(get_device(dev_name, wires=1, seed=seed), **kwargs, gradient_kwargs=gradient_kwargs)
        def circuit(a, p):
            qml.RX(a, wires=0)
            U3(p[0], p[1], p[2], wires=0)
            return qml.expval(qml.PauliX(0))

        res = circuit(a, p)
        expected = np.cos(a) * np.cos(p[1]) * np.sin(p[0]) + np.sin(a) * (
            np.cos(p[2]) * np.sin(p[1]) + np.cos(p[0]) * np.cos(p[1]) * np.sin(p[2])
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = jax.grad(circuit, argnums=1)(a, p)
        expected = np.array(
            [
                np.cos(p[1]) * (np.cos(a) * np.cos(p[0]) - np.sin(a) * np.sin(p[0]) * np.sin(p[2])),
                np.cos(p[1]) * np.cos(p[2]) * np.sin(a)
                - np.sin(p[1])
                * (np.cos(a) * np.sin(p[0]) + np.cos(p[0]) * np.sin(a) * np.sin(p[2])),
                np.sin(a)
                * (np.cos(p[0]) * np.cos(p[1]) * np.cos(p[2]) - np.sin(p[1]) * np.sin(p[2])),
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_jacobian_options(
        self, dev_name, diff_method, grad_on_execution, device_vjp, interface, seed
    ):  # pylint:disable=unused-argument
        """Test setting jacobian options"""
        if diff_method != "finite-diff":
            pytest.skip("Test only applies to finite diff.")

        a = jax.numpy.array([0.1, 0.2])

        gradient_kwargs = {"h": 1e-8, "approx_order": 2}

        @qnode(
            get_device(dev_name, wires=1, seed=seed),
            interface=interface,
            diff_method="finite-diff",
            gradient_kwargs=gradient_kwargs,
        )
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        jax.jacobian(circuit)(a)


@pytest.mark.parametrize("interface", ["auto", "jax"])
@pytest.mark.parametrize(
    "dev_name,diff_method,grad_on_execution, device_vjp", device_and_diff_method
)
class TestVectorValuedQNode:
    """Test that using vector-valued QNodes with JAX integrate with the
    PennyLane stack"""

    def test_diff_expval_expval(
        self, dev_name, diff_method, grad_on_execution, device_vjp, interface, tol, seed
    ):
        """Test jacobian calculation"""
        kwargs = {
            "diff_method": diff_method,
            "interface": interface,
            "grad_on_execution": grad_on_execution,
            "device_vjp": device_vjp,
        }
        gradient_kwargs = {}
        if diff_method == "spsa":
            gradient_kwargs["sampler_rng"] = np.random.default_rng(seed)
            tol = TOL_FOR_SPSA
        if "lightning" in dev_name:
            pytest.xfail("lightning device_vjp not compatible with jax.jacobian.")

        a = jax.numpy.array(0.1)
        b = jax.numpy.array(0.2)

        @qnode(get_device(dev_name, wires=2, seed=seed), **kwargs, gradient_kwargs=gradient_kwargs)
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))

        res = circuit(a, b)

        assert isinstance(res, tuple)
        assert len(res) == 2

        expected = [np.cos(a), -np.cos(a) * np.sin(b)]
        assert np.allclose(res[0], expected[0], atol=tol, rtol=0)
        assert np.allclose(res[1], expected[1], atol=tol, rtol=0)

        res = jax.jacobian(circuit, argnums=[0, 1])(a, b)
        expected = np.array([[-np.sin(a), 0], [np.sin(a) * np.sin(b), -np.cos(a) * np.cos(b)]])
        assert isinstance(res, tuple)
        assert len(res) == 2

        assert isinstance(res[0], tuple)
        assert isinstance(res[0][0], jax.numpy.ndarray)
        assert res[0][0].shape == ()
        assert np.allclose(res[0][0], expected[0][0], atol=tol, rtol=0)
        assert isinstance(res[0][1], jax.numpy.ndarray)
        assert res[0][1].shape == ()
        assert np.allclose(res[0][1], expected[0][1], atol=tol, rtol=0)

        assert isinstance(res[1], tuple)
        assert isinstance(res[1][0], jax.numpy.ndarray)
        assert res[1][0].shape == ()
        assert np.allclose(res[1][0], expected[1][0], atol=tol, rtol=0)
        assert isinstance(res[1][1], jax.numpy.ndarray)
        assert res[1][1].shape == ()
        assert np.allclose(res[1][1], expected[1][1], atol=tol, rtol=0)

    def test_jacobian_no_evaluate(
        self, dev_name, diff_method, grad_on_execution, device_vjp, interface, tol, seed
    ):
        """Test jacobian calculation when no prior circuit evaluation has been performed"""
        kwargs = {
            "diff_method": diff_method,
            "interface": interface,
            "grad_on_execution": grad_on_execution,
            "device_vjp": device_vjp,
        }

        if "lightning" in dev_name:
            pytest.xfail("lightning device_vjp not compatible with jax.jacobian.")

        gradient_kwargs = {}
        if diff_method == "spsa":
            gradient_kwargs["sampler_rng"] = np.random.default_rng(seed)
            tol = TOL_FOR_SPSA

        a = jax.numpy.array(0.1)
        b = jax.numpy.array(0.2)

        @qnode(get_device(dev_name, wires=2, seed=seed), **kwargs, gradient_kwargs=gradient_kwargs)
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))

        jac_fn = jax.jacobian(circuit, argnums=[0, 1])
        res = jac_fn(a, b)

        assert isinstance(res, tuple)
        assert len(res) == 2

        expected = np.array([[-np.sin(a), 0], [np.sin(a) * np.sin(b), -np.cos(a) * np.cos(b)]])

        assert isinstance(res[0][0], jax.numpy.ndarray)
        for i, j in product((0, 1), (0, 1)):
            assert res[i][j].shape == ()
            assert np.allclose(res[i][j], expected[i][j], atol=tol, rtol=0)

        # call the Jacobian with new parameters
        a = jax.numpy.array(0.6)
        b = jax.numpy.array(0.832)

        res = jac_fn(a, b)

        assert isinstance(res, tuple)
        assert len(res) == 2

        expected = np.array([[-np.sin(a), 0], [np.sin(a) * np.sin(b), -np.cos(a) * np.cos(b)]])

        for i, j in product((0, 1), (0, 1)):
            assert res[i][j].shape == ()
            assert np.allclose(res[i][j], expected[i][j], atol=tol, rtol=0)

    def test_diff_single_probs(
        self, dev_name, diff_method, grad_on_execution, device_vjp, interface, tol, seed
    ):
        """Tests correct output shape and evaluation for a tape
        with a single prob output"""
        kwargs = {
            "diff_method": diff_method,
            "interface": interface,
            "grad_on_execution": grad_on_execution,
            "device_vjp": device_vjp,
        }
        gradient_kwargs = {}
        if diff_method == "spsa":
            gradient_kwargs["sampler_rng"] = np.random.default_rng(seed)
            tol = TOL_FOR_SPSA
        if "lightning" in dev_name:
            pytest.xfail("lightning device_vjp not compatible with jax.jacobian.")

        x = jax.numpy.array(0.543)
        y = jax.numpy.array(-0.654)

        @qnode(get_device(dev_name, wires=2, seed=seed), **kwargs, gradient_kwargs=gradient_kwargs)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[1])

        res = jax.jacobian(circuit, argnums=[0, 1])(x, y)

        expected = np.array(
            [
                [-np.sin(x) * np.cos(y) / 2, -np.cos(x) * np.sin(y) / 2],
                [np.cos(y) * np.sin(x) / 2, np.cos(x) * np.sin(y) / 2],
            ]
        )

        assert isinstance(res, tuple)
        assert len(res) == 2

        assert isinstance(res[0], jax.numpy.ndarray)
        assert res[0].shape == (2,)

        assert isinstance(res[1], jax.numpy.ndarray)
        assert res[1].shape == (2,)

        assert np.allclose(res[0], expected.T[0], atol=tol, rtol=0)
        assert np.allclose(res[1], expected.T[1], atol=tol, rtol=0)

    def test_diff_multi_probs(
        self, dev_name, diff_method, grad_on_execution, device_vjp, interface, tol, seed
    ):
        """Tests correct output shape and evaluation for a tape
        with multiple prob outputs"""
        kwargs = {
            "diff_method": diff_method,
            "interface": interface,
            "grad_on_execution": grad_on_execution,
            "device_vjp": device_vjp,
        }

        gradient_kwargs = {}
        if diff_method == "spsa":
            gradient_kwargs["sampler_rng"] = np.random.default_rng(seed)
            tol = TOL_FOR_SPSA
        if "lightning" in dev_name:
            pytest.xfail("lightning device_vjp not compatible with jax.jacobian.")

        x = jax.numpy.array(0.543)
        y = jax.numpy.array(-0.654)

        @qnode(get_device(dev_name, wires=1, seed=seed), **kwargs, gradient_kwargs=gradient_kwargs)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0]), qml.probs(wires=[1, 2])

        res = circuit(x, y)

        assert isinstance(res, tuple)
        assert len(res) == 2

        expected = [
            [np.cos(x / 2) ** 2, np.sin(x / 2) ** 2],
            [(1 + np.cos(x) * np.cos(y)) / 2, 0, (1 - np.cos(x) * np.cos(y)) / 2, 0],
        ]

        assert isinstance(res[0], jax.numpy.ndarray)
        assert res[0].shape == (2,)  # pylint:disable=comparison-with-callable
        assert np.allclose(res[0], expected[0], atol=tol, rtol=0)

        assert isinstance(res[1], jax.numpy.ndarray)
        assert res[1].shape == (4,)  # pylint:disable=comparison-with-callable
        assert np.allclose(res[1], expected[1], atol=tol, rtol=0)

        jac = jax.jacobian(circuit, argnums=[0, 1])(x, y)
        expected_0 = np.array(
            [
                [-np.sin(x) / 2, np.sin(x) / 2],
                [0, 0],
            ]
        )

        expected_1 = np.array(
            [
                [-np.cos(y) * np.sin(x) / 2, 0, np.sin(x) * np.cos(y) / 2, 0],
                [-np.cos(x) * np.sin(y) / 2, 0, np.cos(x) * np.sin(y) / 2, 0],
            ]
        )

        assert isinstance(jac, tuple)
        assert isinstance(jac[0], tuple)

        assert len(jac[0]) == 2
        assert isinstance(jac[0][0], jax.numpy.ndarray)
        assert jac[0][0].shape == (2,)
        assert np.allclose(jac[0][0], expected_0[0], atol=tol, rtol=0)
        assert isinstance(jac[0][1], jax.numpy.ndarray)
        assert jac[0][1].shape == (2,)
        assert np.allclose(jac[0][1], expected_0[1], atol=tol, rtol=0)

        assert isinstance(jac[1], tuple)
        assert len(jac[1]) == 2
        assert isinstance(jac[1][0], jax.numpy.ndarray)
        assert jac[1][0].shape == (4,)

        assert np.allclose(jac[1][0], expected_1[0], atol=tol, rtol=0)
        assert isinstance(jac[1][1], jax.numpy.ndarray)
        assert jac[1][1].shape == (4,)
        assert np.allclose(jac[1][1], expected_1[1], atol=tol, rtol=0)

    def test_diff_expval_probs(
        self, dev_name, diff_method, grad_on_execution, device_vjp, interface, tol, seed
    ):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""
        kwargs = {
            "diff_method": diff_method,
            "interface": interface,
            "grad_on_execution": grad_on_execution,
            "device_vjp": device_vjp,
        }
        gradient_kwargs = {}
        if diff_method == "spsa":
            gradient_kwargs["sampler_rng"] = np.random.default_rng(seed)
            tol = TOL_FOR_SPSA
        if "lightning" in dev_name:
            pytest.xfail("lightning device_vjp not compatible with jax.jacobian.")

        x = jax.numpy.array(0.543)
        y = jax.numpy.array(-0.654)

        @qnode(get_device(dev_name, wires=1, seed=seed), **kwargs, gradient_kwargs=gradient_kwargs)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[1])

        res = circuit(x, y)
        expected = [np.cos(x), [(1 + np.cos(x) * np.cos(y)) / 2, (1 - np.cos(x) * np.cos(y)) / 2]]
        assert isinstance(res, tuple)
        assert len(res) == 2

        assert isinstance(res[0], jax.numpy.ndarray)
        assert res[0].shape == ()  # pylint:disable=comparison-with-callable
        assert np.allclose(res[0], expected[0], atol=tol, rtol=0)

        assert isinstance(res[1], jax.numpy.ndarray)
        assert res[1].shape == (2,)  # pylint:disable=comparison-with-callable
        assert np.allclose(res[1], expected[1], atol=tol, rtol=0)

        jac = jax.jacobian(circuit, argnums=[0, 1])(x, y)
        expected = [
            [-np.sin(x), 0],
            [
                [-np.sin(x) * np.cos(y) / 2, np.cos(y) * np.sin(x) / 2],
                [-np.cos(x) * np.sin(y) / 2, np.cos(x) * np.sin(y) / 2],
            ],
        ]

        assert isinstance(jac, tuple)
        assert len(jac) == 2

        assert isinstance(jac[0], tuple)
        assert len(jac[0]) == 2
        assert isinstance(jac[0][0], jax.numpy.ndarray)
        assert jac[0][0].shape == ()
        assert np.allclose(jac[0][0], expected[0][0], atol=tol, rtol=0)
        assert isinstance(jac[0][1], jax.numpy.ndarray)
        assert jac[0][1].shape == ()
        assert np.allclose(jac[0][1], expected[0][1], atol=tol, rtol=0)

        assert isinstance(jac[1], tuple)
        assert len(jac[1]) == 2
        assert isinstance(jac[1][0], jax.numpy.ndarray)
        assert jac[1][0].shape == (2,)
        assert np.allclose(jac[1][0], expected[1][0], atol=tol, rtol=0)
        assert isinstance(jac[1][1], jax.numpy.ndarray)
        assert jac[1][1].shape == (2,)
        assert np.allclose(jac[1][1], expected[1][1], atol=tol, rtol=0)

    def test_diff_expval_probs_sub_argnums(
        self, dev_name, diff_method, grad_on_execution, device_vjp, interface, tol, seed
    ):
        """Tests correct output shape and evaluation for a tape with prob and expval outputs with less
        trainable parameters (argnums) than parameters."""
        kwargs = {}
        if diff_method == "spsa":
            kwargs["sampler_rng"] = np.random.default_rng(seed)
            tol = TOL_FOR_SPSA

        x = jax.numpy.array(0.543)
        y = jax.numpy.array(-0.654)

        if diff_method == "adjoint":
            x = x + 0j
            y = y + 0j

        @qnode(
            get_device(dev_name, wires=1, seed=seed),
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
            gradient_kwargs=kwargs,
        )
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[1])

        if "lightning" in dev_name:
            pytest.xfail("lightning does not support measuring probabilities with adjoint.")

        jac = jax.jacobian(circuit, argnums=[0])(x, y)

        expected = [
            [-np.sin(x), 0],
            [
                [-np.sin(x) * np.cos(y) / 2, np.cos(y) * np.sin(x) / 2],
                [-np.cos(x) * np.sin(y) / 2, np.cos(x) * np.sin(y) / 2],
            ],
        ]
        assert isinstance(jac, tuple)
        assert len(jac) == 2

        assert isinstance(jac[0], tuple)
        assert len(jac[0]) == 1
        assert isinstance(jac[0][0], jax.numpy.ndarray)
        assert jac[0][0].shape == ()
        assert np.allclose(jac[0][0], expected[0][0], atol=tol, rtol=0)

        assert isinstance(jac[1], tuple)
        assert len(jac[1]) == 1
        assert isinstance(jac[1][0], jax.numpy.ndarray)
        assert jac[1][0].shape == (2,)
        assert np.allclose(jac[1][0], expected[1][0], atol=tol, rtol=0)

    def test_diff_var_probs(
        self, dev_name, diff_method, grad_on_execution, device_vjp, interface, tol, seed
    ):
        """Tests correct output shape and evaluation for a tape
        with prob and variance outputs"""
        kwargs = {
            "diff_method": diff_method,
            "interface": interface,
            "grad_on_execution": grad_on_execution,
            "device_vjp": device_vjp,
        }

        gradient_kwargs = {}
        if diff_method == "hadamard":
            pytest.skip("Hadamard does not support var")
        if "lightning" in dev_name:
            pytest.xfail("lightning device_vjp not compatible with jax.jacobian.")
        elif diff_method == "spsa":
            gradient_kwargs["sampler_rng"] = np.random.default_rng(seed)
            tol = TOL_FOR_SPSA

        x = jax.numpy.array(0.543)
        y = jax.numpy.array(-0.654)

        @qnode(get_device(dev_name, wires=1, seed=seed), **kwargs, gradient_kwargs=gradient_kwargs)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.PauliZ(0)), qml.probs(wires=[1])

        res = circuit(x, y)

        expected = [
            np.sin(x) ** 2,
            [(1 + np.cos(x) * np.cos(y)) / 2, (1 - np.cos(x) * np.cos(y)) / 2],
        ]

        assert isinstance(res[0], jax.numpy.ndarray)
        assert res[0].shape == ()  # pylint:disable=comparison-with-callable
        assert np.allclose(res[0], expected[0], atol=tol, rtol=0)

        assert isinstance(res[1], jax.numpy.ndarray)
        assert res[1].shape == (2,)  # pylint:disable=comparison-with-callable
        assert np.allclose(res[1], expected[1], atol=tol, rtol=0)

        jac = jax.jacobian(circuit, argnums=[0, 1])(x, y)
        expected = [
            [2 * np.cos(x) * np.sin(x), 0],
            [
                [-np.sin(x) * np.cos(y) / 2, np.cos(y) * np.sin(x) / 2],
                [-np.cos(x) * np.sin(y) / 2, np.cos(x) * np.sin(y) / 2],
            ],
        ]

        assert isinstance(jac, tuple)
        assert len(jac) == 2

        assert isinstance(jac[0], tuple)
        assert len(jac[0]) == 2
        assert isinstance(jac[0][0], jax.numpy.ndarray)
        assert jac[0][0].shape == ()
        assert np.allclose(jac[0][0], expected[0][0], atol=tol, rtol=0)
        assert isinstance(jac[0][1], jax.numpy.ndarray)
        assert jac[0][1].shape == ()
        assert np.allclose(jac[0][1], expected[0][1], atol=tol, rtol=0)

        assert isinstance(jac[1], tuple)
        assert len(jac[1]) == 2
        assert isinstance(jac[1][0], jax.numpy.ndarray)
        assert jac[1][0].shape == (2,)
        assert np.allclose(jac[1][0], expected[1][0], atol=tol, rtol=0)
        assert isinstance(jac[1][1], jax.numpy.ndarray)
        assert jac[1][1].shape == (2,)
        assert np.allclose(jac[1][1], expected[1][1], atol=tol, rtol=0)


@pytest.mark.parametrize("interface", ["auto", "jax"])
class TestShotsIntegration:
    """Test that the QNode correctly changes shot value, and
    remains differentiable."""

    def test_diff_method_None(self, interface):
        """Test device works with diff_method=None."""

        @qml.set_shots(shots=10)
        @qml.qnode(DefaultQubit(), diff_method=None, interface=interface)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        assert jax.numpy.allclose(circuit(jax.numpy.array(0.0)), 1)

    def test_changing_shots(self, interface):
        """Test that changing shots works on execution"""
        a, b = jax.numpy.array([0.543, -0.654])

        @qnode(DefaultQubit(), diff_method=qml.gradients.param_shift, interface=interface)
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.sample(wires=(0, 1))

        # execute with device default shots (None)
        with pytest.raises(DeviceError):
            circuit(a, b)

        # execute with shots=100
        res = qml.set_shots(shots=100)(circuit)(a, b)
        assert res.shape == (100, 2)  # pylint: disable=comparison-with-callable

    def test_gradient_integration(self, interface):
        """Test that temporarily setting the shots works
        for gradient computations"""
        a, b = jax.numpy.array([0.543, -0.654])

        @qml.set_shots(shots=30000)
        @qnode(DefaultQubit(), diff_method=qml.gradients.param_shift, interface=interface)
        def cost_fn(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliY(1))

        res = jax.grad(cost_fn, argnums=[0, 1])(a, b)

        expected = [np.sin(a) * np.sin(b), -np.cos(a) * np.cos(b)]
        assert np.allclose(res, expected, atol=0.1, rtol=0)

    def test_update_diff_method(self, interface):
        """Test that temporarily setting the shots updates the diff method"""
        a, b = jax.numpy.array([0.543, -0.654])

        dev = DefaultQubit()

        @qnode(dev, interface=interface)
        def cost_fn(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliY(1))

        with dev.tracker:
            jax.grad(qml.set_shots(shots=100)(cost_fn))(a, b)
        # since we are using finite shots, use parameter shift
        assert dev.tracker.totals["executions"] == 3

        # if we use the default shots value of None, backprop can now be used
        with dev.tracker:
            jax.grad(cost_fn)(a, b)
        assert dev.tracker.totals["executions"] == 1


@pytest.mark.parametrize(
    "dev_name,diff_method,grad_on_execution,device_vjp", device_and_diff_method
)
class TestQubitIntegration:
    """Tests that ensure various qubit circuits integrate correctly"""

    def test_sampling(self, dev_name, diff_method, grad_on_execution, device_vjp, seed):
        """Test sampling works as expected"""
        if grad_on_execution is True:
            pytest.skip("Sampling not possible with grad_on_execution differentiation.")

        if diff_method == "adjoint":
            pytest.skip("Adjoint warns with finite shots")

        @qml.set_shots(shots=10)
        @qnode(
            get_device(dev_name, wires=2, seed=seed),
            diff_method=diff_method,
            interface="jax",
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit():
            qml.Hadamard(wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.sample(qml.PauliZ(0)), qml.sample(qml.PauliX(1))

        res = circuit()

        assert isinstance(res, tuple)

        assert isinstance(res[0], jax.Array)
        assert res[0].shape == (10,)  # pylint:disable=comparison-with-callable
        assert isinstance(res[1], jax.Array)
        assert res[1].shape == (10,)  # pylint:disable=comparison-with-callable

    def test_counts(self, dev_name, diff_method, grad_on_execution, device_vjp, seed):
        """Test counts works as expected"""
        if grad_on_execution is True:
            pytest.skip("Sampling not possible with grad_on_execution differentiation.")

        if diff_method == "adjoint":
            pytest.skip("Adjoint errors with finite shots")

        @qml.set_shots(shots=10)
        @qnode(
            get_device(dev_name, wires=2, seed=seed),
            diff_method=diff_method,
            interface="jax",
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit():
            qml.Hadamard(wires=[0])
            qml.CNOT(wires=[0, 1])
            return (
                qml.counts(qml.PauliZ(0), all_outcomes=True),
                qml.counts(qml.PauliX(1), all_outcomes=True),
            )

        res = circuit()

        assert isinstance(res, tuple)

        assert isinstance(res[0], dict)
        assert len(res[0]) == 2
        assert isinstance(res[1], dict)
        assert len(res[1]) == 2

    def test_chained_qnodes(self, dev_name, diff_method, grad_on_execution, device_vjp, seed):
        """Test that the gradient of chained QNodes works without error"""
        # pylint:disable=too-few-public-methods

        class Template(qml.templates.StronglyEntanglingLayers):
            def decomposition(self):
                return [qml.templates.StronglyEntanglingLayers(*self.parameters, self.wires)]

        @qnode(
            get_device(dev_name, wires=2, seed=seed),
            interface="jax",
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit1(weights):
            Template(weights, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        @qnode(
            get_device(dev_name, wires=2, seed=seed),
            interface="jax",
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit2(data, weights):
            qml.templates.AngleEmbedding(jax.numpy.stack([data, 0.7]), wires=[0, 1])
            Template(weights, wires=[0, 1])
            return qml.expval(qml.PauliX(0))

        def cost(weights):
            w1, w2 = weights
            c1 = circuit1(w1)
            c2 = circuit2(c1, w2)
            return jax.numpy.sum(c2) ** 2

        w1 = qml.templates.StronglyEntanglingLayers.shape(n_wires=2, n_layers=3)
        w2 = qml.templates.StronglyEntanglingLayers.shape(n_wires=2, n_layers=4)

        weights = [
            jax.numpy.array(np.random.random(w1)),
            jax.numpy.array(np.random.random(w2)),
        ]

        grad_fn = jax.grad(cost)
        res = grad_fn(weights)

        assert len(res) == 2

    def test_postselection_differentiation(
        self, dev_name, diff_method, grad_on_execution, device_vjp, seed
    ):
        """Test that when postselecting with default.qubit, differentiation works correctly."""

        if diff_method in ["adjoint", "spsa", "hadamard"]:
            pytest.skip("Diff method does not support postselection.")
        if dev_name == "reference.qubit":
            pytest.xfail("reference.qubit does not support postselection.")

        @qml.qnode(
            get_device(dev_name, wires=2, seed=seed),
            diff_method=diff_method,
            interface="jax",
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(phi, theta):
            qml.RX(phi, wires=0)
            qml.CNOT([0, 1])
            qml.measure(wires=0, postselect=1)
            qml.RX(theta, wires=1)
            return qml.expval(qml.PauliZ(1))

        @qml.qnode(
            get_device(dev_name, wires=2, seed=seed),
            diff_method=diff_method,
            interface="jax",
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def expected_circuit(theta):
            qml.PauliX(1)
            qml.RX(theta, wires=1)
            return qml.expval(qml.PauliZ(1))

        phi = jax.numpy.array(1.23)
        theta = jax.numpy.array(4.56)

        assert np.allclose(circuit(phi, theta), expected_circuit(theta))

        gradient = jax.grad(circuit, argnums=[0, 1])(phi, theta)
        exp_theta_grad = jax.grad(expected_circuit)(theta)
        assert np.allclose(gradient, [0.0, exp_theta_grad])


@pytest.mark.parametrize("interface", ["auto", "jax"])
@pytest.mark.parametrize(
    "dev_name,diff_method,grad_on_execution, device_vjp", device_and_diff_method
)
class TestQubitIntegrationHigherOrder:
    """Tests that ensure various qubit circuits integrate correctly when computing higher-order derivatives"""

    def test_second_derivative(
        self, dev_name, diff_method, grad_on_execution, device_vjp, interface, tol, seed
    ):
        """Test second derivative calculation of a scalar-valued QNode"""
        kwargs = {
            "diff_method": diff_method,
            "interface": interface,
            "grad_on_execution": grad_on_execution,
            "device_vjp": device_vjp,
            "max_diff": 2,
        }

        gradient_kwargs = {}
        if diff_method == "adjoint":
            pytest.skip("Adjoint does not second derivative.")
        elif diff_method == "spsa":
            gradient_kwargs["sampler_rng"] = np.random.default_rng(seed)
            tol = TOL_FOR_SPSA

        @qnode(get_device(dev_name, wires=0, seed=seed), **kwargs, gradient_kwargs=gradient_kwargs)
        def circuit(x):
            qml.RY(x[0], wires=0)
            qml.RX(x[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        x = jax.numpy.array([1.0, 2.0])
        res = circuit(x)
        g = jax.grad(circuit)(x)
        g2 = jax.grad(lambda x: jax.numpy.sum(jax.grad(circuit)(x)))(x)

        a, b = x

        expected_res = np.cos(a) * np.cos(b)
        assert np.allclose(res, expected_res, atol=tol, rtol=0)

        expected_g = [-np.sin(a) * np.cos(b), -np.cos(a) * np.sin(b)]
        assert np.allclose(g, expected_g, atol=tol, rtol=0)

        expected_g2 = [
            -np.cos(a) * np.cos(b) + np.sin(a) * np.sin(b),
            np.sin(a) * np.sin(b) - np.cos(a) * np.cos(b),
        ]
        if diff_method == "finite-diff":
            assert np.allclose(g2, expected_g2, atol=10e-2, rtol=0)
        else:
            assert np.allclose(g2, expected_g2, atol=tol, rtol=0)

    def test_hessian(
        self, dev_name, diff_method, grad_on_execution, device_vjp, interface, tol, seed
    ):
        """Test hessian calculation of a scalar-valued QNode"""
        gradient_kwargs = {}
        if diff_method == "adjoint":
            pytest.skip("Adjoint does not support second derivative.")
        elif diff_method == "spsa":
            gradient_kwargs = {
                "h": H_FOR_SPSA,
                "num_directions": 20,
                "sampler_rng": np.random.default_rng(seed),
            }
            tol = TOL_FOR_SPSA

        @qnode(
            get_device(dev_name, wires=2, seed=seed),
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
            max_diff=2,
            gradient_kwargs=gradient_kwargs,
        )
        def circuit(x):
            qml.RY(x[0], wires=0)
            qml.RX(x[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        x = jax.numpy.array([1.0, 2.0])
        res = circuit(x)

        a, b = x

        expected_res = np.cos(a) * np.cos(b)
        assert np.allclose(res, expected_res, atol=tol, rtol=0)

        grad_fn = jax.grad(circuit)
        g = grad_fn(x)

        expected_g = [-np.sin(a) * np.cos(b), -np.cos(a) * np.sin(b)]
        assert np.allclose(g, expected_g, atol=tol, rtol=0)

        hess = jax.jacobian(grad_fn)(x)

        expected_hess = [
            [-np.cos(a) * np.cos(b), np.sin(a) * np.sin(b)],
            [np.sin(a) * np.sin(b), -np.cos(a) * np.cos(b)],
        ]
        if diff_method == "finite-diff":
            assert np.allclose(hess, expected_hess, atol=10e-2, rtol=0)
        else:
            assert np.allclose(hess, expected_hess, atol=tol, rtol=0)

    def test_hessian_vector_valued(
        self, dev_name, diff_method, grad_on_execution, device_vjp, interface, tol, seed
    ):
        """Test hessian calculation of a vector-valued QNode"""
        gradient_kwargs = {}
        if diff_method == "adjoint":
            pytest.skip("Adjoint does not support second derivative.")
        elif diff_method == "spsa":
            qml.math.random.seed(seed)
            gradient_kwargs = {
                "h": H_FOR_SPSA,
                "num_directions": 20,
                "sampler_rng": np.random.default_rng(seed),
            }
            tol = TOL_FOR_SPSA

        @qnode(
            get_device(dev_name, wires=0, seed=seed),
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
            max_diff=2,
            gradient_kwargs=gradient_kwargs,
        )
        def circuit(x):
            qml.RY(x[0], wires=0)
            qml.RX(x[1], wires=0)
            return qml.probs(wires=0)

        x = jax.numpy.array([1.0, 2.0])
        res = circuit(x)

        a, b = x

        expected_res = [0.5 + 0.5 * np.cos(a) * np.cos(b), 0.5 - 0.5 * np.cos(a) * np.cos(b)]
        assert np.allclose(res, expected_res, atol=tol, rtol=0)

        jac_fn = jax.jacobian(circuit)
        g = jac_fn(x)

        expected_g = [
            [-0.5 * np.sin(a) * np.cos(b), -0.5 * np.cos(a) * np.sin(b)],
            [0.5 * np.sin(a) * np.cos(b), 0.5 * np.cos(a) * np.sin(b)],
        ]
        assert np.allclose(g, expected_g, atol=tol, rtol=0)

        hess = jax.jacobian(jac_fn)(x)

        expected_hess = [
            [
                [-0.5 * np.cos(a) * np.cos(b), 0.5 * np.sin(a) * np.sin(b)],
                [0.5 * np.sin(a) * np.sin(b), -0.5 * np.cos(a) * np.cos(b)],
            ],
            [
                [0.5 * np.cos(a) * np.cos(b), -0.5 * np.sin(a) * np.sin(b)],
                [-0.5 * np.sin(a) * np.sin(b), 0.5 * np.cos(a) * np.cos(b)],
            ],
        ]
        if diff_method == "finite-diff":
            assert np.allclose(hess, expected_hess, atol=10e-2, rtol=0)
        else:
            assert np.allclose(hess, expected_hess, atol=tol, rtol=0)

    def test_hessian_vector_valued_postprocessing(
        self, dev_name, diff_method, interface, grad_on_execution, device_vjp, tol, seed
    ):
        """Test hessian calculation of a vector valued QNode with post-processing"""
        gradient_kwargs = {}
        if diff_method == "adjoint":
            pytest.skip("Adjoint does not support second derivative.")
        elif diff_method == "spsa":
            gradient_kwargs = {
                "h": H_FOR_SPSA,
                "num_directions": 20,
                "sampler_rng": np.random.default_rng(seed),
            }
            tol = TOL_FOR_SPSA

        @qnode(
            get_device(dev_name, wires=2, seed=seed),
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
            max_diff=2,
            gradient_kwargs=gradient_kwargs,
        )
        def circuit(x):
            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=0)
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(0))

        def cost_fn(x):
            return x @ jax.numpy.array(circuit(x))

        x = jax.numpy.array([0.76, -0.87])
        res = cost_fn(x)

        a, b = x

        expected_res = x @ jax.numpy.array([np.cos(a) * np.cos(b), np.cos(a) * np.cos(b)])
        assert np.allclose(res, expected_res, atol=tol, rtol=0)

        grad_fn = jax.grad(cost_fn)
        g = grad_fn(x)

        expected_g = [
            np.cos(b) * (np.cos(a) - (a + b) * np.sin(a)),
            np.cos(a) * (np.cos(b) - (a + b) * np.sin(b)),
        ]
        assert np.allclose(g, expected_g, atol=tol, rtol=0)
        hess = jax.jacobian(grad_fn)(x)

        expected_hess = [
            [
                -(np.cos(b) * ((a + b) * np.cos(a) + 2 * np.sin(a))),
                -(np.cos(b) * np.sin(a)) + (-np.cos(a) + (a + b) * np.sin(a)) * np.sin(b),
            ],
            [
                -(np.cos(b) * np.sin(a)) + (-np.cos(a) + (a + b) * np.sin(a)) * np.sin(b),
                -(np.cos(a) * ((a + b) * np.cos(b) + 2 * np.sin(b))),
            ],
        ]

        if diff_method == "finite-diff":
            assert np.allclose(hess, expected_hess, atol=10e-2, rtol=0)
        else:
            assert np.allclose(hess, expected_hess, atol=tol, rtol=0)

    def test_hessian_vector_valued_separate_args(
        self, dev_name, diff_method, grad_on_execution, device_vjp, interface, tol, seed
    ):
        """Test hessian calculation of a vector valued QNode that has separate input arguments"""
        gradient_kwargs = {}
        if diff_method == "adjoint":
            pytest.skip("Adjoint does not support second derivative.")
        elif diff_method == "spsa":
            gradient_kwargs = {
                "h": H_FOR_SPSA,
                "num_directions": 20,
                "sampler_rng": np.random.default_rng(seed),
            }
            tol = TOL_FOR_SPSA

        @qnode(
            get_device(dev_name, wires=2, seed=seed),
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
            max_diff=2,
            gradient_kwargs=gradient_kwargs,
        )
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            return qml.probs(wires=0)

        a = jax.numpy.array(1.0)
        b = jax.numpy.array(2.0)
        res = circuit(a, b)

        expected_res = [0.5 + 0.5 * np.cos(a) * np.cos(b), 0.5 - 0.5 * np.cos(a) * np.cos(b)]
        assert np.allclose(res, expected_res, atol=tol, rtol=0)

        jac_fn = jax.jacobian(circuit, argnums=[0, 1])
        g = jac_fn(a, b)

        expected_g = np.array(
            [
                [-0.5 * np.sin(a) * np.cos(b), -0.5 * np.cos(a) * np.sin(b)],
                [0.5 * np.sin(a) * np.cos(b), 0.5 * np.cos(a) * np.sin(b)],
            ]
        )
        assert np.allclose(g, expected_g.T, atol=tol, rtol=0)

        hess = jax.jacobian(jac_fn, argnums=[0, 1])(a, b)

        expected_hess = np.array(
            [
                [
                    [-0.5 * np.cos(a) * np.cos(b), 0.5 * np.cos(a) * np.cos(b)],
                    [0.5 * np.sin(a) * np.sin(b), -0.5 * np.sin(a) * np.sin(b)],
                ],
                [
                    [0.5 * np.sin(a) * np.sin(b), -0.5 * np.sin(a) * np.sin(b)],
                    [-0.5 * np.cos(a) * np.cos(b), 0.5 * np.cos(a) * np.cos(b)],
                ],
            ]
        )
        if diff_method == "finite-diff":
            assert np.allclose(hess, expected_hess, atol=10e-2, rtol=0)
        else:
            assert np.allclose(hess, expected_hess, atol=tol, rtol=0)

    def test_state(
        self, dev_name, diff_method, grad_on_execution, device_vjp, interface, tol, seed
    ):
        """Test that the state can be returned and differentiated"""

        if "lightning" in dev_name:
            pytest.xfail("Lightning does not support state adjoint differentiation.")

        x = jax.numpy.array(0.543)
        y = jax.numpy.array(-0.654)

        @qnode(
            get_device(dev_name, wires=2, seed=seed),
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.state()

        def cost_fn(x, y):
            res = circuit(x, y)
            assert res.dtype is np.dtype("complex128")  # pylint:disable=no-member
            probs = jax.numpy.abs(res) ** 2
            return probs[0] + probs[2]

        res = cost_fn(x, y)

        if diff_method not in {"backprop"}:
            pytest.skip("Test only supports backprop")

        res = jax.grad(cost_fn, argnums=[0, 1])(x, y)
        expected = np.array([-np.sin(x) * np.cos(y) / 2, -np.cos(x) * np.sin(y) / 2])
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("state", [[1], [0, 1]])  # Basis state and state vector
    def test_projector(
        self, state, dev_name, diff_method, grad_on_execution, device_vjp, interface, tol, seed
    ):
        """Test that the variance of a projector is correctly returned"""
        gradient_kwargs = {}
        if diff_method == "adjoint":
            pytest.skip("adjoint supports all expvals or only diagonal measurements.")
        if diff_method == "hadamard":
            pytest.skip("Hadamard does not support var.")
        elif diff_method == "spsa":
            gradient_kwargs = {"h": H_FOR_SPSA, "sampler_rng": np.random.default_rng(seed)}
            tol = TOL_FOR_SPSA
        if dev_name == "reference.qubit":
            pytest.xfail("diagonalize_measurements do not support projectors (sc-72911)")

        P = jax.numpy.array(state)
        x, y = 0.765, -0.654

        @qnode(
            get_device(dev_name, wires=2, seed=seed),
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
            gradient_kwargs=gradient_kwargs,
        )
        def circuit(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.Projector(P, wires=0) @ qml.PauliX(1))

        res = circuit(x, y)
        expected = 0.25 * np.sin(x / 2) ** 2 * (3 + np.cos(2 * y) + 2 * np.cos(x) * np.sin(y) ** 2)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = jax.grad(circuit, argnums=[0, 1])(x, y)
        expected = np.array(
            [
                0.5 * np.sin(x) * (np.cos(x / 2) ** 2 + np.cos(2 * y) * np.sin(x / 2) ** 2),
                -2 * np.cos(y) * np.sin(x / 2) ** 4 * np.sin(y),
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)


@pytest.mark.parametrize("interface", ["auto", "jax"])
@pytest.mark.parametrize(
    "dev_name,diff_method,grad_on_execution, device_vjp", device_and_diff_method
)
class TestTapeExpansion:
    """Test that tape expansion within the QNode integrates correctly
    with the JAX interface"""

    @pytest.mark.parametrize("max_diff", [1, 2])
    def test_gradient_expansion_trainable_only(
        self, dev_name, diff_method, grad_on_execution, device_vjp, max_diff, interface, seed
    ):
        """Test that a *supported* operation with no gradient recipe is only
        expanded for parameter-shift and finite-differences when it is trainable."""
        if diff_method not in ("parameter-shift", "finite-diff", "spsa", "hadamard"):
            pytest.skip("Only supports gradient transforms")

        class PhaseShift(qml.PhaseShift):  # pylint:disable=too-few-public-methods
            grad_method = None

            def decomposition(self):
                return [qml.RY(3 * self.data[0], wires=self.wires)]

        @qnode(
            get_device(dev_name, wires=1, seed=seed),
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            max_diff=max_diff,
            interface=interface,
            device_vjp=device_vjp,
        )
        def circuit(x, y):
            qml.Hadamard(wires=0)
            PhaseShift(x, wires=0)
            PhaseShift(2 * y, wires=0)
            return qml.expval(qml.PauliX(0))

        x = jax.numpy.array(0.5)
        y = jax.numpy.array(0.7)
        circuit(x, y)

        jax.grad(circuit, argnums=[0])(x, y)

    @pytest.mark.parametrize("max_diff", [1, 2])
    def test_split_non_commuting_analytic(
        self,
        dev_name,
        diff_method,
        grad_on_execution,
        max_diff,
        interface,
        device_vjp,
        mocker,
        tol,
        seed,
    ):
        """Test that the Hamiltonian is not expanded if there
        are non-commuting groups and the number of shots is None
        and the first and second order gradients are correctly evaluated"""
        gradient_kwargs = {}
        if diff_method == "adjoint":
            pytest.skip("The adjoint method does not yet support Hamiltonians")
        elif diff_method == "hadamard":
            pytest.skip("Hadamard does not yet support Hamiltonians.")
        elif diff_method == "spsa":
            gradient_kwargs = {
                "h": H_FOR_SPSA,
                "num_directions": 20,
                "sampler_rng": np.random.default_rng(seed),
            }
            tol = TOL_FOR_SPSA
        if dev_name == "reference.qubit":
            pytest.skip(
                "Cannot add transform to the transform program in preprocessing"
                "when using mocker.spy on it."
            )

        spy = mocker.spy(qml.transforms, "split_non_commuting")
        obs = [qml.PauliX(0), qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliZ(1)]

        @qnode(
            get_device(dev_name, wires=2, seed=seed),
            interface=interface,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
            max_diff=max_diff,
            gradient_kwargs=gradient_kwargs,
        )
        def circuit(data, weights, coeffs):
            weights = weights.reshape(1, -1)
            qml.templates.AngleEmbedding(data, wires=[0, 1])
            qml.templates.BasicEntanglerLayers(weights, wires=[0, 1])
            return qml.expval(qml.Hamiltonian(coeffs, obs))

        d = jax.numpy.array([0.1, 0.2])
        w = jax.numpy.array([0.654, -0.734])
        c = jax.numpy.array([-0.6543, 0.24, 0.54])

        # test output
        res = circuit(d, w, c)
        expected = c[2] * np.cos(d[1] + w[1]) - c[1] * np.sin(d[0] + w[0]) * np.sin(d[1] + w[1])
        assert np.allclose(res, expected, atol=tol)
        spy.assert_not_called()

        # test gradients
        grad = jax.grad(circuit, argnums=[1, 2])(d, w, c)
        expected_w = [
            -c[1] * np.cos(d[0] + w[0]) * np.sin(d[1] + w[1]),
            -c[1] * np.cos(d[1] + w[1]) * np.sin(d[0] + w[0]) - c[2] * np.sin(d[1] + w[1]),
        ]
        expected_c = [0, -np.sin(d[0] + w[0]) * np.sin(d[1] + w[1]), np.cos(d[1] + w[1])]
        assert np.allclose(grad[0], expected_w, atol=tol)
        assert np.allclose(grad[1], expected_c, atol=tol)

        # TODO: Add parameter shift when the bug with trainable params and hamiltonian_grad is solved.
        # test second-order derivatives
        if diff_method in "backprop" and max_diff == 2:
            grad2_c = jax.jacobian(jax.grad(circuit, argnums=[2]), argnums=[2])(d, w, c)
            assert np.allclose(grad2_c, 0, atol=tol)

            grad2_w_c = jax.jacobian(jax.grad(circuit, argnums=[1]), argnums=[2])(d, w, c)
            expected = [0, -np.cos(d[0] + w[0]) * np.sin(d[1] + w[1]), 0], [
                0,
                -np.cos(d[1] + w[1]) * np.sin(d[0] + w[0]),
                -np.sin(d[1] + w[1]),
            ]
            assert np.allclose(grad2_w_c, expected, atol=tol)

    @pytest.mark.parametrize("max_diff", [1, 2])
    def test_hamiltonian_finite_shots(
        self,
        dev_name,
        diff_method,
        grad_on_execution,
        device_vjp,
        interface,
        max_diff,
        mocker,
        seed,
    ):
        """Test that the Hamiltonian is correctly measured (and not expanded)
        if there are non-commuting groups and the number of shots is finite
        and the first and second order gradients are correctly evaluated"""

        if dev_name == "reference.qubit":
            pytest.skip(
                "Cannot added to a transform to the transform program in "
                "preprocessing when using mocker.spy on it."
            )

        gradient_kwargs = {}
        tol = 0.3
        if diff_method in ("adjoint", "backprop", "finite-diff"):
            pytest.skip("The adjoint and backprop methods do not yet support sampling")
        elif diff_method == "hadamard":
            pytest.skip("Hadamard does not yet support Hamiltonians.")
        elif diff_method == "spsa":
            gradient_kwargs = {
                "h": H_FOR_SPSA,
                "sampler_rng": np.random.default_rng(seed),
                "num_directions": 20,
            }
            tol = TOL_FOR_SPSA

        spy = mocker.spy(qml.transforms, "split_non_commuting")
        obs = [qml.PauliX(0), qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliZ(1)]

        @qml.set_shots(shots=50000)
        @qnode(
            get_device(dev_name, wires=2, seed=seed),
            interface=interface,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
            max_diff=max_diff,
            gradient_kwargs=gradient_kwargs,
        )
        def circuit(data, weights, coeffs):
            weights = weights.reshape(1, -1)
            qml.templates.AngleEmbedding(data, wires=[0, 1])
            qml.templates.BasicEntanglerLayers(weights, wires=[0, 1])
            H = qml.Hamiltonian(coeffs, obs)
            H.compute_grouping()
            return qml.expval(H)

        d = jax.numpy.array([0.1, 0.2])
        w = jax.numpy.array([0.654, -0.734])
        c = jax.numpy.array([-0.6543, 0.24, 0.54])

        # test output
        res = circuit(d, w, c)
        expected = c[2] * np.cos(d[1] + w[1]) - c[1] * np.sin(d[0] + w[0]) * np.sin(d[1] + w[1])
        assert np.allclose(res, expected, atol=tol)
        spy.assert_not_called()

        # test gradients
        grad = jax.grad(circuit, argnums=[1, 2])(d, w, c)
        expected_w = [
            -c[1] * np.cos(d[0] + w[0]) * np.sin(d[1] + w[1]),
            -c[1] * np.cos(d[1] + w[1]) * np.sin(d[0] + w[0]) - c[2] * np.sin(d[1] + w[1]),
        ]
        expected_c = [0, -np.sin(d[0] + w[0]) * np.sin(d[1] + w[1]), np.cos(d[1] + w[1])]
        assert np.allclose(grad[0], expected_w, atol=tol)
        assert np.allclose(grad[1], expected_c, atol=tol)

    #     TODO: Fix hamiltonian grad for parameter shift and jax
    #     # test second-order derivatives
    #     if diff_method == "parameter-shift" and max_diff == 2:

    #         grad2_c = jax.jacobian(jax.grad(circuit, argnum=2), argnum=2)(d, w, c)
    #         assert np.allclose(grad2_c, 0, atol=tol)

    #         grad2_w_c = jax.jacobian(jax.grad(circuit, argnum=1), argnum=2)(d, w, c)
    #         expected = [0, -np.cos(d[0] + w[0]) * np.sin(d[1] + w[1]), 0], [
    #             0,
    #             -np.cos(d[1] + w[1]) * np.sin(d[0] + w[0]),
    #             -np.sin(d[1] + w[1]),
    #         ]
    #         assert np.allclose(grad2_w_c, expected, atol=tol)


jacobian_fn = [jax.jacobian, jax.jacrev, jax.jacfwd]


@pytest.mark.parametrize("shots", [None, 10000])
@pytest.mark.parametrize("interface", ["auto", "jax"])
@pytest.mark.parametrize(
    "dev_name,diff_method,grad_on_execution, device_vjp", device_and_diff_method
)
class TestReturn:  # pylint:disable=too-many-public-methods
    """Class to test the shape of the Grad/Jacobian/Hessian with different return types."""

    def test_grad_single_measurement_param(
        self, dev_name, diff_method, grad_on_execution, device_vjp, shots, interface, seed
    ):
        """For one measurement and one param, the gradient is a float."""
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        @qml.set_shots(shots=shots)
        @qnode(
            get_device(dev_name, wires=2, seed=seed),
            interface=interface,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return qml.expval(qml.PauliZ(0))

        a = jax.numpy.array(0.1)

        grad = jax.grad(circuit)(a)

        assert isinstance(grad, jax.numpy.ndarray)
        assert grad.shape == ()

    def test_grad_single_measurement_multiple_param(
        self, dev_name, diff_method, grad_on_execution, shots, device_vjp, interface, seed
    ):
        """For one measurement and multiple param, the gradient is a tuple of arrays."""
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        @qml.set_shots(shots=shots)
        @qnode(
            get_device(dev_name, wires=2, seed=seed),
            interface=interface,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            return qml.expval(qml.PauliZ(0))

        a = jax.numpy.array(0.1)
        b = jax.numpy.array(0.2)

        grad = jax.grad(circuit, argnums=[0, 1])(a, b)

        assert isinstance(grad, tuple)
        assert len(grad) == 2
        assert grad[0].shape == ()
        assert grad[1].shape == ()

    def test_grad_single_measurement_multiple_param_array(
        self, dev_name, diff_method, grad_on_execution, shots, device_vjp, interface, seed
    ):
        """For one measurement and multiple param as a single array params, the gradient is an array."""
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        @qml.set_shots(shots=shots)
        @qnode(
            get_device(dev_name, wires=2, seed=seed),
            interface=interface,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        a = jax.numpy.array([0.1, 0.2])

        grad = jax.grad(circuit)(a)

        assert isinstance(grad, jax.numpy.ndarray)
        assert grad.shape == (2,)

    @pytest.mark.parametrize("jacobian", jacobian_fn)
    def test_jacobian_single_measurement_param_probs(
        self, dev_name, diff_method, grad_on_execution, device_vjp, jacobian, shots, interface, seed
    ):
        """For a multi dimensional measurement (probs), check that a single array is returned with the correct
        dimension"""
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because of probabilities.")

        @qml.set_shots(shots=shots)
        @qnode(
            get_device(dev_name, wires=2, seed=seed),
            interface=interface,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return qml.probs(wires=[0, 1])

        a = jax.numpy.array(0.1)

        jac = jacobian(circuit)(a)

        assert isinstance(jac, jax.numpy.ndarray)
        assert jac.shape == (4,)

    @pytest.mark.parametrize("jacobian", jacobian_fn)
    def test_jacobian_single_measurement_probs_multiple_param(
        self, dev_name, diff_method, grad_on_execution, device_vjp, jacobian, shots, interface, seed
    ):
        """For a multi dimensional measurement (probs), check that a single tuple is returned containing arrays with
        the correct dimension"""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because of probabilities.")
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        @qml.set_shots(shots=shots)
        @qnode(
            get_device(dev_name, wires=2, seed=seed),
            interface=interface,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            return qml.probs(wires=[0, 1])

        a = jax.numpy.array(0.1)
        b = jax.numpy.array(0.2)

        jac = jacobian(circuit, argnums=[0, 1])(a, b)

        assert isinstance(jac, tuple)

        assert isinstance(jac[0], jax.numpy.ndarray)
        assert jac[0].shape == (4,)

        assert isinstance(jac[1], jax.numpy.ndarray)
        assert jac[1].shape == (4,)

    @pytest.mark.parametrize("jacobian", jacobian_fn)
    def test_jacobian_single_measurement_probs_multiple_param_single_array(
        self, dev_name, diff_method, grad_on_execution, device_vjp, jacobian, shots, interface, seed
    ):
        """For a multi dimensional measurement (probs), check that a single tuple is returned containing arrays with
        the correct dimension"""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because of probabilities.")
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        @qml.set_shots(shots=shots)
        @qnode(
            get_device(dev_name, wires=2, seed=seed),
            interface=interface,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.probs(wires=[0, 1])

        a = jax.numpy.array([0.1, 0.2])
        jac = jacobian(circuit)(a)

        assert isinstance(jac, jax.numpy.ndarray)
        assert jac.shape == (4, 2)

    @pytest.mark.parametrize("jacobian", jacobian_fn)
    def test_jacobian_expval_expval_multiple_params(
        self, dev_name, diff_method, grad_on_execution, jacobian, shots, interface, device_vjp, seed
    ):
        """The hessian of multiple measurements with multiple params return a tuple of arrays."""
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")
        if device_vjp and jacobian is jax.jacfwd:
            pytest.skip("forward pass can't be done with registered vjp.")
        if "lightning" in dev_name:
            pytest.xfail("lightning device_vjp not compatible with jax.jacobian.")
        par_0 = jax.numpy.array(0.1)
        par_1 = jax.numpy.array(0.2)

        @qml.set_shots(shots=shots)
        @qnode(
            get_device(dev_name, wires=2, seed=seed),
            interface=interface,
            diff_method=diff_method,
            max_diff=2,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1)), qml.expval(qml.PauliZ(0))

        jac = jacobian(circuit, argnums=[0, 1])(par_0, par_1)

        assert isinstance(jac, tuple)

        assert isinstance(jac[0], tuple)
        assert len(jac[0]) == 2
        assert isinstance(jac[0][0], jax.numpy.ndarray)
        assert jac[0][0].shape == ()
        assert isinstance(jac[0][1], jax.numpy.ndarray)
        assert jac[0][1].shape == ()

        assert isinstance(jac[1], tuple)
        assert len(jac[1]) == 2
        assert isinstance(jac[1][0], jax.numpy.ndarray)
        assert jac[1][0].shape == ()
        assert isinstance(jac[1][1], jax.numpy.ndarray)
        assert jac[1][1].shape == ()

    @pytest.mark.parametrize("jacobian", jacobian_fn)
    def test_jacobian_expval_expval_multiple_params_array(
        self, dev_name, diff_method, grad_on_execution, jacobian, device_vjp, shots, interface, seed
    ):
        """The jacobian of multiple measurements with a multiple params array return a single array."""
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")
        if device_vjp and jacobian is jax.jacfwd:
            pytest.skip("forward pass can't be done with registered vjp.")
        if "lightning" in dev_name:
            pytest.xfail("lightning device_vjp not compatible with jax.jacobian.")

        @qml.set_shots(shots=shots)
        @qnode(
            get_device(dev_name, wires=2, seed=seed),
            interface=interface,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1)), qml.expval(qml.PauliZ(0))

        a = jax.numpy.array([0.1, 0.2])

        jac = jacobian(circuit)(a)

        assert isinstance(jac, tuple)
        assert len(jac) == 2  # measurements

        assert isinstance(jac[0], jax.numpy.ndarray)
        assert jac[0].shape == (2,)

        assert isinstance(jac[1], jax.numpy.ndarray)
        assert jac[1].shape == (2,)

    @pytest.mark.parametrize("jacobian", jacobian_fn)
    def test_jacobian_var_var_multiple_params(
        self, dev_name, diff_method, grad_on_execution, device_vjp, jacobian, shots, interface, seed
    ):
        """The hessian of multiple measurements with multiple params return a tuple of arrays."""
        if diff_method == "adjoint":
            pytest.skip("adjoint supports either all measurements or only diagonal measurements.")
        if diff_method == "hadamard":
            pytest.skip("Test does not support Hadamard because of var.")
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        par_0 = jax.numpy.array(0.1)
        par_1 = jax.numpy.array(0.2)

        @qml.set_shots(shots=shots)
        @qnode(
            get_device(dev_name, wires=2, seed=seed),
            interface=interface,
            diff_method=diff_method,
            max_diff=2,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.PauliZ(0) @ qml.PauliX(1)), qml.var(qml.PauliZ(0))

        jac = jacobian(circuit, argnums=[0, 1])(par_0, par_1)

        assert isinstance(jac, tuple)
        assert len(jac) == 2

        assert isinstance(jac[0], tuple)
        assert len(jac[0]) == 2
        assert isinstance(jac[0][0], jax.numpy.ndarray)
        assert jac[0][0].shape == ()
        assert isinstance(jac[0][1], jax.numpy.ndarray)
        assert jac[0][1].shape == ()

        assert isinstance(jac[1], tuple)
        assert len(jac[1]) == 2
        assert isinstance(jac[1][0], jax.numpy.ndarray)
        assert jac[1][0].shape == ()
        assert isinstance(jac[1][1], jax.numpy.ndarray)
        assert jac[1][1].shape == ()

    @pytest.mark.parametrize("jacobian", jacobian_fn)
    def test_jacobian_var_var_multiple_params_array(
        self, dev_name, diff_method, grad_on_execution, device_vjp, jacobian, shots, interface, seed
    ):
        """The jacobian of multiple measurements with a multiple params array return a single array."""
        if diff_method == "adjoint":
            pytest.skip("adjoint supports either all expvals or all diagonal measurements.")
        if diff_method == "hadamard":
            pytest.skip("Test does not support Hadamard because of var.")
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        @qml.set_shots(shots=shots)
        @qnode(
            get_device(dev_name, wires=2, seed=seed),
            interface=interface,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.var(qml.PauliZ(0) @ qml.PauliX(1)), qml.var(qml.PauliZ(0))

        a = jax.numpy.array([0.1, 0.2])

        jac = jacobian(circuit)(a)

        assert isinstance(jac, tuple)
        assert len(jac) == 2  # measurements

        assert isinstance(jac[0], jax.numpy.ndarray)
        assert jac[0].shape == (2,)

        assert isinstance(jac[1], jax.numpy.ndarray)
        assert jac[1].shape == (2,)

    @pytest.mark.parametrize("jacobian", jacobian_fn)
    def test_jacobian_multiple_measurement_single_param(
        self, dev_name, diff_method, grad_on_execution, jacobian, device_vjp, shots, interface, seed
    ):
        """The jacobian of multiple measurements with a single params return an array."""
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")
        if "lightning" in dev_name:
            pytest.xfail("lightning device_vjp not compatible with jax.jacobian.")
        if diff_method == "adjoint" and jacobian == jax.jacfwd:
            pytest.skip("jacfwd doesn't like complex numbers")

        @qml.set_shots(shots=shots)
        @qnode(
            get_device(dev_name, wires=2, seed=seed),
            interface=interface,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        a = jax.numpy.array(0.1)

        jac = jacobian(circuit)(a)

        assert isinstance(jac, tuple)
        assert len(jac) == 2

        assert isinstance(jac[0], jax.numpy.ndarray)
        assert jac[0].shape == ()

        assert isinstance(jac[1], jax.numpy.ndarray)
        assert jac[1].shape == (4,)

    @pytest.mark.parametrize("jacobian", jacobian_fn)
    def test_jacobian_multiple_measurement_multiple_param(
        self, dev_name, diff_method, grad_on_execution, jacobian, device_vjp, shots, interface, seed
    ):
        """The jacobian of multiple measurements with a multiple params return a tuple of arrays."""
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")
        if "lightning" in dev_name:
            pytest.xfail("lightning device_vjp not compatible with jax.jacobian.")
        if diff_method == "adjoint" and jacobian == jax.jacfwd:
            pytest.skip("jacfwd doesn't like complex numbers")

        @qml.set_shots(shots=shots)
        @qnode(
            get_device(dev_name, wires=2, seed=seed),
            interface=interface,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        a = jax.numpy.array(0.1)
        b = jax.numpy.array(0.2)

        jac = jacobian(circuit, argnums=[0, 1])(a, b)

        assert isinstance(jac, tuple)
        assert len(jac) == 2

        assert isinstance(jac[0], tuple)
        assert len(jac[0]) == 2
        assert isinstance(jac[0][0], jax.numpy.ndarray)
        assert jac[0][0].shape == ()
        assert isinstance(jac[0][1], jax.numpy.ndarray)
        assert jac[0][1].shape == ()

        assert isinstance(jac[1], tuple)
        assert len(jac[1]) == 2
        assert isinstance(jac[1][0], jax.numpy.ndarray)
        assert jac[1][0].shape == (4,)
        assert isinstance(jac[1][1], jax.numpy.ndarray)
        assert jac[1][1].shape == (4,)

    @pytest.mark.parametrize("jacobian", jacobian_fn)
    def test_jacobian_multiple_measurement_multiple_param_array(
        self, dev_name, diff_method, grad_on_execution, jacobian, device_vjp, shots, interface, seed
    ):
        """The jacobian of multiple measurements with a multiple params array return a single array."""
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")
        if "lightning" in dev_name:
            pytest.xfail("lightning device_vjp not compatible with jax.jacobian.")
        if diff_method == "adjoint" and jacobian == jax.jacfwd:
            pytest.skip("jacfwd doesn't like complex numbers")

        @qml.set_shots(shots=shots)
        @qnode(
            get_device(dev_name, wires=2, seed=seed),
            interface=interface,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        a = jax.numpy.array([0.1, 0.2])

        jac = jacobian(circuit)(a)

        assert isinstance(jac, tuple)
        assert len(jac) == 2  # measurements

        assert isinstance(jac[0], jax.numpy.ndarray)
        assert jac[0].shape == (2,)

        assert isinstance(jac[1], jax.numpy.ndarray)
        assert jac[1].shape == (4, 2)

    def test_hessian_expval_multiple_params(
        self, dev_name, diff_method, grad_on_execution, device_vjp, shots, interface, seed
    ):
        """The hessian of single a measurement with multiple params return a tuple of arrays."""
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because second order diff.")

        par_0 = jax.numpy.array(0.1)
        par_1 = jax.numpy.array(0.2)

        @qml.set_shots(shots=shots)
        @qnode(
            get_device(dev_name, wires=2, seed=seed),
            interface=interface,
            diff_method=diff_method,
            max_diff=2,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        hess = jax.hessian(circuit, argnums=[0, 1])(par_0, par_1)

        assert isinstance(hess, tuple)
        assert len(hess) == 2

        assert isinstance(hess[0], tuple)
        assert len(hess[0]) == 2
        assert isinstance(hess[0][0], jax.numpy.ndarray)
        assert hess[0][0].shape == ()
        assert hess[0][1].shape == ()

        assert isinstance(hess[1], tuple)
        assert len(hess[1]) == 2
        assert isinstance(hess[1][0], jax.numpy.ndarray)
        assert hess[1][0].shape == ()
        assert hess[1][1].shape == ()

    def test_hessian_expval_multiple_param_array(
        self, dev_name, diff_method, grad_on_execution, device_vjp, shots, interface, seed
    ):
        """The hessian of single measurement with a multiple params array return a single array."""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because second order diff.")
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        params = jax.numpy.array([0.1, 0.2])

        @qml.set_shots(shots=shots)
        @qnode(
            get_device(dev_name, wires=2, seed=seed),
            interface=interface,
            diff_method=diff_method,
            max_diff=2,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.RY(x[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        hess = jax.hessian(circuit)(params)

        assert isinstance(hess, jax.numpy.ndarray)
        assert hess.shape == (2, 2)

    def test_hessian_var_multiple_params(
        self, dev_name, diff_method, grad_on_execution, device_vjp, shots, interface, seed
    ):
        """The hessian of single a measurement with multiple params return a tuple of arrays."""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because second order diff.")
        elif diff_method == "hadamard":
            pytest.skip("Test does not support Hadamard because of var.")
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        par_0 = jax.numpy.array(0.1)
        par_1 = jax.numpy.array(0.2)

        @qml.set_shots(shots=shots)
        @qnode(
            get_device(dev_name, wires=2, seed=seed),
            interface=interface,
            diff_method=diff_method,
            max_diff=2,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.PauliZ(0) @ qml.PauliX(1))

        hess = jax.hessian(circuit, argnums=[0, 1])(par_0, par_1)

        assert isinstance(hess, tuple)
        assert len(hess) == 2

        assert isinstance(hess[0], tuple)
        assert len(hess[0]) == 2
        assert isinstance(hess[0][0], jax.numpy.ndarray)
        assert hess[0][0].shape == ()
        assert hess[0][1].shape == ()

        assert isinstance(hess[1], tuple)
        assert len(hess[1]) == 2
        assert isinstance(hess[1][0], jax.numpy.ndarray)
        assert hess[1][0].shape == ()
        assert hess[1][1].shape == ()

    def test_hessian_var_multiple_param_array(
        self, dev_name, diff_method, grad_on_execution, device_vjp, shots, interface, seed
    ):
        """The hessian of single measurement with a multiple params array return a single array."""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because second order diff.")
        elif diff_method == "hadamard":
            pytest.skip("Test does not support Hadamard because of var.")
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        params = jax.numpy.array([0.1, 0.2])

        @qml.set_shots(shots=shots)
        @qnode(
            get_device(dev_name, wires=2, seed=seed),
            interface=interface,
            diff_method=diff_method,
            max_diff=2,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.RY(x[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.PauliZ(0) @ qml.PauliX(1))

        hess = jax.hessian(circuit)(params)

        assert isinstance(hess, jax.numpy.ndarray)
        assert hess.shape == (2, 2)

    def test_hessian_probs_expval_multiple_params(
        self, dev_name, diff_method, grad_on_execution, device_vjp, shots, interface, seed
    ):
        """The hessian of multiple measurements with multiple params return a tuple of arrays."""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because second order diff.")
        elif diff_method == "hadamard":
            pytest.skip("Test does not supports diff of non commuting obs.")

        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        par_0 = jax.numpy.array(0.1)
        par_1 = jax.numpy.array(0.2)

        @qml.set_shots(shots=shots)
        @qnode(
            get_device(dev_name, wires=2, seed=seed),
            interface=interface,
            diff_method=diff_method,
            max_diff=2,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1)), qml.probs(wires=[0])

        hess = jax.hessian(circuit, argnums=[0, 1])(par_0, par_1)

        assert isinstance(hess, tuple)
        assert len(hess) == 2

        assert isinstance(hess[0], tuple)
        assert len(hess[0]) == 2
        assert isinstance(hess[0][0], tuple)
        assert len(hess[0][0]) == 2
        assert isinstance(hess[0][0][0], jax.numpy.ndarray)
        assert hess[0][0][0].shape == ()
        assert isinstance(hess[0][0][1], jax.numpy.ndarray)
        assert hess[0][0][1].shape == ()
        assert isinstance(hess[0][1], tuple)
        assert len(hess[0][1]) == 2
        assert isinstance(hess[0][1][0], jax.numpy.ndarray)
        assert hess[0][1][0].shape == ()
        assert isinstance(hess[0][1][1], jax.numpy.ndarray)
        assert hess[0][1][1].shape == ()

        assert isinstance(hess[1], tuple)
        assert len(hess[1]) == 2
        assert isinstance(hess[1][0], tuple)
        assert len(hess[1][0]) == 2
        assert isinstance(hess[1][0][0], jax.numpy.ndarray)
        assert hess[1][0][0].shape == (2,)
        assert isinstance(hess[1][0][1], jax.numpy.ndarray)
        assert hess[1][0][1].shape == (2,)
        assert isinstance(hess[1][1], tuple)
        assert len(hess[1][1]) == 2
        assert isinstance(hess[1][1][0], jax.numpy.ndarray)
        assert hess[1][1][0].shape == (2,)
        assert isinstance(hess[1][1][1], jax.numpy.ndarray)
        assert hess[1][1][1].shape == (2,)

    def test_hessian_expval_probs_multiple_param_array(
        self, dev_name, diff_method, grad_on_execution, device_vjp, shots, interface, seed
    ):
        """The hessian of multiple measurements with a multiple param array return a single array."""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because second order diff.")
        elif diff_method == "hadamard":
            pytest.skip("Test does not supports diff of non commuting obs.")
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        params = jax.numpy.array([0.1, 0.2])

        @qml.set_shots(shots=shots)
        @qnode(
            get_device(dev_name, wires=2, seed=seed),
            interface=interface,
            diff_method=diff_method,
            device_vjp=device_vjp,
            max_diff=2,
            grad_on_execution=grad_on_execution,
        )
        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.RY(x[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1)), qml.probs(wires=[0])

        hess = jax.hessian(circuit)(params)

        assert isinstance(hess, tuple)
        assert len(hess) == 2

        assert isinstance(hess[0], jax.numpy.ndarray)
        assert hess[0].shape == (2, 2)

        assert isinstance(hess[1], jax.numpy.ndarray)
        assert hess[1].shape == (2, 2, 2)

    def test_hessian_probs_var_multiple_params(
        self, dev_name, diff_method, grad_on_execution, device_vjp, shots, interface, seed
    ):
        """The hessian of multiple measurements with multiple params return a tuple of arrays."""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because second order diff.")
        elif diff_method == "hadamard":
            pytest.skip("Test does not support Hadamard because of var.")
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        par_0 = qml.numpy.array(0.1)
        par_1 = qml.numpy.array(0.2)

        @qml.set_shots(shots=shots)
        @qnode(
            get_device(dev_name, wires=2, seed=seed),
            interface=interface,
            diff_method=diff_method,
            max_diff=2,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.PauliZ(0) @ qml.PauliX(1)), qml.probs(wires=[0])

        hess = jax.hessian(circuit, argnums=[0, 1])(par_0, par_1)

        assert isinstance(hess, tuple)
        assert len(hess) == 2

        assert isinstance(hess[0], tuple)
        assert len(hess[0]) == 2
        assert isinstance(hess[0][0], tuple)
        assert len(hess[0][0]) == 2
        assert isinstance(hess[0][0][0], jax.numpy.ndarray)
        assert hess[0][0][0].shape == ()
        assert isinstance(hess[0][0][1], jax.numpy.ndarray)
        assert hess[0][0][1].shape == ()
        assert isinstance(hess[0][1], tuple)
        assert len(hess[0][1]) == 2
        assert isinstance(hess[0][1][0], jax.numpy.ndarray)
        assert hess[0][1][0].shape == ()
        assert isinstance(hess[0][1][1], jax.numpy.ndarray)
        assert hess[0][1][1].shape == ()

        assert isinstance(hess[1], tuple)
        assert len(hess[1]) == 2
        assert isinstance(hess[1][0], tuple)
        assert len(hess[1][0]) == 2
        assert isinstance(hess[1][0][0], jax.numpy.ndarray)
        assert hess[1][0][0].shape == (2,)
        assert isinstance(hess[1][0][1], jax.numpy.ndarray)
        assert hess[1][0][1].shape == (2,)
        assert isinstance(hess[1][1], tuple)
        assert len(hess[1][1]) == 2
        assert isinstance(hess[1][1][0], jax.numpy.ndarray)
        assert hess[1][1][0].shape == (2,)
        assert isinstance(hess[1][1][1], jax.numpy.ndarray)
        assert hess[1][1][1].shape == (2,)

    def test_hessian_var_probs_multiple_param_array(
        self, dev_name, diff_method, grad_on_execution, device_vjp, shots, interface, seed
    ):
        """The hessian of multiple measurements with a multiple param array return a single array."""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because second order diff.")
        elif diff_method == "hadamard":
            pytest.skip("Test does not support Hadamard because of var.")
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        params = jax.numpy.array([0.1, 0.2])

        @qml.set_shots(shots=shots)
        @qnode(
            get_device(dev_name, wires=2, seed=seed),
            interface=interface,
            diff_method=diff_method,
            max_diff=2,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.RY(x[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.PauliZ(0) @ qml.PauliX(1)), qml.probs(wires=[0])

        hess = jax.hessian(circuit)(params)

        assert isinstance(hess, tuple)
        assert len(hess) == 2

        assert isinstance(hess[0], jax.numpy.ndarray)
        assert hess[0].shape == (2, 2)

        assert isinstance(hess[1], jax.numpy.ndarray)
        assert hess[1].shape == (2, 2, 2)


def test_no_ops():
    """Test that the return value of the QNode matches in the interface
    even if there are no ops"""

    @qml.qnode(DefaultQubit(), interface="jax")
    def circuit():
        qml.Hadamard(wires=0)
        return qml.state()

    res = circuit()
    assert isinstance(res, jax.numpy.ndarray)

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
"""Integration tests for using the JAX-JIT interface with a QNode"""


# pylint: disable=too-many-arguments,too-few-public-methods,protected-access
import pytest
from param_shift_dev import ParamShiftDerivativesDevice

import pennylane as qml
from pennylane import numpy as np
from pennylane import qnode
from pennylane.devices import DefaultQubit
from pennylane.exceptions import DeviceError


def get_device(device_name, wires, seed):
    if device_name == "param_shift.qubit":
        return ParamShiftDerivativesDevice(seed=seed)
    if device_name == "lightning.qubit":
        return qml.device("lightning.qubit", wires=wires)
    return qml.device(device_name, seed=seed)


# device_name, diff_method, grad_on_execution, device_vjp
device_test_cases = [
    ("default.qubit", "backprop", True, False),
    ("default.qubit", "finite-diff", False, False),
    ("default.qubit", "parameter-shift", False, False),
    ("default.qubit", "adjoint", True, False),
    ("default.qubit", "adjoint", True, True),
    ("default.qubit", "adjoint", False, False),
    ("default.qubit", "spsa", False, False),
    ("default.qubit", "hadamard", False, False),
    ("param_shift.qubit", "device", False, True),
    ("lightning.qubit", "adjoint", False, True),
    ("lightning.qubit", "adjoint", True, True),
    ("lightning.qubit", "adjoint", False, False),
    ("lightning.qubit", "adjoint", True, False),
    ("lightning.qubit", "parameter-shift", False, False),
    ("reference.qubit", "parameter-shift", False, False),
]

pytestmark = pytest.mark.jax

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

TOL_FOR_SPSA = 1.0
H_FOR_SPSA = 0.05


@pytest.mark.parametrize("interface", ["auto", "jax-jit"])
@pytest.mark.parametrize(
    "dev_name,diff_method,grad_on_execution,device_vjp",
    device_test_cases,
)
class TestQNode:
    """Test that using the QNode with JAX integrates with the PennyLane
    stack"""

    def test_execution_with_interface(
        self, interface, dev_name, diff_method, grad_on_execution, device_vjp, seed
    ):
        """Test execution works with the interface"""

        if dev_name == "param_shift.qubit":
            pytest.xfail("gradient transforms have a different vjp shape convention")

        dev = get_device(dev_name, wires=1, seed=seed)

        @qnode(
            dev,
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
        jax.jit(circuit)(a)

        assert circuit.interface == interface

        # gradients should work
        grad = jax.jit(jax.grad(circuit))(a)
        assert isinstance(grad, jax.Array)
        assert grad.shape == ()

    def test_changing_trainability(
        self, interface, dev_name, diff_method, grad_on_execution, device_vjp, tol, seed
    ):
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
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Hamiltonian([1, 1], [qml.PauliZ(0), qml.PauliY(1)]))

        grad_fn = jax.jit(jax.grad(circuit, argnums=[0, 1]))
        res = grad_fn(a, b)

        expected = [-np.sin(a) + np.sin(a) * np.sin(b), -np.cos(a) * np.cos(b)]
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # make the second QNode argument a constant
        grad_fn = jax.jit(jax.grad(circuit, argnums=0))
        res = grad_fn(a, b)

        expected = [-np.sin(a) + np.sin(a) * np.sin(b)]
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_classical_processing(
        self, dev_name, diff_method, grad_on_execution, device_vjp, interface, seed
    ):
        """Test classical processing within the quantum tape"""

        if dev_name == "param_shift.qubit":
            pytest.xfail("gradient transforms have a different vjp shape convention.")

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

        res = jax.jit(jax.grad(circuit, argnums=[0, 2]))(a, b, c)

        assert len(res) == 2

    def test_matrix_parameter(
        self, dev_name, diff_method, grad_on_execution, device_vjp, interface, tol, seed
    ):
        """Test that the jax interface works correctly
        with a matrix parameter"""

        if dev_name == "param_shift.qubit":
            pytest.xfail("gradient transforms have a different vjp shape convention.")

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

        res = jax.jit(jax.grad(circuit, argnums=1))(U, a)
        assert np.allclose(res, np.sin(a), atol=tol, rtol=0)

    def test_differentiable_expand(
        self, dev_name, diff_method, grad_on_execution, device_vjp, interface, tol, seed
    ):
        """Test that operation and nested tape expansion
        is differentiable"""

        if dev_name == "param_shift.qubit":
            pytest.xfail("gradient transforms have a different vjp shape convention.")

        gradient_kwargs = {}
        if diff_method == "spsa":
            gradient_kwargs["sampler_rng"] = np.random.default_rng(seed)
            gradient_kwargs["num_directions"] = 20
            tol = TOL_FOR_SPSA

        class U3(qml.U3):
            def decomposition(self):
                theta, phi, lam = self.data
                wires = self.wires
                return [
                    qml.Rot(lam, theta, -lam, wires=wires),
                    qml.PhaseShift(phi + lam, wires=wires),
                ]

        a = jax.numpy.array(0.1)
        p = jax.numpy.array([0.1, 0.2, 0.3])

        @qnode(
            get_device(dev_name, wires=1, seed=seed),
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
            gradient_kwargs=gradient_kwargs,
        )
        def circuit(a, p):
            qml.RX(a, wires=0)
            U3(p[0], p[1], p[2], wires=0)
            return qml.expval(qml.PauliX(0))

        res = jax.jit(circuit)(a, p)
        expected = np.cos(a) * np.cos(p[1]) * np.sin(p[0]) + np.sin(a) * (
            np.cos(p[2]) * np.sin(p[1]) + np.cos(p[0]) * np.cos(p[1]) * np.sin(p[2])
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = jax.jit(jax.grad(circuit, argnums=1))(a, p)
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
    ):
        """Test setting jacobian options"""

        if diff_method != "finite-diff":
            pytest.skip("Test only applies to finite diff.")

        a = np.array([0.1, 0.2], requires_grad=True)

        gradient_kwargs = {"h": 1e-8, "approx_order": 2}

        @qnode(
            get_device(dev_name, wires=1, seed=seed),
            interface=interface,
            diff_method="finite-diff",
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
            gradient_kwargs=gradient_kwargs,
        )
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        if diff_method in {"finite-diff", "parameter-shift", "spsa"} and interface == "jax-jit":
            # No jax.jacobian support for call
            pytest.xfail(reason="batching rules are implemented only for id_tap, not for call.")

        jax.jit(jax.jacobian(circuit))(a)


@pytest.mark.parametrize("interface", ["auto", "jax-jit"])
@pytest.mark.parametrize(
    "dev_name,diff_method,grad_on_execution, device_vjp",
    device_test_cases,
)
class TestVectorValuedQNode:
    """Test that using vector-valued QNodes with JAX integrate with the
    PennyLane stack"""

    def test_diff_expval_expval(
        self, dev_name, diff_method, grad_on_execution, device_vjp, interface, tol, seed
    ):
        """Test jacobian calculation"""

        if dev_name == "param_shift.qubit":
            pytest.xfail("gradient transforms have a different vjp shape convention.")

        if dev_name == "lightning.qubit":
            pytest.xfail("lightning does not support device vjps with jax jacobians.")

        gradient_kwargs = {}

        if diff_method == "spsa":
            gradient_kwargs["sampler_rng"] = np.random.default_rng(seed)
            gradient_kwargs["num_directions"] = 20
            tol = TOL_FOR_SPSA

        a = np.array(0.1, requires_grad=True)
        b = np.array(0.2, requires_grad=True)

        @qnode(
            get_device(dev_name, wires=2, seed=seed),
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
            gradient_kwargs=gradient_kwargs,
        )
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))

        res = jax.jit(circuit)(a, b)

        assert isinstance(res, tuple)
        assert len(res) == 2

        expected = [np.cos(a), -np.cos(a) * np.sin(b)]
        assert np.allclose(res[0], expected[0], atol=tol, rtol=0)
        assert np.allclose(res[1], expected[1], atol=tol, rtol=0)

        res = jax.jit(jax.jacobian(circuit, argnums=[0, 1]))(a, b)

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

        if dev_name == "param_shift.qubit":
            pytest.xfail("gradient transforms have a different vjp shape convention.")

        if dev_name == "lightning.qubit":
            pytest.xfail("lightning does not support device vjps with jax jacobians.")

        gradient_kwargs = {}

        if diff_method == "spsa":
            gradient_kwargs["sampler_rng"] = np.random.default_rng(seed)
            gradient_kwargs["num_directions"] = 20
            tol = TOL_FOR_SPSA

        a = jax.numpy.array(0.1)
        b = jax.numpy.array(0.2)

        @qnode(
            get_device(dev_name, wires=2, seed=seed),
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
            gradient_kwargs=gradient_kwargs,
        )
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))

        jac_fn = jax.jit(jax.jacobian(circuit, argnums=[0, 1]))

        res = jac_fn(a, b)

        assert isinstance(res, tuple)
        assert len(res) == 2

        expected = np.array([[-np.sin(a), 0], [np.sin(a) * np.sin(b), -np.cos(a) * np.cos(b)]])

        for _res, _exp in zip(res, expected):
            for r, e in zip(_res, _exp):
                assert isinstance(r, jax.numpy.ndarray)
                assert r.shape == ()
                assert np.allclose(r, e, atol=tol, rtol=0)

        # call the Jacobian with new parameters
        a = jax.numpy.array(0.6)
        b = jax.numpy.array(0.832)

        res = jac_fn(a, b)

        assert isinstance(res, tuple)
        assert len(res) == 2

        expected = np.array([[-np.sin(a), 0], [np.sin(a) * np.sin(b), -np.cos(a) * np.cos(b)]])

        for _res, _exp in zip(res, expected):
            for r, e in zip(_res, _exp):
                assert isinstance(r, jax.numpy.ndarray)
                assert r.shape == ()
                assert np.allclose(r, e, atol=tol, rtol=0)

    def test_diff_single_probs(
        self, dev_name, diff_method, grad_on_execution, device_vjp, interface, tol, seed
    ):
        """Tests correct output shape and evaluation for a tape
        with a single prob output"""

        if dev_name == "param_shift.qubit":
            pytest.xfail("gradient transforms have a different vjp shape convention.")

        if dev_name == "lightning.qubit":
            pytest.xfail("lightning does not support device vjps with jax jacobians.")

        gradient_kwargs = {}
        if diff_method == "spsa":
            gradient_kwargs["sampler_rng"] = np.random.default_rng(seed)
            gradient_kwargs["num_directions"] = 20
            tol = TOL_FOR_SPSA

        x = jax.numpy.array(0.543)
        y = jax.numpy.array(-0.654)

        @qnode(
            get_device(dev_name, wires=2, seed=seed),
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
            gradient_kwargs=gradient_kwargs,
        )
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[1])

        res = jax.jit(jax.jacobian(circuit, argnums=[0, 1]))(x, y)

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

        if dev_name == "param_shift.qubit":
            pytest.xfail("gradient transforms have a different vjp shape convention.")

        if dev_name == "lightning.qubit":
            pytest.xfail("lightning does not support device vjps with jax jacobians.")

        gradient_kwargs = {}
        if diff_method == "spsa":
            gradient_kwargs["sampler_rng"] = np.random.default_rng(seed)
            gradient_kwargs["num_directions"] = 20
            tol = TOL_FOR_SPSA

        x = jax.numpy.array(0.543)
        y = jax.numpy.array(-0.654)

        @qnode(
            get_device(dev_name, wires=3, seed=seed),
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
            gradient_kwargs=gradient_kwargs,
        )
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

        jac = jax.jit(jax.jacobian(circuit, argnums=[0, 1]))(x, y)
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

        if dev_name == "param_shift.qubit":
            pytest.xfail("gradient transforms have a different vjp shape convention.")

        if dev_name == "lightning.qubit":
            pytest.xfail("lightning does not support device vjps with jax jacobians.")

        gradient_kwargs = {}
        if diff_method == "spsa":
            gradient_kwargs["sampler_rng"] = np.random.default_rng(seed)
            gradient_kwargs["num_directions"] = 20
            tol = TOL_FOR_SPSA

        x = jax.numpy.array(0.543)
        y = jax.numpy.array(-0.654)

        @qnode(
            get_device(dev_name, wires=2, seed=seed),
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
            gradient_kwargs=gradient_kwargs,
        )
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[1])

        res = jax.jit(circuit)(x, y)
        expected = [np.cos(x), [(1 + np.cos(x) * np.cos(y)) / 2, (1 - np.cos(x) * np.cos(y)) / 2]]

        assert isinstance(res, tuple)
        assert len(res) == 2

        assert isinstance(res[0], jax.numpy.ndarray)
        assert res[0].shape == ()
        assert np.allclose(res[0], expected[0], atol=tol, rtol=0)

        assert isinstance(res[1], jax.numpy.ndarray)
        assert res[1].shape == (2,)
        assert np.allclose(res[1], expected[1], atol=tol, rtol=0)

        jac = jax.jit(jax.jacobian(circuit, argnums=[0, 1]))(x, y)
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

        if dev_name == "param_shift.qubit":
            pytest.xfail("gradient transforms have a different vjp shape convention.")

        if dev_name == "lightning.qubit":
            pytest.xfail("lightning does not support device vjps with jax jacobians.")

        kwargs = {}
        if diff_method == "spsa":
            kwargs["sampler_rng"] = np.random.default_rng(seed)
            tol = TOL_FOR_SPSA

        x = jax.numpy.array(0.543)
        y = jax.numpy.array(-0.654)

        @qnode(
            get_device(dev_name, wires=2, seed=seed),
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

        jac = jax.jit(jax.jacobian(circuit, argnums=[0]))(x, y)

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

        if dev_name == "param_shift.qubit":
            pytest.xfail("gradient transforms have a different vjp shape convention.")

        if dev_name == "lightning.qubit":
            pytest.xfail("lightning does not support device vjps with jax jacobians.")

        gradient_kwargs = {}
        if diff_method == "hadamard":
            pytest.skip("Hadamard does not support var")
        elif diff_method == "spsa":
            gradient_kwargs["sampler_rng"] = np.random.default_rng(seed)
            gradient_kwargs["num_directions"] = 20
            tol = TOL_FOR_SPSA

        x = jax.numpy.array(0.543)
        y = jax.numpy.array(-0.654)

        @qnode(
            get_device(dev_name, wires=2, seed=seed),
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
            gradient_kwargs=gradient_kwargs,
        )
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.PauliZ(0)), qml.probs(wires=[1])

        res = jax.jit(circuit)(x, y)

        expected = [
            np.sin(x) ** 2,
            [(1 + np.cos(x) * np.cos(y)) / 2, (1 - np.cos(x) * np.cos(y)) / 2],
        ]

        assert isinstance(res[0], jax.numpy.ndarray)
        assert res[0].shape == ()
        assert np.allclose(res[0], expected[0], atol=tol, rtol=0)

        assert isinstance(res[1], jax.numpy.ndarray)
        assert res[1].shape == (2,)
        assert np.allclose(res[1], expected[1], atol=tol, rtol=0)

        jac = jax.jit(jax.jacobian(circuit, argnums=[0, 1]))(x, y)
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


@pytest.mark.parametrize("interface", ["auto", "jax", "jax-jit"])
class TestShotsIntegration:
    """Test that the QNode correctly changes shot value, and
    remains differentiable."""

    def test_diff_method_None(self, interface):
        """Test device works with diff_method=None."""
        dev = DefaultQubit()

        @jax.jit
        @qml.qnode(dev, diff_method=None, interface=interface)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        assert jax.numpy.allclose(circuit(jax.numpy.array(0.0)), 1)

    @pytest.mark.skip("jax.jit does not work with sample")
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
        res = qml.set_shots(shots=100)(circuit)(a, b)  # pylint: disable=unexpected-keyword-arg
        assert res.shape == (100, 2)  # pylint:disable=comparison-with-callable

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

        jit_cost_fn = jax.jit(cost_fn)
        res = jax.grad(jit_cost_fn, argnums=[0, 1])(a, b)

        expected = [np.sin(a) * np.sin(b), -np.cos(a) * np.cos(b)]
        assert np.allclose(res, expected, atol=0.1, rtol=0)

    def test_update_diff_method(self, interface):
        """Test that temporarily setting the shots updates the diff method"""
        # pylint: disable=unused-argument
        a, b = jax.numpy.array([0.543, -0.654])

        dev = DefaultQubit()

        @qnode(dev, interface=interface)
        def cost_fn(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliY(1))

        with dev.tracker:
            jax.grad(qml.set_shots(cost_fn, shots=100))(a, b)
        # since we are using finite shots, use parameter shift
        assert dev.tracker.totals["executions"] == 3

        # if we use the default shots value of None, backprop can now be used
        with dev.tracker:
            jax.grad(cost_fn)(a, b)
        assert dev.tracker.totals["executions"] == 1

    @pytest.mark.parametrize("shots", [10000, 10005])
    def test_finite_shot_single_measurements(self, interface, shots, seed):
        """Test jax-jit can work with shot vectors and returns correct shapes."""

        dev = qml.device("default.qubit", seed=seed)

        @jax.jit
        @qml.set_shots(shots)
        @qml.qnode(dev, interface=interface, diff_method="parameter-shift")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.var(qml.PauliZ(0))

        res = circuit(0.5)
        expected = 1 - np.cos(0.5) ** 2
        assert qml.math.allclose(res, expected, atol=1 / qml.math.sqrt(shots), rtol=0.03)

        g = jax.jacobian(circuit)(0.5)
        expected_g = 2 * np.cos(0.5) * np.sin(0.5)
        assert qml.math.allclose(g, expected_g, atol=1 / qml.math.sqrt(shots), rtol=0.03)

    @pytest.mark.parametrize("shots", [10000, 10005])
    def test_finite_shot_multiple_measurements(self, interface, shots, seed):
        """Test jax-jit can work with shot vectors and returns correct shapes."""

        dev = qml.device("default.qubit", seed=seed)

        @jax.jit
        @qml.set_shots(shots)
        @qml.qnode(dev, interface=interface, diff_method="parameter-shift")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=0)

        res = circuit(0.5)
        expected = np.cos(0.5)
        assert qml.math.allclose(res[0], expected, atol=1 / qml.math.sqrt(shots), rtol=0.03)

        expected_probs = np.array([np.cos(0.25) ** 2, np.sin(0.25) ** 2])
        assert qml.math.allclose(res[1], expected_probs, atol=1 / qml.math.sqrt(shots), rtol=0.03)
        assert qml.math.allclose(
            res[1][0], expected_probs[0], atol=1 / qml.math.sqrt(shots), rtol=0.03
        )
        # Smaller atol since sin(0.25)**2 is close to zero
        assert qml.math.allclose(res[1][1], expected_probs[1], atol=0.5 * 1 / qml.math.sqrt(shots))

    @pytest.mark.parametrize("shots", [(10, 10), (10, 15)])
    def test_shot_vectors_single_measurements(self, interface, shots, seed):
        """Test jax-jit can work with shot vectors and returns correct shapes."""

        dev = qml.device("default.qubit", seed=seed)

        @jax.jit
        @qml.set_shots(shots)
        @qml.qnode(dev, interface=interface, diff_method="parameter-shift")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.var(qml.PauliZ(0))

        res = circuit(0.5)

        # Test return shapes for shot vectors
        assert isinstance(res, tuple)
        assert len(res) == 2  # Two different shot counts
        assert all(isinstance(r, jax.numpy.ndarray) for r in res)
        assert all(r.shape == () for r in res)  # Scalar outputs

        g = jax.jacobian(circuit)(0.5)

        # Test gradient shapes for shot vectors
        assert isinstance(g, tuple)
        assert len(g) == 2  # Two different shot counts
        assert all(isinstance(gr, jax.numpy.ndarray) for gr in g)
        assert all(gr.shape == () for gr in g)  # Scalar gradients

    @pytest.mark.parametrize("shots", [(10, 10), (10, 15)])
    def test_shot_vectors_multiple_measurements(self, interface, shots, seed):
        """Test jax-jit can work with shot vectors and returns correct shapes for multiple measurements."""

        dev = qml.device("default.qubit", seed=seed)

        @jax.jit
        @qml.set_shots(shots)
        @qml.qnode(dev, interface=interface, diff_method="parameter-shift")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=0)

        res = circuit(0.5)

        # Test return shapes for shot vectors with multiple measurements
        assert isinstance(res, tuple)
        assert len(res) == 2  # Two different shot counts

        # Each shot count should return a tuple of (expval, probs)
        for shot_res in res:
            assert isinstance(shot_res, tuple)
            assert len(shot_res) == 2  # expval and probs

            # expval should be scalar
            assert isinstance(shot_res[0], jax.numpy.ndarray)
            assert shot_res[0].shape == ()

            # probs should be 1D array with 2 elements (for 1 qubit)
            assert isinstance(shot_res[1], jax.numpy.ndarray)
            assert shot_res[1].shape == (2,)


@pytest.mark.parametrize("interface", ["auto", "jax-jit"])
@pytest.mark.parametrize(
    "dev_name,diff_method,grad_on_execution, device_vjp",
    device_test_cases,
)
class TestQubitIntegration:
    """Tests that ensure various qubit circuits integrate correctly"""

    def test_sampling(self, dev_name, diff_method, grad_on_execution, device_vjp, interface, seed):
        """Test sampling works as expected"""

        if grad_on_execution:
            pytest.skip("Sampling not possible with forward grad_on_execution differentiation.")

        if diff_method == "adjoint":
            pytest.skip("Adjoint warns with finite shots")

        @qml.set_shots(shots=10)
        @qml.qnode(
            get_device(dev_name, wires=2, seed=seed),
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit():
            qml.Hadamard(wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.sample(qml.Z(0)), qml.sample(qml.s_prod(2, qml.X(0) @ qml.Y(1)))

        res = jax.jit(circuit)()

        assert isinstance(res, tuple)

        assert isinstance(res[0], jax.Array)
        assert res[0].shape == (10,)
        assert isinstance(res[1], jax.Array)
        assert res[1].shape == (10,)

    def test_counts(self, dev_name, diff_method, grad_on_execution, device_vjp, interface, seed):
        """Test counts works as expected"""

        if grad_on_execution:
            pytest.skip("Sampling not possible with forward grad_on_execution differentiation.")

        if diff_method == "adjoint":
            pytest.skip("Adjoint warns with finite shots")

        @qml.set_shots(shots=10)
        @qnode(
            get_device(dev_name, wires=2, seed=seed),
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit():
            qml.Hadamard(wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.counts(qml.PauliZ(0)), qml.counts(qml.PauliX(1))

        if interface == "jax-jit":
            with pytest.raises(
                NotImplementedError, match="The JAX-JIT interface doesn't support qml.counts."
            ):
                jax.jit(circuit)()
        else:
            res = jax.jit(circuit)()

            assert isinstance(res, tuple)

            assert isinstance(res[0], dict)
            assert len(res[0]) == 2
            assert isinstance(res[1], dict)
            assert len(res[1]) == 2

    def test_chained_qnodes(
        self, dev_name, diff_method, grad_on_execution, device_vjp, interface, seed
    ):
        """Test that the gradient of chained QNodes works without error"""

        if dev_name == "param_shift.qubit":
            pytest.xfail("gradient transforms have a different vjp shape convention.")

        class Template(qml.templates.StronglyEntanglingLayers):
            def decomposition(self):
                return [qml.templates.StronglyEntanglingLayers(*self.parameters, self.wires)]

        dev = get_device(dev_name, wires=2, seed=seed)

        @qnode(
            dev,
            interface=interface,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit1(weights):
            Template(weights, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        @qnode(
            dev,
            interface=interface,
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

        grad_fn = jax.jit(jax.grad(cost))
        res = grad_fn(weights)

        assert len(res) == 2

    def test_postselection_differentiation(
        self, dev_name, diff_method, grad_on_execution, device_vjp, interface, seed
    ):
        """Test that when postselecting with default.qubit, differentiation works correctly."""

        if diff_method in ["adjoint", "spsa", "hadamard"]:
            pytest.skip("Diff method does not support postselection.")
        if dev_name == "param_shift.qubit":
            pytest.xfail("gradient transforms have a different vjp shape convention")
        elif dev_name == "lightning.qubit":
            pytest.xfail("lightning qubit does not support postselection.")
        if dev_name == "reference.qubit":
            pytest.skip("reference.qubit does not support postselection.")

        dev = get_device(dev_name, wires=2, seed=seed)

        @qml.qnode(
            dev, diff_method=diff_method, interface=interface, grad_on_execution=grad_on_execution
        )
        def circuit(phi, theta):
            qml.RX(phi, wires=0)
            qml.CNOT([0, 1])
            qml.measure(wires=0, postselect=1)
            qml.RX(theta, wires=1)
            return qml.expval(qml.PauliZ(1))

        @qml.qnode(
            dev,
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def expected_circuit(theta):
            qml.PauliX(1)
            qml.RX(theta, wires=1)
            return qml.expval(qml.PauliZ(1))

        phi = jax.numpy.array(1.23)
        theta = jax.numpy.array(4.56)

        assert np.allclose(jax.jit(circuit)(phi, theta), jax.jit(expected_circuit)(theta))

        gradient = jax.jit(jax.grad(circuit, argnums=[0, 1]))(phi, theta)
        exp_theta_grad = jax.jit(jax.grad(expected_circuit))(theta)
        assert np.allclose(gradient, [0.0, exp_theta_grad])


@pytest.mark.parametrize("interface", ["auto", "jax-jit"])
@pytest.mark.parametrize(
    "dev_name,diff_method,grad_on_execution,device_vjp",
    device_test_cases,
)
class TestQubitIntegrationHigherOrder:
    """Tests that ensure various qubit circuits integrate correctly when computing higher-order derivatives"""

    def test_second_derivative(
        self, dev_name, diff_method, grad_on_execution, device_vjp, interface, tol, seed
    ):
        """Test second derivative calculation of a scalar-valued QNode"""
        gradient_kwargs = {}
        if diff_method in {"adjoint", "device"}:
            pytest.skip("Adjoint does not support second derivatives.")
        elif diff_method == "spsa":
            gradient_kwargs["sampler_rng"] = np.random.default_rng(seed)
            gradient_kwargs["num_directions"] = 20
            gradient_kwargs["h"] = H_FOR_SPSA
            tol = TOL_FOR_SPSA

        dev = get_device(dev_name, wires=1, seed=seed)

        @qnode(
            dev,
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
        g = jax.jit(jax.grad(circuit))(x)
        g2 = jax.jit(jax.grad(lambda x: jax.numpy.sum(jax.grad(circuit)(x))))(x)

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
        if diff_method in {"adjoint", "device"}:
            pytest.skip("Adjoint does not support second derivative.")
        elif diff_method == "spsa":
            gradient_kwargs = {
                "h": H_FOR_SPSA,
                "num_directions": 40,
                "sampler_rng": np.random.default_rng(seed),
            }
            tol = TOL_FOR_SPSA

        dev = get_device(dev_name, wires=1, seed=seed)

        @qnode(
            dev,
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
        res = jax.jit(circuit)(x)

        a, b = x

        expected_res = np.cos(a) * np.cos(b)
        assert np.allclose(res, expected_res, atol=tol, rtol=0)

        grad_fn = jax.jit(jax.grad(circuit))
        g = grad_fn(x)

        expected_g = [-np.sin(a) * np.cos(b), -np.cos(a) * np.sin(b)]
        assert np.allclose(g, expected_g, atol=tol, rtol=0)

        hess = jax.jit(jax.jacobian(grad_fn))(x)

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
        if diff_method in {"adjoint", "device"}:
            pytest.skip("Adjoint does not support second derivative.")
        elif diff_method == "spsa":
            gradient_kwargs = {
                "h": H_FOR_SPSA,
                "num_directions": 20,
                "sampler_rng": np.random.default_rng(seed),
            }
            tol = TOL_FOR_SPSA

        @qnode(
            get_device(dev_name, wires=1, seed=seed),
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

        jac_fn = jax.jit(jax.jacobian(circuit))
        g = jac_fn(x)

        expected_g = [
            [-0.5 * np.sin(a) * np.cos(b), -0.5 * np.cos(a) * np.sin(b)],
            [0.5 * np.sin(a) * np.cos(b), 0.5 * np.cos(a) * np.sin(b)],
        ]
        assert np.allclose(g, expected_g, atol=tol, rtol=0)

        hess = jax.jit(jax.jacobian(jac_fn))(x)

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
        self, dev_name, diff_method, interface, device_vjp, grad_on_execution, tol, seed
    ):
        """Test hessian calculation of a vector valued QNode with post-processing"""
        gradient_kwargs = {}
        if diff_method in {"adjoint", "device"}:
            pytest.skip("Adjoint does not support second derivative.")
        elif diff_method == "spsa":
            gradient_kwargs = {
                "h": H_FOR_SPSA,
                "num_directions": 20,
                "sampler_rng": np.random.default_rng(seed),
            }
            tol = TOL_FOR_SPSA

        @qnode(
            get_device(dev_name, wires=1, seed=seed),
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
        res = jax.jit(cost_fn)(x)

        a, b = x

        expected_res = x @ jax.numpy.array([np.cos(a) * np.cos(b), np.cos(a) * np.cos(b)])
        assert np.allclose(res, expected_res, atol=tol, rtol=0)

        grad_fn = jax.jit(jax.grad(cost_fn))
        g = grad_fn(x)

        expected_g = [
            np.cos(b) * (np.cos(a) - (a + b) * np.sin(a)),
            np.cos(a) * (np.cos(b) - (a + b) * np.sin(b)),
        ]
        assert np.allclose(g, expected_g, atol=tol, rtol=0)
        hess = jax.jit(jax.jacobian(grad_fn))(x)

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
        if diff_method in {"adjoint", "device"}:
            pytest.skip("Adjoint does not support second derivative.")
        elif diff_method == "spsa":
            gradient_kwargs = {
                "h": H_FOR_SPSA,
                "num_directions": 20,
                "sampler_rng": np.random.default_rng(seed),
            }
            tol = TOL_FOR_SPSA

        @qnode(
            get_device(dev_name, wires=1, seed=seed),
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            max_diff=2,
            device_vjp=device_vjp,
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

        jac_fn = jax.jit(jax.jacobian(circuit, argnums=[0, 1]))
        g = jac_fn(a, b)

        expected_g = np.array(
            [
                [-0.5 * np.sin(a) * np.cos(b), -0.5 * np.cos(a) * np.sin(b)],
                [0.5 * np.sin(a) * np.cos(b), 0.5 * np.cos(a) * np.sin(b)],
            ]
        )
        assert np.allclose(g, expected_g.T, atol=tol, rtol=0)

        hess = jax.jit(jax.jacobian(jac_fn, argnums=[0, 1]))(a, b)

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

        if dev_name == "lightning.qubit" and diff_method == "adjoint":
            pytest.xfail("lightning.qubit does not support adjoint with the state.")

        dev = get_device(dev_name, wires=2, seed=seed)

        x = jax.numpy.array(0.543)
        y = jax.numpy.array(-0.654)

        @qnode(
            dev,
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
            assert res.dtype is np.dtype("complex128")
            probs = jax.numpy.abs(res) ** 2
            return probs[0] + probs[2]

        res = jax.jit(cost_fn)(x, y)

        if diff_method not in {"backprop"}:
            pytest.skip("Test only supports backprop")

        res = jax.jit(jax.grad(cost_fn, argnums=[0, 1]))(x, y)
        expected = np.array([-np.sin(x) * np.cos(y) / 2, -np.cos(x) * np.sin(y) / 2])
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("state", [[1], [0, 1]])  # Basis state and state vector
    def test_projector(
        self, state, dev_name, diff_method, grad_on_execution, device_vjp, interface, tol, seed
    ):
        """Test that the variance of a projector is correctly returned"""
        gradient_kwargs = {}
        if dev_name == "param_shift.qubit":
            pytest.xfail("gradient transforms have a different vjp shape convention.")
        if diff_method == "adjoint":
            pytest.skip("Adjoint does not support projectors")
        elif diff_method == "hadamard":
            pytest.skip("Hadamard does not support var")
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

        res = jax.jit(circuit)(x, y)
        expected = 0.25 * np.sin(x / 2) ** 2 * (3 + np.cos(2 * y) + 2 * np.cos(x) * np.sin(y) ** 2)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = jax.jit(jax.grad(circuit, argnums=[0, 1]))(x, y)
        expected = np.array(
            [
                0.5 * np.sin(x) * (np.cos(x / 2) ** 2 + np.cos(2 * y) * np.sin(x / 2) ** 2),
                -2 * np.cos(y) * np.sin(x / 2) ** 4 * np.sin(y),
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)


@pytest.mark.parametrize("interface", ["auto", "jax-jit"])
@pytest.mark.parametrize(
    "dev_name,diff_method,grad_on_execution, device_vjp",
    device_test_cases,
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

        if diff_method not in ("parameter-shift", "finite-diff", "spsa"):
            pytest.skip("Only supports gradient transforms")

        class PhaseShift(qml.PhaseShift):
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
    def test_hamiltonian_expansion_analytic(
        self,
        dev_name,
        diff_method,
        grad_on_execution,
        max_diff,
        device_vjp,
        interface,
        mocker,
        tol,
        seed,
    ):
        """Test that the Hamiltonian is not expanded if there
        are non-commuting groups and the number of shots is None
        and the first and second order gradients are correctly evaluated"""

        gradient_kwargs = {}
        if dev_name == "reference.qubit":
            pytest.skip(
                "Cannot add transform to the transform program in preprocessing"
                "when using mocker.spy on it."
            )
        if dev_name == "param_shift.qubit":
            pytest.xfail("gradients transforms have a different vjp shape convention.")
        if diff_method == "adjoint":
            pytest.skip("The adjoint method does not yet support Hamiltonians")
        elif diff_method == "hadamard":
            pytest.skip("The Hadamard method does not yet support Hamiltonians")
        elif diff_method == "spsa":
            gradient_kwargs = {
                "h": H_FOR_SPSA,
                "num_directions": 20,
                "sampler_rng": np.random.default_rng(seed),
            }
            tol = TOL_FOR_SPSA

        spy = mocker.spy(qml.transforms, "split_non_commuting")
        obs = [qml.PauliX(0), qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliZ(1)]

        @jax.jit
        @qnode(
            get_device(dev_name, wires=2, seed=seed),
            interface=interface,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            max_diff=max_diff,
            device_vjp=device_vjp,
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
        self, dev_name, diff_method, grad_on_execution, device_vjp, interface, max_diff, seed
    ):
        """Test that the Hamiltonian is correctly measured if there
        are non-commuting groups and the number of shots is finite
        and the first and second order gradients are correctly evaluated"""
        gradient_kwargs = {}
        tol = 0.3
        if dev_name == "param_shift.qubit":
            pytest.xfail("gradient transforms have a different vjp shape convention.")
        if diff_method in ("adjoint", "backprop", "finite-diff"):
            pytest.skip("The adjoint and backprop methods do not yet support sampling")
        elif diff_method == "hadamard":
            pytest.skip("The Hadamard method does not yet support Hamiltonians")
        elif diff_method == "spsa":
            gradient_kwargs = {"sampler_rng": seed, "h": H_FOR_SPSA, "num_directions": 20}
            tol = TOL_FOR_SPSA

        obs = [qml.PauliX(0), qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliZ(1)]

        @qml.set_shots(shots=50000)
        @qnode(
            get_device(dev_name, wires=2, seed=seed),
            interface=interface,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            max_diff=max_diff,
            device_vjp=device_vjp,
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

    def test_vmap_compared_param_broadcasting(
        self, dev_name, diff_method, grad_on_execution, device_vjp, interface, tol, seed
    ):
        """Test that jax.vmap works just as well as parameter-broadcasting with JAX JIT on the forward pass when
        vectorized=True is specified for the callback when caching is disabled."""
        if (
            dev_name == "default.qubit"
            and diff_method == "adjoint"
            and grad_on_execution
            and not device_vjp
        ):
            pytest.xfail("adjoint is incompatible with parameter broadcasting.")
        interface = "jax-jit"

        n_configs = 5
        pars_q = np.random.rand(n_configs, 2)

        def minimal_circ(params):
            @qml.qnode(
                get_device(dev_name, wires=2, seed=seed),
                interface=interface,
                diff_method=diff_method,
                grad_on_execution=grad_on_execution,
                device_vjp=device_vjp,
                cache=None,
            )
            def _measure_operator():
                qml.RY(params[..., 0], wires=0)
                qml.RY(params[..., 1], wires=1)
                return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

            res = _measure_operator()
            return res

        res1 = jax.jit(minimal_circ)(pars_q)
        res2 = jax.jit(jax.vmap(minimal_circ))(pars_q)
        assert np.allclose(res1, res2, tol)

    def test_vmap_compared_param_broadcasting_multi_output(
        self, dev_name, diff_method, grad_on_execution, device_vjp, interface, tol, seed
    ):
        """Test that jax.vmap works just as well as parameter-broadcasting with JAX JIT on the forward pass when
        vectorized=True is specified for the callback when caching is disabled and when multiple output values
        are returned."""
        if (
            dev_name == "default.qubit"
            and diff_method == "adjoint"
            and grad_on_execution
            and not device_vjp
        ):
            pytest.xfail("adjoint is incompatible with parameter broadcasting.")
        interface = "jax-jit"

        n_configs = 5
        pars_q = np.random.rand(n_configs, 2)

        def minimal_circ(params):
            @qml.qnode(
                get_device(dev_name, wires=2, seed=seed),
                interface=interface,
                diff_method=diff_method,
                grad_on_execution=grad_on_execution,
                device_vjp=device_vjp,
                cache=None,
            )
            def _measure_operator():
                qml.RY(params[..., 0], wires=0)
                qml.RY(params[..., 1], wires=1)
                return qml.expval(qml.Z(0) @ qml.Z(1)), qml.expval(qml.X(0) @ qml.X(1))

            res = _measure_operator()
            return res

        res1, res2 = jax.jit(minimal_circ)(pars_q)
        vres1, vres2 = jax.jit(jax.vmap(minimal_circ))(pars_q)
        assert np.allclose(res1, vres1, tol)
        assert np.allclose(res2, vres2, tol)

    def test_vmap_compared_param_broadcasting_probs(
        self, dev_name, diff_method, grad_on_execution, device_vjp, interface, tol, seed
    ):
        """Test that jax.vmap works just as well as parameter-broadcasting with JAX JIT on the forward pass when
        vectorized=True is specified for the callback when caching is disabled and when multiple output values
        are returned."""
        if (
            dev_name == "default.qubit"
            and diff_method == "adjoint"
            and grad_on_execution
            and not device_vjp
        ):
            pytest.xfail("adjoint is incompatible with parameter broadcasting.")
        elif dev_name == "lightning.qubit" and diff_method == "adjoint":
            pytest.xfail("lightning adjoign cannot differentiate probabilities.")
        interface = "jax-jit"

        n_configs = 5
        pars_q = np.random.rand(n_configs, 2)

        def minimal_circ(params):
            @qml.qnode(
                get_device(dev_name, wires=2, seed=seed),
                interface=interface,
                diff_method=diff_method,
                grad_on_execution=grad_on_execution,
                device_vjp=device_vjp,
                cache=None,
            )
            def _measure_operator():
                qml.RY(params[..., 0], wires=0)
                qml.RY(params[..., 1], wires=1)
                return qml.probs(wires=0), qml.probs(wires=1)

            res = _measure_operator()
            return res

        res1, res2 = jax.jit(minimal_circ)(pars_q)
        vres1, vres2 = jax.jit(jax.vmap(minimal_circ))(pars_q)
        assert np.allclose(res1, vres1, tol)
        assert np.allclose(res2, vres2, tol)


jacobian_fn = [jax.jacobian, jax.jacrev, jax.jacfwd]


@pytest.mark.parametrize("interface", ["auto", "jax-jit"])
@pytest.mark.parametrize("jacobian", jacobian_fn)
@pytest.mark.parametrize(
    "dev_name,diff_method,grad_on_execution,device_vjp",
    device_test_cases,
)
class TestJIT:
    """Test JAX JIT integration with the QNode and automatic resolution of the
    correct JAX interface variant."""

    def test_gradient(
        self, dev_name, diff_method, grad_on_execution, device_vjp, jacobian, tol, interface, seed
    ):
        """Test derivative calculation of a scalar valued QNode"""
        gradient_kwargs = {}
        if dev_name == "param_shift.qubit":
            pytest.xfail("gradient transforms have a different vjp shape convention.")
        if "lightning" in dev_name:
            pytest.xfail("lightning device vjps are not compatible with jax jaocbians")
        if device_vjp and jacobian == jax.jacfwd:
            pytest.skip("device vjps not compatible with forward diff.")
        elif diff_method == "spsa":
            gradient_kwargs = {"h": H_FOR_SPSA, "sampler_rng": np.random.default_rng(seed)}
            tol = TOL_FOR_SPSA

        @qnode(
            get_device(dev_name, wires=1, seed=seed),
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
            gradient_kwargs=gradient_kwargs,
        )
        def circuit(x):
            qml.RY(x[0], wires=0)
            qml.RX(x[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        x = jax.numpy.array([1.0, 2.0])
        res = circuit(x)
        g = jax.jit(jacobian(circuit))(x)

        a, b = x

        expected_res = np.cos(a) * np.cos(b)
        assert np.allclose(res, expected_res, atol=tol, rtol=0)

        expected_g = [-np.sin(a) * np.cos(b), -np.cos(a) * np.sin(b)]
        assert np.allclose(g, expected_g, atol=tol, rtol=0)

    @pytest.mark.filterwarnings(
        "ignore:Requested adjoint differentiation to be computed with finite shots."
    )
    @pytest.mark.parametrize("shots", [10, 1000])
    def test_hermitian(
        self, dev_name, diff_method, grad_on_execution, device_vjp, shots, jacobian, interface, seed
    ):
        """Test that the jax device works with qml.Hermitian and jitting even
        when shots>0.

        Note: before a fix, the cases of shots=10 and shots=1000 were failing due
        to different reasons, hence the parametrization in the test.
        """
        # pylint: disable=unused-argument
        if dev_name == "reference.qubit":
            pytest.xfail("diagonalize_measurements do not support Hermitians (sc-72911)")

        if diff_method == "backprop":
            pytest.skip("Backpropagation is unsupported if shots > 0.")

        if diff_method == "adjoint":
            pytest.skip("Computing the gradient for observables is not supported with adjoint.")

        projector = np.array(qml.matrix(qml.PauliZ(0) @ qml.PauliZ(1)))

        @qml.qnode(
            get_device(dev_name, wires=2, seed=seed),
            interface=interface,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circ(projector):
            return qml.expval(qml.Hermitian(projector, wires=range(2)))

        result = jax.jit(circ)(projector)
        assert jax.numpy.allclose(result, 1)

    @pytest.mark.filterwarnings(
        "ignore:Requested adjoint differentiation to be computed with finite shots."
    )
    @pytest.mark.parametrize("shots", [10, 1000])
    def test_probs_obs_none(
        self, dev_name, diff_method, grad_on_execution, device_vjp, shots, jacobian, interface, seed
    ):
        """Test that the jax device works with qml.probs, a MeasurementProcess
        that has obs=None even when shots>0."""
        # pylint: disable=unused-argument
        if diff_method in ["backprop", "adjoint"]:
            pytest.skip("Backpropagation is unsupported if shots > 0.")

        @qml.qnode(
            get_device(dev_name, wires=1, seed=seed),
            interface=interface,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit():
            return qml.probs(wires=0)

        assert jax.numpy.allclose(circuit(), jax.numpy.array([1.0, 0.0]))

    def test_gradient_subset(
        self, dev_name, diff_method, grad_on_execution, device_vjp, jacobian, tol, interface, seed
    ):
        """Test derivative calculation of a scalar valued QNode with respect
        to a subset of arguments"""

        if diff_method == "spsa" and not grad_on_execution and not device_vjp:
            pytest.xfail(reason="incorrect jacobian results")

        if "lightning" in dev_name:
            pytest.xfail("lightning device vjps are not compatible with jax jaocbians")

        if diff_method == "device" and not grad_on_execution and device_vjp:
            pytest.xfail(reason="various runtime-related errors")

        if diff_method == "adjoint" and device_vjp and jacobian is jax.jacfwd:
            pytest.xfail(reason="TypeError applying forward-mode autodiff.")

        a = jax.numpy.array(0.1)
        b = jax.numpy.array(0.2)

        @qnode(
            get_device(dev_name, wires=1, seed=seed),
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(a, b, c):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            qml.RZ(c, wires=0)
            return qml.expval(qml.PauliZ(0))

        res = jax.jit(circuit)(a, b, 0.0)
        expected_res = np.cos(a) * np.cos(b)
        assert np.allclose(res, expected_res, atol=tol, rtol=0)

        g = jax.jit(jacobian(circuit, argnums=[0, 1]))(a, b, 0.0)
        expected_g = [-np.sin(a) * np.cos(b), -np.cos(a) * np.sin(b)]
        assert np.allclose(g, expected_g, atol=tol, rtol=0)

    def test_gradient_scalar_cost_vector_valued_qnode(
        self, dev_name, diff_method, grad_on_execution, device_vjp, jacobian, tol, interface, seed
    ):
        """Test derivative calculation of a scalar valued cost function that
        uses the output of a vector-valued QNode"""

        gradient_kwargs = {}

        if dev_name == "param_shift.qubit":
            pytest.xfail("gradient transforms have a different vjp shape convention.")

        if "lightning" in dev_name:
            pytest.xfail("lightning device vjps are not compatible with jax jaocbians")

        elif jacobian == jax.jacfwd and device_vjp:
            pytest.skip("device vjps are not compatible with forward differentiation.")

        elif diff_method == "spsa":
            gradient_kwargs = {"h": H_FOR_SPSA, "sampler_rng": np.random.default_rng(seed)}
            tol = TOL_FOR_SPSA

        @qnode(
            get_device(dev_name, wires=2, seed=seed),
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
            gradient_kwargs=gradient_kwargs,
        )
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[1])

        def cost(x, y, idx):
            res = circuit(x, y)
            return res[idx]  # pylint:disable=unsubscriptable-object

        x = jax.numpy.array(1.0)
        y = jax.numpy.array(2.0)
        expected_g = (
            np.array([-np.sin(x) * np.cos(y) / 2, np.cos(y) * np.sin(x) / 2]),
            np.array([-np.cos(x) * np.sin(y) / 2, np.cos(x) * np.sin(y) / 2]),
        )

        idx = 0
        g0 = jax.jit(jacobian(cost, argnums=0))(x, y, idx)
        g1 = jax.jit(jacobian(cost, argnums=1))(x, y, idx)
        assert np.allclose(g0, expected_g[0][idx], atol=tol, rtol=0)
        assert np.allclose(g1, expected_g[1][idx], atol=tol, rtol=0)

        idx = 1
        g0 = jax.jit(jacobian(cost, argnums=0))(x, y, idx)
        g1 = jax.jit(jacobian(cost, argnums=1))(x, y, idx)

        assert np.allclose(g0, expected_g[0][idx], atol=tol, rtol=0)
        assert np.allclose(g1, expected_g[1][idx], atol=tol, rtol=0)

    # pylint: disable=unused-argument
    def test_matrix_parameter(
        self, dev_name, diff_method, grad_on_execution, device_vjp, jacobian, tol, interface, seed
    ):
        """Test that the JAX-JIT interface works correctly with a matrix parameter"""

        if dev_name == "param_shift.qubit":
            pytest.xfail("gradient transforms have a different vjp shape convention.")

        if "lightning" in dev_name:
            pytest.xfail("lightning device vjps are not compatible with jax jaocbians")

        if jacobian == jax.jacfwd and device_vjp:
            pytest.skip("device vjps are not compatible with forward differentiation.")

        # pylint: disable=unused-argument
        @qml.qnode(
            get_device(dev_name, wires=1, seed=seed),
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circ(p, U):
            qml.QubitUnitary(U, wires=0)
            qml.RY(p, wires=0)
            return qml.expval(qml.PauliZ(0))

        p = jax.numpy.array(0.1)
        U = jax.numpy.array([[0, 1], [1, 0]])
        res = jax.jit(circ)(p, U)
        assert np.allclose(res, -np.cos(p), atol=tol, rtol=0)

        jac_fn = jax.jit(jacobian(circ, argnums=0))
        res = jac_fn(p, U)
        assert np.allclose(res, np.sin(p), atol=tol, rtol=0)


@pytest.mark.parametrize("shots", [None, 10000])
@pytest.mark.parametrize("jacobian", jacobian_fn)
@pytest.mark.parametrize("interface", ["auto", "jax-jit"])
@pytest.mark.parametrize(
    "dev_name,diff_method,grad_on_execution, device_vjp",
    device_test_cases,
)
class TestReturn:
    """Class to test the shape of the Grad/Jacobian with different return types."""

    def test_grad_single_measurement_param(
        self, dev_name, diff_method, grad_on_execution, device_vjp, jacobian, shots, interface, seed
    ):
        """For one measurement and one param, the gradient is a float."""

        if dev_name == "param_shift.qubit":
            pytest.xfail("gradient transforms have a different vjp shape convention")

        if "lightning" in dev_name:
            pytest.xfail("lightning device vjps are not compatible with jax jaocbians")

        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        if jacobian == jax.jacfwd and device_vjp:
            pytest.skip("jacfwd is not compatible with device_vjp=True.")

        @qml.set_shots(shots=shots)
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

        grad = jax.jit(jacobian(circuit))(a)

        assert isinstance(grad, jax.numpy.ndarray)
        assert grad.shape == ()

    def test_grad_single_measurement_multiple_param(
        self, dev_name, diff_method, grad_on_execution, device_vjp, jacobian, shots, interface, seed
    ):
        """For one measurement and multiple param, the gradient is a tuple of arrays."""

        if dev_name == "param_shift.qubit":
            pytest.xfail("gradient transforms have a different vjp shape convention.")

        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        if jacobian == jax.jacfwd and device_vjp:
            pytest.skip("jacfwd is not compatible with device_vjp=True.")

        if "lightning" in dev_name:
            pytest.xfail("lightning device vjps are not compatible with jax jaocbians")

        @qml.set_shots(shots=shots)
        @qnode(
            get_device(dev_name, wires=1, seed=seed),
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

        grad = jax.jit(jacobian(circuit, argnums=[0, 1]))(a, b)

        assert isinstance(grad, tuple)
        assert len(grad) == 2
        assert grad[0].shape == ()
        assert grad[1].shape == ()

    def test_grad_single_measurement_multiple_param_array(
        self, dev_name, diff_method, grad_on_execution, device_vjp, jacobian, shots, interface, seed
    ):
        """For one measurement and multiple param as a single array params, the gradient is an array."""

        if dev_name == "param_shift.qubit":
            pytest.xfail("gradient transforms have a different vjp shape convention.")

        if "lightning" in dev_name:
            pytest.xfail("lightning device vjps are not compatible with jax jaocbians")

        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        if jacobian == jax.jacfwd and device_vjp:
            pytest.skip("jacfwd is not compatible with device_vjp=True.")

        @qml.set_shots(shots=shots)
        @qnode(
            get_device(dev_name, wires=1, seed=seed),
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

        grad = jax.jit(jacobian(circuit))(a)

        assert isinstance(grad, jax.numpy.ndarray)
        assert grad.shape == (2,)

    def test_jacobian_single_measurement_param_probs(
        self, dev_name, diff_method, grad_on_execution, device_vjp, jacobian, shots, interface, seed
    ):
        """For a multi-dimensional measurement (probs), check that a single array is returned
        with the correct dimension"""

        if dev_name == "param_shift.qubit":
            pytest.xfail("gradient transforms have a different vjp shape convention.")

        if "lightning" in dev_name:
            pytest.xfail("lightning device vjps are not compatible with jax jaocbians")

        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        if jacobian == jax.jacfwd and device_vjp:
            pytest.skip("jacfwd is not compatible with device_vjp=True.")

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

        jac = jax.jit(jacobian(circuit))(a)

        assert isinstance(jac, jax.numpy.ndarray)
        assert jac.shape == (4,)

    def test_jacobian_single_measurement_probs_multiple_param(
        self, dev_name, diff_method, grad_on_execution, device_vjp, jacobian, shots, interface, seed
    ):
        """For a multi-dimensional measurement (probs), check that a single tuple is returned
        containing arrays with the correct dimension"""

        if dev_name == "param_shift.qubit":
            pytest.xfail("gradient transforms have a different vjp shape convention.")

        if "lightning" in dev_name:
            pytest.xfail("lightning device vjps are not compatible with jax jaocbians")

        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        if jacobian == jax.jacfwd and device_vjp:
            pytest.skip("jacfwd is not compatible with device_vjp=True.")

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

        jac = jax.jit(jacobian(circuit, argnums=[0, 1]))(a, b)

        assert isinstance(jac, tuple)

        assert isinstance(jac[0], jax.numpy.ndarray)
        assert jac[0].shape == (4,)

        assert isinstance(jac[1], jax.numpy.ndarray)
        assert jac[1].shape == (4,)

    def test_jacobian_single_measurement_probs_multiple_param_single_array(
        self, dev_name, diff_method, grad_on_execution, device_vjp, jacobian, shots, interface, seed
    ):
        """For a multi-dimensional measurement (probs), check that a single tuple is returned
        containing arrays with the correct dimension"""

        if dev_name == "param_shift.qubit":
            pytest.xfail("gradient transforms have a different vjp shape convention.")

        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        if "lightning" in dev_name:
            pytest.xfail("lightning device vjps are not compatible with jax jaocbians")

        if jacobian == jax.jacfwd and device_vjp:
            pytest.skip("jacfwd is not compatible with device_vjp=True.")

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
        jac = jax.jit(jacobian(circuit))(a)

        assert isinstance(jac, jax.numpy.ndarray)
        assert jac.shape == (4, 2)

    def test_jacobian_expval_expval_multiple_params(
        self, dev_name, diff_method, grad_on_execution, device_vjp, jacobian, shots, interface, seed
    ):
        """The jacobian of multiple measurements with multiple params return a tuple of arrays."""

        if dev_name == "param_shift.qubit":
            pytest.xfail("gradient transforms have a different vjp shape convention.")

        if "lightning" in dev_name:
            pytest.xfail("lightning device vjps are not compatible with jax jaocbians")

        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        if jacobian == jax.jacfwd and device_vjp:
            pytest.skip("jacfwd is not compatible with device_vjp=True.")

        par_0 = jax.numpy.array(0.1)
        par_1 = jax.numpy.array(0.2)

        @qml.set_shots(shots=shots)
        @qnode(
            get_device(dev_name, wires=2, seed=seed),
            interface=interface,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1)), qml.expval(qml.PauliZ(0))

        jac = jax.jit(jacobian(circuit, argnums=[0, 1]))(par_0, par_1)

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

    def test_jacobian_expval_expval_multiple_params_array(
        self, dev_name, diff_method, grad_on_execution, device_vjp, jacobian, shots, interface, seed
    ):
        """The jacobian of multiple measurements with a multiple params array return a single array."""

        if dev_name == "param_shift.qubit":
            pytest.xfail("gradient transforms have a different vjp shape convention.")

        if "lightning" in dev_name:
            pytest.xfail("lightning device vjps are not compatible with jax jaocbians")

        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        if jacobian == jax.jacfwd and device_vjp:
            pytest.skip("jacfwd is not compatible with device_vjp=True.")

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

        jac = jax.jit(jacobian(circuit))(a)

        assert isinstance(jac, tuple)
        assert len(jac) == 2  # measurements

        assert isinstance(jac[0], jax.numpy.ndarray)
        assert jac[0].shape == (2,)

        assert isinstance(jac[1], jax.numpy.ndarray)
        assert jac[1].shape == (2,)

    def test_jacobian_var_var_multiple_params(
        self, dev_name, diff_method, grad_on_execution, device_vjp, jacobian, shots, interface, seed
    ):
        """The jacobian of multiple measurements with multiple params return a tuple of arrays."""

        if dev_name == "param_shift.qubit":
            pytest.xfail("gradient transforms have a different vjp shape convention.")

        elif diff_method == "hadamard":
            pytest.skip("Test does not supports hadamard because of var.")

        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        if jacobian == jax.jacfwd and device_vjp:
            pytest.skip("jacfwd is not compatible with device_vjp=True.")

        if diff_method == "adjoint":
            pytest.skip("adjoint supports either all expvals or only diagonal measurements")

        par_0 = jax.numpy.array(0.1)
        par_1 = jax.numpy.array(0.2)

        @qml.set_shots(shots=shots)
        @qnode(
            get_device(dev_name, wires=2, seed=seed),
            interface=interface,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.PauliZ(0) @ qml.PauliX(1)), qml.var(qml.PauliZ(0))

        jac = jax.jit(jacobian(circuit, argnums=[0, 1]))(par_0, par_1)

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

    def test_jacobian_var_var_multiple_params_array(
        self, dev_name, diff_method, grad_on_execution, device_vjp, jacobian, shots, interface, seed
    ):
        """The jacobian of multiple measurements with a multiple params array return a single array."""

        if dev_name == "param_shift.qubit":
            pytest.xfail("gradient transforms have a different vjp shape convention.")

        elif diff_method == "hadamard":
            pytest.skip("Test does not supports hadamard because of var.")

        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        if jacobian == jax.jacfwd and device_vjp:
            pytest.skip("jacfwd is not compatible with device_vjp=True.")

        if diff_method == "adjoint":
            pytest.skip("adjoint supports either all expvals or only diagonal measurements")

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

        jac = jax.jit(jacobian(circuit))(a)

        assert isinstance(jac, tuple)
        assert len(jac) == 2  # measurements

        assert isinstance(jac[0], jax.numpy.ndarray)
        assert jac[0].shape == (2,)

        assert isinstance(jac[1], jax.numpy.ndarray)
        assert jac[1].shape == (2,)

    def test_jacobian_multiple_measurement_single_param(
        self, dev_name, diff_method, grad_on_execution, device_vjp, jacobian, shots, interface, seed
    ):
        """The jacobian of multiple measurements with a single params return an array."""

        if dev_name == "param_shift.qubit":
            pytest.xfail("gradient transforms have a different vjp shape convention")

        if "lightning" in dev_name:
            pytest.xfail("lightning device vjps are not compatible with jax jaocbians")

        if device_vjp and jacobian == jax.jacfwd:
            pytest.skip("device vjp not compatible with forward differentiation.")

        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        @qml.set_shots(shots=shots)
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
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        a = jax.numpy.array(0.1)

        jac = jax.jit(jacobian(circuit))(a)

        assert isinstance(jac, tuple)
        assert len(jac) == 2

        assert isinstance(jac[0], jax.numpy.ndarray)
        assert jac[0].shape == ()

        assert isinstance(jac[1], jax.numpy.ndarray)
        assert jac[1].shape == (4,)

    def test_jacobian_multiple_measurement_multiple_param(
        self, dev_name, diff_method, grad_on_execution, device_vjp, jacobian, shots, interface, seed
    ):
        """The jacobian of multiple measurements with a multiple params return a tuple of arrays."""

        if dev_name == "param_shift.qubit":
            pytest.xfail("gradient transforms have a different vjp shape convention.")

        if "lightning" in dev_name:
            pytest.xfail("lightning device vjps are not compatible with jax jaocbians")

        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        if jacobian == jax.jacfwd and device_vjp:
            pytest.skip("jacfwd is not compatible with device_vjp=True.")

        @qml.set_shots(shots=shots)
        @qnode(
            get_device(dev_name, wires=1, seed=seed),
            interface=interface,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        a = np.array(0.1, requires_grad=True)
        b = np.array(0.2, requires_grad=True)

        jac = jax.jit(jacobian(circuit, argnums=[0, 1]))(a, b)

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

    def test_jacobian_multiple_measurement_multiple_param_array(
        self, dev_name, diff_method, grad_on_execution, device_vjp, jacobian, shots, interface, seed
    ):
        """The jacobian of multiple measurements with a multiple params array return a single array."""

        if dev_name == "param_shift.qubit":
            pytest.xfail("gradient transforms have a different vjp shape convention.")

        if "lightning" in dev_name:
            pytest.xfail("lightning device vjps are not compatible with jax jaocbians")

        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        if jacobian == jax.jacfwd and device_vjp:
            pytest.skip("jacfwd is not compatible with device_vjp=True.")

        @qml.set_shots(shots=shots)
        @qnode(
            get_device(dev_name, wires=1, seed=seed),
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

        jac = jax.jit(jacobian(circuit))(a)

        assert isinstance(jac, tuple)
        assert len(jac) == 2  # measurements

        assert isinstance(jac[0], jax.numpy.ndarray)
        assert jac[0].shape == (2,)

        assert isinstance(jac[1], jax.numpy.ndarray)
        assert jac[1].shape == (4, 2)


hessian_fn = [
    jax.hessian,
    lambda fn, argnums=0: jax.jacrev(jax.jacfwd(fn, argnums=argnums), argnums=argnums),
    lambda fn, argnums=0: jax.jacfwd(jax.jacrev(fn, argnums=argnums), argnums=argnums),
]


@pytest.mark.parametrize("hessian", hessian_fn)
@pytest.mark.parametrize("interface", ["auto", "jax-jit"])
@pytest.mark.parametrize(
    "dev_name,diff_method,grad_on_execution, device_vjp",
    device_test_cases,
)
class TestReturnHessian:
    """Class to test the shape of the Hessian with different return types."""

    def test_hessian_expval_multiple_params(
        self, dev_name, diff_method, hessian, device_vjp, grad_on_execution, interface, seed
    ):
        """The hessian of single a measurement with multiple params return a tuple of arrays."""
        if diff_method in {"adjoint", "device"}:
            pytest.skip("Test does not supports adjoint because second order diff.")

        par_0 = jax.numpy.array(0.1)
        par_1 = jax.numpy.array(0.2)

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

        hess = jax.jit(hessian(circuit, argnums=[0, 1]))(par_0, par_1)

        assert isinstance(hess, tuple)
        assert len(hess) == 2

        assert isinstance(hess[0], tuple)
        assert len(hess[0]) == 2
        assert isinstance(hess[0][0], jax.numpy.ndarray)
        assert isinstance(hess[0][1], jax.numpy.ndarray)
        assert hess[0][0].shape == ()
        assert hess[0][1].shape == ()

        assert isinstance(hess[1], tuple)
        assert len(hess[1]) == 2
        assert isinstance(hess[1][0], jax.numpy.ndarray)
        assert isinstance(hess[1][1], jax.numpy.ndarray)
        assert hess[1][0].shape == ()
        assert hess[1][1].shape == ()

    def test_hessian_expval_multiple_param_array(
        self, dev_name, diff_method, hessian, grad_on_execution, device_vjp, interface, seed
    ):
        """The hessian of single measurement with a multiple params array return a single array."""

        if diff_method in {"adjoint", "device"}:
            pytest.skip("Test does not supports adjoint because second order diff.")

        params = jax.numpy.array([0.1, 0.2], dtype=jax.numpy.float64)

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

        hess = jax.jit(hessian(circuit))(params)

        assert isinstance(hess, jax.numpy.ndarray)
        assert hess.shape == (2, 2)

    def test_hessian_var_multiple_params(
        self, dev_name, diff_method, hessian, device_vjp, grad_on_execution, interface, seed
    ):
        """The hessian of single a measurement with multiple params return a tuple of arrays."""
        if diff_method in {"adjoint", "device"}:
            pytest.skip("Test does not supports adjoint because second order diff.")
        elif diff_method == "hadamard":
            pytest.skip("Test does not supports hadamard because of var.")

        par_0 = jax.numpy.array(0.1, dtype=jax.numpy.float64)
        par_1 = jax.numpy.array(0.2, dtype=jax.numpy.float64)

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

        hess = jax.jit(hessian(circuit, argnums=[0, 1]))(par_0, par_1)

        assert isinstance(hess, tuple)
        assert len(hess) == 2

        assert isinstance(hess[0], tuple)
        assert len(hess[0]) == 2
        assert isinstance(hess[0][0], jax.numpy.ndarray)
        assert isinstance(hess[0][1], jax.numpy.ndarray)
        assert hess[0][0].shape == ()
        assert hess[0][1].shape == ()

        assert isinstance(hess[1], tuple)
        assert len(hess[1]) == 2
        assert isinstance(hess[1][0], jax.numpy.ndarray)
        assert isinstance(hess[1][1], jax.numpy.ndarray)
        assert hess[1][0].shape == ()
        assert hess[1][1].shape == ()

    def test_hessian_var_multiple_param_array(
        self, dev_name, diff_method, hessian, grad_on_execution, device_vjp, interface, seed
    ):
        """The hessian of single measurement with a multiple params array return a single array."""

        if diff_method in {"adjoint", "device"}:
            pytest.skip("Test does not supports adjoint because second order diff.")

        elif diff_method == "hadamard":
            pytest.skip("Test does not supports hadamard because of var.")

        params = jax.numpy.array([0.1, 0.2], dtype=jax.numpy.float64)

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

        hess = jax.jit(hessian(circuit))(params)

        assert isinstance(hess, jax.numpy.ndarray)
        assert hess.shape == (2, 2)

    def test_hessian_probs_expval_multiple_params(
        self, dev_name, diff_method, hessian, grad_on_execution, device_vjp, interface, seed
    ):
        """The hessian of multiple measurements with multiple params return a tuple of arrays."""

        if diff_method in {"adjoint", "device"}:
            pytest.skip("Test does not supports adjoint because second order diff.")

        elif diff_method == "hadamard":
            pytest.skip("Test does not supports hadamard because of non commuting obs.")

        par_0 = jax.numpy.array(0.1, dtype=jax.numpy.float64)
        par_1 = jax.numpy.array(0.2, dtype=jax.numpy.float64)

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

        hess = jax.jit(hessian(circuit, argnums=[0, 1]))(par_0, par_1)

        assert isinstance(hess, tuple)
        assert len(hess) == 2

        assert isinstance(hess[0], tuple)
        assert len(hess[0]) == 2

        for h in hess[0]:
            assert isinstance(h, tuple)
            for h_comp in h:
                assert h_comp.shape == ()

        for h in hess[1]:
            assert isinstance(h, tuple)
            for h_comp in h:
                assert h_comp.shape == (2,)

    def test_hessian_probs_expval_multiple_param_array(
        self, dev_name, diff_method, hessian, grad_on_execution, device_vjp, interface, seed
    ):
        """The hessian of multiple measurements with a multiple param array return a single array."""

        if diff_method in {"adjoint", "device"}:
            pytest.skip("Test does not supports adjoint because second order diff.")
        elif diff_method == "hadamard":
            pytest.skip("Test does not supports hadamard because of non commuting obs.")

        params = jax.numpy.array([0.1, 0.2], dtype=jax.numpy.float64)

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
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1)), qml.probs(wires=[0])

        hess = jax.jit(hessian(circuit))(params)

        assert isinstance(hess, tuple)
        assert len(hess) == 2
        assert isinstance(hess[0], jax.numpy.ndarray)
        assert hess[0].shape == (2, 2)

        assert isinstance(hess[1], jax.numpy.ndarray)
        assert hess[1].shape == (2, 2, 2)

    def test_hessian_probs_var_multiple_params(
        self, dev_name, diff_method, hessian, grad_on_execution, device_vjp, interface, seed
    ):
        """The hessian of multiple measurements with multiple params return a tuple of arrays."""
        if diff_method in {"adjoint", "device"}:
            pytest.skip("Test does not supports adjoint because second order diff.")
        elif diff_method == "hadamard":
            pytest.skip("Test does not supports hadamard because of var.")

        par_0 = jax.numpy.array(0.1, dtype=jax.numpy.float64)
        par_1 = jax.numpy.array(0.2, dtype=jax.numpy.float64)

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

        hess = jax.jit(hessian(circuit, argnums=[0, 1]))(par_0, par_1)

        assert isinstance(hess, tuple)
        assert len(hess) == 2

        assert isinstance(hess[0], tuple)
        assert len(hess[0]) == 2

        for h in hess[0]:
            assert isinstance(h, tuple)
            for h_comp in h:
                assert h_comp.shape == ()

        for h in hess[1]:
            assert isinstance(h, tuple)
            for h_comp in h:
                assert h_comp.shape == (2,)

    def test_hessian_probs_var_multiple_param_array(
        self, dev_name, diff_method, hessian, grad_on_execution, device_vjp, interface, seed
    ):
        """The hessian of multiple measurements with a multiple param array return a single array."""
        if diff_method in {"adjoint", "device"}:
            pytest.skip("Test does not supports adjoint because second order diff.")
        elif diff_method == "hadamard":
            pytest.skip("Test does not supports hadamard because of var.")

        params = jax.numpy.array([0.1, 0.2], dtype=jax.numpy.float64)

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

        hess = jax.jit(hessian(circuit))(params)

        assert isinstance(hess, tuple)
        assert len(hess) == 2
        assert isinstance(hess[0], jax.numpy.ndarray)
        assert hess[0].shape == (2, 2)

        assert isinstance(hess[1], jax.numpy.ndarray)
        assert hess[1].shape == (2, 2, 2)


@pytest.mark.parametrize("hessian", hessian_fn)
@pytest.mark.parametrize("diff_method", ["parameter-shift", "hadamard"])
def test_jax_device_hessian_shots(hessian, diff_method):
    """The hessian of multiple measurements with a multiple param array return a single array."""

    @jax.jit
    @qml.set_shots(shots=10000)
    @qml.qnode(DefaultQubit(), diff_method=diff_method, max_diff=2)
    def circuit(x):
        qml.RY(x[0], wires=0)
        qml.RX(x[1], wires=0)
        return qml.expval(qml.PauliZ(0))

    x = jax.numpy.array([1.0, 2.0])
    a, b = x

    hess = jax.jit(hessian(circuit))(x)

    expected_hess = [
        [-np.cos(a) * np.cos(b), np.sin(a) * np.sin(b)],
        [np.sin(a) * np.sin(b), -np.cos(a) * np.cos(b)],
    ]
    shots_tol = 0.1
    assert np.allclose(hess, expected_hess, atol=shots_tol, rtol=0)


@pytest.mark.parametrize("jit_inside", [True, False])
@pytest.mark.parametrize("interface", ["auto", "jax-jit"])
@pytest.mark.parametrize("argnums", [0, 1, [0, 1]])
@pytest.mark.parametrize("jacobian", jacobian_fn)
@pytest.mark.parametrize(
    "dev_name,diff_method,grad_on_execution, device_vjp",
    device_test_cases,
)
class TestSubsetArgnums:
    def test_single_measurement(
        self,
        interface,
        dev_name,
        diff_method,
        grad_on_execution,
        device_vjp,
        jacobian,
        argnums,
        jit_inside,
        tol,
        seed,
    ):
        """Test single measurement with different diff methods with argnums."""
        kwargs = {}
        if jacobian == jax.jacfwd and device_vjp:
            pytest.skip("jacfwd is not compatible with device_vjp=True.")
        if "lightning" in dev_name:
            pytest.xfail("lightning device vjps are not compatible with jax jaocbians")
        if dev_name == "param_shift.qubit":
            pytest.xfail("gradient transform have a different vjp shape convention.")
        if diff_method == "spsa":
            kwargs["sampler_rng"] = np.random.default_rng(seed)
            tol = TOL_FOR_SPSA

        @qml.qnode(
            get_device(dev_name, wires=2, seed=seed),
            interface=interface,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            cache=False,
            device_vjp=device_vjp,
            gradient_kwargs=kwargs,
        )
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        a = jax.numpy.array(1.0)
        b = jax.numpy.array(2.0)

        if jit_inside:
            jac = jacobian(jax.jit(circuit), argnums=argnums)(a, b)
        else:
            jac = jax.jit(jacobian(circuit, argnums=argnums))(a, b)

        expected = np.array([-np.sin(a), 0])

        if argnums == 0:
            assert np.allclose(jac, expected[0], atol=tol)
        elif argnums == 1:
            assert np.allclose(jac, expected[1], atol=tol)
        else:
            assert np.allclose(jac[0], expected[0], atol=tol)
            assert np.allclose(jac[1], expected[1], atol=tol)

    def test_multi_measurements(
        self,
        interface,
        dev_name,
        diff_method,
        grad_on_execution,
        device_vjp,
        jacobian,
        argnums,
        jit_inside,
        tol,
        seed,
    ):
        """Test multiple measurements with different diff methods with argnums."""

        if jacobian == jax.jacfwd and device_vjp:
            pytest.skip("jacfwd is not compatible with device_vjp=True.")

        if "lightning" in dev_name:
            pytest.xfail("lightning device vjps are not compatible with jax jaocbians")

        if dev_name == "param_shift.qubit":
            pytest.xfail("gradient transform have a different vjp shape convention.")

        kwargs = {}
        if diff_method == "spsa":
            kwargs["sampler_rng"] = np.random.default_rng(seed)
            tol = TOL_FOR_SPSA

        @qml.qnode(
            get_device(dev_name, wires=2, seed=seed),
            interface=interface,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
            gradient_kwargs=kwargs,
        )
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))

        a = jax.numpy.array(1.0)
        b = jax.numpy.array(2.0)

        if jit_inside:
            jac = jacobian(jax.jit(circuit), argnums=argnums)(a, b)
        else:
            jac = jax.jit(jacobian(circuit, argnums=argnums))(a, b)

        expected = np.array([[-np.sin(a), 0], [np.sin(a) * np.sin(b), -np.cos(a) * np.cos(b)]])

        if argnums == 0:
            assert np.allclose(jac, expected.T[0], atol=tol)
        elif argnums == 1:
            assert np.allclose(jac, expected.T[1], atol=tol)
        else:
            assert np.allclose(jac[0], expected[0], atol=tol)
            assert np.allclose(jac[1], expected[1], atol=tol)


class TestSinglePrecision:
    """Tests for compatibility with single precision mode."""

    # pylint: disable=import-outside-toplevel
    def test_type_conversion_fallback(self):
        """Test that if the type isn't int, float, or complex, we still have a fallback."""
        from pennylane.workflow.interfaces.jax_jit import _jax_dtype

        assert _jax_dtype(bool) == jax.numpy.dtype(bool)

    @pytest.mark.parametrize("diff_method", ("adjoint", "parameter-shift"))
    def test_float32_return(self, diff_method):
        """Test that jax jit works when float64 mode is disabled."""
        jax.config.update("jax_enable_x64", False)

        try:

            @jax.jit
            @qml.qnode(qml.device("default.qubit"), diff_method=diff_method)
            def circuit(x):
                qml.RX(x, wires=0)
                return qml.expval(qml.PauliZ(0))

            grad = jax.grad(circuit)(jax.numpy.array(0.1))
            assert qml.math.allclose(grad, -np.sin(0.1))
        finally:
            jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_enable_x64", True)

    @pytest.mark.parametrize("diff_method", ("adjoint", "finite-diff"))
    def test_complex64_return(self, diff_method):
        """Test that jax jit works with differentiating the state."""
        jax.config.update("jax_enable_x64", False)

        try:
            # finite diff with float32 ...
            tol = 5e-2 if diff_method == "finite-diff" else 1e-6

            @jax.jit
            @qml.qnode(qml.device("default.qubit", wires=1), diff_method=diff_method)
            def circuit(x):
                qml.RX(x, wires=0)
                return qml.state()

            j = jax.jacobian(circuit, holomorphic=True)(jax.numpy.array(0.1 + 0j))
            assert qml.math.allclose(j, [-np.sin(0.05) / 2, -np.cos(0.05) / 2 * 1j], atol=tol)

        finally:
            jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_enable_x64", True)

    def test_int32_return(self):
        """Test that jax jit forward execution works with samples and int32"""

        jax.config.update("jax_enable_x64", False)

        try:

            @jax.jit
            @qml.qnode(qml.device("default.qubit"), diff_method=qml.gradients.param_shift, shots=10)
            def circuit(x):
                qml.RX(x, wires=0)
                return qml.sample(wires=0)

            _ = circuit(jax.numpy.array(0.1))
        finally:
            jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_enable_x64", True)


def test_no_inputs_jitting():
    """Test that if we jit a qnode with no inputs, we can still detect the jitting and proper interface."""

    @jax.jit
    @qml.qnode(qml.device("reference.qubit", wires=1))
    def circuit():
        qml.StatePrep(jax.numpy.array([1, 0]), 0)
        return qml.state()

    res = circuit()
    assert qml.math.allclose(res, jax.numpy.array([1, 0]))

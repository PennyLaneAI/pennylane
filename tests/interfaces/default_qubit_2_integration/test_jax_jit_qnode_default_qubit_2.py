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
# pylint: disable=too-many-arguments,too-few-public-methods
from functools import partial
import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane import qnode
from pennylane.devices.experimental import DefaultQubit2

qubit_device_and_diff_method = [
    [DefaultQubit2(), "backprop", True],
    [DefaultQubit2(), "finite-diff", False],
    [DefaultQubit2(), "parameter-shift", False],
    [DefaultQubit2(), "adjoint", True],
    [DefaultQubit2(), "adjoint", False],
    [DefaultQubit2(), "spsa", False],
    [DefaultQubit2(), "hadamard", False],
]
interface_and_qubit_device_and_diff_method = [
    ["auto"] + inner_list for inner_list in qubit_device_and_diff_method
] + [["jax-jit"] + inner_list for inner_list in qubit_device_and_diff_method]

pytestmark = pytest.mark.jax

jax = pytest.importorskip("jax")
config = pytest.importorskip("jax.config")
config.config.update("jax_enable_x64", True)

TOL_FOR_SPSA = 1.0
SEED_FOR_SPSA = 32651
H_FOR_SPSA = 0.05


@pytest.mark.parametrize(
    "interface,dev,diff_method,grad_on_execution", interface_and_qubit_device_and_diff_method
)
class TestQNode:
    """Test that using the QNode with JAX integrates with the PennyLane
    stack"""

    def test_execution_with_interface(self, dev, diff_method, grad_on_execution, interface):
        """Test execution works with the interface"""
        if diff_method == "backprop":
            pytest.skip("Test does not support backprop")

        @qnode(
            dev, interface=interface, diff_method=diff_method, grad_on_execution=grad_on_execution
        )
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return qml.expval(qml.PauliZ(0))

        a = np.array(0.1, requires_grad=True)
        jax.jit(circuit)(a)

        assert circuit.interface == interface

        # the tape is able to deduce trainable parameters
        assert circuit.qtape.trainable_params == [0]

        # gradients should work
        grad = jax.jit(jax.grad(circuit))(a)
        assert isinstance(grad, jax.Array)
        assert grad.shape == ()

    def test_changing_trainability(
        self, dev, diff_method, grad_on_execution, interface, mocker, tol
    ):
        """Test changing the trainability of parameters changes the
        number of differentiation requests made"""
        if diff_method != "parameter-shift":
            pytest.skip("Test only supports parameter-shift")

        a = jax.numpy.array(0.1)
        b = jax.numpy.array(0.2)

        @qnode(
            dev,
            interface=interface,
            diff_method="parameter-shift",
            grad_on_execution=grad_on_execution,
        )
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Hamiltonian([1, 1], [qml.PauliZ(0), qml.PauliY(1)]))

        grad_fn = jax.jit(jax.grad(circuit, argnums=[0, 1]))
        spy = mocker.spy(qml.gradients.param_shift, "transform_fn")
        res = grad_fn(a, b)

        # the tape has reported both arguments as trainable
        assert circuit.qtape.trainable_params == [0, 1]

        expected = [-np.sin(a) + np.sin(a) * np.sin(b), -np.cos(a) * np.cos(b)]
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # The parameter-shift rule has been called for each argument
        assert len(spy.spy_return[0]) == 4

        # make the second QNode argument a constant
        grad_fn = jax.grad(circuit, argnums=0)
        res = grad_fn(a, b)

        # the tape has reported only the first argument as trainable
        assert circuit.qtape.trainable_params == [0]

        expected = [-np.sin(a) + np.sin(a) * np.sin(b)]
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # The parameter-shift rule has been called only once
        assert len(spy.spy_return[0]) == 2

        # trainability also updates on evaluation
        a = np.array(0.54, requires_grad=False)
        b = np.array(0.8, requires_grad=True)
        circuit(a, b)
        assert circuit.qtape.trainable_params == [1]

    def test_classical_processing(self, dev, diff_method, grad_on_execution, interface):
        """Test classical processing within the quantum tape"""
        a = jax.numpy.array(0.1)
        b = jax.numpy.array(0.2)
        c = jax.numpy.array(0.3)

        @qnode(
            dev, diff_method=diff_method, interface=interface, grad_on_execution=grad_on_execution
        )
        def circuit(a, b, c):
            qml.RY(a * c, wires=0)
            qml.RZ(b, wires=0)
            qml.RX(c + c**2 + jax.numpy.sin(a), wires=0)
            return qml.expval(qml.PauliZ(0))

        res = jax.grad(circuit, argnums=[0, 2])(a, b, c)

        if diff_method == "finite-diff":
            assert circuit.qtape.trainable_params == [0, 2]

        assert len(res) == 2

    def test_matrix_parameter(self, dev, diff_method, grad_on_execution, interface, tol):
        """Test that the jax interface works correctly
        with a matrix parameter"""
        U = jax.numpy.array([[0, 1], [1, 0]])
        a = jax.numpy.array(0.1)

        @qnode(
            dev, diff_method=diff_method, interface=interface, grad_on_execution=grad_on_execution
        )
        def circuit(U, a):
            qml.QubitUnitary(U, wires=0)
            qml.RY(a, wires=0)
            return qml.expval(qml.PauliZ(0))

        res = jax.grad(circuit, argnums=1)(U, a)
        assert np.allclose(res, np.sin(a), atol=tol, rtol=0)

        if diff_method == "finite-diff":
            assert circuit.qtape.trainable_params == [1]

    def test_differentiable_expand(self, dev, diff_method, grad_on_execution, interface, tol):
        """Test that operation and nested tape expansion
        is differentiable"""

        gradient_kwargs = {}
        if diff_method == "spsa":
            gradient_kwargs["sampler_rng"] = np.random.default_rng(SEED_FOR_SPSA)
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
            dev,
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            **gradient_kwargs
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

    def test_jacobian_options(self, dev, diff_method, grad_on_execution, interface, mocker):
        """Test setting jacobian options"""
        if diff_method != "finite-diff":
            pytest.skip("Test only applies to finite diff.")

        spy = mocker.spy(qml.gradients.finite_diff, "transform_fn")

        a = np.array([0.1, 0.2], requires_grad=True)

        @qnode(
            dev,
            interface=interface,
            diff_method="finite-diff",
            h=1e-8,
            approx_order=2,
            grad_on_execution=grad_on_execution,
        )
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        if diff_method in {"finite-diff", "parameter-shift", "spsa"} and interface == "jax-jit":
            # No jax.jacobian support for call
            pytest.xfail(reason="batching rules are implemented only for id_tap, not for call.")

        jax.jit(jax.jacobian(circuit))(a)

        for args in spy.call_args_list:
            assert args[1]["approx_order"] == 2
            assert args[1]["h"] == 1e-8


@pytest.mark.parametrize(
    "interface,dev,diff_method,grad_on_execution", interface_and_qubit_device_and_diff_method
)
class TestVectorValuedQNode:
    """Test that using vector-valued QNodes with JAX integrate with the
    PennyLane stack"""

    def test_diff_expval_expval(self, dev, diff_method, grad_on_execution, interface, mocker, tol):
        """Test jacobian calculation"""

        gradient_kwargs = {}
        if diff_method == "parameter-shift":
            spy = mocker.spy(qml.gradients.param_shift, "transform_fn")
        elif diff_method == "finite-diff":
            spy = mocker.spy(qml.gradients.finite_diff, "transform_fn")
        elif diff_method == "spsa":
            spy = mocker.spy(qml.gradients.spsa_grad, "transform_fn")
            gradient_kwargs["sampler_rng"] = np.random.default_rng(SEED_FOR_SPSA)
            gradient_kwargs["num_directions"] = 20
            tol = TOL_FOR_SPSA
        elif diff_method == "hadamard":
            spy = mocker.spy(qml.gradients.hadamard_grad, "transform_fn")

        a = np.array(0.1, requires_grad=True)
        b = np.array(0.2, requires_grad=True)

        @qnode(
            dev,
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            **gradient_kwargs
        )
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))

        res = jax.jit(circuit)(a, b)

        assert circuit.qtape.trainable_params == [0, 1]
        assert isinstance(res, tuple)
        assert len(res) == 2

        expected = [np.cos(a), -np.cos(a) * np.sin(b)]
        assert np.allclose(res[0], expected[0], atol=tol, rtol=0)
        assert np.allclose(res[1], expected[1], atol=tol, rtol=0)

        res = jax.jit(jax.jacobian(circuit, argnums=[0, 1]))(a, b)
        assert circuit.qtape.trainable_params == [0, 1]

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

        if diff_method in ("parameter-shift", "finite-diff"):
            spy.assert_called()

    def test_jacobian_no_evaluate(
        self, dev, diff_method, grad_on_execution, interface, mocker, tol
    ):
        """Test jacobian calculation when no prior circuit evaluation has been performed"""

        gradient_kwargs = {}
        if diff_method == "parameter-shift":
            spy = mocker.spy(qml.gradients.param_shift, "transform_fn")
        elif diff_method == "finite-diff":
            spy = mocker.spy(qml.gradients.finite_diff, "transform_fn")
        elif diff_method == "spsa":
            spy = mocker.spy(qml.gradients.spsa_grad, "transform_fn")
            gradient_kwargs["sampler_rng"] = np.random.default_rng(SEED_FOR_SPSA)
            gradient_kwargs["num_directions"] = 20
            tol = TOL_FOR_SPSA
        elif diff_method == "hadamard":
            spy = mocker.spy(qml.gradients.hadamard_grad, "transform_fn")

        a = jax.numpy.array(0.1)
        b = jax.numpy.array(0.2)

        @qnode(
            dev,
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            **gradient_kwargs
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

        if diff_method in ("parameter-shift", "finite-diff", "spsa"):
            spy.assert_called()

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

    def test_diff_single_probs(self, dev, diff_method, grad_on_execution, interface, tol):
        """Tests correct output shape and evaluation for a tape
        with a single prob output"""
        gradient_kwargs = {}
        if diff_method == "adjoint":
            pytest.skip("Adjoint does not support probs")
        elif diff_method == "spsa":
            gradient_kwargs["sampler_rng"] = np.random.default_rng(SEED_FOR_SPSA)
            gradient_kwargs["num_directions"] = 20
            tol = TOL_FOR_SPSA

        x = jax.numpy.array(0.543)
        y = jax.numpy.array(-0.654)

        @qnode(
            dev,
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            **gradient_kwargs
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

    def test_diff_multi_probs(self, dev, diff_method, grad_on_execution, interface, tol):
        """Tests correct output shape and evaluation for a tape
        with multiple prob outputs"""
        gradient_kwargs = {}
        if diff_method == "adjoint":
            pytest.skip("Adjoint does not support probs")
        elif diff_method == "spsa":
            gradient_kwargs["sampler_rng"] = np.random.default_rng(SEED_FOR_SPSA)
            gradient_kwargs["num_directions"] = 20
            tol = TOL_FOR_SPSA

        x = jax.numpy.array(0.543)
        y = jax.numpy.array(-0.654)

        @qnode(
            dev,
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            **gradient_kwargs
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

    def test_diff_expval_probs(self, dev, diff_method, grad_on_execution, interface, tol):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""
        gradient_kwargs = {}
        if diff_method == "adjoint":
            pytest.skip("Adjoint does not support probs")
        elif diff_method == "spsa":
            gradient_kwargs["sampler_rng"] = np.random.default_rng(SEED_FOR_SPSA)
            gradient_kwargs["num_directions"] = 20
            tol = TOL_FOR_SPSA

        x = jax.numpy.array(0.543)
        y = jax.numpy.array(-0.654)

        @qnode(
            dev,
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            **gradient_kwargs
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
        self, dev, diff_method, grad_on_execution, interface, tol
    ):
        """Tests correct output shape and evaluation for a tape with prob and expval outputs with less
        trainable parameters (argnums) than parameters."""
        kwargs = {}
        if diff_method == "adjoint":
            pytest.skip("Adjoint does not support probs")
        elif diff_method == "spsa":
            kwargs["sampler_rng"] = np.random.default_rng(SEED_FOR_SPSA)
            tol = TOL_FOR_SPSA

        x = jax.numpy.array(0.543)
        y = jax.numpy.array(-0.654)

        @qnode(
            dev,
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            **kwargs
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

    def test_diff_var_probs(self, dev, diff_method, grad_on_execution, interface, tol):
        """Tests correct output shape and evaluation for a tape
        with prob and variance outputs"""
        gradient_kwargs = {}
        if diff_method == "adjoint":
            pytest.skip("Adjoint does not support probs")
        elif diff_method == "hadamard":
            pytest.skip("Hadamard does not support var")
        elif diff_method == "spsa":
            gradient_kwargs["sampler_rng"] = np.random.default_rng(SEED_FOR_SPSA)
            gradient_kwargs["num_directions"] = 20
            tol = TOL_FOR_SPSA

        x = jax.numpy.array(0.543)
        y = jax.numpy.array(-0.654)

        @qnode(
            dev,
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            **gradient_kwargs
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
        dev = DefaultQubit2()

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

        @qnode(DefaultQubit2(), diff_method=qml.gradients.param_shift, interface=interface)
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.sample(wires=(0, 1))

        # execute with device default shots (None)
        with pytest.raises(qml.DeviceError):
            circuit(a, b)

        # execute with shots=100
        res = circuit(a, b, shots=100)  # pylint: disable=unexpected-keyword-arg
        assert res.shape == (100, 2)  # pylint:disable=comparison-with-callable

    def test_gradient_integration(self, interface):
        """Test that temporarily setting the shots works
        for gradient computations"""
        a, b = jax.numpy.array([0.543, -0.654])

        @qnode(DefaultQubit2(), diff_method=qml.gradients.param_shift, interface=interface)
        def cost_fn(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliY(1))

        # TODO: jit when https://github.com/PennyLaneAI/pennylane/issues/3474 is resolved
        res = jax.grad(cost_fn, argnums=[0, 1])(a, b, shots=30000)

        expected = [np.sin(a) * np.sin(b), -np.cos(a) * np.cos(b)]
        assert np.allclose(res, expected, atol=0.1, rtol=0)

    def test_update_diff_method(self, mocker, interface):
        """Test that temporarily setting the shots updates the diff method"""
        # pylint: disable=unused-argument
        a, b = jax.numpy.array([0.543, -0.654])

        spy = mocker.spy(qml, "execute")

        @qnode(DefaultQubit2(), interface=interface)
        def cost_fn(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliY(1))

        cost_fn(a, b, shots=100)  # pylint:disable=unexpected-keyword-arg
        # since we are using finite shots, parameter-shift will
        # be chosen
        assert cost_fn.gradient_fn == "backprop"
        assert spy.call_args[1]["gradient_fn"] is qml.gradients.param_shift

        cost_fn(a, b)
        # if we set the shots to None, backprop can now be used
        assert spy.call_args[1]["gradient_fn"] == "backprop"


@pytest.mark.parametrize(
    "interface,dev,diff_method,grad_on_execution", interface_and_qubit_device_and_diff_method
)
class TestQubitIntegration:
    """Tests that ensure various qubit circuits integrate correctly"""

    def test_sampling(self, dev, diff_method, grad_on_execution, interface):
        """Test sampling works as expected"""
        if grad_on_execution:
            pytest.skip("Sampling not possible with forward grad_on_execution differentiation.")

        if diff_method == "adjoint":
            pytest.skip("Adjoint warns with finite shots")

        @qnode(
            dev, diff_method=diff_method, interface=interface, grad_on_execution=grad_on_execution
        )
        def circuit():
            qml.Hadamard(wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.sample(qml.PauliZ(0)), qml.sample(qml.PauliX(1))

        res = jax.jit(circuit, static_argnames="shots")(shots=10)

        assert isinstance(res, tuple)

        assert isinstance(res[0], jax.Array)
        assert res[0].shape == (10,)
        assert isinstance(res[1], jax.Array)
        assert res[1].shape == (10,)

    def test_counts(self, dev, diff_method, grad_on_execution, interface):
        """Test counts works as expected"""
        if grad_on_execution:
            pytest.skip("Sampling not possible with forward grad_on_execution differentiation.")

        if diff_method == "adjoint":
            pytest.skip("Adjoint warns with finite shots")

        @qnode(
            dev, diff_method=diff_method, interface=interface, grad_on_execution=grad_on_execution
        )
        def circuit():
            qml.Hadamard(wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.counts(qml.PauliZ(0)), qml.counts(qml.PauliX(1))

        if interface == "jax-jit":
            with pytest.raises(
                NotImplementedError, match="The JAX-JIT interface doesn't support qml.counts."
            ):
                jax.jit(circuit, static_argnames="shots")(shots=10)
        else:
            res = jax.jit(circuit, static_argnames="shots")(shots=10)

            assert isinstance(res, tuple)

            assert isinstance(res[0], dict)
            assert len(res[0]) == 2
            assert isinstance(res[1], dict)
            assert len(res[1]) == 2

    def test_chained_qnodes(self, dev, diff_method, grad_on_execution, interface):
        """Test that the gradient of chained QNodes works without error"""

        class Template(qml.templates.StronglyEntanglingLayers):
            def decomposition(self):
                return [qml.templates.StronglyEntanglingLayers(*self.parameters, self.wires)]

        @qnode(
            dev, interface=interface, diff_method=diff_method, grad_on_execution=grad_on_execution
        )
        def circuit1(weights):
            Template(weights, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        @qnode(
            dev, interface=interface, diff_method=diff_method, grad_on_execution=grad_on_execution
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


@pytest.mark.parametrize(
    "interface,dev,diff_method,grad_on_execution", interface_and_qubit_device_and_diff_method
)
class TestQubitIntegrationHigherOrder:
    """Tests that ensure various qubit circuits integrate correctly when computing higher-order derivatives"""

    def test_second_derivative(self, dev, diff_method, grad_on_execution, interface, tol):
        """Test second derivative calculation of a scalar-valued QNode"""

        gradient_kwargs = {}
        if diff_method == "adjoint":
            pytest.skip("Adjoint does not second derivative.")
        elif diff_method == "spsa":
            gradient_kwargs["sampler_rng"] = np.random.default_rng(SEED_FOR_SPSA)
            gradient_kwargs["num_directions"] = 20
            gradient_kwargs["h"] = H_FOR_SPSA
            tol = TOL_FOR_SPSA

        @qnode(
            dev,
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            max_diff=2,
            **gradient_kwargs
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

    def test_hessian(self, dev, diff_method, grad_on_execution, interface, tol):
        """Test hessian calculation of a scalar-valued QNode"""
        gradient_kwargs = {}
        if diff_method == "adjoint":
            pytest.skip("Adjoint does not support second derivative.")
        elif diff_method == "spsa":
            gradient_kwargs = {
                "h": H_FOR_SPSA,
                "num_directions": 40,
                "sampler_rng": np.random.default_rng(SEED_FOR_SPSA),
            }
            tol = TOL_FOR_SPSA

        @qnode(
            dev,
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            max_diff=2,
            **gradient_kwargs
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

    def test_hessian_vector_valued(self, dev, diff_method, grad_on_execution, interface, tol):
        """Test hessian calculation of a vector-valued QNode"""
        gradient_kwargs = {}
        if diff_method == "adjoint":
            pytest.skip("Adjoint does not support second derivative.")
        elif diff_method == "spsa":
            gradient_kwargs = {
                "h": H_FOR_SPSA,
                "num_directions": 20,
                "sampler_rng": np.random.default_rng(SEED_FOR_SPSA),
            }
            tol = TOL_FOR_SPSA

        @qnode(
            dev,
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            max_diff=2,
            **gradient_kwargs
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
        self, dev, diff_method, interface, grad_on_execution, tol
    ):
        """Test hessian calculation of a vector valued QNode with post-processing"""
        gradient_kwargs = {}
        if diff_method == "adjoint":
            pytest.skip("Adjoint does not support second derivative.")
        elif diff_method == "spsa":
            gradient_kwargs = {
                "h": H_FOR_SPSA,
                "num_directions": 20,
                "sampler_rng": np.random.default_rng(SEED_FOR_SPSA),
            }
            tol = TOL_FOR_SPSA

        @qnode(
            dev,
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            max_diff=2,
            **gradient_kwargs
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
        self, dev, diff_method, grad_on_execution, interface, mocker, tol
    ):
        """Test hessian calculation of a vector valued QNode that has separate input arguments"""
        gradient_kwargs = {}
        if diff_method == "adjoint":
            pytest.skip("Adjoint does not support second derivative.")
        elif diff_method == "spsa":
            gradient_kwargs = {
                "h": H_FOR_SPSA,
                "num_directions": 20,
                "sampler_rng": np.random.default_rng(SEED_FOR_SPSA),
            }
            tol = TOL_FOR_SPSA

        @qnode(
            dev,
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            max_diff=2,
            **gradient_kwargs
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

        spy = mocker.spy(qml.gradients.param_shift, "transform_fn")
        hess = jax.jit(jax.jacobian(jac_fn, argnums=[0, 1]))(a, b)

        if diff_method == "backprop":
            spy.assert_not_called()
        elif diff_method == "parameter-shift":
            spy.assert_called()

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

    def test_state(self, dev, diff_method, grad_on_execution, interface, tol):
        """Test that the state can be returned and differentiated"""
        if diff_method == "adjoint":
            pytest.skip("Adjoint does not support states")

        x = jax.numpy.array(0.543)
        y = jax.numpy.array(-0.654)

        # `qml.state` needs wires with jax-jit because `tape.shape` uses the device wires
        dev._wires = qml.wires.Wires([0, 1])  # pylint:disable=protected-access

        @qnode(
            dev, diff_method=diff_method, interface=interface, grad_on_execution=grad_on_execution
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

        res = jax.jit(cost_fn)(x, y)

        if diff_method not in {"backprop"}:
            pytest.skip("Test only supports backprop")

        res = jax.jit(jax.grad(cost_fn, argnums=[0, 1]))(x, y)
        expected = np.array([-np.sin(x) * np.cos(y) / 2, -np.cos(x) * np.sin(y) / 2])
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("state", [[1], [0, 1]])  # Basis state and state vector
    def test_projector(self, state, dev, diff_method, grad_on_execution, interface, tol):
        """Test that the variance of a projector is correctly returned"""
        gradient_kwargs = {}
        if diff_method == "adjoint":
            pytest.skip("Adjoint does not support projectors")
        elif diff_method == "hadamard":
            pytest.skip("Hadamard does not support var")
        elif diff_method == "spsa":
            gradient_kwargs = {"h": H_FOR_SPSA, "sampler_rng": np.random.default_rng(SEED_FOR_SPSA)}
            tol = TOL_FOR_SPSA

        P = jax.numpy.array(state)
        x, y = 0.765, -0.654

        @qnode(
            dev,
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            **gradient_kwargs
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


@pytest.mark.parametrize(
    "interface,dev,diff_method,grad_on_execution", interface_and_qubit_device_and_diff_method
)
class TestTapeExpansion:
    """Test that tape expansion within the QNode integrates correctly
    with the JAX interface"""

    @pytest.mark.parametrize("max_diff", [1, 2])
    def test_gradient_expansion_trainable_only(
        self, dev, diff_method, grad_on_execution, max_diff, interface, mocker
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
            dev,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            max_diff=max_diff,
            interface=interface,
        )
        def circuit(x, y):
            qml.Hadamard(wires=0)
            PhaseShift(x, wires=0)
            PhaseShift(2 * y, wires=0)
            return qml.expval(qml.PauliX(0))

        x = jax.numpy.array(0.5)
        y = jax.numpy.array(0.7)
        circuit(x, y)

        spy = mocker.spy(circuit.gradient_fn, "transform_fn")
        jax.grad(circuit, argnums=[0])(x, y)

        input_tape = spy.call_args[0][0]
        assert len(input_tape.operations) == 3
        assert input_tape.operations[1].name == "RY"
        assert input_tape.operations[1].data[0] == 3 * x
        assert input_tape.operations[2].name == "PhaseShift"
        assert input_tape.operations[2].grad_method is None

    @pytest.mark.parametrize("max_diff", [1, 2])
    def test_hamiltonian_expansion_analytic(
        self, dev, diff_method, grad_on_execution, max_diff, interface, mocker, tol
    ):
        """Test that the Hamiltonian is not expanded if there
        are non-commuting groups and the number of shots is None
        and the first and second order gradients are correctly evaluated"""
        gradient_kwargs = {}
        if diff_method == "adjoint":
            pytest.skip("The adjoint method does not yet support Hamiltonians")
        elif diff_method == "hadamard":
            pytest.skip("The Hadamard method does not yet support Hamiltonians")
        elif diff_method == "spsa":
            gradient_kwargs = {
                "h": H_FOR_SPSA,
                "num_directions": 20,
                "sampler_rng": np.random.default_rng(SEED_FOR_SPSA),
            }
            tol = TOL_FOR_SPSA

        spy = mocker.spy(qml.transforms, "hamiltonian_expand")
        obs = [qml.PauliX(0), qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliZ(1)]

        @qnode(
            dev,
            interface=interface,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            max_diff=max_diff,
            **gradient_kwargs
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
        self, dev, diff_method, grad_on_execution, interface, max_diff, mocker
    ):
        """Test that the Hamiltonian is correctly measured if there
        are non-commuting groups and the number of shots is finite
        and the first and second order gradients are correctly evaluated"""
        gradient_kwargs = {}
        tol = 0.3
        if diff_method in ("adjoint", "backprop", "finite-diff"):
            pytest.skip("The adjoint and backprop methods do not yet support sampling")
        elif diff_method == "hadamard":
            pytest.skip("The Hadamard method does not yet support Hamiltonians")
        elif diff_method == "spsa":
            gradient_kwargs = {"sampler_rng": SEED_FOR_SPSA, "h": H_FOR_SPSA, "num_directions": 20}
            tol = TOL_FOR_SPSA

        spy = mocker.spy(qml.transforms, "hamiltonian_expand")
        obs = [qml.PauliX(0), qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliZ(1)]

        @qnode(
            dev,
            interface=interface,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            max_diff=max_diff,
            **gradient_kwargs
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
        res = circuit(d, w, c, shots=50000)  # pylint:disable=unexpected-keyword-arg
        expected = c[2] * np.cos(d[1] + w[1]) - c[1] * np.sin(d[0] + w[0]) * np.sin(d[1] + w[1])
        assert np.allclose(res, expected, atol=tol)
        spy.assert_not_called()

        # test gradients
        grad = jax.grad(circuit, argnums=[1, 2])(d, w, c, shots=50000)
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
        self, dev, diff_method, grad_on_execution, interface, tol
    ):
        """Test that jax.vmap works just as well as parameter-broadcasting with JAX JIT on the forward pass when
        vectorized=True is specified for the callback when caching is disabled."""
        interface = "jax-jit"
        if diff_method == "adjoint":
            pytest.skip("The adjoint method does not yet support Hamiltonians")
        elif diff_method == "hadamard":
            pytest.skip("The Hadamard method does not yet support Hamiltonians")

        if diff_method == "backprop":
            pytest.skip(
                "The backprop method does not yet support parameter-broadcasting with Hamiltonians"
            )

        n_configs = 5
        pars_q = np.random.rand(n_configs, 2)

        def minimal_circ(params):
            @qml.qnode(
                dev,
                interface=interface,
                diff_method=diff_method,
                grad_on_execution=grad_on_execution,
                cache=None,
            )
            def _measure_operator():
                qml.RY(params[..., 0], wires=0)
                qml.RY(params[..., 1], wires=1)
                op = qml.Hamiltonian([1.0], [qml.PauliZ(0) @ qml.PauliZ(1)])
                return qml.expval(op)

            res = _measure_operator()
            return res

        assert np.allclose(
            jax.jit(minimal_circ)(pars_q), jax.jit(jax.vmap(minimal_circ))(pars_q), tol
        )

    def test_vmap_compared_param_broadcasting_multi_output(
        self, dev, diff_method, grad_on_execution, interface, tol
    ):
        """Test that jax.vmap works just as well as parameter-broadcasting with JAX JIT on the forward pass when
        vectorized=True is specified for the callback when caching is disabled and when multiple output values
        are returned."""
        interface = "jax-jit"
        if diff_method == "adjoint":
            pytest.skip("The adjoint method does not yet support Hamiltonians")
        elif diff_method == "hadamard":
            pytest.skip("The Hadamard method does not yet support Hamiltonians")

        if diff_method == "backprop":
            pytest.skip(
                "The backprop method does not yet support parameter-broadcasting with Hamiltonians"
            )

        n_configs = 5
        pars_q = np.random.rand(n_configs, 2)

        def minimal_circ(params):
            @qml.qnode(
                dev,
                interface=interface,
                diff_method=diff_method,
                grad_on_execution=grad_on_execution,
                cache=None,
            )
            def _measure_operator():
                qml.RY(params[..., 0], wires=0)
                qml.RY(params[..., 1], wires=1)
                op1 = qml.Hamiltonian([1.0], [qml.PauliZ(0) @ qml.PauliZ(1)])
                op2 = qml.Hamiltonian([1.0], [qml.PauliX(0) @ qml.PauliX(1)])
                return qml.expval(op1), qml.expval(op2)

            res = _measure_operator()
            return res

        res1, res2 = jax.jit(minimal_circ)(pars_q)
        vres1, vres2 = jax.jit(jax.vmap(minimal_circ))(pars_q)
        assert np.allclose(res1, vres1, tol)
        assert np.allclose(res2, vres2, tol)


jacobian_fn = [jax.jacobian, jax.jacrev, jax.jacfwd]


@pytest.mark.parametrize("jacobian", jacobian_fn)
@pytest.mark.parametrize(
    "interface,dev,diff_method,grad_on_execution", interface_and_qubit_device_and_diff_method
)
class TestJIT:
    """Test JAX JIT integration with the QNode and automatic resolution of the
    correct JAX interface variant."""

    def test_gradient(self, dev, diff_method, grad_on_execution, jacobian, tol, interface):
        """Test derivative calculation of a scalar valued QNode"""
        gradient_kwargs = {}
        if diff_method == "adjoint":
            pytest.xfail(reason="The adjoint method is not using host-callback currently")
        elif diff_method == "spsa":
            gradient_kwargs = {"h": H_FOR_SPSA, "sampler_rng": np.random.default_rng(SEED_FOR_SPSA)}
            tol = TOL_FOR_SPSA

        @qnode(
            dev,
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            **gradient_kwargs
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
    def test_hermitian(self, dev, diff_method, grad_on_execution, shots, jacobian, interface):
        """Test that the jax device works with qml.Hermitian and jitting even
        when shots>0.

        Note: before a fix, the cases of shots=10 and shots=1000 were failing due
        to different reasons, hence the parametrization in the test.
        """
        # pylint: disable=unused-argument
        if diff_method == "backprop":
            pytest.skip("Backpropagation is unsupported if shots > 0.")

        if diff_method == "adjoint":
            pytest.skip("Computing the gradient for observables is not supported with adjoint.")

        projector = np.array(qml.matrix(qml.PauliZ(0) @ qml.PauliZ(1)))

        @qml.qnode(
            dev, interface=interface, diff_method=diff_method, grad_on_execution=grad_on_execution
        )
        def circ(projector):
            return qml.expval(qml.Hermitian(projector, wires=range(2)))

        result = jax.jit(circ)(projector)
        assert jax.numpy.allclose(result, 1)

    @pytest.mark.filterwarnings(
        "ignore:Requested adjoint differentiation to be computed with finite shots."
    )
    @pytest.mark.parametrize("shots", [10, 1000])
    def test_probs_obs_none(self, dev, diff_method, grad_on_execution, shots, jacobian, interface):
        """Test that the jax device works with qml.probs, a MeasurementProcess
        that has obs=None even when shots>0."""
        # pylint: disable=unused-argument
        if diff_method in ["backprop", "adjoint"]:
            pytest.skip("Backpropagation is unsupported if shots > 0.")

        @qml.qnode(
            dev, interface=interface, diff_method=diff_method, grad_on_execution=grad_on_execution
        )
        def circuit():
            return qml.probs(wires=0)

        assert jax.numpy.allclose(circuit(), jax.numpy.array([1.0, 0.0]))

    @pytest.mark.xfail(
        reason="Non-trainable parameters are not being correctly unwrapped by the interface"
    )
    def test_gradient_subset(self, dev, diff_method, grad_on_execution, jacobian, tol, interface):
        """Test derivative calculation of a scalar valued QNode with respect
        to a subset of arguments"""
        a = jax.numpy.array(0.1)
        b = jax.numpy.array(0.2)

        @qnode(
            dev, diff_method=diff_method, interface=interface, grad_on_execution=grad_on_execution
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
        self, dev, diff_method, grad_on_execution, jacobian, tol, interface
    ):
        """Test derivative calculation of a scalar valued cost function that
        uses the output of a vector-valued QNode"""
        gradient_kwargs = {}
        if diff_method == "adjoint":
            pytest.xfail(reason="The adjoint method is not using host-callback currently")
        elif diff_method == "spsa":
            gradient_kwargs = {"h": H_FOR_SPSA, "sampler_rng": np.random.default_rng(SEED_FOR_SPSA)}
            tol = TOL_FOR_SPSA

        @qnode(
            dev,
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            **gradient_kwargs
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

    def test_matrix_parameter(self, dev, diff_method, grad_on_execution, jacobian, tol, interface):
        """Test that the JAX-JIT interface works correctly with a matrix
        parameter"""

        # pylint: disable=unused-argument
        @qml.qnode(
            dev, diff_method=diff_method, interface=interface, grad_on_execution=grad_on_execution
        )
        def circ(p, U):
            qml.QubitUnitary(U, wires=0)
            qml.RY(p, wires=0)
            return qml.expval(qml.PauliZ(0))

        p = jax.numpy.array(0.1)
        U = jax.numpy.array([[0, 1], [1, 0]])
        res = jax.jit(circ)(p, U)
        assert np.allclose(res, -np.cos(p), atol=tol, rtol=0)

        jac_fn = jax.jit(jax.grad(circ, argnums=0))
        res = jac_fn(p, U)
        assert np.allclose(res, np.sin(p), atol=tol, rtol=0)


@pytest.mark.parametrize("shots", [None, 10000])
@pytest.mark.parametrize("jacobian", jacobian_fn)
@pytest.mark.parametrize(
    "interface,dev,diff_method,grad_on_execution", interface_and_qubit_device_and_diff_method
)
class TestReturn:
    """Class to test the shape of the Grad/Jacobian with different return types."""

    def test_grad_single_measurement_param(
        self, dev, diff_method, grad_on_execution, jacobian, shots, interface
    ):
        """For one measurement and one param, the gradient is a float."""
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        @qnode(
            dev, interface=interface, diff_method=diff_method, grad_on_execution=grad_on_execution
        )
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return qml.expval(qml.PauliZ(0))

        a = jax.numpy.array(0.1)

        grad = jax.jit(jacobian(circuit), static_argnames="shots")(a, shots=shots)

        assert isinstance(grad, jax.numpy.ndarray)
        assert grad.shape == ()

    def test_grad_single_measurement_multiple_param(
        self, dev, diff_method, grad_on_execution, jacobian, shots, interface
    ):
        """For one measurement and multiple param, the gradient is a tuple of arrays."""
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        @qnode(
            dev, interface=interface, diff_method=diff_method, grad_on_execution=grad_on_execution
        )
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            return qml.expval(qml.PauliZ(0))

        a = jax.numpy.array(0.1)
        b = jax.numpy.array(0.2)

        grad = jax.jit(jacobian(circuit, argnums=[0, 1]), static_argnames="shots")(
            a, b, shots=shots
        )

        assert isinstance(grad, tuple)
        assert len(grad) == 2
        assert grad[0].shape == ()
        assert grad[1].shape == ()

    def test_grad_single_measurement_multiple_param_array(
        self, dev, diff_method, grad_on_execution, jacobian, shots, interface
    ):
        """For one measurement and multiple param as a single array params, the gradient is an array."""
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        @qnode(
            dev, interface=interface, diff_method=diff_method, grad_on_execution=grad_on_execution
        )
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        a = jax.numpy.array([0.1, 0.2])

        grad = jax.jit(jacobian(circuit), static_argnames="shots")(a, shots=shots)

        assert isinstance(grad, jax.numpy.ndarray)
        assert grad.shape == (2,)

    def test_jacobian_single_measurement_param_probs(
        self, dev, diff_method, grad_on_execution, jacobian, shots, interface
    ):
        """For a multi dimensional measurement (probs), check that a single array is returned with the correct
        dimension"""
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because of probabilities.")

        @qnode(
            dev, interface=interface, diff_method=diff_method, grad_on_execution=grad_on_execution
        )
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return qml.probs(wires=[0, 1])

        a = jax.numpy.array(0.1)

        jac = jax.jit(jacobian(circuit), static_argnames="shots")(a, shots=shots)

        assert isinstance(jac, jax.numpy.ndarray)
        assert jac.shape == (4,)

    def test_jacobian_single_measurement_probs_multiple_param(
        self, dev, diff_method, grad_on_execution, jacobian, shots, interface
    ):
        """For a multi dimensional measurement (probs), check that a single tuple is returned containing arrays with
        the correct dimension"""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because of probabilities.")
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        @qnode(
            dev, interface=interface, diff_method=diff_method, grad_on_execution=grad_on_execution
        )
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            return qml.probs(wires=[0, 1])

        a = jax.numpy.array(0.1)
        b = jax.numpy.array(0.2)

        jac = jax.jit(jacobian(circuit, argnums=[0, 1]), static_argnames="shots")(a, b, shots=shots)

        assert isinstance(jac, tuple)

        assert isinstance(jac[0], jax.numpy.ndarray)
        assert jac[0].shape == (4,)

        assert isinstance(jac[1], jax.numpy.ndarray)
        assert jac[1].shape == (4,)

    def test_jacobian_single_measurement_probs_multiple_param_single_array(
        self, dev, diff_method, grad_on_execution, jacobian, shots, interface
    ):
        """For a multi dimensional measurement (probs), check that a single tuple is returned containing arrays with
        the correct dimension"""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because of probabilities.")
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        @qnode(
            dev, interface=interface, diff_method=diff_method, grad_on_execution=grad_on_execution
        )
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.probs(wires=[0, 1])

        a = jax.numpy.array([0.1, 0.2])
        jac = jax.jit(jacobian(circuit), static_argnames="shots")(a, shots=shots)

        assert isinstance(jac, jax.numpy.ndarray)
        assert jac.shape == (4, 2)

    def test_jacobian_expval_expval_multiple_params(
        self, dev, diff_method, grad_on_execution, jacobian, shots, interface
    ):
        """The jacobian of multiple measurements with multiple params return a tuple of arrays."""
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        par_0 = jax.numpy.array(0.1)
        par_1 = jax.numpy.array(0.2)

        @qnode(
            dev, interface=interface, diff_method=diff_method, grad_on_execution=grad_on_execution
        )
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1)), qml.expval(qml.PauliZ(0))

        jac = jax.jit(jacobian(circuit, argnums=[0, 1]), static_argnames="shots")(
            par_0, par_1, shots=shots
        )

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
        self, dev, diff_method, grad_on_execution, jacobian, shots, interface
    ):
        """The jacobian of multiple measurements with a multiple params array return a single array."""
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        @qnode(
            dev, interface=interface, diff_method=diff_method, grad_on_execution=grad_on_execution
        )
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1)), qml.expval(qml.PauliZ(0))

        a = jax.numpy.array([0.1, 0.2])

        jac = jax.jit(jacobian(circuit), static_argnames="shots")(a, shots=shots)

        assert isinstance(jac, tuple)
        assert len(jac) == 2  # measurements

        assert isinstance(jac[0], jax.numpy.ndarray)
        assert jac[0].shape == (2,)

        assert isinstance(jac[1], jax.numpy.ndarray)
        assert jac[1].shape == (2,)

    def test_jacobian_var_var_multiple_params(
        self, dev, diff_method, grad_on_execution, jacobian, shots, interface
    ):
        """The jacobian of multiple measurements with multiple params return a tuple of arrays."""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because of var.")
        elif diff_method == "hadamard":
            pytest.skip("Test does not supports hadamard because of var.")
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        par_0 = jax.numpy.array(0.1)
        par_1 = jax.numpy.array(0.2)

        @qnode(
            dev, interface=interface, diff_method=diff_method, grad_on_execution=grad_on_execution
        )
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.PauliZ(0) @ qml.PauliX(1)), qml.var(qml.PauliZ(0))

        jac = jax.jit(jacobian(circuit, argnums=[0, 1]), static_argnames="shots")(
            par_0, par_1, shots=shots
        )

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
        self, dev, diff_method, grad_on_execution, jacobian, shots, interface
    ):
        """The jacobian of multiple measurements with a multiple params array return a single array."""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because of var.")
        elif diff_method == "hadamard":
            pytest.skip("Test does not supports hadamard because of var.")
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        @qnode(
            dev, interface=interface, diff_method=diff_method, grad_on_execution=grad_on_execution
        )
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.var(qml.PauliZ(0) @ qml.PauliX(1)), qml.var(qml.PauliZ(0))

        a = jax.numpy.array([0.1, 0.2])

        jac = jax.jit(jacobian(circuit), static_argnames="shots")(a, shots=shots)

        assert isinstance(jac, tuple)
        assert len(jac) == 2  # measurements

        assert isinstance(jac[0], jax.numpy.ndarray)
        assert jac[0].shape == (2,)

        assert isinstance(jac[1], jax.numpy.ndarray)
        assert jac[1].shape == (2,)

    def test_jacobian_multiple_measurement_single_param(
        self, dev, diff_method, grad_on_execution, jacobian, shots, interface
    ):
        """The jacobian of multiple measurements with a single params return an array."""
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because of probabilities.")

        @qnode(
            dev, interface=interface, diff_method=diff_method, grad_on_execution=grad_on_execution
        )
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        a = jax.numpy.array(0.1)

        jac = jax.jit(jacobian(circuit), static_argnames="shots")(a, shots=shots)

        assert isinstance(jac, tuple)
        assert len(jac) == 2

        assert isinstance(jac[0], jax.numpy.ndarray)
        assert jac[0].shape == ()

        assert isinstance(jac[1], jax.numpy.ndarray)
        assert jac[1].shape == (4,)

    def test_jacobian_multiple_measurement_multiple_param(
        self, dev, diff_method, grad_on_execution, jacobian, shots, interface
    ):
        """The jacobian of multiple measurements with a multiple params return a tuple of arrays."""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because of probabilities.")
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        @qnode(
            dev, interface=interface, diff_method=diff_method, grad_on_execution=grad_on_execution
        )
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        a = np.array(0.1, requires_grad=True)
        b = np.array(0.2, requires_grad=True)

        jac = jax.jit(jacobian(circuit, argnums=[0, 1]), static_argnames="shots")(a, b, shots=shots)

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
        self, dev, diff_method, grad_on_execution, jacobian, shots, interface
    ):
        """The jacobian of multiple measurements with a multiple params array return a single array."""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because of probabilities.")
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        @qnode(
            dev, interface=interface, diff_method=diff_method, grad_on_execution=grad_on_execution
        )
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        a = jax.numpy.array([0.1, 0.2])

        jac = jax.jit(jacobian(circuit), static_argnames="shots")(a, shots=shots)

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
@pytest.mark.parametrize(
    "interface,dev,diff_method,grad_on_execution", interface_and_qubit_device_and_diff_method
)
class TestReturnHessian:
    """Class to test the shape of the Hessian with different return types."""

    def test_hessian_expval_multiple_params(
        self, dev, diff_method, hessian, grad_on_execution, interface
    ):
        """The hessian of single a measurement with multiple params return a tuple of arrays."""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because second order diff.")

        par_0 = jax.numpy.array(0.1)
        par_1 = jax.numpy.array(0.2)

        @qnode(
            dev,
            interface=interface,
            diff_method=diff_method,
            max_diff=2,
            grad_on_execution=grad_on_execution,
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
        self, dev, diff_method, hessian, grad_on_execution, interface
    ):
        """The hessian of single measurement with a multiple params array return a single array."""

        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because second order diff.")

        params = jax.numpy.array([0.1, 0.2], dtype=jax.numpy.float64)

        @qnode(
            dev,
            interface=interface,
            diff_method=diff_method,
            max_diff=2,
            grad_on_execution=grad_on_execution,
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
        self, dev, diff_method, hessian, grad_on_execution, interface
    ):
        """The hessian of single a measurement with multiple params return a tuple of arrays."""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because second order diff.")
        elif diff_method == "hadamard":
            pytest.skip("Test does not supports hadamard because of var.")

        par_0 = jax.numpy.array(0.1, dtype=jax.numpy.float64)
        par_1 = jax.numpy.array(0.2, dtype=jax.numpy.float64)

        @qnode(
            dev,
            interface=interface,
            diff_method=diff_method,
            max_diff=2,
            grad_on_execution=grad_on_execution,
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
        self, dev, diff_method, hessian, grad_on_execution, interface
    ):
        """The hessian of single measurement with a multiple params array return a single array."""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because second order diff.")
        elif diff_method == "hadamard":
            pytest.skip("Test does not supports hadamard because of var.")

        params = jax.numpy.array([0.1, 0.2], dtype=jax.numpy.float64)

        @qnode(
            dev,
            interface=interface,
            diff_method=diff_method,
            max_diff=2,
            grad_on_execution=grad_on_execution,
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
        self, dev, diff_method, hessian, grad_on_execution, interface
    ):
        """The hessian of multiple measurements with multiple params return a tuple of arrays."""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because second order diff.")
        elif diff_method == "hadamard":
            pytest.skip("Test does not supports hadamard because of non commuting obs.")

        par_0 = jax.numpy.array(0.1, dtype=jax.numpy.float64)
        par_1 = jax.numpy.array(0.2, dtype=jax.numpy.float64)

        @qnode(
            dev,
            interface=interface,
            diff_method=diff_method,
            max_diff=2,
            grad_on_execution=grad_on_execution,
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
        self, dev, diff_method, hessian, grad_on_execution, interface
    ):
        """The hessian of multiple measurements with a multiple param array return a single array."""

        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because second order diff.")
        elif diff_method == "hadamard":
            pytest.skip("Test does not supports hadamard because of non commuting obs.")

        params = jax.numpy.array([0.1, 0.2], dtype=jax.numpy.float64)

        @qnode(
            dev,
            interface=interface,
            diff_method=diff_method,
            max_diff=2,
            grad_on_execution=grad_on_execution,
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
        self, dev, diff_method, hessian, grad_on_execution, interface
    ):
        """The hessian of multiple measurements with multiple params return a tuple of arrays."""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because second order diff.")
        elif diff_method == "hadamard":
            pytest.skip("Test does not supports hadamard because of var.")

        par_0 = jax.numpy.array(0.1, dtype=jax.numpy.float64)
        par_1 = jax.numpy.array(0.2, dtype=jax.numpy.float64)

        @qnode(
            dev,
            interface=interface,
            diff_method=diff_method,
            max_diff=2,
            grad_on_execution=grad_on_execution,
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
        self, dev, diff_method, hessian, grad_on_execution, interface
    ):
        """The hessian of multiple measurements with a multiple param array return a single array."""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because second order diff.")
        elif diff_method == "hadamard":
            pytest.skip("Test does not supports hadamard because of var.")

        params = jax.numpy.array([0.1, 0.2], dtype=jax.numpy.float64)

        @qnode(
            dev,
            interface=interface,
            diff_method=diff_method,
            max_diff=2,
            grad_on_execution=grad_on_execution,
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

    @partial(jax.jit, static_argnames="shots")
    @qml.qnode(DefaultQubit2(), diff_method=diff_method, max_diff=2)
    def circuit(x):
        qml.RY(x[0], wires=0)
        qml.RX(x[1], wires=0)
        return qml.expval(qml.PauliZ(0))

    x = jax.numpy.array([1.0, 2.0])
    a, b = x

    hess = jax.jit(hessian(circuit), static_argnames="shots")(x, shots=10000)

    expected_hess = [
        [-np.cos(a) * np.cos(b), np.sin(a) * np.sin(b)],
        [np.sin(a) * np.sin(b), -np.cos(a) * np.cos(b)],
    ]
    shots_tol = 0.1
    assert np.allclose(hess, expected_hess, atol=shots_tol, rtol=0)


@pytest.mark.parametrize("jit_inside", [True, False])
@pytest.mark.parametrize("argnums", [0, 1, [0, 1]])
@pytest.mark.parametrize("jacobian", jacobian_fn)
@pytest.mark.parametrize(
    "interface,dev,diff_method,grad_on_execution", interface_and_qubit_device_and_diff_method
)
class TestSubsetArgnums:
    def test_single_measurement(
        self,
        interface,
        dev,
        diff_method,
        grad_on_execution,
        jacobian,
        argnums,
        jit_inside,
        tol,
    ):
        """Test single measurement with different diff methods with argnums."""
        kwargs = {}
        if diff_method == "spsa":
            kwargs["sampler_rng"] = np.random.default_rng(SEED_FOR_SPSA)
            tol = TOL_FOR_SPSA

        @qml.qnode(
            dev,
            interface=interface,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            cache=False,
            **kwargs
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
        dev,
        diff_method,
        grad_on_execution,
        jacobian,
        argnums,
        jit_inside,
        tol,
    ):
        """Test multiple measurements with different diff methods with argnums."""
        kwargs = {}
        if diff_method == "spsa":
            kwargs["sampler_rng"] = np.random.default_rng(SEED_FOR_SPSA)
            tol = TOL_FOR_SPSA

        @qml.qnode(
            dev,
            interface=interface,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            **kwargs
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

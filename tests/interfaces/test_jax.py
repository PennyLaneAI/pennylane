# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the JAX-Python interface"""
import numpy as np

# pylint: disable=protected-access,too-few-public-methods
import pytest

import pennylane as qml
from pennylane import execute
from pennylane.gradients import param_shift
from pennylane.typing import TensorLike

pytestmark = pytest.mark.jax

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)


class TestJaxExecuteUnitTests:
    """Unit tests for jax execution"""

    def test_jacobian_options(self, mocker):
        """Test setting jacobian options"""
        spy = mocker.spy(qml.gradients, "param_shift")

        a = jax.numpy.array([0.1, 0.2])

        dev = qml.device("default.qubit.legacy", wires=1)

        def cost(a, device):
            with qml.queuing.AnnotatedQueue() as q:
                qml.RY(a[0], wires=0)
                qml.RX(a[1], wires=0)
                qml.expval(qml.PauliZ(0))

            tape = qml.tape.QuantumScript.from_queue(q)
            return execute(
                [tape],
                device,
                gradient_fn=param_shift,
                gradient_kwargs={"shifts": [(np.pi / 4,)] * 2},
            )[0]

        jax.grad(cost)(a, device=dev)

        for args in spy.call_args_list:
            assert args[1]["shifts"] == [(np.pi / 4,)] * 2

    def test_incorrect_grad_on_execution(self):
        """Test that an error is raised if an gradient transform
        is used with grad_on_execution=True"""
        a = jax.numpy.array([0.1, 0.2])

        dev = qml.device("default.qubit.legacy", wires=1)

        def cost(a, device):
            with qml.queuing.AnnotatedQueue() as q:
                qml.RY(a[0], wires=0)
                qml.RX(a[1], wires=0)
                qml.expval(qml.PauliZ(0))

            tape = qml.tape.QuantumScript.from_queue(q)
            return execute(
                [tape],
                device,
                gradient_fn=param_shift,
                grad_on_execution=True,
            )[0]

        with pytest.raises(
            ValueError, match="Gradient transforms cannot be used with grad_on_execution=True"
        ):
            jax.grad(cost)(a, device=dev)

    def test_unknown_interface(self):
        """Test that an error is raised if the interface is unknown"""
        a = jax.numpy.array([0.1, 0.2])

        dev = qml.device("default.qubit.legacy", wires=1)

        def cost(a, device):
            with qml.queuing.AnnotatedQueue() as q:
                qml.RY(a[0], wires=0)
                qml.RX(a[1], wires=0)
                qml.expval(qml.PauliZ(0))

            tape = qml.tape.QuantumScript.from_queue(q)
            return execute(
                [tape],
                device,
                gradient_fn=param_shift,
                interface="None",
            )[0]

        with pytest.raises(ValueError, match="Unknown interface"):
            cost(a, device=dev)

    def test_grad_on_execution(self, mocker):
        """Test that grad_on_execution uses the `device.execute_and_gradients` pathway"""
        dev = qml.device("default.qubit.legacy", wires=2)
        spy = mocker.spy(dev, "execute_and_gradients")

        def cost(params):
            tape1 = qml.tape.QuantumScript(
                [qml.RY(params[0], 0), qml.RX(params[1], 0)],
                [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))],
            )

            tape2 = qml.tape.QuantumScript(
                [qml.RY(np.array(0.5), 0)],
                [qml.expval(qml.PauliZ(0))],
            )

            tape3 = qml.tape.QuantumScript(
                [qml.RY(params[0], 0), qml.RX(params[1], 0)],
                [qml.expval(qml.PauliZ(0))],
            )
            return execute(
                [tape1, tape2, tape3],
                dev,
                gradient_fn="device",
                gradient_kwargs={
                    "method": "adjoint_jacobian",
                    "use_device_state": True,
                },
            )

        a = jax.numpy.array([0.1, 0.2])
        res = cost(a)

        x, y = a
        assert np.allclose(res[0][0], np.cos(x) * np.cos(y))
        assert np.allclose(res[0][1], 1)
        assert np.allclose(res[1], np.cos(0.5))
        assert np.allclose(res[2], np.cos(x) * np.cos(y))

        # adjoint method only performs a single device execution per tape, but gets both result and gradient
        assert dev.num_executions == 3
        spy.assert_not_called()

        g = jax.jacobian(cost)(a)
        spy.assert_called()
        expected_g = (-np.sin(x) * np.cos(y), -np.cos(x) * np.sin(y))
        assert qml.math.allclose(g[0][0], expected_g)
        assert qml.math.allclose(g[0][1], np.zeros(2))
        assert qml.math.allclose(g[1], np.zeros(2))
        assert qml.math.allclose(g[2], expected_g)

    def test_no_grad_on_execution(self, mocker):
        """Test that `grad_on_execution=False` uses the `device.execute_and_gradients`."""
        dev = qml.device("default.qubit.legacy", wires=1)
        spy_execute = mocker.spy(qml.devices.DefaultQubitLegacy, "batch_execute")
        spy_gradients = mocker.spy(qml.devices.DefaultQubitLegacy, "execute_and_gradients")

        def cost(a):
            with qml.queuing.AnnotatedQueue() as q:
                qml.RY(a[0], wires=0)
                qml.RX(a[1], wires=0)
                qml.expval(qml.PauliZ(0))

            tape = qml.tape.QuantumScript.from_queue(q)
            return execute(
                [tape],
                dev,
                gradient_fn="device",
                grad_on_execution=False,
                gradient_kwargs={"method": "adjoint_jacobian"},
            )[0]

        a = jax.numpy.array([0.1, 0.2])
        cost(a)

        assert dev.num_executions == 1
        spy_execute.assert_called()
        spy_gradients.assert_not_called()

        jax.grad(cost)(a)
        spy_gradients.assert_called()


class TestCaching:
    """Test for caching behaviour"""

    def test_cache_maxsize(self, mocker):
        """Test the cachesize property of the cache"""
        dev = qml.device("default.qubit.legacy", wires=1)
        spy = mocker.spy(qml.workflow.execution._cache_transform, "_transform")

        def cost(a, cachesize):
            with qml.queuing.AnnotatedQueue() as q:
                qml.RY(a[0], wires=0)
                qml.RX(a[1], wires=0)
                qml.expval(qml.PauliZ(0))

            tape = qml.tape.QuantumScript.from_queue(q)
            return execute(
                [tape],
                dev,
                gradient_fn=param_shift,
                cachesize=cachesize,
            )[0]

        params = jax.numpy.array([0.1, 0.2])
        jax.grad(cost)(params, cachesize=2)
        cache = spy.call_args.kwargs["cache"]

        assert cache.maxsize == 2
        assert cache.currsize == 2
        assert len(cache) == 2

    def test_custom_cache(self, mocker):
        """Test the use of a custom cache object"""
        dev = qml.device("default.qubit.legacy", wires=1)
        spy = mocker.spy(qml.workflow.execution._cache_transform, "_transform")

        def cost(a, cache):
            with qml.queuing.AnnotatedQueue() as q:
                qml.RY(a[0], wires=0)
                qml.RX(a[1], wires=0)
                qml.expval(qml.PauliZ(0))

            tape = qml.tape.QuantumScript.from_queue(q)
            return execute(
                [tape],
                dev,
                gradient_fn=param_shift,
                cache=cache,
            )[0]

        custom_cache = {}
        params = jax.numpy.array([0.1, 0.2])
        jax.grad(cost)(params, cache=custom_cache)

        cache = spy.call_args.kwargs["cache"]
        assert cache is custom_cache

    def test_custom_cache_multiple(self, mocker):
        """Test the use of a custom cache object with multiple tapes"""
        dev = qml.device("default.qubit.legacy", wires=1)
        spy = mocker.spy(qml.workflow.execution._cache_transform, "_transform")

        a = jax.numpy.array(0.1)
        b = jax.numpy.array(0.2)

        def cost(a, b, cache):
            with qml.queuing.AnnotatedQueue() as q1:
                qml.RY(a, wires=0)
                qml.RX(b, wires=0)
                qml.expval(qml.PauliZ(0))

            tape1 = qml.tape.QuantumScript.from_queue(q1)
            with qml.queuing.AnnotatedQueue() as q2:
                qml.RY(a, wires=0)
                qml.RX(b, wires=0)
                qml.expval(qml.PauliZ(0))

            tape2 = qml.tape.QuantumScript.from_queue(q2)
            res = execute(
                [tape1, tape2],
                dev,
                gradient_fn=param_shift,
                cache=cache,
            )
            return res[0]

        custom_cache = {}
        jax.grad(cost)(a, b, cache=custom_cache)

        cache = spy.call_args.kwargs["cache"]
        assert cache is custom_cache

    def test_caching_param_shift(self, tol):
        """Test that, when using parameter-shift transform,
        caching produces the optimum number of evaluations."""
        dev = qml.device("default.qubit.legacy", wires=1)

        def cost(a, cache):
            with qml.queuing.AnnotatedQueue() as q:
                qml.RY(a[0], wires=0)
                qml.RX(a[1], wires=0)
                qml.expval(qml.PauliZ(0))

            tape = qml.tape.QuantumScript.from_queue(q)
            return execute(
                [tape],
                dev,
                gradient_fn=param_shift,
                cache=cache,
            )[0]

        # Without caching, 5 evaluations are required to compute
        # the Jacobian: 1 (forward pass) + 2 (backward pass) * (2 shifts * 2 params)
        params = jax.numpy.array([0.1, 0.2])
        jax.grad(cost)(params, cache=None)
        assert dev.num_executions == 5

        # With caching, 5 evaluations are required to compute
        # the Jacobian: 1 (forward pass) + (2 shifts * 2 params)
        dev._num_executions = 0
        jac_fn = jax.grad(cost)
        grad1 = jac_fn(params, cache=True)
        assert dev.num_executions == 5

        # Check that calling the cost function again
        # continues to evaluate the device (that is, the cache
        # is emptied between calls)
        grad2 = jac_fn(params, cache=True)
        assert dev.num_executions == 10
        assert np.allclose(grad1, grad2, atol=tol, rtol=0)

        # Check that calling the cost function again
        # with different parameters produces a different Jacobian
        grad2 = jac_fn(2 * params, cache=True)
        assert dev.num_executions == 15
        assert not np.allclose(grad1, grad2, atol=tol, rtol=0)

    def test_caching_adjoint_backward(self):
        """Test that caching produces the optimum number of adjoint evaluations
        when no grad on execution."""
        dev = qml.device("default.qubit.legacy", wires=2)
        params = jax.numpy.array([0.1, 0.2, 0.3])

        def cost(a, cache):
            with qml.queuing.AnnotatedQueue() as q:
                qml.RY(a[0], wires=0)
                qml.RX(a[1], wires=0)
                qml.RY(a[2], wires=0)
                qml.expval(qml.PauliZ(0))

            tape = qml.tape.QuantumScript.from_queue(q)
            return execute(
                [tape],
                dev,
                gradient_fn="device",
                cache=cache,
                grad_on_execution=False,
                gradient_kwargs={"method": "adjoint_jacobian"},
            )[0]

        # Without caching, 2 evaluations are required.
        # 1 for the forward pass, and one per output dimension
        # on the backward pass.
        jax.grad(cost)(params, cache=None)
        assert dev.num_executions == 2

        # With caching, also 2 evaluations are required. One
        # for the forward pass, and one for the backward pass.
        dev._num_executions = 0
        jac_fn = jax.grad(cost)
        jac_fn(params, cache=True)
        assert dev.num_executions == 2


execute_kwargs_integration = [
    {"gradient_fn": param_shift},
    {
        "gradient_fn": "device",
        "grad_on_execution": True,
        "gradient_kwargs": {"method": "adjoint_jacobian", "use_device_state": True},
    },
    {
        "gradient_fn": "device",
        "grad_on_execution": False,
        "gradient_kwargs": {"method": "adjoint_jacobian"},
    },
]


@pytest.mark.parametrize("execute_kwargs", execute_kwargs_integration)
class TestJaxExecuteIntegration:
    """Test the jax interface execute function
    integrates well for both forward and backward execution"""

    def test_execution(self, execute_kwargs):
        """Test execution"""
        dev = qml.device("default.qubit.legacy", wires=1)

        def cost(a, b):
            with qml.queuing.AnnotatedQueue() as q1:
                qml.RY(a, wires=0)
                qml.RX(b, wires=0)
                qml.expval(qml.PauliZ(0))

            tape1 = qml.tape.QuantumScript.from_queue(q1)
            with qml.queuing.AnnotatedQueue() as q2:
                qml.RY(a, wires=0)
                qml.RX(b, wires=0)
                qml.expval(qml.PauliZ(0))

            tape2 = qml.tape.QuantumScript.from_queue(q2)

            return execute([tape1, tape2], dev, **execute_kwargs)

        a = jax.numpy.array(0.1)
        b = jax.numpy.array(0.2)
        res = cost(a, b)

        assert len(res) == 2
        assert res[0].shape == ()
        assert res[1].shape == ()

    def test_scalar_jacobian(self, execute_kwargs, tol):
        """Test scalar jacobian calculation"""
        a = jax.numpy.array(0.1)
        dev = qml.device("default.qubit.legacy", wires=2)

        def cost(a):
            with qml.queuing.AnnotatedQueue() as q:
                qml.RY(a, wires=0)
                qml.expval(qml.PauliZ(0))

            tape = qml.tape.QuantumScript.from_queue(q)

            return execute([tape], dev, **execute_kwargs)[0]

        res = jax.grad(cost)(a)
        assert res.shape == ()

        # compare to standard tape jacobian
        with qml.queuing.AnnotatedQueue() as q:
            qml.RY(a, wires=0)
            qml.expval(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        tape.trainable_params = [0]
        tapes, fn = param_shift(tape)
        expected = fn(dev.batch_execute(tapes))

        assert expected.shape == ()
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_reusing_quantum_tape(self, execute_kwargs, tol):
        """Test re-using a quantum tape by passing new parameters"""
        a = jax.numpy.array(0.1)
        b = jax.numpy.array(0.2)

        dev = qml.device("default.qubit.legacy", wires=2)

        with qml.queuing.AnnotatedQueue() as q:
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        assert tape.trainable_params == [0, 1]

        def cost(a, b):
            # An explicit call to _update() is required here to update the
            # trainable parameters in between tape executions.
            # This is different from how the autograd interface works.
            # Unless the update is issued, the validation check related to the
            # number of provided parameters fails in the tape: (len(params) !=
            # required_length) and the tape produces incorrect results.
            tape._update()
            new_tape = tape.bind_new_parameters([a, b], [0, 1])
            return execute([new_tape], dev, **execute_kwargs)[0]

        jac_fn = jax.grad(cost)
        jac = jac_fn(a, b)

        a = jax.numpy.array(0.54)
        b = jax.numpy.array(0.8)

        # check that the cost function continues to depend on the
        # values of the parameters for subsequent calls
        res2 = cost(2 * a, b)
        expected = [np.cos(2 * a)]
        assert np.allclose(res2, expected, atol=tol, rtol=0)

        jac_fn = jax.grad(lambda a, b: cost(2 * a, b))
        jac = jac_fn(a, b)
        expected = -2 * np.sin(2 * a)
        assert np.allclose(jac, expected, atol=tol, rtol=0)

    def test_grad_with_different_grad_on_execution(self, execute_kwargs):
        """Test jax grad for adjoint diff method with different execution kwargs."""
        dev = qml.device("default.qubit.legacy", wires=2)
        params = jax.numpy.array([0.1, 0.2, 0.3])
        expected_results = jax.numpy.array([-0.3875172, -0.18884787, -0.38355705])

        def cost(a, cache):
            with qml.queuing.AnnotatedQueue() as q:
                qml.RY(a[0], wires=0)
                qml.RX(a[1], wires=0)
                qml.RY(a[2], wires=0)
                qml.expval(qml.PauliZ(0))

            tape = qml.tape.QuantumScript.from_queue(q)
            res = qml.execute([tape], dev, cache=cache, **execute_kwargs)[0]
            return res

        results = jax.grad(cost)(params, cache=None)
        for r, e in zip(results, expected_results):
            assert jax.numpy.allclose(r, e, atol=1e-7)

    def test_classical_processing_single_tape(self, execute_kwargs):
        """Test classical processing within the quantum tape for a single tape"""
        a = jax.numpy.array(0.1)
        b = jax.numpy.array(0.2)
        c = jax.numpy.array(0.3)

        def cost(a, b, c, device):
            with qml.queuing.AnnotatedQueue() as q:
                qml.RY(a * c, wires=0)
                qml.RZ(b, wires=0)
                qml.RX(c + c**2 + jax.numpy.sin(a), wires=0)
                qml.expval(qml.PauliZ(0))

            tape = qml.tape.QuantumScript.from_queue(q)

            return execute([tape], device, **execute_kwargs)[0]

        dev = qml.device("default.qubit.legacy", wires=2)
        res = jax.grad(cost, argnums=(0, 1, 2))(a, b, c, device=dev)
        assert len(res) == 3

    def test_classical_processing_multiple_tapes(self, execute_kwargs):
        """Test classical processing within the quantum tape for multiple
        tapes"""
        dev = qml.device("default.qubit.legacy", wires=2)
        params = jax.numpy.array([0.3, 0.2])

        def cost_fn(x):
            with qml.queuing.AnnotatedQueue() as q1:
                qml.Hadamard(0)
                qml.RY(x[0], wires=[0])
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0))

            tape1 = qml.tape.QuantumScript.from_queue(q1)
            with qml.queuing.AnnotatedQueue() as q2:
                qml.Hadamard(0)
                qml.CRX(2 * x[0] * x[1], wires=[0, 1])
                qml.RX(2 * x[1], wires=[1])
                qml.expval(qml.PauliZ(0))

            tape2 = qml.tape.QuantumScript.from_queue(q2)
            result = execute(tapes=[tape1, tape2], device=dev, **execute_kwargs)
            return result[0] + result[1] - 7 * result[1]

        res = jax.grad(cost_fn)(params)
        assert res.shape == (2,)

    def test_multiple_tapes_output(self, execute_kwargs):
        """Test the output types for the execution of multiple quantum tapes"""
        dev = qml.device("default.qubit.legacy", wires=2)
        params = jax.numpy.array([0.3, 0.2])

        def cost_fn(x):
            with qml.queuing.AnnotatedQueue() as q1:
                qml.Hadamard(0)
                qml.RY(x[0], wires=[0])
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0))

            tape1 = qml.tape.QuantumScript.from_queue(q1)
            with qml.queuing.AnnotatedQueue() as q2:
                qml.Hadamard(0)
                qml.CRX(2 * x[0] * x[1], wires=[0, 1])
                qml.RX(2 * x[1], wires=[1])
                qml.expval(qml.PauliZ(0))

            tape2 = qml.tape.QuantumScript.from_queue(q2)

            return execute(tapes=[tape1, tape2], device=dev, **execute_kwargs)

        res = cost_fn(params)
        assert isinstance(res, TensorLike)
        assert all(isinstance(r, jax.numpy.ndarray) for r in res)
        assert all(r.shape == () for r in res)

    def test_matrix_parameter(self, execute_kwargs, tol):
        """Test that the jax interface works correctly
        with a matrix parameter"""
        a = jax.numpy.array(0.1)
        U = jax.numpy.array([[0, 1], [1, 0]])

        def cost(a, U, device):
            with qml.queuing.AnnotatedQueue() as q:
                qml.QubitUnitary(U, wires=0)
                qml.RY(a, wires=0)
                qml.expval(qml.PauliZ(0))

            tape = qml.tape.QuantumScript.from_queue(q)
            tape.trainable_params = [0]
            return execute([tape], device, **execute_kwargs)[0]

        dev = qml.device("default.qubit.legacy", wires=2)
        res = cost(a, U, device=dev)
        assert np.allclose(res, -np.cos(a), atol=tol, rtol=0)

        jac_fn = jax.grad(cost, argnums=0)
        res = jac_fn(a, U, device=dev)
        assert np.allclose(res, np.sin(a), atol=tol, rtol=0)

    def test_differentiable_expand(self, execute_kwargs, tol):
        """Test that operation and nested tapes expansion
        is differentiable"""

        class U3(qml.U3):
            def decomposition(self):
                theta, phi, lam = self.data
                wires = self.wires
                return [
                    qml.Rot(lam, theta, -lam, wires=wires),
                    qml.PhaseShift(phi + lam, wires=wires),
                ]

        def cost_fn(a, p, device):
            with qml.queuing.AnnotatedQueue() as q_tape:
                qml.RX(a, wires=0)
                U3(*p, wires=0)
                qml.expval(qml.PauliX(0))

            tape = qml.tape.QuantumScript.from_queue(q_tape)
            tape = tape.expand(stop_at=lambda obj: device.supports_operation(obj.name))
            return execute([tape], device, **execute_kwargs)[0]

        a = jax.numpy.array(0.1)
        p = jax.numpy.array([0.1, 0.2, 0.3])

        dev = qml.device("default.qubit.legacy", wires=1)
        res = cost_fn(a, p, device=dev)
        expected = np.cos(a) * np.cos(p[1]) * np.sin(p[0]) + np.sin(a) * (
            np.cos(p[2]) * np.sin(p[1]) + np.cos(p[0]) * np.cos(p[1]) * np.sin(p[2])
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

        jac_fn = jax.grad(cost_fn, argnums=1)
        res = jac_fn(a, p, device=dev)
        expected = jax.numpy.array(
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

    def test_independent_expval(self, execute_kwargs):
        """Tests computing an expectation value that is independent of trainable
        parameters."""
        dev = qml.device("default.qubit.legacy", wires=2)
        params = jax.numpy.array([0.1, 0.2, 0.3])

        def cost(a, cache):
            with qml.queuing.AnnotatedQueue() as q:
                qml.RY(a[0], wires=0)
                qml.RX(a[1], wires=0)
                qml.RY(a[2], wires=0)
                qml.expval(qml.PauliZ(1))

            tape = qml.tape.QuantumScript.from_queue(q)

            res = execute([tape], dev, cache=cache, **execute_kwargs)
            return res[0]

        res = jax.grad(cost)(params, cache=None)
        assert res.shape == (3,)


@pytest.mark.parametrize("execute_kwargs", execute_kwargs_integration)
class TestVectorValued:
    """Test vector-valued jacobian returns for the JAX Python interface."""

    def test_multiple_expvals(self, execute_kwargs):
        """Tests computing multiple expectation values in a tape."""

        dev = qml.device("default.qubit.legacy", wires=2)
        params = jax.numpy.array([0.1, 0.2, 0.3])

        def cost(a, cache):
            with qml.queuing.AnnotatedQueue() as q:
                qml.RY(a[0], wires=0)
                qml.RX(a[1], wires=0)
                qml.RY(a[2], wires=0)
                qml.expval(qml.PauliZ(0))
                qml.expval(qml.PauliZ(1))

            tape = qml.tape.QuantumScript.from_queue(q)
            res = qml.execute([tape], dev, cache=cache, **execute_kwargs)
            return res[0]

        res = jax.jacobian(cost)(params, cache=None)

        assert isinstance(res, tuple)
        assert len(res) == 2

        assert res[0].shape == (3,)
        assert isinstance(res[0], jax.numpy.ndarray)

        assert res[1].shape == (3,)
        assert isinstance(res[1], jax.numpy.ndarray)

    def test_multiple_expvals_single_par(self, execute_kwargs):
        """Tests computing multiple expectation values in a tape with a single
        trainable parameter."""
        dev = qml.device("default.qubit.legacy", wires=2)
        params = jax.numpy.array([0.1])

        def cost(a, cache):
            with qml.queuing.AnnotatedQueue() as q:
                qml.RY(a[0], wires=0)
                qml.expval(qml.PauliZ(0))
                qml.expval(qml.PauliZ(1))

            tape = qml.tape.QuantumScript.from_queue(q)
            res = qml.execute([tape], dev, cache=cache, **execute_kwargs)
            return res[0]

        res = jax.jacobian(cost)(params, cache=None)

        assert isinstance(res, tuple)

        assert isinstance(res[0], jax.numpy.ndarray)
        assert res[0].shape == (1,)

        assert isinstance(res[1], jax.numpy.ndarray)
        assert res[1].shape == (1,)

    def test_multi_tape_fwd(self, execute_kwargs):
        """Test the forward evaluation of a cost function that uses the output
        of multiple tapes that be vector-valued."""
        dev = qml.device("default.qubit.legacy", wires=2)
        params = jax.numpy.array([0.3, 0.2])

        def cost_fn(x):
            with qml.queuing.AnnotatedQueue() as q1:
                qml.RX(x[0], wires=[0])
                qml.expval(qml.PauliY(0))

            tape1 = qml.tape.QuantumScript.from_queue(q1)
            with qml.queuing.AnnotatedQueue() as q2:
                qml.RX(x[1], wires=[0])
                qml.RX(x[1], wires=[0])
                qml.RX(-x[1], wires=[0])
                qml.expval(qml.PauliY(0))
                qml.expval(qml.PauliY(1))

            tape2 = qml.tape.QuantumScript.from_queue(q2)
            result = qml.execute(tapes=[tape1, tape2], device=dev, **execute_kwargs)
            return result[0] + result[1][0]

        expected = -jax.numpy.sin(params[0]) + -jax.numpy.sin(params[1])
        res = cost_fn(params)
        assert jax.numpy.allclose(expected, res)

    def test_multi_tape_jacobian(self, execute_kwargs):
        """Test the jacobian computation with multiple tapes."""

        def cost(x, y, device, interface, ek):
            with qml.queuing.AnnotatedQueue() as q1:
                qml.RX(x, wires=[0])
                qml.RY(y, wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0))
                qml.expval(qml.PauliZ(1))

            tape1 = qml.tape.QuantumScript.from_queue(q1)
            with qml.queuing.AnnotatedQueue() as q2:
                qml.RX(x, wires=[0])
                qml.RY(y, wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0))
                qml.expval(qml.PauliZ(1))

            tape2 = qml.tape.QuantumScript.from_queue(q2)
            return qml.execute([tape1, tape2], device, **ek, interface=interface)

        dev = qml.device("default.qubit.legacy", wires=2)
        x = jax.numpy.array(0.543)
        y = jax.numpy.array(-0.654)

        x_ = np.array(0.543)
        y_ = np.array(-0.654)

        exec_jax = cost(x, y, dev, interface="jax-python", ek=execute_kwargs)
        exec_autograd = cost(x_, y_, dev, interface="autograd", ek=execute_kwargs)

        assert np.allclose(exec_jax, exec_autograd)

        res = jax.jacobian(cost, argnums=(0, 1))(
            x, y, dev, interface="jax-python", ek=execute_kwargs
        )

        import autograd.numpy as anp

        def cost_stack(x, y, device, interface, ek):
            return anp.hstack(cost(x, y, device, interface, ek))

        exp = qml.jacobian(cost_stack, argnum=(0, 1))(
            x_, y_, dev, interface="autograd", ek=execute_kwargs
        )
        res_0 = jax.numpy.array([res[0][0][0], res[0][1][0], res[1][0][0], res[1][1][0]])
        res_1 = jax.numpy.array([res[0][0][1], res[0][1][1], res[1][0][1], res[1][1][1]])

        assert np.allclose(res_0, exp[0])
        assert np.allclose(res_1, exp[1])

    def test_multi_tape_jacobian_probs_expvals(self, execute_kwargs):
        """Test the jacobian computation with multiple tapes with probability
        and expectation value computations."""

        adjoint = execute_kwargs.get("gradient_kwargs", {}).get("method", "") == "adjoint_jacobian"
        if adjoint:
            pytest.skip("The adjoint diff method doesn't support probabilities.")

        def cost(x, y, device, interface, ek):
            with qml.queuing.AnnotatedQueue() as q1:
                qml.RX(x, wires=[0])
                qml.RY(y, wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0))
                qml.expval(qml.PauliZ(1))

            tape1 = qml.tape.QuantumScript.from_queue(q1)
            with qml.queuing.AnnotatedQueue() as q2:
                qml.RX(x, wires=[0])
                qml.RY(y, wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.probs(wires=[0])
                qml.probs(wires=[1])

            tape2 = qml.tape.QuantumScript.from_queue(q2)
            return qml.execute([tape1, tape2], device, **ek, interface=interface)

        dev = qml.device("default.qubit.legacy", wires=2)
        x = jax.numpy.array(0.543)
        y = jax.numpy.array(-0.654)

        x_ = np.array(0.543)
        y_ = np.array(-0.654)

        exec_jax = cost(x, y, dev, interface="jax-python", ek=execute_kwargs)
        exec_autograd = cost(x_, y_, dev, interface="autograd", ek=execute_kwargs)

        assert all(
            np.allclose(exec_jax[i][j], exec_autograd[i][j]) for i in range(2) for j in range(2)
        )

        res = jax.jacobian(cost, argnums=(0, 1))(
            x, y, dev, interface="jax-python", ek=execute_kwargs
        )

        assert isinstance(res, TensorLike)
        assert len(res) == 2

        for r, exp_shape in zip(res, [(), (2,)]):
            assert isinstance(r, tuple)
            assert len(r) == 2
            assert len(r[0]) == 2
            assert isinstance(r[0][0], jax.numpy.ndarray)
            assert r[0][0].shape == exp_shape
            assert isinstance(r[0][1], jax.numpy.ndarray)
            assert r[0][1].shape == exp_shape
            assert len(r[1]) == 2
            assert isinstance(r[1][0], jax.numpy.ndarray)
            assert r[1][0].shape == exp_shape
            assert isinstance(r[1][1], jax.numpy.ndarray)
            assert r[1][1].shape == exp_shape

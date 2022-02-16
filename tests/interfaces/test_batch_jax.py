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
"""Unit tests for the jax interface"""
import functools

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
import numpy as np

import pennylane as qml
from pennylane.gradients import param_shift
from pennylane.interfaces.batch import execute
from pennylane.interfaces.batch import InterfaceUnsupportedError


@pytest.mark.parametrize("interface", ["jax-jit", "jax-python"])
class TestJaxExecuteUnitTests:
    """Unit tests for jax execution"""

    def test_jacobian_options(self, mocker, interface, tol):
        """Test setting jacobian options"""
        spy = mocker.spy(qml.gradients, "param_shift")

        a = jnp.array([0.1, 0.2])

        dev = qml.device("default.qubit", wires=1)

        def cost(a, device):
            with qml.tape.JacobianTape() as tape:
                qml.RY(a[0], wires=0)
                qml.RX(a[1], wires=0)
                qml.expval(qml.PauliZ(0))

            return execute(
                [tape],
                device,
                gradient_fn=param_shift,
                gradient_kwargs={"shift": np.pi / 4},
                interface=interface,
            )[0][0]

        res = jax.grad(cost)(a, device=dev)

        for args in spy.call_args_list:
            assert args[1]["shift"] == np.pi / 4

    def test_incorrect_mode(self, interface):
        """Test that an error is raised if an gradient transform
        is used with mode=forward"""
        a = jnp.array([0.1, 0.2])

        dev = qml.device("default.qubit", wires=1)

        def cost(a, device):
            with qml.tape.JacobianTape() as tape:
                qml.RY(a[0], wires=0)
                qml.RX(a[1], wires=0)
                qml.expval(qml.PauliZ(0))

            return execute(
                [tape],
                device,
                gradient_fn=param_shift,
                mode="forward",
                interface=interface,
            )[0]

        with pytest.raises(
            ValueError, match="Gradient transforms cannot be used with mode='forward'"
        ):
            res = jax.grad(cost)(a, device=dev)

    def test_unknown_interface(self, interface):
        """Test that an error is raised if the interface is unknown"""
        a = jnp.array([0.1, 0.2])

        dev = qml.device("default.qubit", wires=1)

        def cost(a, device):
            with qml.tape.JacobianTape() as tape:
                qml.RY(a[0], wires=0)
                qml.RX(a[1], wires=0)
                qml.expval(qml.PauliZ(0))

            return execute(
                [tape],
                device,
                gradient_fn=param_shift,
                interface="None",
            )[0]

        with pytest.raises(ValueError, match="Unknown interface"):
            cost(a, device=dev)

    def test_forward_mode(self, interface, mocker):
        """Test that forward mode uses the `device.execute_and_gradients` pathway"""
        dev = qml.device("default.qubit", wires=1)
        spy = mocker.spy(dev, "execute_and_gradients")

        def cost(a):
            with qml.tape.JacobianTape() as tape:
                qml.RY(a[0], wires=0)
                qml.RX(a[1], wires=0)
                qml.expval(qml.PauliZ(0))

            return execute(
                [tape],
                dev,
                gradient_fn="device",
                interface=interface,
                gradient_kwargs={
                    "method": "adjoint_jacobian",
                    "use_device_state": True,
                },
            )[0]

        a = jnp.array([0.1, 0.2])
        cost(a)

        # adjoint method only performs a single device execution, but gets both result and gradient
        assert dev.num_executions == 1
        spy.assert_called()

    def test_backward_mode(self, interface, mocker):
        """Test that backward mode uses the `device.batch_execute` and `device.gradients` pathway"""
        dev = qml.device("default.qubit", wires=1)
        spy_execute = mocker.spy(qml.devices.DefaultQubit, "batch_execute")
        spy_gradients = mocker.spy(qml.devices.DefaultQubit, "gradients")

        def cost(a):
            with qml.tape.JacobianTape() as tape:
                qml.RY(a[0], wires=0)
                qml.RX(a[1], wires=0)
                qml.expval(qml.PauliZ(0))

            return execute(
                [tape],
                dev,
                gradient_fn="device",
                mode="backward",
                interface=interface,
                gradient_kwargs={"method": "adjoint_jacobian"},
            )[0][0]

        a = jnp.array([0.1, 0.2])
        cost(a)

        assert dev.num_executions == 1
        spy_execute.assert_called()
        spy_gradients.assert_not_called()

        jax.grad(cost)(a)
        spy_gradients.assert_called()

    def test_max_diff_error(self, interface):
        """Test that an error is being raised if max_diff > 1 for the JAX
        interface."""
        a = jnp.array([0.1, 0.2])

        dev = qml.device("default.qubit", wires=1)

        with pytest.raises(
            InterfaceUnsupportedError,
            match="The JAX interface only supports first order derivatives.",
        ):
            with qml.tape.JacobianTape() as tape:
                qml.RY(a[0], wires=0)
                qml.RX(a[1], wires=0)
                qml.expval(qml.PauliZ(0))

            execute(
                [tape],
                dev,
                interface=interface,
                gradient_fn=param_shift,
                gradient_kwargs={"shift": np.pi / 4},
                max_diff=2,
            )


@pytest.mark.parametrize("interface", ["jax-jit", "jax-python"])
class TestCaching:
    """Test for caching behaviour"""

    def test_cache_maxsize(self, interface, mocker):
        """Test the cachesize property of the cache"""
        dev = qml.device("default.qubit", wires=1)
        spy = mocker.spy(qml.interfaces.batch, "cache_execute")

        def cost(a, cachesize):
            with qml.tape.JacobianTape() as tape:
                qml.RY(a[0], wires=0)
                qml.RX(a[1], wires=0)
                qml.expval(qml.PauliZ(0))

            return execute(
                [tape],
                dev,
                gradient_fn=param_shift,
                cachesize=cachesize,
                interface=interface,
            )[0][0]

        params = jnp.array([0.1, 0.2])
        jax.grad(cost)(params, cachesize=2)
        cache = spy.call_args[0][1]

        assert cache.maxsize == 2
        assert cache.currsize == 2
        assert len(cache) == 2

    def test_custom_cache(self, interface, mocker):
        """Test the use of a custom cache object"""
        dev = qml.device("default.qubit", wires=1)
        spy = mocker.spy(qml.interfaces.batch, "cache_execute")

        def cost(a, cache):
            with qml.tape.JacobianTape() as tape:
                qml.RY(a[0], wires=0)
                qml.RX(a[1], wires=0)
                qml.expval(qml.PauliZ(0))

            return execute([tape], dev, gradient_fn=param_shift, cache=cache, interface=interface,)[
                0
            ][0]

        custom_cache = {}
        params = jnp.array([0.1, 0.2])
        jax.grad(cost)(params, cache=custom_cache)

        cache = spy.call_args[0][1]
        assert cache is custom_cache

    def test_custom_cache_multiple(self, interface, mocker):
        """Test the use of a custom cache object with multiple tapes"""
        dev = qml.device("default.qubit", wires=1)
        spy = mocker.spy(qml.interfaces.batch, "cache_execute")

        a = jnp.array(0.1)
        b = jnp.array(0.2)

        def cost(a, b, cache):
            with qml.tape.JacobianTape() as tape1:
                qml.RY(a, wires=0)
                qml.RX(b, wires=0)
                qml.expval(qml.PauliZ(0))

            with qml.tape.JacobianTape() as tape2:
                qml.RY(a, wires=0)
                qml.RX(b, wires=0)
                qml.expval(qml.PauliZ(0))

            res = execute(
                [tape1, tape2],
                dev,
                gradient_fn=param_shift,
                cache=cache,
                interface=interface,
            )
            return res[0][0]

        custom_cache = {}
        jax.grad(cost)(a, b, cache=custom_cache)

        cache = spy.call_args[0][1]
        assert cache is custom_cache

    def test_caching_param_shift(self, interface, tol):
        """Test that, when using parameter-shift transform,
        caching produces the optimum number of evaluations."""
        dev = qml.device("default.qubit", wires=1)

        def cost(a, cache):
            with qml.tape.JacobianTape() as tape:
                qml.RY(a[0], wires=0)
                qml.RX(a[1], wires=0)
                qml.expval(qml.PauliZ(0))

            return execute([tape], dev, gradient_fn=param_shift, cache=cache, interface=interface,)[
                0
            ][0]

        # Without caching, 5 evaluations are required to compute
        # the Jacobian: 1 (forward pass) + 2 (backward pass) * (2 shifts * 2 params)
        params = jnp.array([0.1, 0.2])
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

    def test_caching_adjoint_backward(self, interface):
        """Test that caching produces the optimum number of adjoint evaluations
        when mode=backward"""
        dev = qml.device("default.qubit", wires=2)
        params = jnp.array([0.1, 0.2, 0.3])

        def cost(a, cache):
            with qml.tape.JacobianTape() as tape:
                qml.RY(a[0], wires=0)
                qml.RX(a[1], wires=0)
                qml.RY(a[2], wires=0)
                qml.expval(qml.PauliZ(0))

            return execute(
                [tape],
                dev,
                gradient_fn="device",
                cache=cache,
                mode="backward",
                interface=interface,
                gradient_kwargs={"method": "adjoint_jacobian"},
            )[0][0]

        # Without caching, 2 evaluations are required.
        # 1 for the forward pass, and one per output dimension
        # on the backward pass.
        jax.grad(cost)(params, cache=None)
        assert dev.num_executions == 2

        # With caching, also 2 evaluations are required. One
        # for the forward pass, and one for the backward pass.
        dev._num_executions = 0
        jac_fn = jax.grad(cost)
        grad1 = jac_fn(params, cache=True)
        assert dev.num_executions == 2


execute_kwargs = [
    {"gradient_fn": param_shift},
    {
        "gradient_fn": "device",
        "mode": "forward",
        "gradient_kwargs": {"method": "adjoint_jacobian", "use_device_state": True},
    },
    {
        "gradient_fn": "device",
        "mode": "backward",
        "gradient_kwargs": {"method": "adjoint_jacobian"},
    },
]


@pytest.mark.parametrize("execute_kwargs", execute_kwargs)
@pytest.mark.parametrize("interface", ["jax-jit", "jax-python"])
class TestJaxExecuteIntegration:
    """Test the jax interface execute function
    integrates well for both forward and backward execution"""

    def test_execution(self, execute_kwargs, interface):
        """Test execution"""
        dev = qml.device("default.qubit", wires=1)

        def cost(a, b):
            with qml.tape.JacobianTape() as tape1:
                qml.RY(a, wires=0)
                qml.RX(b, wires=0)
                qml.expval(qml.PauliZ(0))

            with qml.tape.JacobianTape() as tape2:
                qml.RY(a, wires=0)
                qml.RX(b, wires=0)
                qml.expval(qml.PauliZ(0))

            return execute([tape1, tape2], dev, interface=interface, **execute_kwargs)

        a = jnp.array(0.1)
        b = jnp.array(0.2)
        res = cost(a, b)

        assert len(res) == 2
        assert res[0].shape == (1,)
        assert res[1].shape == (1,)

    def test_scalar_jacobian(self, execute_kwargs, interface, tol):
        """Test scalar jacobian calculation"""
        a = jnp.array(0.1)
        dev = qml.device("default.qubit", wires=2)

        def cost(a):
            with qml.tape.JacobianTape() as tape:
                qml.RY(a, wires=0)
                qml.expval(qml.PauliZ(0))
            return execute([tape], dev, interface=interface, **execute_kwargs)[0][0]

        res = jax.grad(cost)(a)
        assert res.shape == ()

        # compare to standard tape jacobian
        with qml.tape.JacobianTape() as tape:
            qml.RY(a, wires=0)
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = [0]
        tapes, fn = param_shift(tape)
        expected = fn(dev.batch_execute(tapes))

        assert expected.shape == (1, 1)
        assert np.allclose(res, np.squeeze(expected), atol=tol, rtol=0)

    def test_reusing_quantum_tape(self, execute_kwargs, interface, tol):
        """Test re-using a quantum tape by passing new parameters"""
        a = jnp.array(0.1)
        b = jnp.array(0.2)

        dev = qml.device("default.qubit", wires=2)

        with qml.tape.JacobianTape() as tape:
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        assert tape.trainable_params == [0, 1]

        def cost(a, b):

            # An explicit call to _update() is required here to update the
            # trainable parameters in between tape executions.
            # This is different from how the autograd interface works.
            # Unless the update is issued, the validation check related to the
            # number of provided parameters fails in the tape: (len(params) !=
            # required_length) and the tape produces incorrect results.
            tape._update()
            tape.set_parameters([a, b])
            return execute([tape], dev, interface=interface, **execute_kwargs)[0][0]

        jac_fn = jax.grad(cost)
        jac = jac_fn(a, b)

        a = jnp.array(0.54)
        b = jnp.array(0.8)

        # check that the cost function continues to depend on the
        # values of the parameters for subsequent calls
        res2 = cost(2 * a, b)
        expected = [np.cos(2 * a)]
        assert np.allclose(res2, expected, atol=tol, rtol=0)

        jac_fn = jax.grad(lambda a, b: cost(2 * a, b))
        jac = jac_fn(a, b)
        expected = -2 * np.sin(2 * a)
        assert np.allclose(jac, expected, atol=tol, rtol=0)

    def test_classical_processing_single_tape(self, execute_kwargs, interface, tol):
        """Test classical processing within the quantum tape for a single tape"""
        a = jnp.array(0.1)
        b = jnp.array(0.2)
        c = jnp.array(0.3)

        def cost(a, b, c, device):
            with qml.tape.JacobianTape() as tape:
                qml.RY(a * c, wires=0)
                qml.RZ(b, wires=0)
                qml.RX(c + c**2 + jnp.sin(a), wires=0)
                qml.expval(qml.PauliZ(0))

            return execute([tape], device, interface=interface, **execute_kwargs)[0][0]

        dev = qml.device("default.qubit", wires=2)
        res = jax.grad(cost, argnums=(0, 1, 2))(a, b, c, device=dev)
        assert len(res) == 3

    def test_classical_processing_multiple_tapes(self, execute_kwargs, interface, tol):
        """Test classical processing within the quantum tape for multiple
        tapes"""
        dev = qml.device("default.qubit", wires=2)
        params = jax.numpy.array([0.3, 0.2])

        def cost_fn(x):
            with qml.tape.JacobianTape() as tape1:
                qml.Hadamard(0)
                qml.RY(x[0], wires=[0])
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0))

            with qml.tape.JacobianTape() as tape2:
                qml.Hadamard(0)
                qml.CRX(2 * x[0] * x[1], wires=[0, 1])
                qml.RX(2 * x[1], wires=[1])
                qml.expval(qml.PauliZ(0))

            result = execute(
                tapes=[tape1, tape2], device=dev, interface=interface, **execute_kwargs
            )
            return (result[0] + result[1] - 7 * result[1])[0]

        res = jax.grad(cost_fn)(params)
        assert res.shape == (2,)

    def test_multiple_tapes_output(self, execute_kwargs, interface, tol):
        """Test the output types for the execution of multiple quantum tapes"""
        dev = qml.device("default.qubit", wires=2)
        params = jax.numpy.array([0.3, 0.2])

        def cost_fn(x):
            with qml.tape.JacobianTape() as tape1:
                qml.Hadamard(0)
                qml.RY(x[0], wires=[0])
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0))

            with qml.tape.JacobianTape() as tape2:
                qml.Hadamard(0)
                qml.CRX(2 * x[0] * x[1], wires=[0, 1])
                qml.RX(2 * x[1], wires=[1])
                qml.expval(qml.PauliZ(0))

            return execute(tapes=[tape1, tape2], device=dev, interface=interface, **execute_kwargs)

        res = cost_fn(params)
        assert isinstance(res, list)
        assert all(isinstance(r, jnp.ndarray) for r in res)
        assert all(r.shape == (1,) for r in res)

    @pytest.mark.xfail
    def test_matrix_parameter(self, execute_kwargs, interface, tol):
        """Test that the jax interface works correctly
        with a matrix parameter"""
        a = jnp.array(0.1)
        U = qml.RY(a, wires=0).get_matrix()

        def cost(U, device):
            with qml.tape.JacobianTape() as tape:
                qml.PauliX(0)
                qml.QubitUnitary(U, wires=0)
                qml.expval(qml.PauliZ(0))

            tape.trainable_params = [0]
            return execute([tape], device, interface=interface, **execute_kwargs)[0][0]

        dev = qml.device("default.qubit", wires=2)
        res = cost(U, device=dev)
        assert np.allclose(res, -np.cos(a), atol=tol, rtol=0)

        jac_fn = jax.grad(cost, argnums=(0))
        res = jac_fn(U, device=dev)
        assert np.allclose(res, np.sin(a), atol=tol, rtol=0)

    def test_differentiable_expand(self, execute_kwargs, interface, tol):
        """Test that operation and nested tapes expansion
        is differentiable"""

        class U3(qml.U3):
            def expand(self):
                tape = qml.tape.JacobianTape()
                theta, phi, lam = self.data
                wires = self.wires
                tape._ops += [
                    qml.Rot(lam, theta, -lam, wires=wires),
                    qml.PhaseShift(phi + lam, wires=wires),
                ]
                return tape

        def cost_fn(a, p, device):
            tape = qml.tape.JacobianTape()

            with tape:
                qml.RX(a, wires=0)
                U3(*p, wires=0)
                qml.expval(qml.PauliX(0))

            tape = tape.expand(stop_at=lambda obj: device.supports_operation(obj.name))
            return execute([tape], device, interface=interface, **execute_kwargs)[0][0]

        a = jnp.array(0.1)
        p = jnp.array([0.1, 0.2, 0.3])

        dev = qml.device("default.qubit", wires=1)
        res = cost_fn(a, p, device=dev)
        expected = np.cos(a) * np.cos(p[1]) * np.sin(p[0]) + np.sin(a) * (
            np.cos(p[2]) * np.sin(p[1]) + np.cos(p[0]) * np.cos(p[1]) * np.sin(p[2])
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

        jac_fn = jax.grad(cost_fn, argnums=(1))
        res = jac_fn(a, p, device=dev)
        expected = jnp.array(
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

    def test_independent_expval(self, execute_kwargs, interface):
        """Tests computing an expectation value that is independent of trainable
        parameters."""
        dev = qml.device("default.qubit", wires=2)
        params = jnp.array([0.1, 0.2, 0.3])

        def cost(a, cache):
            with qml.tape.JacobianTape() as tape:
                qml.RY(a[0], wires=0)
                qml.RX(a[1], wires=0)
                qml.RY(a[2], wires=0)
                qml.expval(qml.PauliZ(1))

            res = qml.interfaces.batch.execute(
                [tape], dev, cache=cache, interface=interface, **execute_kwargs
            )
            return res[0][0]

        res = jax.grad(cost)(params, cache=None)
        assert res.shape == (3,)

    @pytest.mark.parametrize(
        "ret, mes",
        [
            ([qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))], "single return type"),
            ([qml.state()], "Only Variance and Expectation"),
        ],
    )
    def test_raises_for_jax_jit(self, execute_kwargs, interface, ret, mes):
        """Tests multiple measurements and unsupported measurements raise an
        error for the jit JAX interface."""
        dev = qml.device("default.qubit", wires=2)
        params = jnp.array([0.1, 0.2, 0.3])

        def cost(a, cache):
            with qml.tape.JacobianTape() as tape:
                qml.RY(a[0], wires=0)
                qml.RX(a[1], wires=0)
                qml.RY(a[2], wires=0)
                [qml.apply(r) for r in ret]

            res = qml.interfaces.batch.execute(
                # Test only applicable for the jax jit interface
                [tape],
                dev,
                cache=cache,
                interface="jax-jit",
                **execute_kwargs
            )
            return res[0][0]

        with pytest.raises(InterfaceUnsupportedError, match=mes):
            cost(params, cache=None)


@pytest.mark.parametrize("execute_kwargs", execute_kwargs)
class TestVectorValued:
    """Test vector-valued returns for the JAX Python interface."""

    def test_multiple_expvals(self, execute_kwargs):
        """Tests computing multiple expectation values in a tape."""
        fwd_mode = execute_kwargs.get("mode", "not forward") == "forward"
        if fwd_mode:
            pytest.skip("The forward mode is tested separately as it should raise an error.")

        dev = qml.device("default.qubit", wires=2)
        params = jnp.array([0.1, 0.2, 0.3])

        def cost(a, cache):
            with qml.tape.JacobianTape() as tape:
                qml.RY(a[0], wires=0)
                qml.RX(a[1], wires=0)
                qml.RY(a[2], wires=0)
                qml.expval(qml.PauliZ(0))
                qml.expval(qml.PauliZ(1))

            res = qml.interfaces.batch.execute(
                [tape], dev, cache=cache, interface="jax-python", **execute_kwargs
            )
            return res[0]

        res = jax.jacobian(cost)(params, cache=None)
        assert res.shape == (2, 3)

    def test_multiple_expvals_single_par(self, execute_kwargs):
        """Tests computing multiple expectation values in a tape with a single
        trainable parameter."""
        fwd_mode = execute_kwargs.get("mode", "not forward") == "forward"
        if fwd_mode:
            pytest.skip("The forward mode is tested separately as it should raise an error.")

        dev = qml.device("default.qubit", wires=2)
        params = jnp.array([0.1])

        def cost(a, cache):
            with qml.tape.JacobianTape() as tape:
                qml.RY(a[0], wires=0)
                qml.expval(qml.PauliZ(0))
                qml.expval(qml.PauliZ(1))

            res = qml.interfaces.batch.execute(
                [tape], dev, cache=cache, interface="jax-python", **execute_kwargs
            )
            return res[0]

        res = jax.jacobian(cost)(params, cache=None)
        assert res.shape == (2, 1)

    def test_multi_tape_jacobian(self, execute_kwargs):
        """Test the jacobian computation with multiple tapes."""
        fwd_mode = execute_kwargs.get("mode", "not forward") == "forward"
        if fwd_mode:
            pytest.skip("The forward mode is tested separately as it should raise an error.")

        def cost(x, y, device, interface, ek):
            with qml.tape.JacobianTape() as tape1:
                qml.RX(x, wires=[0])
                qml.RY(y, wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0))
                qml.expval(qml.PauliZ(1))

            with qml.tape.JacobianTape() as tape2:
                qml.RX(x, wires=[0])
                qml.RY(y, wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0))
                qml.expval(qml.PauliZ(1))

            return qml.execute([tape1, tape2], device, **ek, interface=interface)[0]

        dev = qml.device("default.qubit", wires=2)
        x = jnp.array(0.543)
        y = jnp.array(-0.654)

        x_ = np.array(0.543)
        y_ = np.array(-0.654)

        res = jax.jacobian(cost, argnums=(0, 1))(
            x, y, dev, interface="jax-python", ek=execute_kwargs
        )

        exp = qml.jacobian(cost, argnum=(0, 1))(
            x_, y_, dev, interface="autograd", ek=execute_kwargs
        )
        for r, e in zip(res, exp):
            assert jnp.allclose(r, e, atol=1e-7)

    def test_multi_tape_jacobian_probs_expvals(self, execute_kwargs):
        """Test the jacobian computation with multiple tapes with probability
        and expectation value computations."""
        fwd_mode = execute_kwargs.get("mode", "not forward") == "forward"
        if fwd_mode:
            pytest.skip("The forward mode is tested separately as it should raise an error.")

        adjoint = execute_kwargs.get("gradient_kwargs", {}).get("method", "") == "adjoint_jacobian"
        if adjoint:
            pytest.skip("The adjoint diff method doesn't support probabilities.")

        def cost(x, y, device, interface, ek):
            with qml.tape.JacobianTape() as tape1:
                qml.RX(x, wires=[0])
                qml.RY(y, wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0))
                qml.expval(qml.PauliZ(1))

            with qml.tape.JacobianTape() as tape2:
                qml.RX(x, wires=[0])
                qml.RY(y, wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.probs(wires=[0])
                qml.probs(wires=[1])

            return qml.execute([tape1, tape2], device, **ek, interface=interface)[0]

        dev = qml.device("default.qubit", wires=2)
        x = jnp.array(0.543)
        y = jnp.array(-0.654)

        x_ = np.array(0.543)
        y_ = np.array(-0.654)

        res = jax.jacobian(cost, argnums=(0, 1))(
            x, y, dev, interface="jax-python", ek=execute_kwargs
        )

        exp = qml.jacobian(cost, argnum=(0, 1))(
            x_, y_, dev, interface="autograd", ek=execute_kwargs
        )
        for r, e in zip(res, exp):
            assert jnp.allclose(r, e, atol=1e-7)

    def test_multiple_expvals_raises_fwd_device_grad(self, execute_kwargs):
        """Tests computing multiple expectation values in a tape."""
        fwd_mode = execute_kwargs.get("mode", "not forward") == "forward"
        if not fwd_mode:
            pytest.skip("Forward mode is not turned on.")

        dev = qml.device("default.qubit", wires=2)
        params = jnp.array([0.1, 0.2, 0.3])

        def cost(a, cache):
            with qml.tape.JacobianTape() as tape:
                qml.RY(a[0], wires=0)
                qml.RX(a[1], wires=0)
                qml.RY(a[2], wires=0)
                qml.expval(qml.PauliZ(0))
                qml.expval(qml.PauliZ(1))

            res = qml.interfaces.batch.execute(
                [tape], dev, cache=cache, interface="jax-python", **execute_kwargs
            )
            return res[0]

        with pytest.raises(InterfaceUnsupportedError):
            jax.jacobian(cost)(params, cache=None)


def test_diff_method_None_jit():
    """Test that jitted execution works when `gradient_fn=None`."""

    dev = qml.device("default.qubit.jax", wires=1, shots=10)

    @jax.jit
    def wrapper(x):
        with qml.tape.QuantumTape() as tape:
            qml.RX(x, wires=0)
            qml.expval(qml.PauliZ(0))

        return qml.execute([tape], dev, gradient_fn=None)

    assert jnp.allclose(wrapper(jnp.array(0.0))[0], 1.0)

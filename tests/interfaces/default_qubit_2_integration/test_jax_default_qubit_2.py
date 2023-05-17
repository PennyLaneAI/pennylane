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
"""Jax specific tests for execute and default qubit 2."""
# pylint: disable=invalid-sequence-index
import pytest
import numpy as np


import pennylane as qml
from pennylane.devices.experimental import DefaultQubit2
from pennylane.gradients import param_shift
from pennylane.interfaces import execute

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
config = pytest.importorskip("jax.config")
config.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.jax


# pylint: disable=too-few-public-methods
class TestCaching:
    """Tests for caching behaviour"""

    @pytest.mark.skip("caching is not implemented for jax")
    @pytest.mark.parametrize("num_params", [2, 3])
    def test_caching_param_shift_hessian(self, num_params):
        """Test that, when using parameter-shift transform,
        caching reduces the number of evaluations to their optimum
        when computing Hessians."""
        dev = DefaultQubit2()
        params = jnp.arange(1, num_params + 1) / 10

        N = len(params)

        def cost(x, cache):
            with qml.queuing.AnnotatedQueue() as q:
                qml.RX(x[0], wires=[0])
                qml.RY(x[1], wires=[1])

                for i in range(2, num_params):
                    qml.RZ(x[i], wires=[i % 2])

                qml.CNOT(wires=[0, 1])
                qml.var(qml.prod(qml.PauliZ(0), qml.PauliX(1)))

            tape = qml.tape.QuantumScript.from_queue(q)
            return qml.execute(
                [tape], dev, gradient_fn=qml.gradients.param_shift, cache=cache, max_diff=2
            )[0]

        # No caching: number of executions is not ideal
        with qml.Tracker(dev) as tracker:
            hess1 = jax.jacobian(jax.grad(cost))(params, cache=False)

        if num_params == 2:
            # compare to theoretical result
            x, y, *_ = params
            expected = jnp.array(
                [
                    [2 * jnp.cos(2 * x) * jnp.sin(y) ** 2, jnp.sin(2 * x) * jnp.sin(2 * y)],
                    [jnp.sin(2 * x) * jnp.sin(2 * y), -2 * jnp.cos(x) ** 2 * jnp.cos(2 * y)],
                ]
            )
            assert np.allclose(expected, hess1)

        expected_runs = 1  # forward pass

        # Jacobian of an involutory observable:
        # ------------------------------------
        #
        # 2 * N execs: evaluate the analytic derivative of <A>
        # 1 execs: Get <A>, the expectation value of the tape with unshifted parameters.
        num_shifted_evals = 2 * N
        runs_for_jacobian = num_shifted_evals + 1
        expected_runs += runs_for_jacobian

        # Each tape used to compute the Jacobian is then shifted again
        expected_runs += runs_for_jacobian * num_shifted_evals
        assert tracker.totals["executions"] == expected_runs

        # Use caching: number of executions is ideal

        with qml.Tracker(dev) as tracker2:
            hess2 = jax.jacobian(jax.grad(cost))(params, cache=True)
        assert np.allclose(hess1, hess2)

        expected_runs_ideal = 1  # forward pass
        expected_runs_ideal += 2 * N  # Jacobian
        expected_runs_ideal += N + 1  # Hessian diagonal
        expected_runs_ideal += 4 * N * (N - 1) // 2  # Hessian off-diagonal
        assert tracker2.totals["executions"] == expected_runs_ideal
        assert expected_runs_ideal < expected_runs


execute_kwargs_iterable = [
    {"gradient_fn": param_shift},
    {"gradient_fn": "backprop"},
    # no device gradient yet
]


@pytest.mark.parametrize("execute_kwargs", execute_kwargs_iterable)
class TestJaxExecuteIntegration:
    """Test the jax interface execute function
    integrates well for both forward and backward execution"""

    def test_execution(self, execute_kwargs):
        """Test execution"""
        dev = DefaultQubit2()

        def cost(a, b):
            ops1 = [qml.RY(a, wires=0), qml.RX(b, wires=0)]
            tape1 = qml.tape.QuantumScript(ops1, [qml.expval(qml.PauliZ(0))])

            ops2 = [qml.RY(a, wires="a"), qml.RX(b, wires="a")]
            tape2 = qml.tape.QuantumScript(ops2, [qml.expval(qml.PauliZ("a"))])

            return execute([tape1, tape2], dev, **execute_kwargs)

        a = jnp.array(0.1)
        b = np.array(0.2)
        with dev.tracker:
            res = cost(a, b)

        assert dev.tracker.totals["batches"] == 1
        assert dev.tracker.totals["executions"] == 2  # different wires so different hashes

        assert len(res) == 2
        assert res[0].shape == ()
        assert res[1].shape == ()

        assert qml.math.allclose(res[0], jnp.cos(a) * jnp.cos(b))
        assert qml.math.allclose(res[1], jnp.cos(a) * jnp.cos(b))

    def test_scalar_jacobian(self, execute_kwargs, tol):
        """Test scalar jacobian calculation"""
        a = jnp.array(0.1)
        dev = DefaultQubit2()

        def cost(a):
            tape = qml.tape.QuantumScript([qml.RY(a, 0)], [qml.expval(qml.PauliZ(0))])
            return execute([tape], dev, **execute_kwargs)[0]

        res = jax.jacobian(cost)(a)
        assert res.shape == ()  # pylint: disable=no-member

        # compare to standard tape jacobian
        tape = qml.tape.QuantumScript([qml.RY(a, wires=0)], [qml.expval(qml.PauliZ(0))])
        tape.trainable_params = [0]
        tapes, fn = param_shift(tape)
        expected = fn(dev.execute(tapes))

        assert expected.shape == ()
        assert np.allclose(res, expected, atol=tol, rtol=0)
        assert np.allclose(res, -jnp.sin(a))

    def test_jacobian(self, execute_kwargs, tol):
        """Test jacobian calculation"""
        a = jnp.array(0.1)
        b = jnp.array(0.2)

        def cost(a, b, device):
            ops = [qml.RY(a, wires=0), qml.RX(b, wires=1), qml.CNOT(wires=[0, 1])]
            m = [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))]
            tape = qml.tape.QuantumScript(ops, m)
            return jnp.hstack(execute([tape], device, **execute_kwargs)[0])

        dev = DefaultQubit2()

        res = cost(a, b, device=dev)
        expected = [jnp.cos(a), -jnp.cos(a) * jnp.sin(b)]
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = jax.jacobian(cost, argnums=[0, 1])(a, b, device=dev)
        assert isinstance(res, tuple) and len(res) == 2
        assert res[0].shape == (2,)
        assert res[1].shape == (2,)

        expected = ([-jnp.sin(a), jnp.sin(a) * jnp.sin(b)], [0, -jnp.cos(a) * jnp.cos(b)])
        assert all(np.allclose(_r, _e, atol=tol, rtol=0) for _r, _e in zip(res, expected))

    def test_tape_no_parameters(self, execute_kwargs, tol):
        """Test that a tape with no parameters is correctly
        ignored during the gradient computation"""

        if execute_kwargs["gradient_fn"] == "device":
            pytest.skip("Adjoint differentiation does not yet support probabilities")

        dev = DefaultQubit2()

        def cost(params):
            tape1 = qml.tape.QuantumScript([qml.Hadamard(0)], [qml.expval(qml.PauliX(0))])

            tape2 = qml.tape.QuantumScript(
                [qml.RY(jnp.array(0.5), wires=0)], [qml.expval(qml.PauliZ(0))]
            )

            tape3 = qml.tape.QuantumScript(
                [qml.RY(params[0], 0), qml.RX(params[1], 0)], [qml.expval(qml.PauliZ(0))]
            )

            tape4 = qml.tape.QuantumScript([qml.RY(jnp.array(0.5), 0)], [qml.probs(wires=[0, 1])])
            return sum(jnp.hstack(execute([tape1, tape2, tape3, tape4], dev, **execute_kwargs)))

        params = jnp.array([0.1, 0.2])
        x, y = params

        res = cost(params)
        expected = 2 + jnp.cos(0.5) + jnp.cos(x) * jnp.cos(y)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # TODO: jax does not allow computing tapes with different gradient
        # shapes like [None, None, *, None] here

        # grad = jax.grad(cost)(params)
        # expected = [-jnp.cos(y) * jnp.sin(x), -jnp.cos(x) * jnp.sin(y)]
        # assert np.allclose(grad, expected, atol=tol, rtol=0)

    def test_tapes_with_different_return_size(self, execute_kwargs):
        """Test that tapes wit different can be executed and differentiated."""

        # TODO: Will probably fail if we update jax to fix this test
        # if execute_kwargs["gradient_fn"] == "backprop":
        #     pytest.xfail("backprop is not compatible with something about this situation.")

        dev = DefaultQubit2()

        def cost(params):
            tape1 = qml.tape.QuantumScript(
                [qml.RY(params[0], 0), qml.RX(params[1], 0)],
                [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))],
            )

            tape2 = qml.tape.QuantumScript([qml.RY(np.array(0.5), 0)], [qml.expval(qml.PauliZ(0))])

            tape3 = qml.tape.QuantumScript(
                [qml.RY(params[0], 0), qml.RX(params[1], 0)], [qml.expval(qml.PauliZ(0))]
            )
            return jnp.hstack(execute([tape1, tape2, tape3], dev, **execute_kwargs))

        params = jnp.array([0.1, 0.2])
        x, y = params

        res = cost(params)
        assert isinstance(res, jax.Array)
        assert res.shape == (4,)

        assert np.allclose(res[0], jnp.cos(x) * jnp.cos(y))
        assert np.allclose(res[1], 1)
        assert np.allclose(res[2], jnp.cos(0.5))
        assert np.allclose(res[3], jnp.cos(x) * jnp.cos(y))

        # TODO: jax does not allow computing tapes with different gradient shapes

        # jac = jax.jacobian(cost)(params)
        # assert isinstance(jac, jnp.ndarray)
        # assert jac.shape == (4, 2)  # pylint: disable=no-member

        # assert np.allclose(jac[1:3], 0)

        # d1 = -jnp.sin(x) * jnp.cos(y)
        # assert np.allclose(jac[0, 0], d1)
        # assert np.allclose(jac[3, 0], d1)

        # d2 = -jnp.cos(x) * jnp.sin(y)
        # assert np.allclose(jac[0, 1], d2)
        # assert np.allclose(jac[3, 1], d2)

    def test_reusing_quantum_tape(self, execute_kwargs, tol):
        """Test re-using a quantum tape by passing new parameters"""
        if execute_kwargs["gradient_fn"] == param_shift:
            pytest.skip("Basic QNode execution wipes out trainable params with param-shift")

        a = jnp.array(0.1)
        b = jnp.array(0.2)

        dev = DefaultQubit2()

        tape = qml.tape.QuantumScript(
            [qml.RY(a, 0), qml.RX(b, 1), qml.CNOT((0, 1))],
            [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))],
        )
        assert tape.trainable_params == [0, 1]

        def cost(a, b):
            tape.set_parameters([a, b])
            return jnp.hstack(execute([tape], dev, **execute_kwargs)[0])

        jac_fn = jax.jacobian(cost, argnums=[0, 1])
        jac = jac_fn(a, b)

        a = jnp.array(0.54)
        b = jnp.array(0.8)

        # check that the cost function continues to depend on the
        # values of the parameters for subsequent calls
        res2 = cost(2 * a, b)
        expected = [jnp.cos(2 * a), -jnp.cos(2 * a) * jnp.sin(b)]
        assert np.allclose(res2, expected, atol=tol, rtol=0)

        jac_fn = jax.jacobian(lambda a, b: cost(2 * a, b), argnums=[0, 1])
        jac = jac_fn(a, b)
        expected = (
            [-2 * jnp.sin(2 * a), 2 * jnp.sin(2 * a) * jnp.sin(b)],
            [0, -jnp.cos(2 * a) * jnp.cos(b)],
        )
        assert isinstance(jac, tuple) and len(jac) == 2
        assert all(np.allclose(_j, _e, atol=tol, rtol=0) for _j, _e in zip(jac, expected))

    def test_classical_processing(self, execute_kwargs):
        """Test classical processing within the quantum tape"""
        a = jnp.array(0.1)
        b = jnp.array(0.2)
        c = jnp.array(0.3)

        def cost(a, b, c, device):
            ops = [
                qml.RY(a * c, wires=0),
                qml.RZ(b, wires=0),
                qml.RX(c + c**2 + jnp.sin(a), wires=0),
            ]

            tape = qml.tape.QuantumScript(ops, [qml.expval(qml.PauliZ(0))])
            return execute([tape], device, **execute_kwargs)[0]

        dev = DefaultQubit2()
        res = jax.jacobian(cost, argnums=[0, 2])(a, b, c, device=dev)

        # Only two arguments are trainable
        assert isinstance(res, tuple) and len(res) == 2
        assert res[0].shape == ()
        assert res[1].shape == ()

        # I tried getting analytic results for this circuit but I kept being wrong and am giving up

    def test_matrix_parameter(self, execute_kwargs, tol):
        """Test that the jax interface works correctly
        with a matrix parameter"""
        U = jnp.array([[0, 1], [1, 0]])
        a = jnp.array(0.1)

        def cost(a, U, device):
            ops = [qml.QubitUnitary(U, wires=0), qml.RY(a, wires=0)]
            tape = qml.tape.QuantumScript(ops, [qml.expval(qml.PauliZ(0))])
            return execute([tape], device, **execute_kwargs)[0]

        dev = DefaultQubit2()
        res = cost(a, U, device=dev)
        assert np.allclose(res, -jnp.cos(a), atol=tol, rtol=0)

        jac_fn = jax.jacobian(cost)
        jac = jac_fn(a, U, device=dev)
        assert isinstance(jac, jnp.ndarray)
        assert np.allclose(jac, jnp.sin(a), atol=tol, rtol=0)

    def test_differentiable_expand(self, execute_kwargs, tol):
        """Test that operation and nested tapes expansion
        is differentiable"""

        class U3(qml.U3):
            """Dummy operator."""

            def expand(self):
                theta, phi, lam = self.data
                wires = self.wires
                return qml.tape.QuantumScript(
                    [
                        qml.Rot(lam, theta, -lam, wires=wires),
                        qml.PhaseShift(phi + lam, wires=wires),
                    ]
                )

        def cost_fn(a, p, device):
            tape = qml.tape.QuantumScript(
                [qml.RX(a, wires=0), U3(*p, wires=0)], [qml.expval(qml.PauliX(0))]
            )
            return execute([tape], device, **execute_kwargs)[0]

        a = jnp.array(0.1)
        p = jnp.array([0.1, 0.2, 0.3])

        dev = DefaultQubit2()
        res = cost_fn(a, p, device=dev)
        expected = jnp.cos(a) * jnp.cos(p[1]) * jnp.sin(p[0]) + jnp.sin(a) * (
            jnp.cos(p[2]) * jnp.sin(p[1]) + jnp.cos(p[0]) * jnp.cos(p[1]) * jnp.sin(p[2])
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

        jac_fn = jax.jacobian(cost_fn, argnums=[1])
        res = jac_fn(a, p, device=dev)
        expected = jnp.array(
            [
                jnp.cos(p[1])
                * (jnp.cos(a) * jnp.cos(p[0]) - jnp.sin(a) * jnp.sin(p[0]) * jnp.sin(p[2])),
                jnp.cos(p[1]) * jnp.cos(p[2]) * jnp.sin(a)
                - jnp.sin(p[1])
                * (jnp.cos(a) * jnp.sin(p[0]) + jnp.cos(p[0]) * jnp.sin(a) * jnp.sin(p[2])),
                jnp.sin(a)
                * (jnp.cos(p[0]) * jnp.cos(p[1]) * jnp.cos(p[2]) - jnp.sin(p[1]) * jnp.sin(p[2])),
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_probability_differentiation(self, execute_kwargs, tol):
        """Tests correct output shape and evaluation for a tape
        with prob outputs"""

        def cost(x, y, device):
            ops = [qml.RX(x, 0), qml.RY(y, 1), qml.CNOT((0, 1))]
            m = [qml.probs(wires=0), qml.probs(wires=1)]
            tape = qml.tape.QuantumScript(ops, m)
            return jnp.hstack(execute([tape], device, **execute_kwargs)[0])

        dev = DefaultQubit2()
        x = jnp.array(0.543)
        y = jnp.array(-0.654)

        res = cost(x, y, device=dev)
        expected = jnp.array(
            [
                [
                    jnp.cos(x / 2) ** 2,
                    jnp.sin(x / 2) ** 2,
                    (1 + jnp.cos(x) * jnp.cos(y)) / 2,
                    (1 - jnp.cos(x) * jnp.cos(y)) / 2,
                ],
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

        jac_fn = jax.jacobian(cost, argnums=[0, 1])
        res = jac_fn(x, y, device=dev)
        assert isinstance(res, tuple) and len(res) == 2
        assert res[0].shape == (4,)
        assert res[1].shape == (4,)

        expected = (
            jnp.array(
                [
                    [
                        -jnp.sin(x) / 2,
                        jnp.sin(x) / 2,
                        -jnp.sin(x) * jnp.cos(y) / 2,
                        jnp.sin(x) * jnp.cos(y) / 2,
                    ],
                ]
            ),
            jnp.array(
                [
                    [0, 0, -jnp.cos(x) * jnp.sin(y) / 2, jnp.cos(x) * jnp.sin(y) / 2],
                ]
            ),
        )

        assert np.allclose(res[0], expected[0], atol=tol, rtol=0)
        assert np.allclose(res[1], expected[1], atol=tol, rtol=0)

    def test_ragged_differentiation(self, execute_kwargs, tol):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""
        if execute_kwargs["gradient_fn"] == "device":
            pytest.skip("Adjoint differentiation does not yet support probabilities")

        def cost(x, y, device):
            ops = [qml.RX(x, wires=0), qml.RY(y, 1), qml.CNOT((0, 1))]
            m = [qml.expval(qml.PauliZ(0)), qml.probs(wires=1)]
            tape = qml.tape.QuantumScript(ops, m)
            return jnp.hstack(execute([tape], device, **execute_kwargs)[0])

        dev = DefaultQubit2()
        x = jnp.array(0.543)
        y = jnp.array(-0.654)

        res = cost(x, y, device=dev)
        expected = jnp.array(
            [jnp.cos(x), (1 + jnp.cos(x) * jnp.cos(y)) / 2, (1 - jnp.cos(x) * jnp.cos(y)) / 2]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

        jac_fn = jax.jacobian(cost, argnums=[0, 1])
        res = jac_fn(x, y, device=dev)
        assert isinstance(res, tuple) and len(res) == 2
        assert res[0].shape == (3,)
        assert res[1].shape == (3,)

        expected = (
            jnp.array([-jnp.sin(x), -jnp.sin(x) * jnp.cos(y) / 2, jnp.sin(x) * jnp.cos(y) / 2]),
            jnp.array([0, -jnp.cos(x) * jnp.sin(y) / 2, jnp.cos(x) * jnp.sin(y) / 2]),
        )
        assert np.allclose(res[0], expected[0], atol=tol, rtol=0)
        assert np.allclose(res[1], expected[1], atol=tol, rtol=0)


class TestHigherOrderDerivatives:
    """Test that the jax execute function can be differentiated"""

    @pytest.mark.parametrize(
        "params",
        [
            jnp.array([0.543, -0.654]),
            jnp.array([0, -0.654]),
            jnp.array([-2.0, 0]),
        ],
    )
    def test_parameter_shift_hessian(self, params, tol):
        """Tests that the output of the parameter-shift transform
        can be differentiated using jax, yielding second derivatives."""
        dev = DefaultQubit2()

        def cost_fn(x):
            ops1 = [qml.RX(x[0], 0), qml.RY(x[1], 1), qml.CNOT((0, 1))]
            tape1 = qml.tape.QuantumScript(ops1, [qml.var(qml.PauliZ(0) @ qml.PauliX(1))])

            ops2 = [qml.RX(x[0], 0), qml.RY(x[0], 1), qml.CNOT((0, 1))]
            tape2 = qml.tape.QuantumScript(ops2, [qml.probs(wires=1)])
            result = execute([tape1, tape2], dev, gradient_fn=param_shift, max_diff=2)
            return result[0] + result[1][0]

        res = cost_fn(params)
        x, y = params
        expected = 0.5 * (3 + jnp.cos(x) ** 2 * jnp.cos(2 * y))
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = jax.grad(cost_fn)(params)
        expected = jnp.array(
            [-jnp.cos(x) * jnp.cos(2 * y) * jnp.sin(x), -jnp.cos(x) ** 2 * jnp.sin(2 * y)]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = jax.jacobian(jax.grad(cost_fn))(params)
        expected = jnp.array(
            [
                [-jnp.cos(2 * x) * jnp.cos(2 * y), jnp.sin(2 * x) * jnp.sin(2 * y)],
                [jnp.sin(2 * x) * jnp.sin(2 * y), -2 * jnp.cos(x) ** 2 * jnp.cos(2 * y)],
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_max_diff(self, tol):
        """Test that setting the max_diff parameter blocks higher-order
        derivatives"""
        dev = DefaultQubit2()
        params = jnp.array([0.543, -0.654])

        def cost_fn(x):
            ops = [qml.RX(x[0], 0), qml.RY(x[1], 1), qml.CNOT((0, 1))]
            tape1 = qml.tape.QuantumScript(ops, [qml.var(qml.PauliZ(0) @ qml.PauliX(1))])

            ops2 = [qml.RX(x[0], 0), qml.RY(x[0], 1), qml.CNOT((0, 1))]
            tape2 = qml.tape.QuantumScript(ops2, [qml.probs(wires=1)])

            result = execute([tape1, tape2], dev, gradient_fn=param_shift, max_diff=1)
            return result[0] + result[1][0]

        res = cost_fn(params)
        x, y = params
        expected = 0.5 * (3 + jnp.cos(x) ** 2 * jnp.cos(2 * y))
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = jax.grad(cost_fn)(params)
        expected = jnp.array(
            [-jnp.cos(x) * jnp.cos(2 * y) * jnp.sin(x), -jnp.cos(x) ** 2 * jnp.sin(2 * y)]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = jax.jacobian(jax.grad(cost_fn))(params)
        expected = jnp.zeros([2, 2])
        assert np.allclose(res, expected, atol=tol, rtol=0)


@pytest.mark.parametrize("execute_kwargs", execute_kwargs_iterable)
class TestHamiltonianWorkflows:
    """Test that tapes ending with expectations
    of Hamiltonians provide correct results and gradients"""

    @pytest.fixture
    def cost_fn(self, execute_kwargs):
        """Cost function for gradient tests"""

        def _cost_fn(weights, coeffs1, coeffs2, dev=None):
            obs1 = [qml.PauliZ(0), qml.PauliZ(0) @ qml.PauliX(1), qml.PauliY(0)]
            H1 = qml.Hamiltonian(coeffs1, obs1)

            obs2 = [qml.PauliZ(0)]
            H2 = qml.Hamiltonian(coeffs2, obs2)

            with qml.queuing.AnnotatedQueue() as q:
                qml.RX(weights[0], wires=0)
                qml.RY(weights[1], wires=1)
                qml.CNOT(wires=[0, 1])
                qml.expval(H1)
                qml.expval(H2)

            tape = qml.tape.QuantumScript.from_queue(q)
            return jnp.hstack(execute([tape], dev, **execute_kwargs)[0])

        return _cost_fn

    @staticmethod
    def cost_fn_expected(weights, coeffs1, coeffs2):
        """Analytic value of cost_fn above"""
        a, b, c = coeffs1
        d = coeffs2[0]
        x, y = weights
        return [-c * jnp.sin(x) * jnp.sin(y) + jnp.cos(x) * (a + b * jnp.sin(y)), d * jnp.cos(x)]

    @staticmethod
    def cost_fn_jacobian(weights, coeffs1, coeffs2):
        """Analytic jacobian of cost_fn above"""
        a, b, c = coeffs1
        d = coeffs2[0]
        x, y = weights
        return jnp.array(
            [
                [
                    -c * jnp.cos(x) * jnp.sin(y) - jnp.sin(x) * (a + b * jnp.sin(y)),
                    b * jnp.cos(x) * jnp.cos(y) - c * jnp.cos(y) * jnp.sin(x),
                    jnp.cos(x),
                    jnp.cos(x) * jnp.sin(y),
                    -(jnp.sin(x) * jnp.sin(y)),
                    0,
                ],
                [-d * jnp.sin(x), 0, 0, 0, 0, jnp.cos(x)],
            ]
        )

    def test_multiple_hamiltonians_not_trainable(self, cost_fn, tol):
        """Test hamiltonian with no trainable parameters."""
        coeffs1 = jnp.array([0.1, 0.2, 0.3])
        coeffs2 = jnp.array([0.7])
        weights = jnp.array([0.4, 0.5])
        dev = DefaultQubit2()

        res = cost_fn(weights, coeffs1, coeffs2, dev=dev)
        expected = self.cost_fn_expected(weights, coeffs1, coeffs2)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = jax.jacobian(cost_fn)(weights, coeffs1, coeffs2, dev=dev)
        expected = self.cost_fn_jacobian(weights, coeffs1, coeffs2)[:, :2]
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_multiple_hamiltonians_trainable(self, cost_fn, tol):
        """Test hamiltonian with trainable parameters."""
        coeffs1 = jnp.array([0.1, 0.2, 0.3])
        coeffs2 = jnp.array([0.7])
        weights = jnp.array([0.4, 0.5])
        dev = DefaultQubit2()

        res = cost_fn(weights, coeffs1, coeffs2, dev=dev)
        expected = self.cost_fn_expected(weights, coeffs1, coeffs2)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = jnp.hstack(
            jax.jacobian(cost_fn, argnums=[0, 1, 2])(weights, coeffs1, coeffs2, dev=dev)
        )
        expected = self.cost_fn_jacobian(weights, coeffs1, coeffs2)
        assert np.allclose(res, expected, atol=tol, rtol=0)

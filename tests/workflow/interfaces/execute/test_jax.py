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
import numpy as np
import pytest
from param_shift_dev import ParamShiftDerivativesDevice

import pennylane as qml
from pennylane import execute
from pennylane.devices import DefaultQubit
from pennylane.gradients import param_shift
from pennylane.measurements import Shots

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.jax


def get_device(device_name, seed):
    if device_name == "param_shift.qubit":
        return ParamShiftDerivativesDevice(seed=seed)
    return qml.device(device_name, seed=seed)


def test_jit_execution():
    """Test that qml.execute can be directly jitted."""
    dev = qml.device("default.qubit")

    tape = qml.tape.QuantumScript(
        [qml.RX(jax.numpy.array(0.1), 0)], [qml.expval(qml.s_prod(2.0, qml.PauliZ(0)))]
    )

    out = jax.jit(qml.execute, static_argnames=("device", "diff_method"))(
        (tape,), device=dev, diff_method=qml.gradients.param_shift
    )
    expected = 2.0 * jax.numpy.cos(jax.numpy.array(0.1))
    assert qml.math.allclose(out[0], expected)


# pylint: disable=too-few-public-methods
class TestCaching:
    """Tests for caching behaviour"""

    @pytest.mark.skip("caching is not implemented for jax")
    @pytest.mark.parametrize("num_params", [2, 3])
    def test_caching_param_shift_hessian(self, num_params):
        """Test that, when using parameter-shift transform,
        caching reduces the number of evaluations to their optimum
        when computing Hessians."""
        device = DefaultQubit()
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
                [tape], device, diff_method=qml.gradients.param_shift, cache=cache, max_diff=2
            )[0]

        # No caching: number of executions is not ideal
        with qml.Tracker(device) as tracker:
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

        with qml.Tracker(device) as tracker2:
            hess2 = jax.jacobian(jax.grad(cost))(params, cache=True)
        assert np.allclose(hess1, hess2)

        expected_runs_ideal = 1  # forward pass
        expected_runs_ideal += 2 * N  # Jacobian
        expected_runs_ideal += N + 1  # Hessian diagonal
        expected_runs_ideal += 4 * N * (N - 1) // 2  # Hessian off-diagonal
        assert tracker2.totals["executions"] == expected_runs_ideal
        assert expected_runs_ideal < expected_runs


# add tests for lightning 2 when possible
# set rng for device when possible
no_shots = Shots(None)
shots_10k = Shots(10000)
shots_2_10k = Shots((10000, 10000))
test_matrix = [
    ({"diff_method": param_shift}, shots_10k, "default.qubit"),  # 0
    ({"diff_method": param_shift}, shots_2_10k, "default.qubit"),  # 1
    ({"diff_method": param_shift}, no_shots, "default.qubit"),  # 2
    ({"diff_method": "backprop"}, no_shots, "default.qubit"),  # 3
    ({"diff_method": "adjoint"}, no_shots, "default.qubit"),  # 4
    ({"diff_method": "adjoint", "device_vjp": True}, no_shots, "default.qubit"),  # 5
    ({"diff_method": "device"}, shots_2_10k, "param_shift.qubit"),  # 6
    ({"diff_method": param_shift}, no_shots, "reference.qubit"),  # 7
    ({"diff_method": param_shift}, shots_10k, "reference.qubit"),  # 8
    ({"diff_method": param_shift}, shots_2_10k, "reference.qubit"),  # 9
    ({"diff_method": "best"}, shots_10k, "default.qubit"),  # 10
    ({"diff_method": "best"}, no_shots, "default.qubit"),  # 11
    ({"diff_method": "best"}, no_shots, "reference.qubit"),  # 12
]


def atol_for_shots(shots):
    """Return higher tolerance if finite shots."""
    return 3e-2 if shots else 1e-6


@pytest.mark.parametrize("execute_kwargs, shots, device_name", test_matrix)
class TestJaxExecuteIntegration:
    """Test the jax interface execute function
    integrates well for both forward and backward execution"""

    def test_execution(self, execute_kwargs, shots, device_name, seed):
        """Test execution"""

        device = get_device(device_name, seed)

        def cost(a, b):
            ops1 = [qml.RY(a, wires=0), qml.RX(b, wires=0)]
            tape1 = qml.tape.QuantumScript(ops1, [qml.expval(qml.PauliZ(0))], shots=shots)

            ops2 = [qml.RY(a, wires="a"), qml.RX(b, wires="a")]
            tape2 = qml.tape.QuantumScript(ops2, [qml.expval(qml.PauliZ("a"))], shots=shots)

            return execute([tape1, tape2], device, **execute_kwargs)

        a = jnp.array(0.1)
        b = np.array(0.2)
        with device.tracker:
            res = cost(a, b)

        if execute_kwargs.get("diff_method", None) == "adjoint":
            assert device.tracker.totals.get("execute_and_derivative_batches", 0) == 0
        else:
            assert device.tracker.totals["batches"] == 1
        assert device.tracker.totals["executions"] == 2  # different wires so different hashes

        assert len(res) == 2
        if not shots.has_partitioned_shots:
            assert res[0].shape == ()
            assert res[1].shape == ()

        assert qml.math.allclose(res[0], jnp.cos(a) * jnp.cos(b), atol=atol_for_shots(shots))
        assert qml.math.allclose(res[1], jnp.cos(a) * jnp.cos(b), atol=atol_for_shots(shots))

    def test_scalar_jacobian(self, execute_kwargs, shots, device_name, seed):
        """Test scalar jacobian calculation"""
        a = jnp.array(0.1)

        device = get_device(device_name, seed)

        def cost(a):
            tape = qml.tape.QuantumScript([qml.RY(a, 0)], [qml.expval(qml.PauliZ(0))], shots=shots)
            return execute([tape], device, **execute_kwargs)[0]

        res = jax.jacobian(cost)(a)
        if not shots.has_partitioned_shots:
            assert res.shape == ()  # pylint: disable=no-member

        expected = -qml.math.sin(a)

        assert expected.shape == ()
        assert np.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)
        assert np.allclose(res, -jnp.sin(a), atol=atol_for_shots(shots))

    def test_jacobian(self, execute_kwargs, shots, device_name, seed):
        """Test jacobian calculation"""

        a = jnp.array(0.1)
        b = jnp.array(0.2)

        device = get_device(device_name, seed)

        def cost(a, b):
            ops = [qml.RY(a, wires=0), qml.RX(b, wires=1), qml.CNOT(wires=[0, 1])]
            m = [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))]
            tape = qml.tape.QuantumScript(ops, m, shots=shots)
            return execute([tape], device, **execute_kwargs)[0]

        res = cost(a, b)
        expected = [jnp.cos(a), -jnp.cos(a) * jnp.sin(b)]
        if shots.has_partitioned_shots:
            assert np.allclose(res[0], expected, atol=atol_for_shots(shots), rtol=0)
            assert np.allclose(res[1], expected, atol=2 * atol_for_shots(shots), rtol=0)
        else:
            assert np.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)

        g = jax.jacobian(cost, argnums=[0, 1])(a, b)
        assert isinstance(g, tuple) and len(g) == 2

        expected = ([-jnp.sin(a), jnp.sin(a) * jnp.sin(b)], [0, -jnp.cos(a) * jnp.cos(b)])

        if shots.has_partitioned_shots:
            for i in (0, 1):
                assert np.allclose(g[i][0][0], expected[0][0], atol=atol_for_shots(shots), rtol=0)
                assert np.allclose(g[i][1][0], expected[0][1], atol=atol_for_shots(shots), rtol=0)
                assert np.allclose(g[i][0][1], expected[1][0], atol=atol_for_shots(shots), rtol=0)
                assert np.allclose(g[i][1][1], expected[1][1], atol=atol_for_shots(shots), rtol=0)
        else:
            assert np.allclose(g[0][0], expected[0][0], atol=atol_for_shots(shots), rtol=0)
            assert np.allclose(g[1][0], expected[0][1], atol=atol_for_shots(shots), rtol=0)
            assert np.allclose(g[0][1], expected[1][0], atol=atol_for_shots(shots), rtol=0)
            assert np.allclose(g[1][1], expected[1][1], atol=atol_for_shots(shots), rtol=0)

    def test_tape_no_parameters(self, execute_kwargs, shots, device_name, seed):
        """Test that a tape with no parameters is correctly
        ignored during the gradient computation"""

        device = get_device(device_name, seed)

        def cost(params):
            tape1 = qml.tape.QuantumScript(
                [qml.Hadamard(0)], [qml.expval(qml.PauliX(0))], shots=shots
            )

            tape2 = qml.tape.QuantumScript(
                [qml.RY(jnp.array(0.5), wires=0)],
                [qml.expval(qml.PauliZ(0))],
                shots=shots,
            )

            tape3 = qml.tape.QuantumScript(
                [qml.RY(params[0], 0), qml.RX(params[1], 0)],
                [qml.expval(qml.PauliZ(0))],
                shots=shots,
            )

            tape4 = qml.tape.QuantumScript(
                [qml.RY(jnp.array(0.5), 0)], [qml.probs(wires=[0, 1])], shots=shots
            )
            res = execute([tape1, tape2, tape3, tape4], device, **execute_kwargs)
            res = jax.tree_util.tree_leaves(res)
            out = sum(jnp.hstack(res))
            if shots.has_partitioned_shots:
                out = out / shots.num_copies
            return out

        params = jnp.array([0.1, 0.2])

        x, y = params

        res = cost(params)
        expected = 2 + jnp.cos(0.5) + jnp.cos(x) * jnp.cos(y)
        assert np.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)

        grad = jax.grad(cost)(params)
        expected = [-jnp.cos(y) * jnp.sin(x), -jnp.cos(x) * jnp.sin(y)]
        assert np.allclose(grad, expected, atol=atol_for_shots(shots), rtol=0)

    # pylint: disable=too-many-statements
    def test_tapes_with_different_return_size(self, execute_kwargs, shots, device_name, seed):
        """Test that tapes wit different can be executed and differentiated."""

        device = get_device(device_name, seed)

        def cost(params):
            tape1 = qml.tape.QuantumScript(
                [qml.RY(params[0], 0), qml.RX(params[1], 0)],
                [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))],
                shots=shots,
            )

            tape2 = qml.tape.QuantumScript(
                [qml.RY(np.array(0.5), 0)], [qml.expval(qml.PauliZ(0))], shots=shots
            )

            tape3 = qml.tape.QuantumScript(
                [qml.RY(params[0], 0), qml.RX(params[1], 0)],
                [qml.expval(qml.PauliZ(0))],
                shots=shots,
            )
            res = qml.execute([tape1, tape2, tape3], device, **execute_kwargs)
            leaves = jax.tree_util.tree_leaves(res)
            return jnp.hstack(leaves)

        params = jnp.array([0.1, 0.2])
        x, y = params

        res = cost(params)
        assert isinstance(res, jax.Array)
        assert res.shape == (4 * shots.num_copies,) if shots.has_partitioned_shots else (4,)

        if shots.has_partitioned_shots:
            assert np.allclose(res[0], jnp.cos(x) * jnp.cos(y), atol=atol_for_shots(shots))
            assert np.allclose(res[2], jnp.cos(x) * jnp.cos(y), atol=atol_for_shots(shots))
            assert np.allclose(res[1], 1, atol=atol_for_shots(shots))
            assert np.allclose(res[3], 1, atol=atol_for_shots(shots))
            assert np.allclose(res[4], jnp.cos(0.5), atol=atol_for_shots(shots))
            assert np.allclose(res[5], jnp.cos(0.5), atol=atol_for_shots(shots))
            assert np.allclose(res[6], jnp.cos(x) * jnp.cos(y), atol=atol_for_shots(shots))
            assert np.allclose(res[7], jnp.cos(x) * jnp.cos(y), atol=atol_for_shots(shots))
        else:
            assert np.allclose(res[0], jnp.cos(x) * jnp.cos(y), atol=atol_for_shots(shots))
            assert np.allclose(res[1], 1, atol=atol_for_shots(shots))
            assert np.allclose(res[2], jnp.cos(0.5), atol=atol_for_shots(shots))
            assert np.allclose(res[3], jnp.cos(x) * jnp.cos(y), atol=atol_for_shots(shots))

        jac = jax.jacobian(cost)(params)
        assert isinstance(jac, jnp.ndarray)
        assert (
            jac.shape == (8, 2) if shots.has_partitioned_shots else (4, 2)
        )  # pylint: disable=no-member

        if shots.has_partitioned_shots:
            assert np.allclose(jac[1], 0, atol=atol_for_shots(shots))
            assert np.allclose(jac[3:5], 0, atol=atol_for_shots(shots))
        else:
            assert np.allclose(jac[1:3], 0, atol=atol_for_shots(shots))

        d1 = -jnp.sin(x) * jnp.cos(y)
        if shots.has_partitioned_shots:
            assert np.allclose(jac[0, 0], d1, atol=atol_for_shots(shots))
            assert np.allclose(jac[2, 0], d1, atol=atol_for_shots(shots))
            assert np.allclose(jac[6, 0], d1, atol=atol_for_shots(shots))
            assert np.allclose(jac[7, 0], d1, atol=atol_for_shots(shots))
        else:
            assert np.allclose(jac[0, 0], d1, atol=atol_for_shots(shots))
            assert np.allclose(jac[3, 0], d1, atol=atol_for_shots(shots))

        d2 = -jnp.cos(x) * jnp.sin(y)
        if shots.has_partitioned_shots:
            assert np.allclose(jac[0, 1], d2, atol=atol_for_shots(shots))
            assert np.allclose(jac[2, 1], d2, atol=atol_for_shots(shots))
            assert np.allclose(jac[6, 1], d2, atol=atol_for_shots(shots))
            assert np.allclose(jac[7, 1], d2, atol=atol_for_shots(shots))
        else:
            assert np.allclose(jac[0, 1], d2, atol=atol_for_shots(shots))
            assert np.allclose(jac[3, 1], d2, atol=atol_for_shots(shots))

    def test_reusing_quantum_tape(self, execute_kwargs, shots, device_name, seed):
        """Test re-using a quantum tape by passing new parameters"""

        device = get_device(device_name, seed)

        a = jnp.array(0.1)
        b = jnp.array(0.2)

        tape = qml.tape.QuantumScript(
            [qml.RY(a, 0), qml.RX(b, 1), qml.CNOT((0, 1))],
            [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))],
            shots=shots,
        )
        assert tape.trainable_params == [0, 1]

        def cost(a, b):
            new_tape = tape.bind_new_parameters([a, b], [0, 1])
            return jnp.hstack(jnp.array(execute([new_tape], device, **execute_kwargs)[0]))

        jac_fn = jax.jacobian(cost, argnums=[0, 1])
        jac = jac_fn(a, b)

        a = jnp.array(0.54)
        b = jnp.array(0.8)

        # check that the cost function continues to depend on the
        # values of the parameters for subsequent calls
        res2 = cost(2 * a, b)
        expected = [jnp.cos(2 * a), -jnp.cos(2 * a) * jnp.sin(b)]
        if shots.has_partitioned_shots:
            assert np.allclose(res2[:2], expected, atol=atol_for_shots(shots), rtol=0)
            assert np.allclose(res2[2:], expected, atol=atol_for_shots(shots), rtol=0)
        else:
            assert np.allclose(res2, expected, atol=atol_for_shots(shots), rtol=0)

        jac_fn = jax.jacobian(lambda a, b: cost(2 * a, b), argnums=[0, 1])
        jac = jac_fn(a, b)
        expected = (
            [-2 * jnp.sin(2 * a), 2 * jnp.sin(2 * a) * jnp.sin(b)],
            [0, -jnp.cos(2 * a) * jnp.cos(b)],
        )
        assert isinstance(jac, tuple) and len(jac) == 2
        if shots.has_partitioned_shots:
            for offset in (0, 2):
                assert np.allclose(jac[0][0 + offset], expected[0][0], atol=atol_for_shots(shots))
                assert np.allclose(jac[0][1 + offset], expected[0][1], atol=atol_for_shots(shots))
                assert np.allclose(jac[1][0 + offset], expected[1][0], atol=atol_for_shots(shots))
                assert np.allclose(jac[1][1 + offset], expected[1][1], atol=atol_for_shots(shots))
        else:
            for _j, _e in zip(jac, expected):
                assert np.allclose(_j, _e, atol=atol_for_shots(shots), rtol=0)

    def test_classical_processing(self, execute_kwargs, shots, device_name, seed):
        """Test classical processing within the quantum tape"""
        a = jnp.array(0.1)
        b = jnp.array(0.2)
        c = jnp.array(0.3)

        device = get_device(device_name, seed)

        def cost(a, b, c):
            ops = [
                qml.RY(a * c, wires=0),
                qml.RZ(b, wires=0),
                qml.RX(c + c**2 + jnp.sin(a), wires=0),
            ]

            tape = qml.tape.QuantumScript(ops, [qml.expval(qml.PauliZ(0))], shots=shots)
            return execute([tape], device, **execute_kwargs)[0]

        res = jax.jacobian(cost, argnums=[0, 2])(a, b, c)

        # Only two arguments are trainable
        assert isinstance(res, tuple) and len(res) == 2
        if not shots.has_partitioned_shots:
            assert res[0].shape == ()
            assert res[1].shape == ()

        # I tried getting analytic results for this circuit but I kept being wrong and am giving up

    def test_matrix_parameter(self, execute_kwargs, device_name, seed, shots):
        """Test that the jax interface works correctly
        with a matrix parameter"""
        U = jnp.array([[0, 1], [1, 0]])
        a = jnp.array(0.1)

        device = get_device(device_name, seed)

        def cost(a, U):
            ops = [qml.QubitUnitary(U, wires=0), qml.RY(a, wires=0)]
            tape = qml.tape.QuantumScript(ops, [qml.expval(qml.PauliZ(0))], shots=shots)
            return execute([tape], device, **execute_kwargs)[0]

        res = cost(a, U)
        assert np.allclose(res, -jnp.cos(a), atol=atol_for_shots(shots), rtol=0)

        jac_fn = jax.jacobian(cost)
        jac = jac_fn(a, U)
        if not shots.has_partitioned_shots:
            assert isinstance(jac, jnp.ndarray)
        assert np.allclose(jac, jnp.sin(a), atol=atol_for_shots(shots), rtol=0)

    def test_differentiable_expand(self, execute_kwargs, device_name, seed, shots):
        """Test that operation and nested tapes expansion
        is differentiable"""

        device = get_device(device_name, seed)

        class U3(qml.U3):
            """Dummy operator."""

            def decomposition(self):
                theta, phi, lam = self.data
                wires = self.wires
                return [
                    qml.Rot(lam, theta, -lam, wires=wires),
                    qml.PhaseShift(phi + lam, wires=wires),
                ]

        def cost_fn(a, p):
            tape = qml.tape.QuantumScript(
                [qml.RX(a, wires=0), U3(*p, wires=0)],
                [qml.expval(qml.PauliX(0))],
                shots=shots,
            )
            return execute([tape], device, **execute_kwargs)[0]

        a = jnp.array(0.1)
        p = jnp.array([0.1, 0.2, 0.3])

        res = cost_fn(a, p)
        expected = jnp.cos(a) * jnp.cos(p[1]) * jnp.sin(p[0]) + jnp.sin(a) * (
            jnp.cos(p[2]) * jnp.sin(p[1]) + jnp.cos(p[0]) * jnp.cos(p[1]) * jnp.sin(p[2])
        )
        assert np.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)

        jac_fn = jax.jacobian(cost_fn, argnums=[1])
        res = jac_fn(a, p)
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
        assert np.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)

    def test_probability_differentiation(self, execute_kwargs, device_name, seed, shots):
        """Tests correct output shape and evaluation for a tape
        with prob outputs"""

        device = get_device(device_name, seed)

        def cost(x, y):
            ops = [qml.RX(x, 0), qml.RY(y, 1), qml.CNOT((0, 1))]
            m = [qml.probs(wires=0), qml.probs(wires=1)]
            tape = qml.tape.QuantumScript(ops, m, shots=shots)
            return jnp.hstack(jnp.array(execute([tape], device, **execute_kwargs)[0]))

        x = jnp.array(0.543)
        y = jnp.array(-0.654)

        res = cost(x, y)
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
        if shots.has_partitioned_shots:
            assert np.allclose(res[:, 0:2].flatten(), expected, atol=atol_for_shots(shots))
            assert np.allclose(res[:, 2:].flatten(), expected, atol=atol_for_shots(shots))
        else:
            assert np.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)

        jac_fn = jax.jacobian(cost, argnums=[0, 1])
        res = jac_fn(x, y)
        assert isinstance(res, tuple) and len(res) == 2
        assert res[0].shape == (2, 4) if shots.has_partitioned_shots else (4,)
        assert res[1].shape == (2, 4) if shots.has_partitioned_shots else (4,)

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

        if shots.has_partitioned_shots:
            assert np.allclose(res[0][:, 0:2].flatten(), expected[0], atol=atol_for_shots(shots))
            assert np.allclose(res[0][:, 2:].flatten(), expected[0], atol=atol_for_shots(shots))
            assert np.allclose(res[1][:, :2].flatten(), expected[1], atol=atol_for_shots(shots))
            assert np.allclose(res[1][:, 2:].flatten(), expected[1], atol=atol_for_shots(shots))
        else:
            assert np.allclose(res[0], expected[0], atol=atol_for_shots(shots), rtol=0)
            assert np.allclose(res[1], expected[1], atol=atol_for_shots(shots), rtol=0)

    def test_ragged_differentiation(self, execute_kwargs, device_name, seed, shots):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""

        device = get_device(device_name, seed)

        def cost(x, y):
            ops = [qml.RX(x, wires=0), qml.RY(y, 1), qml.CNOT((0, 1))]
            m = [qml.expval(qml.PauliZ(0)), qml.probs(wires=1)]
            tape = qml.tape.QuantumScript(ops, m, shots=shots)
            res = qml.execute([tape], device, **execute_kwargs)[0]
            return jnp.hstack(jax.tree_util.tree_leaves(res))

        x = jnp.array(0.543)
        y = jnp.array(-0.654)

        res = cost(x, y)
        expected = jnp.array(
            [jnp.cos(x), (1 + jnp.cos(x) * jnp.cos(y)) / 2, (1 - jnp.cos(x) * jnp.cos(y)) / 2]
        )
        if shots.has_partitioned_shots:
            assert np.allclose(res[:3], expected, atol=atol_for_shots(shots), rtol=0)
            assert np.allclose(res[3:], expected, atol=atol_for_shots(shots), rtol=0)
        else:
            assert np.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)

        jac_fn = jax.jacobian(cost, argnums=[0, 1])
        res = jac_fn(x, y)
        assert isinstance(res, tuple) and len(res) == 2
        assert res[0].shape == (3 * shots.num_copies,) if shots.has_partitioned_shots else (3,)
        assert res[1].shape == (3 * shots.num_copies,) if shots.has_partitioned_shots else (3,)

        expected = (
            jnp.array([-jnp.sin(x), -jnp.sin(x) * jnp.cos(y) / 2, jnp.sin(x) * jnp.cos(y) / 2]),
            jnp.array([0, -jnp.cos(x) * jnp.sin(y) / 2, jnp.cos(x) * jnp.sin(y) / 2]),
        )
        if shots.has_partitioned_shots:
            assert np.allclose(res[0][:3], expected[0], atol=atol_for_shots(shots), rtol=0)
            assert np.allclose(res[0][3:], expected[0], atol=atol_for_shots(shots), rtol=0)
            assert np.allclose(res[1][:3], expected[1], atol=atol_for_shots(shots), rtol=0)
            assert np.allclose(res[1][3:], expected[1], atol=atol_for_shots(shots), rtol=0)
        else:
            assert np.allclose(res[0], expected[0], atol=atol_for_shots(shots), rtol=0)
            assert np.allclose(res[1], expected[1], atol=atol_for_shots(shots), rtol=0)


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
        dev = DefaultQubit()

        def cost_fn(x):
            ops1 = [qml.RX(x[0], 0), qml.RY(x[1], 1), qml.CNOT((0, 1))]
            tape1 = qml.tape.QuantumScript(ops1, [qml.var(qml.PauliZ(0) @ qml.PauliX(1))])

            ops2 = [qml.RX(x[0], 0), qml.RY(x[0], 1), qml.CNOT((0, 1))]
            tape2 = qml.tape.QuantumScript(ops2, [qml.probs(wires=1)])
            result = execute([tape1, tape2], dev, diff_method=param_shift, max_diff=2)
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
        dev = DefaultQubit()
        params = jnp.array([0.543, -0.654])

        def cost_fn(x):
            ops = [qml.RX(x[0], 0), qml.RY(x[1], 1), qml.CNOT((0, 1))]
            tape1 = qml.tape.QuantumScript(
                ops, [qml.var(qml.PauliZ(0) @ qml.PauliX(1))], shots=50000
            )

            ops2 = [qml.RX(x[0], 0), qml.RY(x[0], 1), qml.CNOT((0, 1))]
            tape2 = qml.tape.QuantumScript(ops2, [qml.probs(wires=1)], shots=50000)

            result = execute([tape1, tape2], dev, diff_method=param_shift, max_diff=1)
            return result[0] + result[1][0]

        res = cost_fn(params)
        x, y = params
        expected = 0.5 * (3 + jnp.cos(x) ** 2 * jnp.cos(2 * y))
        assert np.allclose(res, expected, atol=2e-2, rtol=0)

        res = jax.grad(cost_fn)(params)
        expected = jnp.array(
            [-jnp.cos(x) * jnp.cos(2 * y) * jnp.sin(x), -jnp.cos(x) ** 2 * jnp.sin(2 * y)]
        )
        assert np.allclose(res, expected, atol=2e-2, rtol=0)

        res = jax.jacobian(jax.grad(cost_fn))(params)
        expected = jnp.zeros([2, 2])
        assert np.allclose(res, expected, atol=tol, rtol=0)


@pytest.mark.parametrize("execute_kwargs, shots, device_name", test_matrix)
@pytest.mark.parametrize("constructor", (qml.Hamiltonian, qml.dot, "dunders"))
class TestHamiltonianWorkflows:
    """Test that tapes ending with expectations
    of Hamiltonians provide correct results and gradients"""

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    @pytest.fixture
    def cost_fn(self, execute_kwargs, shots, device_name, seed, constructor):
        """Cost function for gradient tests"""

        device = get_device(device_name, seed)

        def _cost_fn(weights, coeffs1, coeffs2):
            if constructor == "dunders":
                H1 = (
                    coeffs1[0] * qml.Z(0) + coeffs1[1] * qml.Z(0) @ qml.X(1) + coeffs1[2] * qml.Y(0)
                )
                H2 = coeffs2[0] * qml.Z(0)
            else:

                obs1 = [qml.Z(0), qml.Z(0) @ qml.X(1), qml.Y(0)]
                H1 = constructor(coeffs1, obs1)

                obs2 = [qml.PauliZ(0)]
                H2 = constructor(coeffs2, obs2)

            with qml.queuing.AnnotatedQueue() as q:
                qml.RX(weights[0], wires=0)
                qml.RY(weights[1], wires=1)
                qml.CNOT(wires=[0, 1])
                qml.expval(H1)
                qml.expval(H2)

            tape = qml.tape.QuantumScript.from_queue(q, shots=shots)
            res = execute([tape], device, **execute_kwargs)[0]
            if shots.has_partitioned_shots:
                return jnp.hstack(res[0] + res[1])
            return jnp.hstack(res)

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

    def test_multiple_hamiltonians_not_trainable(self, cost_fn, shots):
        """Test hamiltonian with no trainable parameters."""

        coeffs1 = jnp.array([0.1, 0.2, 0.3])
        coeffs2 = jnp.array([0.7])
        weights = jnp.array([0.4, 0.5])

        res = cost_fn(weights, coeffs1, coeffs2)
        expected = self.cost_fn_expected(weights, coeffs1, coeffs2)
        if shots.has_partitioned_shots:
            assert np.allclose(res[:2], expected, atol=atol_for_shots(shots), rtol=0)
            assert np.allclose(res[2:], expected, atol=atol_for_shots(shots), rtol=0)
        else:
            assert np.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)

        res = jax.jacobian(cost_fn)(weights, coeffs1, coeffs2)
        expected = self.cost_fn_jacobian(weights, coeffs1, coeffs2)[:, :2]
        if shots.has_partitioned_shots:
            assert np.allclose(res[:2, :], expected, atol=atol_for_shots(shots), rtol=0)
            assert np.allclose(res[2:, :], expected, atol=atol_for_shots(shots), rtol=0)
        else:
            assert np.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)

    def test_multiple_hamiltonians_trainable(self, execute_kwargs, cost_fn, shots):
        """Test hamiltonian with trainable parameters."""
        if execute_kwargs["diff_method"] == "adjoint":
            pytest.xfail("trainable hamiltonians not supported with adjoint")

        coeffs1 = jnp.array([0.1, 0.2, 0.3])
        coeffs2 = jnp.array([0.7])
        weights = jnp.array([0.4, 0.5])

        res = cost_fn(weights, coeffs1, coeffs2)
        expected = self.cost_fn_expected(weights, coeffs1, coeffs2)
        if shots.has_partitioned_shots:
            assert np.allclose(res[:2], expected, atol=atol_for_shots(shots), rtol=0)
            assert np.allclose(res[2:], expected, atol=atol_for_shots(shots), rtol=0)
        else:
            assert np.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)

        res = jnp.hstack(jax.jacobian(cost_fn, argnums=[0, 1, 2])(weights, coeffs1, coeffs2))
        expected = self.cost_fn_jacobian(weights, coeffs1, coeffs2)
        if shots.has_partitioned_shots:
            assert np.allclose(res[:2, :], expected, atol=atol_for_shots(shots), rtol=0)
            assert np.allclose(res[2:, :], expected, atol=atol_for_shots(shots), rtol=0)
        else:
            assert np.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)

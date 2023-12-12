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
"""Tensorflow specific tests for execute and default qubit 2."""
import pytest
import numpy as np

import pennylane as qml
from pennylane.devices import DefaultQubit
from pennylane.gradients import param_shift
from pennylane.interfaces import execute

pytestmark = pytest.mark.tf
tf = pytest.importorskip("tensorflow")


# pylint: disable=too-few-public-methods
class TestCaching:
    """Tests for caching behaviour"""

    @pytest.mark.parametrize("num_params", [2, 3])
    def test_caching_param_shift_hessian(self, num_params):
        """Test that, when using parameter-shift transform,
        caching reduces the number of evaluations to their optimum
        when computing Hessians."""
        dev = DefaultQubit()
        params = tf.Variable(tf.range(1, num_params + 1) / 10)

        N = num_params

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
            with tf.GradientTape() as jac_tape:
                with tf.GradientTape() as grad_tape:
                    res = cost(params, cache=False)
                grad = grad_tape.gradient(res, params)
            hess1 = jac_tape.jacobian(grad, params)

        if num_params == 2:
            # compare to theoretical result
            x, y, *_ = params
            expected = tf.convert_to_tensor(
                [
                    [2 * tf.cos(2 * x) * tf.sin(y) ** 2, tf.sin(2 * x) * tf.sin(2 * y)],
                    [tf.sin(2 * x) * tf.sin(2 * y), -2 * tf.cos(x) ** 2 * tf.cos(2 * y)],
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
            with tf.GradientTape() as jac_tape:
                with tf.GradientTape() as grad_tape:
                    res = cost(params, cache=True)
                grad = grad_tape.gradient(res, params)
            hess2 = jac_tape.jacobian(grad, params)
        assert np.allclose(hess1, hess2)

        expected_runs_ideal = 1  # forward pass
        expected_runs_ideal += 2 * N  # Jacobian
        expected_runs_ideal += N + 1  # Hessian diagonal
        expected_runs_ideal += 4 * N * (N - 1) // 2  # Hessian off-diagonal
        assert tracker2.totals["executions"] == expected_runs_ideal
        assert expected_runs_ideal < expected_runs


# add tests for lightning 2 when possible
# set rng for device when possible
test_matrix = [
    ({"gradient_fn": param_shift, "interface": "tensorflow"}, 100000, DefaultQubit(seed=42)),
    ({"gradient_fn": param_shift, "interface": "tensorflow"}, None, DefaultQubit()),
    ({"gradient_fn": "backprop", "interface": "tensorflow"}, None, DefaultQubit()),
    ({"gradient_fn": "adjoint", "interface": "tensorflow"}, None, DefaultQubit()),
    ({"gradient_fn": param_shift, "interface": "tf-autograph"}, 100000, DefaultQubit(seed=42)),
    ({"gradient_fn": param_shift, "interface": "tf-autograph"}, None, DefaultQubit()),
    ({"gradient_fn": "backprop", "interface": "tf-autograph"}, None, DefaultQubit()),
    ({"gradient_fn": "adjoint", "interface": "tf-autograph"}, None, DefaultQubit()),
]


def atol_for_shots(shots):
    """Return higher tolerance if finite shots."""
    return 1e-2 if shots else 1e-6


@pytest.mark.parametrize("execute_kwargs, shots, device", test_matrix)
class TestTensorflowExecuteIntegration:
    """Test the tensorflow interface execute function
    integrates well for both forward and backward execution"""

    def test_execution(self, execute_kwargs, shots, device):
        """Test execution"""

        def cost(a, b):
            ops1 = [qml.RY(a, wires=0), qml.RX(b, wires=0)]
            tape1 = qml.tape.QuantumScript(ops1, [qml.expval(qml.PauliZ(0))], shots=shots)

            ops2 = [qml.RY(a, wires="a"), qml.RX(b, wires="a")]
            tape2 = qml.tape.QuantumScript(ops2, [qml.expval(qml.PauliZ("a"))], shots=shots)

            return execute([tape1, tape2], device, **execute_kwargs)

        a = tf.Variable(0.1)
        b = tf.constant(0.2)
        with device.tracker:
            res = cost(a, b)

        if execute_kwargs.get("gradient_fn", None) == "adjoint":
            assert device.tracker.totals["execute_and_derivative_batches"] == 1
        else:
            assert device.tracker.totals["batches"] == 1
        assert device.tracker.totals["executions"] == 2  # different wires so different hashes

        assert len(res) == 2
        assert res[0].shape == ()
        assert res[1].shape == ()

        assert qml.math.allclose(res[0], tf.cos(a) * tf.cos(b), atol=atol_for_shots(shots))
        assert qml.math.allclose(res[1], tf.cos(a) * tf.cos(b), atol=atol_for_shots(shots))

    def test_scalar_jacobian(self, execute_kwargs, shots, device):
        """Test scalar jacobian calculation"""
        a = tf.Variable(0.1, dtype=tf.float64)

        def cost(a):
            tape = qml.tape.QuantumScript([qml.RY(a, 0)], [qml.expval(qml.PauliZ(0))], shots=shots)
            return execute([tape], device, **execute_kwargs)[0]

        with tf.GradientTape() as tape:
            cost_res = cost(a)
        res = tape.jacobian(cost_res, a)
        assert res.shape == ()  # pylint: disable=no-member

        # compare to standard tape jacobian
        tape = qml.tape.QuantumScript([qml.RY(a, wires=0)], [qml.expval(qml.PauliZ(0))])
        tape.trainable_params = [0]
        tapes, fn = param_shift(tape)
        expected = fn(device.execute(tapes))

        assert expected.shape == ()
        assert np.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)
        assert np.allclose(res, -tf.sin(a), atol=atol_for_shots(shots))

    def test_jacobian(self, execute_kwargs, shots, device):
        """Test jacobian calculation"""
        a = tf.Variable(0.1)
        b = tf.Variable(0.2)

        def cost(a, b):
            ops = [qml.RY(a, wires=0), qml.RX(b, wires=1), qml.CNOT(wires=[0, 1])]
            m = [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))]
            tape = qml.tape.QuantumScript(ops, m, shots=shots)
            return qml.math.hstack(execute([tape], device, **execute_kwargs)[0], like="tensorflow")

        with tf.GradientTape() as tape:
            res = cost(a, b)
        expected = [tf.cos(a), -tf.cos(a) * tf.sin(b)]
        assert np.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)

        jac = tape.jacobian(res, [a, b])
        assert isinstance(jac, list) and len(jac) == 2
        assert jac[0].shape == (2,)
        assert jac[1].shape == (2,)

        expected = ([-tf.sin(a), tf.sin(a) * tf.sin(b)], [0, -tf.cos(a) * tf.cos(b)])
        for _r, _e in zip(jac, expected):
            assert np.allclose(_r, _e, atol=atol_for_shots(shots))

    def test_tape_no_parameters(self, execute_kwargs, shots, device):
        """Test that a tape with no parameters is correctly
        ignored during the gradient computation"""

        if execute_kwargs["gradient_fn"] == "adjoint":
            pytest.skip("Adjoint differentiation does not yet support probabilities")

        def cost(params):
            tape1 = qml.tape.QuantumScript(
                [qml.Hadamard(0)], [qml.expval(qml.PauliX(0))], shots=shots
            )

            tape2 = qml.tape.QuantumScript(
                [qml.RY(tf.constant(0.5), wires=0)],
                [qml.expval(qml.PauliZ(0))],
                shots=shots,
            )

            tape3 = qml.tape.QuantumScript(
                [qml.RY(params[0], 0), qml.RX(params[1], 0)],
                [qml.expval(qml.PauliZ(0))],
                shots=shots,
            )

            tape4 = qml.tape.QuantumScript(
                [qml.RY(tf.constant(0.5), 0)],
                [qml.probs(wires=[0, 1])],
                shots=shots,
            )
            return tf.reduce_sum(
                qml.math.hstack(
                    execute([tape1, tape2, tape3, tape4], device, **execute_kwargs),
                    like="tensorflow",
                )
            )

        params = tf.Variable([0.1, 0.2])
        x, y = params

        with tf.GradientTape() as tape:
            res = cost(params)
        expected = 2 + tf.cos(0.5) + tf.cos(x) * tf.cos(y)
        assert np.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)

        grad = tape.gradient(res, params)
        expected = [-tf.cos(y) * tf.sin(x), -tf.cos(x) * tf.sin(y)]
        assert np.allclose(grad, expected, atol=atol_for_shots(shots), rtol=0)

    def test_tapes_with_different_return_size(self, execute_kwargs, shots, device):
        """Test that tapes wit different can be executed and differentiated."""

        if (
            execute_kwargs["gradient_fn"] == "adjoint"
            and execute_kwargs["interface"] == "tf-autograph"
        ):
            pytest.skip("Cannot compute the jacobian with adjoint-differentation and tf-autograph")

        def cost(params):
            tape1 = qml.tape.QuantumScript(
                [qml.RY(params[0], 0), qml.RX(params[1], 0)],
                [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))],
                shots=shots,
            )

            tape2 = qml.tape.QuantumScript(
                [qml.RY(tf.constant(0.5), 0)],
                [qml.expval(qml.PauliZ(0))],
                shots=shots,
            )

            tape3 = qml.tape.QuantumScript(
                [qml.RY(params[0], 0), qml.RX(params[1], 0)],
                [qml.expval(qml.PauliZ(0))],
                shots=shots,
            )
            return qml.math.hstack(
                execute([tape1, tape2, tape3], device, **execute_kwargs), like="tensorflow"
            )

        params = tf.Variable([0.1, 0.2])
        x, y = params

        with tf.GradientTape() as tape:
            res = cost(params)

        assert isinstance(res, tf.Tensor)
        assert res.shape == (4,)

        assert np.allclose(res[0], tf.cos(x) * tf.cos(y), atol=atol_for_shots(shots))
        assert np.allclose(res[1], 1, atol=atol_for_shots(shots))
        assert np.allclose(res[2], tf.cos(0.5), atol=atol_for_shots(shots))
        assert np.allclose(res[3], tf.cos(x) * tf.cos(y), atol=atol_for_shots(shots))

        jac = tape.jacobian(res, params)
        assert isinstance(jac, tf.Tensor)
        assert jac.shape == (4, 2)  # pylint: disable=no-member

        assert np.allclose(jac[1:3], 0, atol=atol_for_shots(shots))

        d1 = -tf.sin(x) * tf.cos(y)
        assert np.allclose(jac[0, 0], d1, atol=atol_for_shots(shots))
        assert np.allclose(jac[3, 0], d1, atol=atol_for_shots(shots))

        d2 = -tf.cos(x) * tf.sin(y)
        assert np.allclose(jac[0, 1], d2, atol=atol_for_shots(shots))
        assert np.allclose(jac[3, 1], d2, atol=atol_for_shots(shots))

    def test_reusing_quantum_tape(self, execute_kwargs, shots, device):
        """Test re-using a quantum tape by passing new parameters"""
        a = tf.Variable(0.1)
        b = tf.Variable(0.2)

        tape = qml.tape.QuantumScript(
            [qml.RY(a, 0), qml.RX(b, 1), qml.CNOT((0, 1))],
            [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))],
        )
        assert tape.trainable_params == [0, 1]

        def cost(a, b):
            new_tape = tape.bind_new_parameters([a, b], [0, 1])
            return qml.math.hstack(
                execute([new_tape], device, **execute_kwargs)[0], like="tensorflow"
            )

        with tf.GradientTape() as t:
            res = cost(a, b)

        jac = t.jacobian(res, [a, b])
        a = tf.Variable(0.54, dtype=tf.float64)
        b = tf.Variable(0.8, dtype=tf.float64)

        # check that the cost function continues to depend on the
        # values of the parameters for subsequent calls

        with tf.GradientTape():
            res2 = cost(2 * a, b)

        expected = [tf.cos(2 * a), -tf.cos(2 * a) * tf.sin(b)]
        assert np.allclose(res2, expected, atol=atol_for_shots(shots), rtol=0)

        with tf.GradientTape() as t:
            res = cost(2 * a, b)

        jac = t.jacobian(res, [a, b])
        expected = (
            [-2 * tf.sin(2 * a), 2 * tf.sin(2 * a) * tf.sin(b)],
            [0, -tf.cos(2 * a) * tf.cos(b)],
        )
        assert isinstance(jac, list) and len(jac) == 2
        for _j, _e in zip(jac, expected):
            assert np.allclose(_j, _e, atol=atol_for_shots(shots), rtol=0)

    def test_classical_processing(self, execute_kwargs, device, shots):
        """Test classical processing within the quantum tape"""
        a = tf.Variable(0.1, dtype=tf.float64)
        b = tf.constant(0.2, dtype=tf.float64)
        c = tf.Variable(0.3, dtype=tf.float64)

        def cost(a, b, c):
            ops = [
                qml.RY(a * c, wires=0),
                qml.RZ(b, wires=0),
                qml.RX(c + c**2 + tf.sin(a), wires=0),
            ]

            tape = qml.tape.QuantumScript(ops, [qml.expval(qml.PauliZ(0))], shots=shots)
            return execute([tape], device, **execute_kwargs)[0]

        with tf.GradientTape() as tape:
            cost_res = cost(a, b, c)

        res = tape.jacobian(cost_res, [a, c])

        # Only two arguments are trainable
        assert isinstance(res, list) and len(res) == 2
        assert res[0].shape == ()
        assert res[1].shape == ()

        # I tried getting analytic results for this circuit but I kept being wrong and am giving up

    def test_no_trainable_parameters(self, execute_kwargs, shots, device):
        """Test evaluation and Jacobian if there are no trainable parameters"""
        a = tf.constant(0.1)
        b = tf.constant(0.2)

        def cost(a, b):
            ops = [qml.RY(a, 0), qml.RX(b, 0), qml.CNOT((0, 1))]
            m = [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]
            tape = qml.tape.QuantumScript(ops, m, shots=shots)
            return qml.math.hstack(execute([tape], device, **execute_kwargs)[0], like="tensorflow")

        with tf.GradientTape() as tape:
            cost_res = cost(a, b)

        assert cost_res.shape == (2,)

        res = tape.jacobian(cost_res, [a, b])
        assert len(res) == 2
        assert all(r is None for r in res)

        def loss(a, b):
            return tf.reduce_sum(cost(a, b))

        with tf.GradientTape() as tape:
            loss_res = loss(a, b)

        res = tape.gradient(loss_res, [a, b])
        assert all(r is None for r in res)

    def test_matrix_parameter(self, execute_kwargs, device, shots):
        """Test that the tensorflow interface works correctly
        with a matrix parameter"""
        U = tf.constant([[0, 1], [1, 0]], dtype=tf.complex128)
        a = tf.Variable(0.1)

        def cost(a, U):
            ops = [qml.QubitUnitary(U, wires=0), qml.RY(a, wires=0)]
            tape = qml.tape.QuantumScript(ops, [qml.expval(qml.PauliZ(0))])
            return execute([tape], device, **execute_kwargs)[0]

        with tf.GradientTape() as tape:
            res = cost(a, U)

        assert np.allclose(res, -tf.cos(a), atol=atol_for_shots(shots))

        jac = tape.jacobian(res, a)
        assert isinstance(jac, tf.Tensor)
        assert np.allclose(jac, tf.sin(a), atol=atol_for_shots(shots), rtol=0)

    def test_differentiable_expand(self, execute_kwargs, device, shots):
        """Test that operation and nested tapes expansion
        is differentiable"""

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
                [qml.RX(a, wires=0), U3(*p, wires=0)], [qml.expval(qml.PauliX(0))]
            )
            gradient_fn = execute_kwargs["gradient_fn"]
            if gradient_fn is None:
                _gradient_method = None
            elif isinstance(gradient_fn, str):
                _gradient_method = gradient_fn
            else:
                _gradient_method = "gradient-transform"
            config = qml.devices.ExecutionConfig(
                interface="autograd",
                gradient_method=_gradient_method,
                grad_on_execution=execute_kwargs.get("grad_on_execution", None),
            )
            program, _ = device.preprocess(execution_config=config)
            return execute([tape], device, **execute_kwargs, transform_program=program)[0]

        a = tf.constant(0.1)
        p = tf.Variable([0.1, 0.2, 0.3])

        with tf.GradientTape() as tape:
            cost_res = cost_fn(a, p)

        expected = tf.cos(a) * tf.cos(p[1]) * tf.sin(p[0]) + tf.sin(a) * (
            tf.cos(p[2]) * tf.sin(p[1]) + tf.cos(p[0]) * tf.cos(p[1]) * tf.sin(p[2])
        )
        assert np.allclose(cost_res, expected, atol=atol_for_shots(shots), rtol=0)

        res = tape.jacobian(cost_res, p)
        expected = tf.convert_to_tensor(
            [
                tf.cos(p[1]) * (tf.cos(a) * tf.cos(p[0]) - tf.sin(a) * tf.sin(p[0]) * tf.sin(p[2])),
                tf.cos(p[1]) * tf.cos(p[2]) * tf.sin(a)
                - tf.sin(p[1])
                * (tf.cos(a) * tf.sin(p[0]) + tf.cos(p[0]) * tf.sin(a) * tf.sin(p[2])),
                tf.sin(a)
                * (tf.cos(p[0]) * tf.cos(p[1]) * tf.cos(p[2]) - tf.sin(p[1]) * tf.sin(p[2])),
            ]
        )
        assert np.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)

    def test_probability_differentiation(self, execute_kwargs, device, shots):
        """Tests correct output shape and evaluation for a tape
        with prob outputs"""

        if execute_kwargs["gradient_fn"] == "adjoint":
            pytest.skip("adjoint differentiation does not support probabilities")

        def cost(x, y):
            ops = [qml.RX(x, 0), qml.RY(y, 1), qml.CNOT((0, 1))]
            m = [qml.probs(wires=0), qml.probs(wires=1)]
            tape = qml.tape.QuantumScript(ops, m)
            return qml.math.hstack(execute([tape], device, **execute_kwargs)[0], like="tensorflow")

        x = tf.Variable(0.543)
        y = tf.Variable(-0.654)

        with tf.GradientTape() as tape:
            cost_res = cost(x, y)

        expected = tf.convert_to_tensor(
            [
                [
                    tf.cos(x / 2) ** 2,
                    tf.sin(x / 2) ** 2,
                    (1 + tf.cos(x) * tf.cos(y)) / 2,
                    (1 - tf.cos(x) * tf.cos(y)) / 2,
                ],
            ]
        )
        assert np.allclose(cost_res, expected, atol=atol_for_shots(shots), rtol=0)

        res = tape.jacobian(cost_res, [x, y])
        assert isinstance(res, list) and len(res) == 2
        assert res[0].shape == (4,)
        assert res[1].shape == (4,)

        expected = (
            tf.convert_to_tensor(
                [
                    [
                        -tf.sin(x) / 2,
                        tf.sin(x) / 2,
                        -tf.sin(x) * tf.cos(y) / 2,
                        tf.sin(x) * tf.cos(y) / 2,
                    ],
                ]
            ),
            tf.convert_to_tensor(
                [
                    [0, 0, -tf.cos(x) * tf.sin(y) / 2, tf.cos(x) * tf.sin(y) / 2],
                ]
            ),
        )

        assert np.allclose(res[0], expected[0], atol=atol_for_shots(shots), rtol=0)
        assert np.allclose(res[1], expected[1], atol=atol_for_shots(shots), rtol=0)

    def test_ragged_differentiation(self, execute_kwargs, device, shots):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""
        if execute_kwargs["gradient_fn"] == "adjoint":
            pytest.skip("Adjoint differentiation does not yet support probabilities")

        def cost(x, y):
            ops = [qml.RX(x, wires=0), qml.RY(y, 1), qml.CNOT((0, 1))]
            m = [qml.expval(qml.PauliZ(0)), qml.probs(wires=1)]
            tape = qml.tape.QuantumScript(ops, m)
            return qml.math.hstack(execute([tape], device, **execute_kwargs)[0], like="tensorflow")

        x = tf.Variable(0.543)
        y = tf.Variable(-0.654)

        with tf.GradientTape() as tape:
            cost_res = cost(x, y)

        expected = tf.convert_to_tensor(
            [tf.cos(x), (1 + tf.cos(x) * tf.cos(y)) / 2, (1 - tf.cos(x) * tf.cos(y)) / 2]
        )
        assert np.allclose(cost_res, expected, atol=atol_for_shots(shots), rtol=0)

        res = tape.jacobian(cost_res, [x, y])
        assert isinstance(res, list) and len(res) == 2
        assert res[0].shape == (3,)
        assert res[1].shape == (3,)

        expected = (
            tf.convert_to_tensor(
                [-tf.sin(x), -tf.sin(x) * tf.cos(y) / 2, tf.sin(x) * tf.cos(y) / 2]
            ),
            tf.convert_to_tensor([0, -tf.cos(x) * tf.sin(y) / 2, tf.cos(x) * tf.sin(y) / 2]),
        )
        assert np.allclose(res[0], expected[0], atol=atol_for_shots(shots), rtol=0)
        assert np.allclose(res[1], expected[1], atol=atol_for_shots(shots), rtol=0)


class TestHigherOrderDerivatives:
    """Test that the tensorflow execute function can be differentiated"""

    @pytest.mark.parametrize(
        "params",
        [
            tf.Variable([0.543, -0.654], dtype=tf.float64),
            tf.Variable([0, -0.654], dtype=tf.float64),
            tf.Variable([-2.0, 0], dtype=tf.float64),
        ],
    )
    def test_parameter_shift_hessian(self, params, tol):
        """Tests that the output of the parameter-shift transform
        can be differentiated using tensorflow, yielding second derivatives."""
        dev = DefaultQubit()

        def cost_fn(x):
            ops1 = [qml.RX(x[0], 0), qml.RY(x[1], 1), qml.CNOT((0, 1))]
            tape1 = qml.tape.QuantumScript(ops1, [qml.var(qml.PauliZ(0) @ qml.PauliX(1))])

            ops2 = [qml.RX(x[0], 0), qml.RY(x[0], 1), qml.CNOT((0, 1))]
            tape2 = qml.tape.QuantumScript(ops2, [qml.probs(wires=1)])
            result = execute([tape1, tape2], dev, gradient_fn=param_shift, max_diff=2)
            return result[0] + result[1][0]

        with tf.GradientTape() as jac_tape:
            with tf.GradientTape() as grad_tape:
                res = cost_fn(params)
            grad = grad_tape.gradient(res, params)
        hess = jac_tape.jacobian(grad, params)

        x, y = params
        expected = 0.5 * (3 + tf.cos(x) ** 2 * tf.cos(2 * y))
        assert np.allclose(res, expected, atol=tol, rtol=0)

        expected = tf.convert_to_tensor(
            [-tf.cos(x) * tf.cos(2 * y) * tf.sin(x), -tf.cos(x) ** 2 * tf.sin(2 * y)]
        )
        assert np.allclose(grad, expected, atol=tol, rtol=0)

        expected = tf.convert_to_tensor(
            [
                [-tf.cos(2 * x) * tf.cos(2 * y), tf.sin(2 * x) * tf.sin(2 * y)],
                [tf.sin(2 * x) * tf.sin(2 * y), -2 * tf.cos(x) ** 2 * tf.cos(2 * y)],
            ]
        )
        assert np.allclose(hess, expected, atol=tol, rtol=0)

    def test_max_diff(self, tol):
        """Test that setting the max_diff parameter blocks higher-order
        derivatives"""
        dev = DefaultQubit()
        params = tf.Variable([0.543, -0.654])

        def cost_fn(x):
            ops = [qml.RX(x[0], 0), qml.RY(x[1], 1), qml.CNOT((0, 1))]
            tape1 = qml.tape.QuantumScript(ops, [qml.var(qml.PauliZ(0) @ qml.PauliX(1))])

            ops2 = [qml.RX(x[0], 0), qml.RY(x[0], 1), qml.CNOT((0, 1))]
            tape2 = qml.tape.QuantumScript(ops2, [qml.probs(wires=1)])

            result = execute([tape1, tape2], dev, gradient_fn=param_shift, max_diff=1)
            return result[0] + result[1][0]

        with tf.GradientTape() as jac_tape:
            with tf.GradientTape() as grad_tape:
                res = cost_fn(params)
            grad = grad_tape.gradient(res, params)
        hess = jac_tape.gradient(grad, params)

        x, y = params
        expected = 0.5 * (3 + tf.cos(x) ** 2 * tf.cos(2 * y))
        assert np.allclose(res, expected, atol=tol, rtol=0)

        expected = tf.convert_to_tensor(
            [-tf.cos(x) * tf.cos(2 * y) * tf.sin(x), -tf.cos(x) ** 2 * tf.sin(2 * y)]
        )
        assert np.allclose(grad, expected, atol=tol, rtol=0)
        assert hess is None


@pytest.mark.parametrize("execute_kwargs, shots, device", test_matrix)
@pytest.mark.parametrize("use_new_op_math", (True, False))
class TestHamiltonianWorkflows:
    """Test that tapes ending with expectations
    of Hamiltonians provide correct results and gradients"""

    @pytest.fixture
    def cost_fn(self, execute_kwargs, shots, device, use_new_op_math):
        """Cost function for gradient tests"""

        def _cost_fn(weights, coeffs1, coeffs2):
            obs1 = [qml.PauliZ(0), qml.PauliZ(0) @ qml.PauliX(1), qml.PauliY(0)]
            H1 = qml.Hamiltonian(coeffs1, obs1)
            if use_new_op_math:
                H1 = qml.pauli.pauli_sentence(H1).operation()

            obs2 = [qml.PauliZ(0)]
            H2 = qml.Hamiltonian(coeffs2, obs2)
            if use_new_op_math:
                H2 = qml.pauli.pauli_sentence(H2).operation()

            with qml.queuing.AnnotatedQueue() as q:
                qml.RX(weights[0], wires=0)
                qml.RY(weights[1], wires=1)
                qml.CNOT(wires=[0, 1])
                qml.expval(H1)
                qml.expval(H2)

            tape = qml.tape.QuantumScript.from_queue(q, shots=shots)
            return qml.math.hstack(execute([tape], device, **execute_kwargs)[0], like="tensorflow")

        return _cost_fn

    @staticmethod
    def cost_fn_expected(weights, coeffs1, coeffs2):
        """Analytic value of cost_fn above"""
        a, b, c = coeffs1
        d = coeffs2[0]
        x, y = weights
        return [-c * tf.sin(x) * tf.sin(y) + tf.cos(x) * (a + b * tf.sin(y)), d * tf.cos(x)]

    @staticmethod
    def cost_fn_jacobian(weights, coeffs1, coeffs2):
        """Analytic jacobian of cost_fn above"""
        a, b, c = coeffs1
        d = coeffs2[0]
        x, y = weights
        return tf.convert_to_tensor(
            [
                [
                    -c * tf.cos(x) * tf.sin(y) - tf.sin(x) * (a + b * tf.sin(y)),
                    b * tf.cos(x) * tf.cos(y) - c * tf.cos(y) * tf.sin(x),
                    tf.cos(x),
                    tf.cos(x) * tf.sin(y),
                    -(tf.sin(x) * tf.sin(y)),
                    0,
                ],
                [-d * tf.sin(x), 0, 0, 0, 0, tf.cos(x)],
            ]
        )

    def test_multiple_hamiltonians_not_trainable(
        self, execute_kwargs, cost_fn, shots, use_new_op_math
    ):
        """Test hamiltonian with no trainable parameters."""

        if execute_kwargs["gradient_fn"] == "adjoint" and not use_new_op_math:
            pytest.skip("adjoint differentiation does not suppport hamiltonians.")

        coeffs1 = tf.constant([0.1, 0.2, 0.3], dtype=tf.float64)
        coeffs2 = tf.constant([0.7], dtype=tf.float64)
        weights = tf.Variable([0.4, 0.5], dtype=tf.float64)

        with tf.GradientTape() as tape:
            res = cost_fn(weights, coeffs1, coeffs2)

        expected = self.cost_fn_expected(weights, coeffs1, coeffs2)
        assert np.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)

        jac = tape.jacobian(res, [weights])
        expected = self.cost_fn_jacobian(weights, coeffs1, coeffs2)[:, :2]
        assert np.allclose(jac, expected, atol=atol_for_shots(shots), rtol=0)

    def test_multiple_hamiltonians_trainable(self, cost_fn, execute_kwargs, shots, use_new_op_math):
        """Test hamiltonian with trainable parameters."""
        if execute_kwargs["gradient_fn"] == "adjoint":
            pytest.skip("trainable hamiltonians not supported with adjoint")
        if use_new_op_math:
            pytest.skip("parameter shift derivatives do not yet support sums.")

        coeffs1 = tf.Variable([0.1, 0.2, 0.3], dtype=tf.float64)
        coeffs2 = tf.Variable([0.7], dtype=tf.float64)
        weights = tf.Variable([0.4, 0.5], dtype=tf.float64)

        with tf.GradientTape() as tape:
            res = cost_fn(weights, coeffs1, coeffs2)

        expected = self.cost_fn_expected(weights, coeffs1, coeffs2)
        assert np.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)

        jac = qml.math.hstack(tape.jacobian(res, [weights, coeffs1, coeffs2]), like="tensorflow")
        expected = self.cost_fn_jacobian(weights, coeffs1, coeffs2)
        assert np.allclose(jac, expected, atol=atol_for_shots(shots), rtol=0)

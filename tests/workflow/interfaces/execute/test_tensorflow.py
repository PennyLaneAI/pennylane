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
import numpy as np
import pytest

import pennylane as qml
from pennylane import execute
from pennylane.devices import DefaultQubit
from pennylane.gradients import param_shift

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
                [tape], dev, diff_method=qml.gradients.param_shift, cache=cache, max_diff=2
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
    ({"diff_method": param_shift, "interface": "tensorflow"}, 100000, "default.qubit"),  # 0
    ({"diff_method": param_shift, "interface": "tensorflow"}, None, "default.qubit"),  # 1
    ({"diff_method": "backprop", "interface": "tensorflow"}, None, "default.qubit"),  # 2
    ({"diff_method": "adjoint", "interface": "tensorflow"}, None, "default.qubit"),  # 3
    ({"diff_method": param_shift, "interface": "tf-autograph"}, 100000, "default.qubit"),  # 4
    ({"diff_method": param_shift, "interface": "tf-autograph"}, None, "default.qubit"),  # 5
    ({"diff_method": "backprop", "interface": "tf-autograph"}, None, "default.qubit"),  # 6
    ({"diff_method": "adjoint", "interface": "tf-autograph"}, None, "default.qubit"),  # 7
    ({"diff_method": "adjoint", "interface": "tf", "device_vjp": True}, None, "default.qubit"),  # 8
    ({"diff_method": param_shift, "interface": "tensorflow"}, None, "reference.qubit"),  # 9
    (
        {"diff_method": param_shift, "interface": "tensorflow"},
        100000,
        "reference.qubit",
    ),  # 10
]


def atol_for_shots(shots):
    """Return higher tolerance if finite shots."""
    return 1e-2 if shots else 1e-6


@pytest.mark.parametrize("execute_kwargs, shots, device_name", test_matrix)
class TestTensorflowExecuteIntegration:
    """Test the tensorflow interface execute function
    integrates well for both forward and backward execution"""

    def test_execution(self, execute_kwargs, shots, device_name, seed):
        """Test execution"""

        device = qml.device(device_name, seed=seed)

        def cost(a, b):
            ops1 = [qml.RY(a, wires=0), qml.RX(b, wires=0)]
            tape1 = qml.tape.QuantumScript(ops1, [qml.expval(qml.PauliZ(0))], shots=shots)

            ops2 = [qml.RY(a, wires="a"), qml.RX(b, wires="a")]
            tape2 = qml.tape.QuantumScript(ops2, [qml.expval(qml.PauliZ("a"))], shots=shots)

            return execute([tape1, tape2], device, **execute_kwargs)

        a = tf.Variable(0.1, dtype="float64")
        b = tf.constant(0.2, dtype="float64")
        with device.tracker:
            res = cost(a, b)

        if execute_kwargs.get("diff_method", None) == "adjoint" and not execute_kwargs.get(
            "device_vjp", False
        ):
            assert device.tracker.totals["execute_and_derivative_batches"] == 1
        else:
            assert device.tracker.totals["batches"] == 1
        assert device.tracker.totals["executions"] == 2  # different wires so different hashes

        assert len(res) == 2
        assert res[0].shape == ()
        assert res[1].shape == ()

        assert qml.math.allclose(res[0], tf.cos(a) * tf.cos(b), atol=atol_for_shots(shots))
        assert qml.math.allclose(res[1], tf.cos(a) * tf.cos(b), atol=atol_for_shots(shots))

    def test_scalar_jacobian(self, execute_kwargs, shots, device_name, seed):
        """Test scalar jacobian calculation"""
        a = tf.Variable(0.1, dtype=tf.float64)

        device_vjp = execute_kwargs.get("device_vjp", False)

        device = qml.device(device_name, seed=seed)

        def cost(a):
            tape = qml.tape.QuantumScript([qml.RY(a, 0)], [qml.expval(qml.PauliZ(0))], shots=shots)
            return execute([tape], device, **execute_kwargs)[0]

        with tf.GradientTape(persistent=device_vjp) as tape:
            cost_res = cost(a)
        res = tape.jacobian(cost_res, a, experimental_use_pfor=not device_vjp)
        assert res.shape == ()  # pylint: disable=no-member

        expected = -qml.math.sin(a)

        assert expected.shape == ()
        assert np.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)
        assert np.allclose(res, -tf.sin(a), atol=atol_for_shots(shots))

    def test_jacobian(self, execute_kwargs, shots, device_name, seed):
        """Test jacobian calculation"""
        a = tf.Variable(0.1)
        b = tf.Variable(0.2)

        device = qml.device(device_name, seed=seed)
        device_vjp = execute_kwargs.get("device_vjp", False)

        def cost(a, b):
            ops = [qml.RY(a, wires=0), qml.RX(b, wires=1), qml.CNOT(wires=[0, 1])]
            m = [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))]
            tape = qml.tape.QuantumScript(ops, m, shots=shots)
            return qml.math.hstack(execute([tape], device, **execute_kwargs)[0], like="tensorflow")

        with tf.GradientTape(persistent=device_vjp) as tape:
            res = cost(a, b)
        expected = [tf.cos(a), -tf.cos(a) * tf.sin(b)]
        assert np.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)

        jac = tape.jacobian(res, [a, b], experimental_use_pfor=not device_vjp)
        assert isinstance(jac, list) and len(jac) == 2
        assert jac[0].shape == (2,)
        assert jac[1].shape == (2,)

        expected = ([-tf.sin(a), tf.sin(a) * tf.sin(b)], [0, -tf.cos(a) * tf.cos(b)])
        for _r, _e in zip(jac, expected):
            assert np.allclose(_r, _e, atol=atol_for_shots(shots))

    def test_tape_no_parameters(self, execute_kwargs, shots, device_name, seed):
        """Test that a tape with no parameters is correctly
        ignored during the gradient computation"""

        device = qml.device(device_name, seed=seed)

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

        if (
            execute_kwargs.get("interface", "") == "tf-autograph"
            and execute_kwargs.get("diff_method", "") == "adjoint"
        ):
            with pytest.raises(NotImplementedError):
                tape.gradient(res, params)
            return

        grad = tape.gradient(res, params)
        expected = [-tf.cos(y) * tf.sin(x), -tf.cos(x) * tf.sin(y)]
        assert np.allclose(grad, expected, atol=atol_for_shots(shots), rtol=0)

    def test_tapes_with_different_return_size(self, execute_kwargs, shots, device_name, seed):
        """Test that tapes wit different can be executed and differentiated."""

        device = qml.device(device_name, seed=seed)

        if (
            execute_kwargs["diff_method"] == "adjoint"
            and execute_kwargs["interface"] == "tf-autograph"
        ):
            pytest.skip("Cannot compute the jacobian with adjoint-differentation and tf-autograph")

        device_vjp = execute_kwargs.get("device_vjp", False)

        def cost(params):
            tape1 = qml.tape.QuantumScript(
                [qml.RY(params[0], 0), qml.RX(params[1], 0)],
                [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))],
                shots=shots,
            )

            tape2 = qml.tape.QuantumScript(
                [qml.RY(tf.constant(0.5, dtype=tf.float64), 0)],
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

        params = tf.Variable([0.1, 0.2], dtype=tf.float64)
        x, y = params

        with tf.GradientTape(persistent=device_vjp) as tape:
            res = cost(params)

        assert isinstance(res, tf.Tensor)
        assert res.shape == (4,)

        assert np.allclose(res[0], tf.cos(x) * tf.cos(y), atol=atol_for_shots(shots))
        assert np.allclose(res[1], 1, atol=atol_for_shots(shots))
        assert np.allclose(res[2], tf.cos(0.5), atol=atol_for_shots(shots))
        assert np.allclose(res[3], tf.cos(x) * tf.cos(y), atol=atol_for_shots(shots))

        jac = tape.jacobian(res, params, experimental_use_pfor=not device_vjp)
        assert isinstance(jac, tf.Tensor)
        assert jac.shape == (4, 2)  # pylint: disable=no-member

        assert np.allclose(jac[1:3], 0, atol=atol_for_shots(shots))

        d1 = -tf.sin(x) * tf.cos(y)
        assert np.allclose(jac[0, 0], d1, atol=atol_for_shots(shots))
        assert np.allclose(jac[3, 0], d1, atol=atol_for_shots(shots))

        d2 = -tf.cos(x) * tf.sin(y)
        assert np.allclose(jac[0, 1], d2, atol=atol_for_shots(shots))
        assert np.allclose(jac[3, 1], d2, atol=atol_for_shots(shots))

    def test_reusing_quantum_tape(self, execute_kwargs, shots, device_name, seed):
        """Test re-using a quantum tape by passing new parameters"""
        a = tf.Variable(0.1)
        b = tf.Variable(0.2)
        device = qml.device(device_name, seed=seed)

        tape = qml.tape.QuantumScript(
            [qml.RY(a, 0), qml.RX(b, 1), qml.CNOT((0, 1))],
            [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))],
        )
        assert tape.trainable_params == [0, 1]

        device_vjp = execute_kwargs.get("device_vjp", False)

        def cost(a, b):
            new_tape = tape.bind_new_parameters([a, b], [0, 1])
            return qml.math.hstack(
                execute([new_tape], device, **execute_kwargs)[0], like="tensorflow"
            )

        with tf.GradientTape(persistent=device_vjp) as t:
            res = cost(a, b)

        jac = t.jacobian(res, [a, b], experimental_use_pfor=not device_vjp)
        a = tf.Variable(0.54, dtype=tf.float64)
        b = tf.Variable(0.8, dtype=tf.float64)

        # check that the cost function continues to depend on the
        # values of the parameters for subsequent calls

        with tf.GradientTape(persistent=device_vjp):
            res2 = cost(2 * a, b)

        expected = [tf.cos(2 * a), -tf.cos(2 * a) * tf.sin(b)]
        assert np.allclose(res2, expected, atol=atol_for_shots(shots), rtol=0)

        with tf.GradientTape(persistent=device_vjp) as t:
            res = cost(2 * a, b)

        jac = t.jacobian(res, [a, b], experimental_use_pfor=not device_vjp)
        expected = (
            [-2 * tf.sin(2 * a), 2 * tf.sin(2 * a) * tf.sin(b)],
            [0, -tf.cos(2 * a) * tf.cos(b)],
        )
        assert isinstance(jac, list) and len(jac) == 2
        for _j, _e in zip(jac, expected):
            assert np.allclose(_j, _e, atol=atol_for_shots(shots), rtol=0)

    def test_classical_processing(self, execute_kwargs, device_name, seed, shots):
        """Test classical processing within the quantum tape"""
        a = tf.Variable(0.1, dtype=tf.float64)
        b = tf.constant(0.2, dtype=tf.float64)
        c = tf.Variable(0.3, dtype=tf.float64)
        device = qml.device(device_name, seed=seed)

        device_vjp = execute_kwargs.get("device_vjp", False)

        def cost(a, b, c):
            ops = [
                qml.RY(a * c, wires=0),
                qml.RZ(b, wires=0),
                qml.RX(c + c**2 + tf.sin(a), wires=0),
            ]

            tape = qml.tape.QuantumScript(ops, [qml.expval(qml.PauliZ(0))], shots=shots)
            return execute([tape], device, **execute_kwargs)[0]

        with tf.GradientTape(persistent=device_vjp) as tape:
            cost_res = cost(a, b, c)

        res = tape.jacobian(cost_res, [a, c], experimental_use_pfor=not device_vjp)

        # Only two arguments are trainable
        assert isinstance(res, list) and len(res) == 2
        assert res[0].shape == ()
        assert res[1].shape == ()

        # I tried getting analytic results for this circuit but I kept being wrong and am giving up

    def test_no_trainable_parameters(self, execute_kwargs, shots, device_name, seed):
        """Test evaluation and Jacobian if there are no trainable parameters"""
        a = tf.constant(0.1)
        b = tf.constant(0.2)
        device = qml.device(device_name, seed=seed)

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

    def test_matrix_parameter(self, execute_kwargs, device_name, seed, shots):
        """Test that the tensorflow interface works correctly
        with a matrix parameter"""
        U = tf.constant([[0, 1], [1, 0]], dtype=tf.complex128)
        a = tf.Variable(0.1)
        device = qml.device(device_name, seed=seed)
        device_vjp = execute_kwargs.get("device_vjp", False)

        def cost(a, U):
            ops = [qml.QubitUnitary(U, wires=0), qml.RY(a, wires=0)]
            tape = qml.tape.QuantumScript(ops, [qml.expval(qml.PauliZ(0))])
            return execute([tape], device, **execute_kwargs)[0]

        with tf.GradientTape(persistent=device_vjp) as tape:
            res = cost(a, U)

        assert np.allclose(res, -tf.cos(a), atol=atol_for_shots(shots))

        jac = tape.jacobian(res, a, experimental_use_pfor=not device_vjp)
        assert isinstance(jac, tf.Tensor)
        assert np.allclose(jac, tf.sin(a), atol=atol_for_shots(shots), rtol=0)

    def test_differentiable_expand(self, execute_kwargs, device_name, seed, shots):
        """Test that operation and nested tapes expansion
        is differentiable"""

        device = qml.device(device_name, seed=seed)
        device_vjp = execute_kwargs.get("device_vjp", False)

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
            diff_method = execute_kwargs["diff_method"]
            if diff_method is None:
                _gradient_method = None
            elif isinstance(diff_method, str):
                _gradient_method = diff_method
            else:
                _gradient_method = "gradient-transform"
            config = qml.devices.ExecutionConfig(
                interface="autograd",
                gradient_method=_gradient_method,
                grad_on_execution=execute_kwargs.get("grad_on_execution", None),
            )
            program = device.preprocess_transforms(execution_config=config)
            return execute([tape], device, **execute_kwargs, transform_program=program)[0]

        a = tf.constant(0.1)
        p = tf.Variable([0.1, 0.2, 0.3])

        with tf.GradientTape(persistent=device_vjp) as tape:
            cost_res = cost_fn(a, p)

        expected = tf.cos(a) * tf.cos(p[1]) * tf.sin(p[0]) + tf.sin(a) * (
            tf.cos(p[2]) * tf.sin(p[1]) + tf.cos(p[0]) * tf.cos(p[1]) * tf.sin(p[2])
        )
        assert np.allclose(cost_res, expected, atol=atol_for_shots(shots), rtol=0)

        res = tape.jacobian(cost_res, p, experimental_use_pfor=not device_vjp)
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

    def test_probability_differentiation(self, execute_kwargs, device_name, seed, shots):
        """Tests correct output shape and evaluation for a tape
        with prob outputs"""

        device = qml.device(device_name, seed=seed)

        def cost(x, y):
            ops = [qml.RX(x, 0), qml.RY(y, 1), qml.CNOT((0, 1))]
            m = [qml.probs(wires=0), qml.probs(wires=1)]
            tape = qml.tape.QuantumScript(ops, m)
            return qml.math.hstack(execute([tape], device, **execute_kwargs)[0], like="tensorflow")

        device_vjp = execute_kwargs.get("device_vjp", False)

        x = tf.Variable(0.543, dtype=tf.float64)
        y = tf.Variable(-0.654, dtype=tf.float64)

        with tf.GradientTape(persistent=device_vjp) as tape:
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

        if (
            execute_kwargs.get("interface", "") == "tf-autograph"
            and execute_kwargs.get("diff_method", "") == "adjoint"
        ):
            with pytest.raises(tf.errors.UnimplementedError):
                tape.jacobian(cost_res, [x, y])
            return
        res = tape.jacobian(cost_res, [x, y], experimental_use_pfor=not device_vjp)
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

    def test_ragged_differentiation(self, execute_kwargs, device_name, seed, shots):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""

        device_vjp = execute_kwargs.get("device_vjp", False)
        device = qml.device(device_name, seed=seed)

        def cost(x, y):
            ops = [qml.RX(x, wires=0), qml.RY(y, 1), qml.CNOT((0, 1))]
            m = [qml.expval(qml.PauliZ(0)), qml.probs(wires=1)]
            tape = qml.tape.QuantumScript(ops, m)
            return qml.math.hstack(execute([tape], device, **execute_kwargs)[0], like="tensorflow")

        x = tf.Variable(0.543, dtype=tf.float64)
        y = tf.Variable(-0.654, dtype=tf.float64)

        with tf.GradientTape(persistent=device_vjp) as tape:
            cost_res = cost(x, y)

        expected = tf.convert_to_tensor(
            [tf.cos(x), (1 + tf.cos(x) * tf.cos(y)) / 2, (1 - tf.cos(x) * tf.cos(y)) / 2]
        )
        assert np.allclose(cost_res, expected, atol=atol_for_shots(shots), rtol=0)

        if (
            execute_kwargs.get("interface", "") == "tf-autograph"
            and execute_kwargs.get("diff_method", "") == "adjoint"
        ):
            with pytest.raises(tf.errors.UnimplementedError):
                tape.jacobian(cost_res, [x, y])
            return
        res = tape.jacobian(cost_res, [x, y], experimental_use_pfor=not device_vjp)
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
            result = execute([tape1, tape2], dev, diff_method=param_shift, max_diff=2)
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

            result = execute([tape1, tape2], dev, diff_method=param_shift, max_diff=1)
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


@pytest.mark.parametrize("execute_kwargs, shots, device_name", test_matrix)
@pytest.mark.parametrize("constructor", (qml.Hamiltonian, qml.dot, "dunders"))
class TestHamiltonianWorkflows:
    """Test that tapes ending with expectations
    of Hamiltonians provide correct results and gradients"""

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    @pytest.fixture
    def cost_fn(self, execute_kwargs, shots, device_name, seed, constructor):
        """Cost function for gradient tests"""

        device = qml.device(device_name, seed=seed)

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

    def test_multiple_hamiltonians_not_trainable(self, execute_kwargs, cost_fn, shots):
        """Test hamiltonian with no trainable parameters."""

        device_vjp = execute_kwargs.get("device_vjp", False)

        coeffs1 = tf.constant([0.1, 0.2, 0.3], dtype=tf.float64)
        coeffs2 = tf.constant([0.7], dtype=tf.float64)
        weights = tf.Variable([0.4, 0.5], dtype=tf.float64)

        with tf.GradientTape(persistent=device_vjp) as tape:
            res = cost_fn(weights, coeffs1, coeffs2)

        expected = self.cost_fn_expected(weights, coeffs1, coeffs2)
        assert np.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)

        jac = tape.jacobian(res, [weights], experimental_use_pfor=not device_vjp)
        expected = self.cost_fn_jacobian(weights, coeffs1, coeffs2)[:, :2]
        assert np.allclose(jac, expected, atol=atol_for_shots(shots), rtol=0)

    def test_multiple_hamiltonians_trainable(self, cost_fn, execute_kwargs, shots):
        """Test hamiltonian with trainable parameters."""
        if execute_kwargs["diff_method"] == "adjoint":
            pytest.skip("trainable hamiltonians not supported with adjoint")
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


@pytest.mark.parametrize("diff_method", ("adjoint", "parameter-shift"))
def test_device_returns_float32(diff_method):
    """Test that if the device returns float32, the derivative succeeds."""

    def _to_float32(results):
        if isinstance(results, (list, tuple)):
            return tuple(_to_float32(r) for r in results)
        return np.array(results, dtype=np.float32)

    class Float32Dev(qml.devices.DefaultQubit):
        def execute(self, circuits, execution_config=qml.devices.DefaultExecutionConfig):
            results = super().execute(circuits, execution_config)
            return _to_float32(results)

    dev = Float32Dev()

    @qml.qnode(dev, diff_method=diff_method)
    def circuit(x):
        qml.RX(tf.cos(x), wires=0)
        return qml.expval(qml.Z(0))

    x = tf.Variable(0.1, dtype=tf.float64)

    with tf.GradientTape() as tape:
        y = circuit(x)

    assert qml.math.allclose(y, np.cos(np.cos(0.1)))

    g = tape.gradient(y, x)
    expected_g = np.sin(np.cos(0.1)) * np.sin(0.1)
    assert qml.math.allclose(g, expected_g)


def test_autograph_with_sample():
    """Test tensorflow autograph with sampling."""

    @tf.function
    @qml.qnode(qml.device("default.qubit", shots=50))
    def circuit(x):
        qml.RX(x, 0)
        return qml.sample(wires=0)

    res = circuit(tf.Variable(0.0))
    assert qml.math.allclose(res, np.zeros(50))

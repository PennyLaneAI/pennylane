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
"""
Tests for the gradients.pulse_gradient module.
"""
# pylint:disable=import-outside-toplevel

import copy
import pytest
import numpy as np
import pennylane as qml

from pennylane.gradients.pulse_gradient import split_evol_ops, split_evol_tapes, stoch_pulse_grad
from pennylane.pulse import ParametrizedEvolution

pytestmark = pytest.mark.jax



def equal(op1, op2, check_t=True):
    """Helper function to check whether two operations are the same via qml.equal, optionally
    including the time window for ParametrizedEvolution.

    TODO: Remove once #3846 is resolved/ #3870 is merged.
    """
    if (
        check_t
        and isinstance(op1, ParametrizedEvolution)
        and isinstance(op2, ParametrizedEvolution)
    ):
        return qml.equal(op1, op2) and qml.math.allclose(op1.t, op2.t)
    return qml.equal(op1, op2)


class TestSplitEvolOps:
    """Tests for the helper method split_evol_ops that samples a splitting time and splits up
    a ParametrizedEvolution operation at the sampled time, inserting a Pauli rotation about the
    provided Pauli word with angles +- pi/2."""

    ham_single_q_fixed = 0.4 * qml.PauliX(0)
    ham_single_q_const = qml.pulse.constant * qml.PauliY(0)
    ham_single_q_pwc = qml.pulse.pwc((2.0, 4.0)) * qml.PauliZ(0)
    ham_two_q_pwc = qml.pulse.pwc((2.0, 4.0)) * (qml.PauliZ(0) @ qml.PauliX(1))

    split_evol_ops_test_cases = [
        (ham_single_q_const, [0.3], 2.3, "X", 0),
        (ham_single_q_const, [0.3], 2.3, "X", ["aux"]),
        (ham_single_q_pwc, [np.linspace(0, 1, 13)], (0.6, 1.2), "Y", [1]),
        (ham_two_q_pwc, [np.linspace(0, 1, 13)], 3, "YX", [0, "a"]),
    ]

    @pytest.mark.parametrize("test_case", split_evol_ops_test_cases)
    def test_output_properties(self, test_case):
        """Test that split_evol_ops returns the right objects with correct
        relations to the input operation."""

        import jax

        ham, params, time, word, word_wires = test_case
        key = jax.random.PRNGKey(5324)
        op = qml.evolve(ham)(params, time)
        op_copy = copy.deepcopy(op)
        exp_time = [0, time] if qml.math.isscalar(time) else time
        assert qml.math.allclose(op.t, exp_time)
        output = split_evol_ops(op, word, word_wires, key)
        # Check that the original operation was not altered
        assert equal(op, op_copy)

        # Check that tau and ops is returned
        assert isinstance(output, tuple) and len(output) == 2

        tau, ops = output

        # Check format and range of tau
        assert isinstance(tau, jax.Array) and tau.shape == ()
        assert op.t[0] <= tau <= op.t[1]

        assert isinstance(ops, tuple) and len(ops) == 2

        for sign, _ops in zip([1, -1], ops):
            assert isinstance(_ops, list) and len(_ops) == 3
            # Check that the split-up time evolution is correct
            assert equal(_ops[0], op, check_t=False)
            assert qml.math.allclose(_ops[0].t, [op.t[0], tau])
            assert _ops[0].wires == op.wires

            assert equal(_ops[2], op, check_t=False)
            assert qml.math.allclose(_ops[2].t, [tau, op.t[1]])
            assert _ops[2].wires == op.wires

            # Check that the inserted Pauli rotation is correct
            assert isinstance(_ops[1], qml.PauliRot)
            assert qml.math.allclose(_ops[1].data, sign * np.pi / 2)
            assert _ops[1].hyperparameters["pauli_word"] == word
            assert _ops[1].wires == qml.wires.Wires(word_wires)

    @pytest.mark.parametrize("test_case", split_evol_ops_test_cases)
    def test_randomness_key(self, test_case):
        """Test that split_evol_ops returns the same when used twice with the same
        randomness key and different results when used with different keys."""
        import jax

        ham, params, time, word, word_wires = test_case
        op = qml.evolve(ham)(params, time)
        key_a = jax.random.PRNGKey(5324)
        key_b = jax.random.PRNGKey(7281)
        tau_a_0, ops_a_0 = split_evol_ops(op, word, word_wires, key_a)
        tau_a_1, ops_a_1 = split_evol_ops(op, word, word_wires, key_a)
        tau_b, ops_b = split_evol_ops(op, word, word_wires, key_b)

        # Check same output for same key
        assert qml.math.isclose(tau_a_0, tau_a_1)
        assert all(
            equal(o_0, o_1) for o_0, o_1 in zip(np.array(ops_a_0).flat, np.array(ops_a_1).flat)
        )

        # Check different output for different key
        assert not qml.math.isclose(tau_a_0, tau_b)
        assert not all(
            equal(o_0, o_1) for o_0, o_1 in zip(np.array(ops_a_0).flat, np.array(ops_b).flat)
        )
        # Check positive and negative Pauli rotations are the same even for different keys
        assert equal(ops_a_0[0][1], ops_b[0][1])
        assert equal(ops_a_0[1][1], ops_b[1][1])


class TestSplitEvolTapes:
    """Tests for the helper method split_evol_tapes that replaces an indicated operation by other operations
    and creates a new tape for each provided set of replacing operations."""

    ham_single_q_pwc = qml.pulse.pwc((2.0, 4.0)) * qml.PauliZ(0)

    def test_with_standard_ops(self):
        """Test basic behaviour of the operation replacement with standard qml ops."""
        ops = [qml.RX(0.4, 2), qml.PauliZ(0), qml.CNOT([0, 2])]
        tape = qml.tape.QuantumScript(ops)
        split_evolve_ops = (
            [qml.RX(0.6, 2), qml.PauliY(0), qml.RZ(0.0, 0)],
            [qml.PauliX(0), qml.PauliZ(2)],
        )
        new_tapes = split_evol_tapes(tape, split_evolve_ops, 1)
        assert len(new_tapes) == 2
        for t, new_ops in zip(new_tapes, split_evolve_ops):
            assert qml.equal(t.operations[0], ops[0])
            assert all(qml.equal(o1, o2) for o1, o2 in zip(t.operations[1:-1], new_ops))
            assert qml.equal(t.operations[-1], ops[2])

    def test_with_parametrized_evolution(self):
        """Test basic behaviour of the operation replacement with ParametrizedEvolution."""

        ops = [qml.evolve(self.ham_single_q_pwc)([np.linspace(0, 1, 9)], 0.4)]
        tape = qml.tape.QuantumScript(ops)
        split_evolve_ops = (
            [qml.RX(0.6, 2), qml.PauliY(0), qml.RZ(0.0, 0)],
            [qml.PauliX(0), qml.PauliZ(2)],
        )
        new_tapes = split_evol_tapes(tape, split_evolve_ops, 0)
        assert len(new_tapes) == 2
        for t, new_ops in zip(new_tapes, split_evolve_ops):
            assert all(qml.equal(o1, o2) for o1, o2 in zip(t.operations, new_ops))

        ops = [qml.evolve(self.ham_single_q_pwc)([np.linspace(0, 1, 9)], 0.4), qml.CNOT([0, 2])]
        tape = qml.tape.QuantumScript(ops)
        split_evolve_ops = ([qml.RX(0.6, 2), qml.PauliY(0), qml.RZ(0.0, 0)], [])
        new_tapes = split_evol_tapes(tape, split_evolve_ops, 0)
        assert len(new_tapes) == 2
        for t, new_ops in zip(new_tapes, split_evolve_ops):
            assert all(qml.equal(o1, o2) for o1, o2 in zip(t.operations[:-1], new_ops))
            assert qml.equal(t.operations[-1], ops[1])

        ops = [
            qml.RX(0.4, 2),
            qml.evolve(self.ham_single_q_pwc)([np.linspace(0, 1, 9)], 0.4),
            qml.CNOT([0, 2]),
        ]
        tape = qml.tape.QuantumScript(ops)
        split_evolve_ops = (
            [qml.RX(0.6, 2), qml.PauliY(0), qml.RZ(0.0, 0)],
            [qml.PauliX(0), qml.PauliZ(2)],
        )
        new_tapes = split_evol_tapes(tape, split_evolve_ops, 1)
        assert len(new_tapes) == 2
        for t, new_ops in zip(new_tapes, split_evolve_ops):
            assert qml.equal(t.operations[0], ops[0])
            assert all(qml.equal(o1, o2) for o1, o2 in zip(t.operations[1:-1], new_ops))
            assert qml.equal(t.operations[-1], ops[2])


class TestStochPulseGradErrors:
    """Test errors raised by stoch_pulse_grad."""

    def test_raises_if_jax_not_installed(self):
        """Test that an error is raised if a convenience function is called without jax installed"""
        try:
            import jax  # pylint: disable=unused-import

            pytest.skip()
        except ImportError:
            tape = qml.tape.QuantumScript([])
            with pytest.raises(ImportError, match="Module jax is required"):
                stoch_pulse_grad(tape)

    def test_raises_for_variance(self):
        """Test that an error is raised when attempting to differentiate a tape that measures a variance."""
        tape = qml.tape.QuantumScript(
            measurements=[qml.expval(qml.PauliX(2)), qml.var(qml.PauliZ(0))]
        )
        with pytest.raises(
            ValueError, match="Computing the gradient of variances with the stochastic"
        ):
            stoch_pulse_grad(tape)

    @pytest.mark.parametrize(
        "measurement", [qml.vn_entropy(0), qml.state(), qml.mutual_info([0], 1)]
    )
    def test_raises_for_state_measurements(self, measurement):
        """Test that an error is raised when attempting to differentiate a tape that measures a
        state, or returns a state indirectly via entropy/mutual info measurements."""
        tape = qml.tape.QuantumScript(measurements=[measurement])
        with pytest.raises(
            ValueError, match="Computing the gradient of circuits that return the state"
        ):
            stoch_pulse_grad(tape)

    @pytest.mark.parametrize("num_samples", [-1, 0, np.array(-2)])
    def test_raises_for_less_than_one_sample(self, num_samples):
        """Test that an error is raised if fewer than one samples for the stochastic shift rule are requested."""
        tape = qml.tape.QuantumScript([])
        with pytest.raises(ValueError, match="Expected a positive number of samples"):
            stoch_pulse_grad(tape, num_samples=num_samples)

    @pytest.mark.parametrize("num_meas", [0, 1, 2])
    def test_warning_no_trainable_params(self, num_meas):
        """Test that an empty gradient is returned when there are no trainable parameters."""
        measurements = [qml.expval(qml.PauliX(w)) for w in range(num_meas)]
        # No parameters at all
        tape = qml.tape.QuantumScript([], measurements=measurements)
        with pytest.warns(
            UserWarning, match="Attempted to compute the gradient of a tape with no trainable"
        ):
            tapes, fn = stoch_pulse_grad(tape)
        assert not tapes
        assert qml.math.allclose(fn([]), tuple(qml.math.zeros([0]) for _ in range(num_meas)))

        # parameters but none are trainable
        ops = [qml.RX(0.4, 2), qml.CRY(0.1, [1, 0])]
        tape = qml.tape.QuantumScript(ops, measurements=measurements)
        tape.trainable_params = []
        with pytest.warns(
            UserWarning, match="Attempted to compute the gradient of a tape with no trainable"
        ):
            tapes, fn = stoch_pulse_grad(tape)
        assert not tapes
        assert qml.math.allclose(fn([]), tuple(qml.math.zeros([0]) for _ in range(num_meas)))

    def test_raises_non_pulse_marked_as_trainable(self):
        """Test that an empty gradient is returned when there are no trainable parameters."""
        import jax.numpy as jnp

        ops = [qml.RX(jnp.array(0.4), wires=0)]
        tape = qml.tape.QuantumScript(ops, measurements=[qml.expval(qml.PauliZ(0))])
        tape.trainable_params = [0]
        with pytest.raises(ValueError, match="stoch_pulse_grad does not support differentiating"):
            stoch_pulse_grad(tape)

    def test_raises_for_non_pauli_term(self):
        """Test that an error is raised if a ParametrizedHamiltonian contains a paramatrized
        term that is not a Pauli word."""
        import jax.numpy as jnp

        ham = qml.dot([qml.pulse.constant], [qml.PauliX(0) + qml.PauliY(2)])
        ops = [qml.evolve(ham)([jnp.array(0.152)], 0.3)]
        tape = qml.tape.QuantumScript(ops, measurements=[qml.expval(qml.PauliZ(0))])
        tape.trainable_params = [0]
        with pytest.raises(ValueError, match="stoch_pulse_grad currently only supports Pauli"):
            stoch_pulse_grad(tape)


class TestStochPulseGrad:
    """Test working cases of stoch_pulse_grad."""

    ham_single_q_const = qml.pulse.constant * qml.PauliY(0)

    def test_all_zero_grads(self):
        """Test that a zero gradient is returned when the trainable parameters are
        identified to have zero gradient in advance."""
        measurements = [qml.expval(qml.PauliZ("a")), qml.probs(["b", "c"])]
        # No parameters at all
        tape = qml.tape.QuantumScript([qml.RX(0.4, 0), qml.RZ(0.9, 0)], measurements=measurements)
        tapes, fn = stoch_pulse_grad(tape)
        assert not tapes

        res = fn([])
        assert isinstance(res, tuple) and len(res) == 2
        assert qml.math.allclose(res[0], np.zeros(2))
        assert qml.math.allclose(res[1], np.zeros((2, 4)))

    @pytest.mark.parametrize("num_samples", [1, 3])
    @pytest.mark.parametrize("t", [2.0, 3, (0.5, 0.6)])
    def test_constant_rx(self, num_samples, t):
        """Test that the derivative of a pulse generated by a constant Hamiltonian
        is computed correctly."""
        import jax

        jax.config.update("jax_enable_x64", True)
        jnp = jax.numpy
        params = [jnp.array(0.24)]
        T = t if isinstance(t, tuple) else (0, t)
        op = qml.evolve(self.ham_single_q_const)(params, t)
        tape = qml.tape.QuantumScript([op], [qml.expval(qml.PauliZ(0))])

        dev = qml.device("default.qubit.jax", wires=1)
        # Effective rotation parameter
        p = params[0] * (T[1] - T[0])
        r = qml.execute([tape], dev, None)
        assert qml.math.isclose(r, jnp.cos(2 * p), atol=1e-4)
        tapes, fn = stoch_pulse_grad(tape, num_samples=num_samples)
        assert len(tapes) == num_samples * 2

        res = fn(qml.execute(tapes, dev, None))
        assert qml.math.isclose(res, -2 * jnp.sin(2 * p) * (T[1] - T[0]))

    @pytest.mark.parametrize("t", [0.02, (0.5, 0.6)])
    def test_sin_envelope_rx(self, t):
        """Test that the derivative of a pulse with a sine wave envelope
        is computed correctly."""
        import jax

        jnp = jax.numpy

        T = t if isinstance(t, tuple) else (0, t)

        dev = qml.device("default.qubit.jax", wires=1)
        params = [jnp.array([2.3, -0.245])]

        def sine(p, t):
            return p[0] * jnp.sin(p[1] * t)

        ham = sine * qml.PauliZ(0)
        op = qml.evolve(ham)(params, t)
        tape = qml.tape.QuantumScript([qml.Hadamard(0), op], [qml.expval(qml.PauliX(0))])

        # Effective rotation parameter
        x, y = params[0]
        theta = -x / y * (jnp.cos(y * T[1]) - jnp.cos(y * T[0]))
        theta_jac = jnp.array(
            [
                theta / x,
                x / y**2 * (jnp.cos(y * T[1]) - jnp.cos(y * T[0]))
                + x / y * (jnp.sin(y * T[1]) * T[1] - jnp.sin(y * T[0]) * T[0]),
            ]
        )
        r = qml.execute([tape], dev, None)
        assert qml.math.isclose(r, jnp.cos(2 * theta))

        tapes, fn = stoch_pulse_grad(tape, num_samples=100)
        assert len(tapes) == 200

        res = fn(qml.execute(tapes, dev, None))
        exp_grad = -2 * jnp.sin(2 * theta) * theta_jac
        # classical Jacobian is being estimated with the Monte Carlo sampling -> coarse tolerance
        assert qml.math.allclose(res, exp_grad, atol=0.2)

    # TODO
    @pytest.mark.parametrize("t", [0.02, (0.5, 0.6)])
    def test_sin_envelope_rx_expval_probs(self, t):
        """Test that the derivative of a pulse with a sine wave envelope
        is computed correctly."""
        import jax

        jnp = jax.numpy

        T = t if isinstance(t, tuple) else (0, t)

        dev = qml.device("default.qubit.jax", wires=2)
        params = [jnp.array([2.3, -0.245])]

        def sine(p, t):
            return p[0] * jnp.sin(p[1] * t)

        ham = sine * (qml.PauliZ(0) @ qml.PauliX(0))
        op = qml.evolve(ham)(params, t)
        tape = qml.tape.QuantumScript([qml.Hadamard(0), op], [qml.expval(qml.PauliX(0)), qml.probs(wires=1)])

        # Effective rotation parameter
        x, y = params[0]
        theta = -x / y * (jnp.cos(y * T[1]) - jnp.cos(y * T[0]))
        theta_jac = jnp.array(
            [
                theta / x,
                x / y**2 * (jnp.cos(y * T[1]) - jnp.cos(y * T[0]))
                + x / y * (jnp.sin(y * T[1]) * T[1] - jnp.sin(y * T[0]) * T[0]),
            ]
        )
        r = qml.execute([tape], dev, None)
        assert qml.math.isclose(r, jnp.cos(2 * theta))

        tapes, fn = stoch_pulse_grad(tape, num_samples=100)
        assert len(tapes) == 200

        res = fn(qml.execute(tapes, dev, None))
        exp_grad = -2 * jnp.sin(2 * theta) * theta_jac
        # classical Jacobian is being estimated with the Monte Carlo sampling -> coarse tolerance
        assert qml.math.allclose(res, exp_grad, atol=0.2)

    @pytest.mark.parametrize("t", [0.02, (0.5, 0.6)])
    def test_pwc_envelope_rx(self, t):
        """Test that the derivative of a pulse generated by a piecewise constant Hamiltonian
        is computed correctly."""
        import jax

        jnp = jax.numpy

        T = t if isinstance(t, tuple) else (0, t)

        dev = qml.device("default.qubit.jax", wires=1)
        params = [jnp.array([0.24, 0.9, -0.1, 2.3, -0.245])]
        op = qml.evolve(qml.pulse.pwc(t) * qml.PauliZ(0))(params, t)
        tape = qml.tape.QuantumScript([qml.Hadamard(0), op], [qml.expval(qml.PauliX(0))])

        # Effective rotation parameter
        p = jnp.mean(params[0]) * (T[1] - T[0])
        r = qml.execute([tape], dev, None)
        assert qml.math.isclose(r, jnp.cos(2 * p))
        tapes, fn = stoch_pulse_grad(tape, num_samples=100, sampler_seed=7512)
        assert len(tapes) == 200

        res = fn(qml.execute(tapes, dev, None))
        # The sampling of pwc functions does not automatically reduce to the analytically
        # correct time integrals, leading to approximations -> coarse tolerance
        assert qml.math.allclose(
            res, -2 * jnp.sin(2 * p) * (T[1] - T[0]) / len(params[0]), atol=0.01
        )

    @pytest.mark.parametrize("t", [2.0, 3, (0.5, 0.6)])
    def test_constant_commuting(self, t):
        """Test that the derivative of a pulse generated by two constant commuting Hamiltonians
        is computed correctly."""
        import jax

        jax.config.update("jax_enable_x64", True)
        jnp = jax.numpy
        params = [jnp.array(0.24), jnp.array(-0.672)]
        T = t if isinstance(t, tuple) else (0, t)
        op = qml.evolve(qml.pulse.constant * qml.PauliX(0) + qml.pulse.constant * qml.PauliY(1))(
            params, t
        )
        tape = qml.tape.QuantumScript([op], [qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))])

        dev = qml.device("default.qubit.jax", wires=2)
        r = qml.execute([tape], dev, None)
        # Effective rotation parameters
        p = [_p * (T[1] - T[0]) for _p in params]
        assert qml.math.isclose(r, jnp.cos(2 * p[0]) * jnp.cos(2 * p[1]))
        tapes, fn = stoch_pulse_grad(tape)
        assert len(tapes) == 4

        res = fn(qml.execute(tapes, dev, None))
        exp_grad = [
            -2 * jnp.sin(2 * p[0]) * jnp.cos(2 * p[1]) * (T[1] - T[0]),
            -2 * jnp.sin(2 * p[1]) * jnp.cos(2 * p[0]) * (T[1] - T[0]),
        ]
        assert qml.math.allclose(res, exp_grad)

    def test_advanced_pulse(self):
        """Test the derivative of a more complex pulse."""
        import jax

        jax.config.update("jax_enable_x64", True)
        jnp = jax.numpy

        ham = (
            qml.pulse.constant * qml.PauliX(0)
            + (lambda p, t: jnp.sin(p * t)) * qml.PauliZ(0)
            + jnp.polyval * (qml.PauliY(0) @ qml.PauliY(1))
        )
        params = [jnp.array(1.51), jnp.array(-0.371), jnp.array([0.2, 0.2, -0.4])]
        dev = qml.device("default.qubit.jax", wires=2)

        @qml.qnode(dev, interface="jax")
        def qnode(params):
            qml.evolve(ham)(params, 0.4)
            return qml.expval(qml.PauliY(0) @ qml.PauliX(1))

        qnode.construct((params,), {})

        num_samples = 100
        tapes, fn = stoch_pulse_grad(qnode.tape, argnums=[0, 1, 2], num_samples=num_samples, sampler_seed=7123)
        assert len(tapes) == 3 * 2 * num_samples

        res = fn(qml.execute(tapes, dev, None))
        exp_grad = jax.grad(qnode)(params)
        assert all(qml.math.allclose(r, e, rtol=0.4) for r, e in zip(res, exp_grad))

    def test_randomness(self):
        """Test that the derivative of a pulse is exactly the same when reusing a seed and
        that it differs when using a different seed."""
        import jax

        jax.config.update("jax_enable_x64", True)
        jnp = jax.numpy

        params = [jnp.array(1.51)]
        op = qml.evolve((lambda p, t: p * t) * qml.PauliX(0))(params, 0.4)
        tape = qml.tape.QuantumScript([op], [qml.expval(qml.PauliY(0))])

        seed_a = 8752
        seed_b = 8753
        tapes_a_0, fn_a_0 = stoch_pulse_grad(tape, num_samples=2, sampler_seed=seed_a)
        tapes_a_1, fn_a_1 = stoch_pulse_grad(tape, num_samples=2, sampler_seed=seed_a)
        tapes_b, fn_b = stoch_pulse_grad(tape, num_samples=2, sampler_seed=seed_b)

        for tape_a_0, tape_a_1, tape_b in zip(tapes_a_0, tapes_a_1, tapes_b):
            assert all(equal(o0, o1) for o0, o1 in zip(tape_a_0, tape_a_1))
            assert all(equal(o0, o1, check_t=False) for o0, o1 in zip(tape_a_0, tape_b))
            assert not equal(tape_a_0[0], tape_b[0], check_t=True)
            assert not equal(tape_a_0[2], tape_b[2], check_t=True)

        dev = qml.device("default.qubit.jax", wires=1)
        res_a_0 = fn_a_0(qml.execute(tapes_a_0, dev, None))
        res_a_1 = fn_a_1(qml.execute(tapes_a_1, dev, None))
        res_b = fn_b(qml.execute(tapes_b, dev, None))

        assert res_a_0 == res_a_1
        assert not res_a_0 == res_b

    def test_two_pulses(self):
        """Test that the derivatives of two pulses in a circuit are computed correctly."""
        import jax

        jax.config.update("jax_enable_x64", True)
        jnp = jax.numpy

        ham_0 = qml.pulse.constant * qml.PauliX(0) + (lambda p, t: jnp.sin(p * t)) * qml.PauliY(0)
        ham_1 = 0.3 * qml.PauliZ(0) + jnp.polyval * (qml.PauliY(0) @ qml.PauliY(1))
        params_0 = [jnp.array(1.51), jnp.array(-0.371)]
        params_1 = [jnp.array([0.2, 0.2, -0.4])]
        dev = qml.device("default.qubit.jax", wires=2)

        @qml.qnode(dev, interface="jax")
        def qnode(params_0, params_1):
            qml.evolve(ham_0)(params_0, 0.2)
            qml.evolve(ham_1)(params_1, 0.4)
            return qml.expval(qml.PauliY(0) @ qml.PauliZ(1))

        qnode.construct((params_0, params_1), {})

        num_samples = 50
        qnode.tape.trainable_params = [0, 1, 2]
        tapes, fn = stoch_pulse_grad(qnode.tape, num_samples=num_samples, sampler_seed=7123)
        assert len(tapes) == 3 * 2 * num_samples

        res = fn(qml.execute(tapes, dev, None))
        exp_grad = jax.grad(qnode, argnums=(0, 1))(params_0, params_1)
        exp_grad = exp_grad[0] + exp_grad[1]
        assert all(qml.math.allclose(r, e, rtol=0.2) for r, e in zip(res, exp_grad))

    def test_with_jit(self):
        """Test that the stochastic parameter-shift rule works with JITting."""
        import jax

        jax.config.update("jax_enable_x64", True)
        dev = qml.device("default.qubit.jax", wires=1)
        jnp = jax.numpy
        T = (0.2, 0.5)

        def fun(params):
            op = qml.evolve(self.ham_single_q_const)(params, T)
            tape = qml.tape.QuantumScript([op], [qml.expval(qml.PauliZ(0))])
            tapes, fn = stoch_pulse_grad(tape)
            assert len(tapes) == 2
            res = fn(qml.execute(tapes, dev, None))
            return res

        params = [jnp.array(0.24)]
        # Effective rotation parameter
        p = params[0] * (T[1] - T[0])
        res = fun(params)
        assert qml.math.isclose(res, -2 * jnp.sin(2 * p) * (T[1] - T[0]))
        res_jit = jax.jit(fun)(params)
        assert qml.math.isclose(res, res_jit)


@pytest.mark.jax
class TestStochPulseGradQNodeIntegration:
    """Test that stoch_pulse_grad integrates correctly with QNodes."""

    ham_single_q_const = qml.pulse.constant * qml.PauliY(0)

    @pytest.mark.parametrize("num_samples", [1, 3])
    def test_simple_qnode(self, num_samples):
        """Test that a simple qnode can be differentiated with stoch_pulse_grad."""
        import jax

        jnp = jax.numpy
        jax.config.update("jax_enable_x64", True)
        dev = qml.device("default.qubit.jax", wires=1)
        T = 0.5

        @qml.qnode(dev, interface="jax", diff_method=stoch_pulse_grad, num_samples=num_samples)
        def circuit(params):
            qml.evolve(self.ham_single_q_const)(params, T)
            return qml.expval(qml.PauliZ(0))

        params = [jnp.array(0.4)]
        grad = jax.grad(circuit)(params)
        p = params[0] * T
        exp_grad = -2 * jnp.sin(2 * p) * T
        assert qml.math.isclose(grad, exp_grad)

    @pytest.mark.xfail # TODO: Make 
    @pytest.mark.parametrize("num_samples", [1, 3])
    def test_simple_qnode_jit(self, num_samples):
        """Test that a simple qnode can be differentiated with stoch_pulse_grad."""
        import jax

        jnp = jax.numpy
        jax.config.update("jax_enable_x64", True)
        dev = qml.device("default.qubit.jax", wires=1)
        T = 0.5

        @qml.qnode(dev, interface="jax", diff_method=stoch_pulse_grad, num_samples=num_samples)
        def circuit(params):
            qml.evolve(self.ham_single_q_const)(params, T)
            return qml.expval(qml.PauliZ(0))

        params = [jnp.array(0.4)]
        p = params[0] * T
        exp_grad = -2 * jnp.sin(2 * p) * T
        jit_grad = jax.jit(jax.grad(circuit))(params)
        assert qml.math.isclose(jit_grad, exp_grad)

    def test_advanced_qnode(self):
        """Test that an advanced qnode can be differentiated with stoch_pulse_grad."""
        import jax

        jnp = jax.numpy
        jax.config.update("jax_enable_x64", True)

        params = [jnp.array(1.51), jnp.array(-0.371), jnp.array([0.2, 0.2, -0.4])]
        dev = qml.device("default.qubit.jax", wires=2)
        ham = (
            qml.pulse.constant * qml.PauliX(0)
            + (lambda p, t: jnp.sin(p * t)) * qml.PauliZ(0)
            + jnp.polyval * (qml.PauliY(0) @ qml.PauliY(1))
        )

        def ansatz(params):
            qml.evolve(ham)(params, 0.4)
            return qml.expval(qml.PauliY(0) @ qml.PauliX(1))

        num_samples = 10
        qnode_pulse_grad = qml.QNode(
            ansatz,
            dev,
            interface="jax",
            diff_method=stoch_pulse_grad,
            num_samples=num_samples,
            sampler_seed=7123,
            cache=False,# remove once 3870 is merged
        )
        qnode_backprop = qml.QNode(ansatz, dev, interface="jax")

        grad_pulse_grad = jax.grad(qnode_pulse_grad)(params)
        assert dev.num_executions == 1 + 2 * 3 * num_samples
        grad_backprop = jax.grad(qnode_backprop)(params)

        assert all(
            qml.math.allclose(r, e, rtol=0.4) for r, e in zip(grad_pulse_grad, grad_backprop)
        )


@pytest.mark.jax
class TestStochPulseGradDiff:
    """Test that stoch_pulse_grad is differentiable."""

    # pylint: disable=too-few-public-methods
    ham_single_q_const = qml.pulse.constant * qml.PauliY(0)

    def test_jax(self):
        """Test that stoch_pulse_grad is differentiable with JAX."""
        import jax

        jnp = jax.numpy
        jax.config.update("jax_enable_x64", True)
        dev = qml.device("default.qubit.jax", wires=1)
        T = 0.5
        def fun(params):
            op = qml.evolve(self.ham_single_q_const)(params, T)
            tape = qml.tape.QuantumScript([op], [qml.expval(qml.PauliZ(0))])
            tape.trainable_params = [0]
            tapes, fn = stoch_pulse_grad(tape)
            return fn(qml.execute(tapes, dev, None))

        params = [jnp.array(0.4)]
        p = params[0] * T
        exp_grad = -2 * jnp.sin(2 * p) * T
        grad = fun(params)
        assert qml.math.isclose(grad, exp_grad)

        exp_diff_of_grad = -4 * jnp.cos(2 * p) * T**2
        diff_of_grad = jax.grad(fun)(params)
        assert qml.math.isclose(diff_of_grad, exp_diff_of_grad)

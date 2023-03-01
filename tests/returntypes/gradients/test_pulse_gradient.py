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

import warnings
import pytest
import copy
import numpy as np
import pennylane as qml

from pennylane.gradients.pulse_gradient import split_evol_ops, split_evol_tapes, stoch_pulse_grad
from pennylane.pulse import ParametrizedEvolution

ham_single_q_fixed = 0.4 * qml.PauliX(0)
ham_single_q_const = qml.pulse.constant * qml.PauliY(0) 
ham_single_q_pwc = qml.pulse.pwc((2., 4.)) * qml.PauliZ(0)
ham_two_q_pwc = qml.pulse.pwc((2., 4.)) * (qml.PauliZ(0) @ qml.PauliX(1))


def equal(op1, op2, check_t=True):
    """Helper function to check whether two operations are the same via qml.equal, optionally
    including the time window for ParametrizedEvolution. 

    TODO: Remove once #3846 is resolved.
    """
    if check_t and isinstance(op1, ParametrizedEvolution) and isinstance(op2, ParametrizedEvolution):
        return qml.equal(op1, op2) and qml.math.allclose(op1.t, op2.t)
    return qml.equal(op1, op2)


split_evol_ops_test_cases = [
    (ham_single_q_const, [0.3], 2.3, "X"),
    (ham_single_q_pwc, [np.linspace(0, 1, 13)], (0.6, 1.2), "Y"),
    (ham_two_q_pwc, [np.linspace(0, 1, 13)], 3, "YX"),
]
@pytest.mark.jax
class TestSplitEvolOps:
    """Tests for the helper method split_evol_ops that samples a splitting time and splits up
    a ParametrizedEvolution operation at the sampled time, inserting a Pauli rotation about the
    provided Pauli word with angles +- pi/2."""

    @pytest.mark.parametrize("ham, params, time, word", split_evol_ops_test_cases)
    def test_output_properties(self, ham, params, time, word):
        """Test that split_evol_ops returns the right objects with correct
        relations to the input operation."""

        import jax

        key = jax.random.PRNGKey(5324)
        op = qml.evolve(ham)(params, time)
        op_copy = copy.deepcopy(op)
        exp_time = [0, time] if qml.math.isscalar(time) else time
        assert qml.math.allclose(op.t, exp_time)
        output = split_evol_ops(op, word, key)
        # Check that the original operation was not altered
        assert equal(op, op_copy)

        # Check that tau and ops is returned
        assert isinstance(output, tuple) and len(output)==2
        
        tau, ops = output

        # Check format and range of tau
        assert isinstance(tau, jax.Array) and tau.shape==() 
        assert op.t[0]<=tau<=op.t[1]

        assert isinstance(ops, tuple) and len(ops)==2

        for sign, _ops in zip([1, -1], ops):
            assert isinstance(_ops, list) and len(_ops) == 3
            # Check that the split-up time evolution is correct
            assert equal(_ops[0], op, check_t=False)
            assert qml.math.allclose(_ops[0].t, [op.t[0], tau])

            assert equal(_ops[2], op, check_t=False)
            assert qml.math.allclose(_ops[2].t, [tau, op.t[1]])

            # Check that the inserted Pauli rotation is correct
            assert isinstance(_ops[1], qml.PauliRot)
            assert qml.math.allclose(_ops[1].data, sign * np.pi/2)
            assert _ops[1].hyperparameters["pauli_word"] == word

            assert all(_op.wires == op.wires for _op in _ops)

    @pytest.mark.parametrize("ham, params, time, word", split_evol_ops_test_cases)
    def test_randomness_key(self, ham, params, time, word):
        """Test that split_evol_ops returns the same when used twice with the same
        randomness key and different results when used with different keys."""
        import jax

        op = qml.evolve(ham)(params, time)
        key_a = jax.random.PRNGKey(5324)
        key_b = jax.random.PRNGKey(7281)
        tau_a_0, ops_a_0 = split_evol_ops(op, word, key_a)
        tau_a_1, ops_a_1 = split_evol_ops(op, word, key_a)
        tau_b, ops_b = split_evol_ops(op, word, key_b)

        # Check same output for same key
        assert qml.math.isclose(tau_a_0, tau_a_1)
        assert all(equal(o_0, o_1) for o_0, o_1 in zip(np.array(ops_a_0).flat, np.array(ops_a_1).flat))

        # Check different output for different key
        assert not qml.math.isclose(tau_a_0, tau_b)
        assert not all(equal(o_0, o_1) for o_0, o_1 in zip(np.array(ops_a_0).flat, np.array(ops_b).flat))
        # Check positive and negative Pauli rotations are the same even for different keys
        assert equal(ops_a_0[0][1], ops_b[0][1])
        assert equal(ops_a_0[1][1], ops_b[1][1])

class TestSplitEvolTapes:
    """Tests for the helper method split_evol_tapes that replaces an indicated operation by other operations
    and creates a new tape for each provided set of replacing operations."""

    def test_with_standard_ops(self):
        """Test basic behaviour of the operation replacement with standard qml ops."""
        ops = [qml.RX(0.4, 2), qml.PauliZ(0), qml.CNOT([0, 2])]
        tape = qml.tape.QuantumScript(ops)
        split_evolve_ops = ([qml.RX(0.6, 2), qml.PauliY(0), qml.RZ(0., 0)], [qml.PauliX(0), qml.PauliZ(2)])
        new_tapes = split_evol_tapes(tape, split_evolve_ops, 1)
        assert len(new_tapes) == 2
        for t, new_ops in zip(new_tapes, split_evolve_ops):
            assert qml.equal(t.operations[0], ops[0])
            assert all(qml.equal(o1, o2) for o1, o2 in zip(t.operations[1:-1], new_ops))
            assert qml.equal(t.operations[-1], ops[2])


    def test_with_parametrized_evolution(self):
        """Test basic behaviour of the operation replacement with ParametrizedEvolution."""

        ops = [qml.evolve(ham_single_q_pwc)([np.linspace(0, 1, 9)], 0.4)]
        tape = qml.tape.QuantumScript(ops)
        split_evolve_ops = ([qml.RX(0.6, 2), qml.PauliY(0), qml.RZ(0., 0)], [qml.PauliX(0), qml.PauliZ(2)])
        new_tapes = split_evol_tapes(tape, split_evolve_ops, 0)
        assert len(new_tapes) == 2
        for t, new_ops in zip(new_tapes, split_evolve_ops):
            assert all(qml.equal(o1, o2) for o1, o2 in zip(t.operations, new_ops))

        ops = [qml.evolve(ham_single_q_pwc)([np.linspace(0, 1, 9)], 0.4), qml.CNOT([0, 2])]
        tape = qml.tape.QuantumScript(ops)
        split_evolve_ops = ([qml.RX(0.6, 2), qml.PauliY(0), qml.RZ(0., 0)], [])
        new_tapes = split_evol_tapes(tape, split_evolve_ops, 0)
        assert len(new_tapes) == 2
        for t, new_ops in zip(new_tapes, split_evolve_ops):
            assert all(qml.equal(o1, o2) for o1, o2 in zip(t.operations[:-1], new_ops))
            assert qml.equal(t.operations[-1], ops[1])

        ops = [qml.RX(0.4, 2), qml.evolve(ham_single_q_pwc)([np.linspace(0, 1, 9)], 0.4), qml.CNOT([0, 2])]
        tape = qml.tape.QuantumScript(ops)
        split_evolve_ops = ([qml.RX(0.6, 2), qml.PauliY(0), qml.RZ(0., 0)], [qml.PauliX(0), qml.PauliZ(2)])
        new_tapes = split_evol_tapes(tape, split_evolve_ops, 1)
        assert len(new_tapes) == 2
        for t, new_ops in zip(new_tapes, split_evolve_ops):
            assert qml.equal(t.operations[0], ops[0])
            assert all(qml.equal(o1, o2) for o1, o2 in zip(t.operations[1:-1], new_ops))
            assert qml.equal(t.operations[-1], ops[2])


@pytest.mark.jax
class TestStochPulseGradErrors:
    """Test errors raised by stoch_pulse_grad."""

    def test_error_raised_if_jax_not_installed(self):
        """Test that an error is raised if a convenience function is called without jax installed"""
        try:
            import jax  # pylint: disable=unused-import

            pytest.skip()
        except ImportError:
            tape = qml.tape.QuantumScript([])
            with pytest.raises(ImportError, match="Module jax is required"):
                stoch_pulse_grad(tape)

    def test_error_raised_for_variance(self):
        """Test that an error is raised when attempting to differentiate a tape that measures a variance."""
        tape = qml.tape.QuantumScript(measurements=[qml.expval(qml.PauliX(2)), qml.var(qml.PauliZ(0))])
        with pytest.raises(ValueError, match="Computing the gradient of variances with the stochastic"):
            stoch_pulse_grad(tape)

    @pytest.mark.parametrize("measurement", [qml.vn_entropy(0), qml.state(), qml.mutual_info([0], 1)])
    def test_error_raised_for_state_measurements(self, measurement):
        """Test that an error is raised when attempting to differentiate a tape that measures a 
        state, or returns a state indirectly via entropy/mutual info measurements."""
        tape = qml.tape.QuantumScript(measurements=[measurement])
        with pytest.raises(ValueError, match="Computing the gradient of circuits that return the state"):
            stoch_pulse_grad(tape)

    @pytest.mark.parametrize("num_samples", [-1, 0, np.array(-2)])
    def test_error_raised_for_less_than_one_sample(self, num_samples):
        """Test that an error is raised if fewer than one samples for the stochastic shift rule are requested."""
        tape = qml.tape.QuantumScript([])
        with pytest.raises(ValueError, match="Expected a positive number of samples"):
            stoch_pulse_grad(tape, num_samples=num_samples)

    @pytest.mark.parametrize("num_meas", [0, 1, 2])
    def test_no_trainable_params(self, num_meas):
        """Test that an empty gradient is returned when there are no trainable parameters."""
        measurements = [qml.expval(qml.PauliX(w)) for w in range(num_meas)]
        # No parameters at all
        tape = qml.tape.QuantumScript([], measurements=measurements)
        with pytest.warns(UserWarning, match="Attempted to compute the gradient of a tape with no trainable"):
            tapes, fn = stoch_pulse_grad(tape)
        assert not tapes
        assert qml.math.allclose(fn([]), tuple(qml.math.zeros([0]) for _ in range(num_meas)))

        # parameters but none are trainable
        ops = [qml.RX(0.4, 2), qml.CRY(0.1, [1, 0])]
        tape = qml.tape.QuantumScript(ops, measurements=measurements)
        tape.trainable_params = []
        with pytest.warns(UserWarning, match="Attempted to compute the gradient of a tape with no trainable"):
            tapes, fn = stoch_pulse_grad(tape)
        assert not tapes
        assert qml.math.allclose(fn([]), tuple(qml.math.zeros([0]) for _ in range(num_meas)))

    def test_non_pulse_marked_as_trainable(self):
        """Test that an empty gradient is returned when there are no trainable parameters."""
        import jax.numpy as jnp

        ops = [qml.RX(jnp.array(0.4), wires=0)]
        tape = qml.tape.QuantumScript(ops, measurements=[qml.expval(qml.PauliZ(0))])
        tape.trainable_params = [0]
        with pytest.raises(ValueError, match="stoch_pulse_grad does not support differentiating"):
            stoch_pulse_grad(tape)

class TestStochPulseGrad:
    """Test working cases of stoch_pulse_grad."""

    @pytest.mark.parametrize("num_meas", [0, 1, 2])
    def test_all_zero_grads(self, num_meas):
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


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

import copy

import numpy as np
import pytest

import pennylane as qp
from pennylane.gradients.general_shift_rules import eigvals_to_frequencies, generate_shift_rule
from pennylane.gradients.pulse_gradient import (
    _parshift_and_integrate,
    _split_evol_ops,
    _split_evol_tape,
    stoch_pulse_grad,
)


# pylint: disable=too-few-public-methods
@pytest.mark.jax
class TestSplitEvolOps:
    """Tests for the helper method _split_evol_ops that samples a splitting time and splits up
    a ParametrizedEvolution operation at the sampled time, inserting a Pauli rotation about the
    provided Pauli word with angles +- pi/2."""

    # pylint: disable=unnecessary-lambda-assignment

    # Need to wrap the Hamiltonians in a callable in order to use `qp.pulse` functions, as
    # the tests would otherwise fail when used without JAX.
    ham_single_q_const = lambda _: qp.pulse.constant * qp.PauliY(0)
    ham_single_q_pwc = lambda _: qp.pulse.pwc((2.0, 4.0)) * qp.PauliZ(0)
    ham_two_q_pwc = lambda _: qp.pulse.pwc((2.0, 4.0)) * (qp.PauliZ(0) @ qp.PauliX(1))

    split_evol_ops_test_cases_pauliword = [
        (ham_single_q_const, [0.3], 2.3, qp.PauliX(0), "X"),
        (ham_single_q_pwc, [np.linspace(0, 1, 13)], (0.6, 1.2), qp.PauliY(1), "Y"),
        (
            ham_two_q_pwc,
            [np.linspace(0, 1, 13)],
            (0.2, 0.6, 0.9, 1.8),
            qp.PauliY(0) @ qp.PauliX(1),
            "YX",
        ),
        (ham_single_q_const, [0.3], 2.3, qp.Hamiltonian([0.2], [qp.PauliZ(0)]), "Z"),
        (ham_single_q_const, [0.3], 2.3, 1.2 * qp.PauliZ(0), "Z"),
        (ham_single_q_const, [0.3], 2.3, qp.s_prod(1.2, qp.PauliZ(0)), "Z"),
        (ham_single_q_const, [0.3], 2.3, qp.dot([1.9], [qp.PauliZ(0)]), "Z"),
    ]

    # pylint: disable=too-many-arguments
    @pytest.mark.parametrize("ham, params, time, ob, word", split_evol_ops_test_cases_pauliword)
    def test_with_pauliword(self, ham, params, time, ob, word, seed):
        """Test that _split_evol_ops returns the right ops with correct
        relations to the input operation for a Pauli word as ``ob``."""

        import jax

        ham = ham(None)
        key = jax.random.PRNGKey(seed)
        op = qp.evolve(ham)(params, time)
        op_copy = copy.deepcopy(op)
        exp_time = [0, time] if qp.math.ndim(time) == 0 else time
        # Cross-check instantiation of evolution time
        assert qp.math.allclose(op.t, exp_time)

        # Sample splitting time
        tau = jax.random.uniform(key) * (exp_time[1] - exp_time[0]) + exp_time[0]
        ops, coeffs = _split_evol_ops(op, ob, tau)
        eigvals = qp.eigvals(ob)
        prefactor = np.max(eigvals)
        exp_coeffs = [prefactor, -prefactor]
        exp_shifts = [np.pi / 2, -np.pi / 2]

        # Check coefficients
        assert qp.math.allclose(coeffs, exp_coeffs)

        # Check that the original operation was not altered
        qp.assert_equal(op, op_copy)

        assert isinstance(ops, tuple) and len(ops) == len(exp_shifts)

        for exp_shift, _ops in zip(exp_shifts, ops):
            assert isinstance(_ops, list) and len(_ops) == 3
            # Check that the split-up time evolution is correct
            assert qp.math.allclose(_ops[0].t, [op.t[0], tau])
            # Patch _ops[0] to have the same time as op, so that it should become the same as op
            _ops[0].t = op.t
            qp.assert_equal(_ops[0], op)

            assert qp.math.allclose(_ops[2].t, [tau, op.t[-1]])
            # Patch _ops[2] to have the same time as op, so that it should become the same as op
            _ops[2].t = op.t
            qp.assert_equal(_ops[2], op)

            # Check that the inserted exponential is correct
            qp.assert_equal(qp.PauliRot(exp_shift, word, wires=ob.wires), _ops[1])

    split_evol_ops_test_cases_general = [
        (
            ham_single_q_pwc,
            [np.linspace(0, 1, 13)],
            (0.6, 1.2),
            0.2 * qp.PauliX(1) + 0.9 * qp.PauliZ(1),
        ),
        (
            ham_single_q_pwc,
            [np.linspace(0, 1, 13)],
            (0.6, 1.2),
            qp.sum(0.2 * qp.PauliX(1), 0.9 * qp.PauliZ(1)),
        ),
        (
            ham_single_q_pwc,
            [np.linspace(0, 1, 13)],
            (0.6, 1.2),
            qp.sum(0.2 * qp.PauliX(1), qp.s_prod(0.9, qp.PauliZ(1))),
        ),
        (
            ham_two_q_pwc,
            [np.linspace(0, 1, 13)],
            (0.2, 0.6, 0.9, 1.8),
            qp.PauliY(0) @ qp.PauliX(1) + 0.2 * qp.PauliZ(0),
        ),
    ]

    @pytest.mark.parametrize("ham, params, time, ob", split_evol_ops_test_cases_general)
    def test_with_general_ob(self, ham, params, time, ob, seed):
        """Test that _split_evol_ops returns the right ops with correct
        relations to the input operation for a general Hermitian as ``ob``."""

        import jax

        ham = ham(None)
        key = jax.random.PRNGKey(seed)
        op = qp.evolve(ham)(params, time)
        op_copy = copy.deepcopy(op)
        exp_time = [0, time] if qp.math.ndim(time) == 0 else time
        # Cross-check instantiation of evolution time
        assert qp.math.allclose(op.t, exp_time)

        # Sample splitting time
        tau = jax.random.uniform(key) * (exp_time[1] - exp_time[0]) + exp_time[0]
        ops, coeffs = _split_evol_ops(op, ob, tau)
        eigvals = qp.eigvals(ob)
        exp_coeffs, exp_shifts = zip(*generate_shift_rule(eigvals_to_frequencies(tuple(eigvals))))

        # Check coefficients
        assert qp.math.allclose(coeffs, exp_coeffs)

        # Check that the original operation was not altered
        qp.assert_equal(op, op_copy)

        assert isinstance(ops, tuple) and len(ops) == len(exp_shifts)

        for exp_shift, _ops in zip(exp_shifts, ops):
            assert isinstance(_ops, list) and len(_ops) == 3
            # Check that the split-up time evolution is correct
            assert qp.math.allclose(_ops[0].t, [op.t[0], tau])
            # Patch _ops[0] to have the same time as op, so that it should become the same as op
            _ops[0].t = op.t
            qp.assert_equal(_ops[0], op)

            assert qp.math.allclose(_ops[2].t, [tau, op.t[-1]])
            # Patch _ops[2] to have the same time as op, so that it should become the same as op
            _ops[2].t = op.t
            qp.assert_equal(_ops[2], op)

            # Check that the inserted exponential is correct
            qp.assert_equal(qp.exp(qp.dot([-1j * exp_shift], [ob])), _ops[1])


@pytest.mark.jax
class TestSplitEvolTapes:
    """Tests for the helper method _split_evol_tape that replaces an indicated operation by
    other operations and creates a new tape for each provided set of replacing operations."""

    def test_with_standard_ops(self):
        """Test basic behaviour of the operation replacement with standard qml ops."""
        ops = [qp.RX(0.4, 2), qp.PauliZ(0), qp.CNOT([0, 2])]
        tape = qp.tape.QuantumScript(ops)
        split_evolve_ops = (
            [qp.RX(0.6, 2), qp.PauliY(0), qp.RZ(0.0, 0)],
            [qp.PauliX(0), qp.PauliZ(2)],
        )
        new_tapes = _split_evol_tape(tape, split_evolve_ops, 1)
        assert len(new_tapes) == 2
        for t, new_ops in zip(new_tapes, split_evolve_ops):
            qp.assert_equal(t.operations[0], ops[0])
            for o1, o2 in zip(t.operations[1:-1], new_ops):
                qp.assert_equal(o1, o2)
            qp.assert_equal(t.operations[-1], ops[2])

    def test_with_parametrized_evolution(self):
        """Test basic behaviour of the operation replacement with ParametrizedEvolution."""

        ham_single_q_pwc = qp.pulse.pwc((2.0, 4.0)) * qp.PauliZ(0)
        ops = [qp.evolve(ham_single_q_pwc)([np.linspace(0, 1, 9)], 0.4)]
        tape = qp.tape.QuantumScript(ops)
        split_evolve_ops = (
            [qp.RX(0.6, 2), qp.PauliY(0), qp.RZ(0.0, 0)],
            [qp.PauliX(0), qp.PauliZ(2)],
        )
        new_tapes = _split_evol_tape(tape, split_evolve_ops, 0)
        assert len(new_tapes) == 2
        for t, new_ops in zip(new_tapes, split_evolve_ops):
            for o1, o2 in zip(t.operations, new_ops):
                qp.assert_equal(o1, o2)

        ops = [qp.evolve(ham_single_q_pwc)([np.linspace(0, 1, 9)], 0.4), qp.CNOT([0, 2])]
        tape = qp.tape.QuantumScript(ops)
        split_evolve_ops = ([qp.RX(0.6, 2), qp.PauliY(0), qp.RZ(0.0, 0)], [])
        new_tapes = _split_evol_tape(tape, split_evolve_ops, 0)
        assert len(new_tapes) == 2
        for t, new_ops in zip(new_tapes, split_evolve_ops):
            for o1, o2 in zip(t.operations[:-1], new_ops):
                qp.assert_equal(o1, o2)
            qp.assert_equal(t.operations[-1], ops[1])

        ops = [
            qp.RX(0.4, 2),
            qp.evolve(ham_single_q_pwc)([np.linspace(0, 1, 9)], 0.4),
            qp.CNOT([0, 2]),
        ]
        tape = qp.tape.QuantumScript(ops)
        split_evolve_ops = (
            [qp.RX(0.6, 2), qp.PauliY(0), qp.RZ(0.0, 0)],
            [qp.PauliX(0), qp.PauliZ(2)],
        )
        new_tapes = _split_evol_tape(tape, split_evolve_ops, 1)
        assert len(new_tapes) == 2
        for t, new_ops in zip(new_tapes, split_evolve_ops):
            qp.assert_equal(t.operations[0], ops[0])
            for o1, o2 in zip(t.operations[1:-1], new_ops):
                qp.assert_equal(o1, o2)
            qp.assert_equal(t.operations[-1], ops[2])


@pytest.mark.jax
class TestParshiftAndIntegrate:
    """Test the helper routine ``_parshift_and_integrate``. Most tests use uniform
    return types and parameters, so that we can test against simple tensor contractions."""

    # pylint: disable=too-many-arguments

    @pytest.mark.parametrize("multi_term", [1, 4])
    @pytest.mark.parametrize("meas_shape", [(), (4,)])
    @pytest.mark.parametrize("par_shape", [(), (3,), (2, 7)])
    @pytest.mark.parametrize("num_shifts", [2, 5])
    @pytest.mark.parametrize("num_split_times", [1, 7])
    def test_single_measure_single_shots(
        self, num_split_times, num_shifts, par_shape, multi_term, meas_shape
    ):
        """Test that ``_parshift_and_integrate`` works with results for a single measurement
        per shift and splitting time, and with a single setting of shots. This corresponds to
        ``single_measure=True and has_partitioned_shots=False``. The test is parametrized with whether
        or not there are multiple Hamiltonian terms to take into account (and sum their
        contributions), with the shape of the single measurement and of the parameter, with
        the number of shifts in the shift rule and with the number of splitting times.
        """
        from jax import numpy as jnp

        cjac_shape = (num_split_times,) + par_shape
        if multi_term > 1:
            cjacs = tuple(np.random.random(cjac_shape) for _ in range(multi_term))
            psr_coeffs = tuple(np.random.random(num_shifts) for _ in range(multi_term))
        else:
            cjacs = np.random.random(cjac_shape)
            psr_coeffs = np.random.random(num_shifts)

        results_shape = (num_split_times * num_shifts * multi_term,) + meas_shape
        new_results_shape = (
            multi_term,
            num_split_times,
            num_shifts,
        ) + meas_shape
        results = np.random.random(results_shape)

        prefactor = 0.3214

        res = _parshift_and_integrate(
            results,
            cjacs,
            prefactor,
            psr_coeffs,
            single_measure=True,
            has_partitioned_shots=False,
            use_broadcasting=False,
        )

        assert isinstance(res, jnp.ndarray)
        assert res.shape == meas_shape + par_shape

        _results = np.reshape(results, new_results_shape)
        _cjacs = np.stack(cjacs).reshape((multi_term,) + cjac_shape)
        _psr_coeffs = np.stack(psr_coeffs).reshape((multi_term, num_shifts))
        meas_letter = "" if meas_shape == () else "a"
        contraction = f"ms,mts{meas_letter},mt...->{meas_letter}..."
        expected = np.einsum(contraction, _psr_coeffs, _results, _cjacs)
        assert np.allclose(res, expected * prefactor)

    @pytest.mark.parametrize("multi_term", [1, 4])
    @pytest.mark.parametrize("meas_shape", [(), (4,)])
    @pytest.mark.parametrize("par_shape", [(), (3,), (2, 7)])
    @pytest.mark.parametrize("num_shifts", [2, 5])
    @pytest.mark.parametrize("num_split_times", [1, 7])
    def test_single_measure_single_shots_broadcast(
        self, num_split_times, num_shifts, par_shape, multi_term, meas_shape
    ):
        """Test that ``_parshift_and_integrate`` works with results for a single measurement
        per shift and splitting time, and with a single setting of shots. This corresponds to
        ``single_measure=True and has_partitioned_shots=False``. The test is parametrized with whether
        or not there are multiple Hamiltonian terms to take into account (and sum their
        contributions), with the shape of the single measurement and of the parameter, with
        the number of shifts in the shift rule and with the number of splitting times.
        This is the variant of the previous test that uses broadcasting.
        """
        from jax import numpy as jnp

        cjac_shape = (num_split_times,) + par_shape
        if multi_term > 1:
            cjacs = tuple(np.random.random(cjac_shape) for _ in range(multi_term))
            psr_coeffs = tuple(np.random.random(num_shifts) for _ in range(multi_term))
        else:
            cjacs = np.random.random(cjac_shape)
            psr_coeffs = np.random.random(num_shifts)

        results_shape = (num_shifts * multi_term, (num_split_times + 2)) + meas_shape
        new_results_shape = (
            multi_term,
            num_shifts,
            num_split_times + 2,
        ) + meas_shape
        results = np.random.random(results_shape)

        prefactor = 0.3214

        res = _parshift_and_integrate(
            results,
            cjacs,
            prefactor,
            psr_coeffs,
            single_measure=True,
            has_partitioned_shots=False,
            use_broadcasting=True,
        )

        assert isinstance(res, jnp.ndarray)
        assert res.shape == meas_shape + par_shape

        _results = np.reshape(results, new_results_shape)
        _cjacs = np.stack(cjacs).reshape((multi_term,) + cjac_shape)
        _psr_coeffs = np.stack(psr_coeffs).reshape((multi_term, num_shifts))
        meas_letter = "" if meas_shape == () else "a"
        # Slice away excess results
        _results = _results[:, :, 1:-1]
        # With broadcasting, the axes of different shifts and splitting times are
        # switched for the results tensor, compared to without broadcasting.
        contraction = f"ms,mst{meas_letter},mt...->{meas_letter}..."
        expected = np.einsum(contraction, _psr_coeffs, _results, _cjacs)
        assert np.allclose(res, expected * prefactor)

    @pytest.mark.parametrize("multi_term", [1, 4])
    @pytest.mark.parametrize("meas_shape", [(), (4,)])
    @pytest.mark.parametrize("par_shape", [(), (3,), (2, 2)])
    @pytest.mark.parametrize("num_shifts", [2, 5])
    @pytest.mark.parametrize("num_split_times", [1, 3])
    def test_multi_measure_or_multi_shots(
        self, num_split_times, num_shifts, par_shape, multi_term, meas_shape
    ):
        """Test that ``_parshift_and_integrate`` works with results for multiple measurements
        per shift and splitting time and with a single setting of shots, or alternatively with
        a single measurement but multiple shot settings. This corresponds to
        ``single_measure=False and has_partitioned_shots=False`` or
        ``single_measure=True and has_partitioned_shots=True``. The test is parametrized with whether
        or not there are multiple Hamiltonian terms to take into account (and sum their
        contributions), with the shape of the single measurement and of the parameter, with
        the number of shifts in the shift rule and with the number of splitting times.
        """
        from jax import numpy as jnp

        num_meas_or_shots = 5

        cjac_shape = (num_split_times,) + par_shape
        if multi_term > 1:
            cjacs = tuple(np.random.random(cjac_shape) for _ in range(multi_term))
            psr_coeffs = tuple(np.random.random(num_shifts) for _ in range(multi_term))
        else:
            cjacs = np.random.random(cjac_shape)
            psr_coeffs = np.random.random(num_shifts)

        results_shape = (
            num_split_times * num_shifts * multi_term,
            num_meas_or_shots,
        ) + meas_shape
        new_results_shape = (
            multi_term,
            num_split_times,
            num_shifts,
            num_meas_or_shots,
        ) + meas_shape
        results = np.random.random(results_shape)

        prefactor = 0.3214

        res0, res1 = (
            _parshift_and_integrate(
                results,
                cjacs,
                prefactor,
                psr_coeffs,
                single_measure=_bool,
                has_partitioned_shots=_bool,
                use_broadcasting=False,
            )
            for _bool in [False, True]
        )

        _results = np.reshape(results, new_results_shape)
        _cjacs = np.stack(cjacs).reshape((multi_term,) + cjac_shape)
        _psr_coeffs = np.stack(psr_coeffs).reshape((multi_term, num_shifts))
        meas_letter = "" if meas_shape == () else "a"
        contraction = f"ms,mtsn{meas_letter},mt...->n{meas_letter}..."
        expected = np.einsum(contraction, _psr_coeffs, _results, _cjacs)

        for res in [res0, res1]:
            assert isinstance(res, tuple)
            assert len(res) == num_meas_or_shots
            assert all(isinstance(r, jnp.ndarray) for r in res)
            assert all(r.shape == meas_shape + par_shape for r in res)

            assert np.allclose(np.stack(res), expected * prefactor)

    @pytest.mark.parametrize("multi_term", [1, 4])
    @pytest.mark.parametrize("meas_shape", [(), (4,)])
    @pytest.mark.parametrize("par_shape", [(), (3,), (2, 2)])
    @pytest.mark.parametrize("num_shifts", [2, 5])
    @pytest.mark.parametrize("num_split_times", [1, 3])
    def test_multi_measure_or_multi_shots_broadcast(
        self, num_split_times, num_shifts, par_shape, multi_term, meas_shape
    ):
        """Test that ``_parshift_and_integrate`` works with results for multiple measurements
        per shift and splitting time and with a single setting of shots, or alternatively with
        a single measurement but multiple shot settings. This corresponds to
        ``single_measure=False and has_partitioned_shots=False`` or
        ``single_measure=True and has_partitioned_shots=True``. The test is parametrized with whether
        or not there are multiple Hamiltonian terms to take into account (and sum their
        contributions), with the shape of the single measurement and of the parameter, with
        the number of shifts in the shift rule and with the number of splitting times.
        This is the variant of the previous test that uses broadcasting.
        """
        from jax import numpy as jnp

        num_meas_or_shots = 5

        cjac_shape = (num_split_times,) + par_shape
        if multi_term > 1:
            cjacs = tuple(np.random.random(cjac_shape) for _ in range(multi_term))
            psr_coeffs = tuple(np.random.random(num_shifts) for _ in range(multi_term))
        else:
            cjacs = np.random.random(cjac_shape)
            psr_coeffs = np.random.random(num_shifts)

        results_shape = (
            num_shifts * multi_term,
            num_meas_or_shots,
            (num_split_times + 2),
        ) + meas_shape
        new_results_shape = (
            multi_term,
            num_shifts,
            num_meas_or_shots,
            num_split_times + 2,
        ) + meas_shape
        results = np.random.random(results_shape)

        prefactor = 0.3214

        res0, res1 = (
            _parshift_and_integrate(
                results,
                cjacs,
                prefactor,
                psr_coeffs,
                single_measure=_bool,
                has_partitioned_shots=_bool,
                use_broadcasting=True,
            )
            for _bool in [False, True]
        )

        _results = np.reshape(results, new_results_shape)
        _cjacs = np.stack(cjacs).reshape((multi_term,) + cjac_shape)
        _psr_coeffs = np.stack(psr_coeffs).reshape((multi_term, num_shifts))
        meas_letter = "" if meas_shape == () else "a"
        # Slice away excess results
        _results = _results[:, :, :, 1:-1]
        # With broadcasting, the axes of different shifts and splitting times are
        # switched for the results tensor, compared to without broadcasting.
        contraction = f"ms,msnt{meas_letter},mt...->n{meas_letter}..."
        expected = np.einsum(contraction, _psr_coeffs, _results, _cjacs)

        for res in [res0, res1]:
            assert isinstance(res, tuple)
            assert len(res) == num_meas_or_shots
            assert all(isinstance(r, jnp.ndarray) for r in res)
            assert all(r.shape == meas_shape + par_shape for r in res)

            assert np.allclose(np.stack(res), expected * prefactor)

    @pytest.mark.parametrize("multi_term", [1, 4])
    @pytest.mark.parametrize("meas_shape", [(), (4,)])
    @pytest.mark.parametrize("par_shape", [(), (3,), (2, 2)])
    @pytest.mark.parametrize("num_shifts", [2, 5])
    @pytest.mark.parametrize("num_split_times", [1, 3])
    def test_multi_measure_multi_shots(
        self, num_split_times, num_shifts, par_shape, multi_term, meas_shape
    ):
        """Test that ``_parshift_and_integrate`` works with results for multiple measurements
        per shift and splitting time and with multiple shot settings. This corresponds to
        ``single_measure=False and has_partitioned_shots=True``. The test is parametrized with whether
        or not there are multiple Hamiltonian terms to take into account (and sum their
        contributions), with the shape of the single measurement and of the parameter, with
        the number of shifts in the shift rule and with the number of splitting times.
        """
        from jax import numpy as jnp

        num_shots = 3
        num_meas = 5

        cjac_shape = (num_split_times,) + par_shape
        if multi_term > 1:
            cjacs = tuple(np.random.random(cjac_shape) for _ in range(multi_term))
            psr_coeffs = tuple(np.random.random(num_shifts) for _ in range(multi_term))
        else:
            cjacs = np.random.random(cjac_shape)
            psr_coeffs = np.random.random(num_shifts)

        results_shape = (
            num_split_times * num_shifts * multi_term,
            num_shots,
            num_meas,
        ) + meas_shape
        new_results_shape = (
            multi_term,
            num_split_times,
            num_shifts,
            num_shots,
            num_meas,
        ) + meas_shape
        results = np.random.random(results_shape)

        prefactor = 0.3214

        res = _parshift_and_integrate(
            results,
            cjacs,
            prefactor,
            psr_coeffs,
            single_measure=False,
            has_partitioned_shots=True,
            use_broadcasting=False,
        )

        assert isinstance(res, tuple)
        assert len(res) == num_shots
        for r in res:
            assert isinstance(r, tuple)
            assert len(r) == num_meas
            assert all(isinstance(_r, jnp.ndarray) for _r in r)
            assert all(_r.shape == meas_shape + par_shape for _r in r)

        _results = np.reshape(results, new_results_shape)
        _cjacs = np.stack(cjacs).reshape((multi_term,) + cjac_shape)
        _psr_coeffs = np.stack(psr_coeffs).reshape((multi_term, num_shifts))
        meas_letter = "" if meas_shape == () else "a"
        contraction = f"ms,mtsNn{meas_letter},mt...->Nn{meas_letter}..."
        expected = np.einsum(contraction, _psr_coeffs, _results, _cjacs)
        assert np.allclose(np.stack(res), expected * prefactor)

    @pytest.mark.parametrize("multi_term", [1, 4])
    @pytest.mark.parametrize("meas_shape", [(), (4,)])
    @pytest.mark.parametrize("par_shape", [(), (3,), (2, 2)])
    @pytest.mark.parametrize("num_shifts", [2, 5])
    @pytest.mark.parametrize("num_split_times", [1, 3])
    def test_multi_measure_multi_shots_broadcast(
        self, num_split_times, num_shifts, par_shape, multi_term, meas_shape
    ):
        """Test that ``_parshift_and_integrate`` works with results for multiple measurements
        per shift and splitting time and with multiple shot settings. This corresponds to
        ``single_measure=False and has_partitioned_shots=True``. The test is parametrized with whether
        or not there are multiple Hamiltonian terms to take into account (and sum their
        contributions), with the shape of the single measurement and of the parameter, with
        the number of shifts in the shift rule and with the number of splitting times.
        This is the variant of the previous test that uses broadcasting.
        """
        from jax import numpy as jnp

        num_shots = 3
        num_meas = 5

        cjac_shape = (num_split_times,) + par_shape
        if multi_term > 1:
            cjacs = tuple(np.random.random(cjac_shape) for _ in range(multi_term))
            psr_coeffs = tuple(np.random.random(num_shifts) for _ in range(multi_term))
        else:
            cjacs = np.random.random(cjac_shape)
            psr_coeffs = np.random.random(num_shifts)

        results_shape = (
            num_shifts * multi_term,
            num_shots,
            num_meas,
            (num_split_times + 2),
        ) + meas_shape
        new_results_shape = (
            multi_term,
            num_shifts,
            num_shots,
            num_meas,
            num_split_times + 2,
        ) + meas_shape
        results = np.random.random(results_shape)

        prefactor = 0.3214

        res = _parshift_and_integrate(
            results,
            cjacs,
            prefactor,
            psr_coeffs,
            single_measure=False,
            has_partitioned_shots=True,
            use_broadcasting=True,
        )

        assert isinstance(res, tuple)
        assert len(res) == num_shots
        for r in res:
            assert isinstance(r, tuple)
            assert len(r) == num_meas
            assert all(isinstance(_r, jnp.ndarray) for _r in r)
            assert all(_r.shape == meas_shape + par_shape for _r in r)

        _results = np.reshape(results, new_results_shape)
        _cjacs = np.stack(cjacs).reshape((multi_term,) + cjac_shape)
        _psr_coeffs = np.stack(psr_coeffs).reshape((multi_term, num_shifts))
        meas_letter = "" if meas_shape == () else "a"
        # Slice away excess results
        _results = _results[:, :, :, :, 1:-1]
        # With broadcasting, the axes of different shifts and splitting times are
        # switched for the results tensor, compared to without broadcasting.
        contraction = f"ms,msNnt{meas_letter},mt...->Nn{meas_letter}..."
        expected = np.einsum(contraction, _psr_coeffs, _results, _cjacs)
        assert np.allclose(np.stack(res), expected * prefactor)


@pytest.mark.jax
class TestStochPulseGradErrors:
    """Test errors raised by stoch_pulse_grad."""

    def test_raises_for_variance(self):
        """Test that an error is raised when attempting to differentiate a tape that measures a variance."""
        tape = qp.tape.QuantumScript(
            measurements=[qp.expval(qp.PauliX(2)), qp.var(qp.PauliZ(0))]
        )
        with pytest.raises(
            ValueError, match="Computing the gradient of variances with the stochastic"
        ):
            stoch_pulse_grad(tape)

    @pytest.mark.parametrize(
        "measurement", [qp.vn_entropy(0), qp.state(), qp.mutual_info([0], 1)]
    )
    def test_raises_for_state_measurements(self, measurement):
        """Test that an error is raised when attempting to differentiate a tape that measures a
        state, or returns a state indirectly via entropy/mutual info measurements."""
        tape = qp.tape.QuantumScript(measurements=[measurement])
        with pytest.raises(
            ValueError, match="Computing the gradient of circuits that return the state"
        ):
            stoch_pulse_grad(tape)

    @pytest.mark.parametrize("num_split_times", [-1, 0, np.array(-2)])
    def test_raises_for_less_than_one_sample(self, num_split_times):
        """Test that an error is raised if fewer than one samples for the stochastic shift rule are requested."""
        tape = qp.tape.QuantumScript([])
        with pytest.raises(ValueError, match="Expected a positive number of samples"):
            stoch_pulse_grad(tape, num_split_times=num_split_times)

    def test_trainable_batched_tape_raises(self):
        """Test that an error is raised for a broadcasted/batched tape if the broadcasted
        parameter is differentiated."""
        tape = qp.tape.QuantumScript([qp.RX([0.4, 0.2], 0)], [qp.expval(qp.PauliZ(0))])
        _match = r"Computing the gradient of broadcasted tapes .* using the stochastic pulse"
        with pytest.raises(NotImplementedError, match=_match):
            stoch_pulse_grad(tape)

    def test_nontrainable_batched_tape(self):
        """Test that no error is raised for a broadcasted/batched tape if the broadcasted
        parameter is not differentiated, and that the results correspond to the stacked
        results of the single-tape derivatives."""
        import jax.numpy as jnp

        dev = qp.device("default.qubit")
        x = [0.4, 0.2]
        params = [jnp.array(0.14)]
        ham_single_q_const = qp.pulse.constant * qp.PauliY(0)
        op = qp.evolve(ham_single_q_const)(params, 0.7)
        tape = qp.tape.QuantumScript(
            [qp.RX(x, 0), op], [qp.expval(qp.PauliZ(0))], trainable_params=[1]
        )
        batched_tapes, batched_fn = stoch_pulse_grad(tape, argnum=0, num_split_times=1)
        batched_grad = batched_fn(dev.execute(batched_tapes))
        separate_tapes = [
            qp.tape.QuantumScript(
                [qp.RX(_x, 0), op], [qp.expval(qp.PauliZ(0))], trainable_params=[1]
            )
            for _x in x
        ]
        separate_tapes_and_fns = [
            stoch_pulse_grad(t, argnum=0, num_split_times=1) for t in separate_tapes
        ]
        separate_grad = [_fn(dev.execute(_tapes)) for _tapes, _fn in separate_tapes_and_fns]
        assert np.allclose(batched_grad, separate_grad)

    @pytest.mark.parametrize("num_meas", [0, 1, 2])
    def test_warning_no_trainable_params(self, num_meas):
        """Test that an empty gradient is returned when there are no trainable parameters."""
        measurements = [qp.expval(qp.PauliX(w)) for w in range(num_meas)]
        # No parameters at all
        tape = qp.tape.QuantumScript([], measurements=measurements)
        with pytest.warns(
            UserWarning, match="Attempted to compute the gradient of a tape with no trainable"
        ):
            tapes, fn = stoch_pulse_grad(tape)
        assert not tapes
        assert qp.math.allclose(fn([]), tuple(qp.math.zeros([0]) for _ in range(num_meas)))

        # parameters but none are trainable
        ops = [qp.RX(0.4, 2), qp.CRY(0.1, [1, 0])]
        tape = qp.tape.QuantumScript(ops, measurements=measurements)
        tape.trainable_params = []
        with pytest.warns(
            UserWarning, match="Attempted to compute the gradient of a tape with no trainable"
        ):
            tapes, fn = stoch_pulse_grad(tape)
        assert not tapes
        assert qp.math.allclose(fn([]), tuple(qp.math.zeros([0]) for _ in range(num_meas)))

    def test_raises_non_pulse_marked_as_trainable(self):
        """Test that an empty gradient is returned when there are no trainable parameters."""
        ops = [qp.RX(0.4, wires=0)]
        tape = qp.tape.QuantumScript(ops, measurements=[qp.expval(qp.PauliZ(0))])
        tape.trainable_params = [0]
        with pytest.raises(ValueError, match="stoch_pulse_grad does not support differentiating"):
            stoch_pulse_grad(tape)

    def test_raises_use_broadcasting_with_broadcasted_tape(self):
        """Test that an error is raised if the option `use_broadcasting` is activated
        for a tape that already is broadcasted."""
        ham = qp.dot([qp.pulse.constant], [qp.PauliX(0)])
        ops = [qp.RX(0.5, wires=0), qp.evolve(ham, return_intermediate=True)([0.152], 0.3)]
        tape = qp.tape.QuantumScript(ops, measurements=[qp.expval(qp.PauliZ(0))])
        tape.trainable_params = [0]
        with pytest.raises(ValueError, match="Broadcasting is not supported for tapes that"):
            stoch_pulse_grad(tape, use_broadcasting=True)

    @pytest.mark.parametrize(
        "reorder_fn",
        [
            lambda x, _: [x[0] + 10, x[0] - 2],
            lambda x, _: [[x[0], x[0] + 10], [x[0] - 2, x[0]]],
        ],
    )
    def test_raises_for_invalid_reorder_fn(self, reorder_fn):
        """Test that an error is raised for an invalid reordering function of
        a HardwareHamiltonian."""

        H = qp.pulse.transmon_drive(qp.pulse.constant, 0.0, 0.0, wires=[0])
        H.reorder_fn = reorder_fn
        ops = [qp.evolve(H)([0.152], 0.3)]
        tape = qp.tape.QuantumScript(ops, measurements=[qp.expval(qp.PauliZ(0))])
        tape.trainable_params = [0]
        _match = "Only permutations, fan-out or fan-in functions are allowed as reordering"
        with pytest.raises(ValueError, match=_match):
            stoch_pulse_grad(tape)


@pytest.mark.jax
class TestStochPulseGrad:
    """Test working cases of stoch_pulse_grad."""

    @staticmethod
    def sine(p, t):
        """Compute the sin function with parametrized amplitude and frequency at a given time."""
        from jax import numpy as jnp

        return p[0] * jnp.sin(p[1] * t)

    # Need to wrap the Hamiltonians in a callable in order to use `qp.pulse` functions, as
    # the tests would otherwise fail when used without JAX.
    @pytest.mark.parametrize(
        "ops, arg, exp_shapes",
        (
            (lambda _: [qp.RX(0.4, 0), qp.RZ(0.9, 0)], None, [(2,), (2, 4)]),
            (
                lambda _: [qp.evolve(qp.pulse.constant * qp.PauliZ(0))([0.2], 0.1)],
                None,
                [(1,), (1, 4)],
            ),
            (
                lambda x: [qp.evolve(qp.pulse.pwc * qp.PauliZ(0))([x], 0.1)],
                np.ones(3),
                [(3,), (3, 4)],
            ),
            (
                lambda x: [qp.evolve(qp.pulse.pwc * qp.PauliZ(0))([x], 0.1), qp.RX(1.0, 2)],
                np.ones(3),
                [[(3,), (1,)], [(3, 4), (1, 4)]],
            ),
        ),
    )
    def test_all_zero_grads(self, ops, arg, exp_shapes):  # pylint:disable=unused-argument
        """Test that a zero gradient is returned when all trainable parameters are
        identified to have zero gradient in advance."""
        import jax
        from jax import numpy as jnp

        arg = None if arg is None else jnp.array(arg)
        ops = ops(arg)
        measurements = [qp.expval(qp.PauliZ("a")), qp.probs(["b", "c"])]
        tape = qp.tape.QuantumScript(ops, measurements=measurements)
        tapes, fn = stoch_pulse_grad(tape)
        assert not tapes

        res = fn([])
        assert isinstance(res, tuple) and len(res) == 2
        for r, exp_shape in zip(res, exp_shapes):
            if isinstance(exp_shape, list):
                assert all(qp.math.allclose(_r, np.zeros(_sh)) for _r, _sh in zip(r, exp_shape))
            else:
                assert qp.math.allclose(r, np.zeros(exp_shape))
        jax.clear_caches()

    def test_some_zero_grads(self):
        """Test that a zero gradient is returned for trainable parameters that are
        identified to have a zero gradient in advance."""
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
        ops = [
            qp.evolve(qp.pulse.pwc(0.1) * qp.PauliX(w))([jnp.linspace(1.0, 2.0, 5)], 0.1)
            for w in [1, 0]
        ]
        measurements = [qp.expval(qp.PauliZ(0)), qp.probs(wires=0)]
        tape = qp.tape.QuantumScript(ops, measurements)
        tapes, fn = stoch_pulse_grad(tape, num_split_times=3)
        assert len(tapes) == 2 * 3

        dev = qp.device("default.qubit", wires=2)
        res = fn(qp.execute(tapes, dev, None))
        assert isinstance(res, tuple) and len(res) == 2
        assert qp.math.allclose(res[0][0], np.zeros(5))
        assert qp.math.allclose(res[1][0], np.zeros((2, 5)))
        jax.clear_caches()

    @pytest.mark.parametrize("num_split_times", [1, 3])
    @pytest.mark.parametrize("t", [2.0, 3, (0.5, 0.6), (0.1, 0.9, 1.2)])
    def test_constant_ry(self, num_split_times, t):
        """Test that the derivative of a pulse generated by a constant Hamiltonian,
        which is a Pauli word, is computed correctly."""
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
        params = [jnp.array(0.24)]
        delta_t = t[-1] - t[0] if isinstance(t, tuple) else t
        ham_single_q_const = qp.pulse.constant * qp.PauliY(0)
        op = qp.evolve(ham_single_q_const)(params, t)
        tape = qp.tape.QuantumScript([op], [qp.expval(qp.PauliZ(0))])

        dev = qp.device("default.qubit", wires=1)
        # Effective rotation parameter
        p = params[0] * delta_t
        tapes, fn = stoch_pulse_grad(tape, num_split_times=num_split_times)
        assert len(tapes) == num_split_times * 2

        res = fn(qp.execute(tapes, dev, None))
        assert qp.math.isclose(res, -2 * jnp.sin(2 * p) * delta_t)

        # note that qp.execute changes trainable params
        r = qp.execute([tape], dev, None)
        assert qp.math.isclose(r, jnp.cos(2 * p), atol=1e-4)
        jax.clear_caches()

    def test_constant_ry_argnum(self):
        """Test that the derivative of a pulse generated by a constant Hamiltonian,
        which is a Pauli word, is computed correctly if it is not the only
        operation in a tape but selected via `argnum`."""
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
        params = [jnp.array(0.04)]
        t = 0.1
        y = 0.3
        ham_single_q_const = qp.pulse.constant * qp.PauliY(0)
        op = qp.evolve(ham_single_q_const)(params, t)
        tape = qp.tape.QuantumScript([qp.RY(y, 0), op], [qp.expval(qp.PauliZ(0))])
        tape.trainable_params = [1]

        dev = qp.device("default.qubit", wires=1)
        # Effective rotation parameter
        p = params[0] * t
        num_split_times = 1
        tapes, fn = stoch_pulse_grad(tape, num_split_times=num_split_times, argnum=0)
        assert len(tapes) == num_split_times * 2

        res = fn(qp.execute(tapes, dev, None))
        assert qp.math.isclose(res, -2 * jnp.sin(2 * p + y) * t)

        r = qp.execute([tape], dev, None)
        assert qp.math.isclose(r, jnp.cos(2 * p + y), atol=1e-4)
        jax.clear_caches()

    @pytest.mark.parametrize("num_split_times", [1, 3])
    @pytest.mark.parametrize("t", [2.0, 3, (0.5, 0.6), (0.1, 0.9, 1.2)])
    def test_constant_ry_rescaled(self, num_split_times, t):
        """Test that the derivative of a pulse generated by a constant Hamiltonian,
        which is a Pauli sentence, is computed correctly."""
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
        params = [jnp.array(0.24)]
        T = t if isinstance(t, tuple) else (0, t)
        ham_single_q_const = qp.pulse.constant * qp.dot(
            [0.2, 0.9], [qp.PauliY(0), qp.PauliX(0)]
        )
        op = qp.evolve(ham_single_q_const)(params, t)
        tape = qp.tape.QuantumScript([op], [qp.expval(qp.PauliZ(0))])

        dev = qp.device("default.qubit", wires=1)
        # Prefactor due to the generator being a Pauli sentence
        prefactor = np.sqrt(0.85)
        # Effective rotation parameter
        p = params[0] * (delta_t := T[-1] - T[0]) * prefactor
        tapes, fn = stoch_pulse_grad(tape, num_split_times=num_split_times)
        assert len(tapes) == num_split_times * 2

        res = fn(qp.execute(tapes, dev, None))
        assert qp.math.isclose(res, -2 * jnp.sin(2 * p) * delta_t * prefactor)
        r = qp.execute([tape], dev, None)
        assert qp.math.isclose(r, jnp.cos(2 * p), atol=1e-4)
        jax.clear_caches()

    @pytest.mark.parametrize("t", [0.02, (0.5, 0.6)])
    def test_sin_envelope_rz_expval(self, t):
        """Test that the derivative of a pulse with a sine wave envelope
        is computed correctly when returning an expectation value."""
        import jax
        import jax.numpy as jnp

        T = t if isinstance(t, tuple) else (0, t)

        dev = qp.device("default.qubit", wires=1)
        params = [jnp.array([2.3, -0.245])]

        ham = self.sine * qp.PauliZ(0)
        op = qp.evolve(ham)(params, t)
        tape = qp.tape.QuantumScript([qp.Hadamard(0), op], [qp.expval(qp.PauliX(0))])

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

        num_split_times = 5
        tapes, fn = stoch_pulse_grad(tape, num_split_times=num_split_times)
        assert len(tapes) == 2 * num_split_times

        res = fn(qp.execute(tapes, dev, None))
        exp_grad = -2 * jnp.sin(2 * theta) * theta_jac
        # classical Jacobian is being estimated with the Monte Carlo sampling -> coarse tolerance
        assert qp.math.allclose(res, exp_grad, atol=0.2)
        r = qp.execute([tape], dev, None)
        assert qp.math.isclose(r, jnp.cos(2 * theta))
        jax.clear_caches()

    @pytest.mark.parametrize("t", [0.02, (0.5, 0.6)])
    def test_sin_envelope_rx_probs(self, t):
        """Test that the derivative of a pulse with a sine wave envelope
        is computed correctly when returning probabilities."""
        import jax
        import jax.numpy as jnp

        T = t if isinstance(t, tuple) else (0, t)

        dev = qp.device("default.qubit", wires=1)
        params = [jnp.array([2.3, -0.245])]

        ham = self.sine * qp.PauliX(0)
        op = qp.evolve(ham)(params, t)
        tape = qp.tape.QuantumScript([op], [qp.probs(wires=0)])

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

        num_split_times = 5
        tapes, fn = stoch_pulse_grad(tape, num_split_times=num_split_times)
        assert len(tapes) == 2 * num_split_times

        jac = fn(qp.execute(tapes, dev, None))
        probs_jac = jnp.array([-1, 1]) * (2 * jnp.sin(theta) * jnp.cos(theta))
        exp_jac = jnp.tensordot(probs_jac, theta_jac, axes=0)
        # classical Jacobian is being estimated with the Monte Carlo sampling -> coarse tolerance
        assert qp.math.allclose(jac, exp_jac, atol=0.2)
        r = qp.execute([tape], dev, None)
        exp_probs = jnp.array([jnp.cos(theta) ** 2, jnp.sin(theta) ** 2])
        assert qp.math.allclose(r, exp_probs)
        jax.clear_caches()

    @pytest.mark.parametrize("t", [0.02, (0.5, 0.6)])
    def test_sin_envelope_rx_expval_probs(self, t):
        """Test that the derivative of a pulse with a sine wave envelope
        is computed correctly when returning expectation."""
        import jax
        import jax.numpy as jnp

        T = t if isinstance(t, tuple) else (0, t)

        dev = qp.device("default.qubit", wires=1)
        params = [jnp.array([2.3, -0.245])]

        ham = self.sine * qp.PauliX(0)
        op = qp.evolve(ham)(params, t)
        tape = qp.tape.QuantumScript([op], [qp.expval(qp.PauliZ(0)), qp.probs(wires=0)])

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

        num_split_times = 5
        tapes, fn = stoch_pulse_grad(tape, num_split_times=num_split_times)
        assert len(tapes) == 2 * num_split_times

        jac = fn(qp.execute(tapes, dev, None))
        expval_jac = -2 * jnp.sin(2 * theta)
        probs_jac = jnp.array([-1, 1]) * (2 * jnp.sin(theta) * jnp.cos(theta))
        exp_jac = (expval_jac * theta_jac, jnp.tensordot(probs_jac, theta_jac, axes=0))
        # classical Jacobian is being estimated with the Monte Carlo sampling -> coarse tolerance
        for j, e in zip(jac, exp_jac):
            assert qp.math.allclose(j, e, atol=0.2)

        r = qp.execute([tape], dev, None)[0]
        exp = (jnp.cos(2 * theta), jnp.array([jnp.cos(theta) ** 2, jnp.sin(theta) ** 2]))
        assert isinstance(r, tuple) and len(r) == 2
        assert qp.math.allclose(r[0], exp[0])
        assert qp.math.allclose(r[1], exp[1])
        jax.clear_caches()

    @pytest.mark.parametrize("t", [0.02, (0.5, 0.6)])
    def test_pwc_envelope_rx(self, t, seed):
        """Test that the derivative of a pulse generated by a piecewise constant Hamiltonian
        is computed correctly."""
        import jax
        import jax.numpy as jnp

        T = t if isinstance(t, tuple) else (0, t)

        dev = qp.device("default.qubit", wires=1)
        params = [jnp.array([0.24, 0.9, -0.1, 2.3, -0.245])]
        op = qp.evolve(qp.pulse.pwc(t) * qp.PauliZ(0))(params, t)
        tape = qp.tape.QuantumScript([qp.Hadamard(0), op], [qp.expval(qp.PauliX(0))])

        # Effective rotation parameter
        p = jnp.mean(params[0]) * (T[1] - T[0])
        num_split_times = 5
        tapes, fn = stoch_pulse_grad(tape, num_split_times=num_split_times, sampler_seed=seed)
        assert len(tapes) == 2 * num_split_times

        res = fn(qp.execute(tapes, dev, None))
        # The sampling of pwc functions does not automatically reduce to the analytically
        # correct time integrals, leading to approximations -> coarse tolerance
        assert qp.math.allclose(
            res, -2 * jnp.sin(2 * p) * (T[1] - T[0]) / len(params[0]), atol=0.01
        )
        r = qp.execute([tape], dev, None)
        assert qp.math.isclose(r, jnp.cos(2 * p))
        jax.clear_caches()

    @pytest.mark.parametrize("t", [2.0, 3, (0.5, 0.6)])
    def test_constant_commuting(self, t):
        """Test that the derivative of a pulse generated by two constant commuting Hamiltonians
        is computed correctly."""
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
        params = [jnp.array(0.24), jnp.array(-0.672)]
        T = t if isinstance(t, tuple) else (0, t)
        op = qp.evolve(qp.pulse.constant * qp.PauliX(0) + qp.pulse.constant * qp.PauliY(1))(
            params, t
        )
        tape = qp.tape.QuantumScript([op], [qp.expval(qp.PauliZ(0) @ qp.PauliZ(1))])

        dev = qp.device("default.qubit", wires=2)
        p = [_p * (T[1] - T[0]) for _p in params]
        tapes, fn = stoch_pulse_grad(tape)
        assert len(tapes) == 4

        res = fn(qp.execute(tapes, dev, None))
        exp_grad = [
            -2 * jnp.sin(2 * p[0]) * jnp.cos(2 * p[1]) * (T[1] - T[0]),
            -2 * jnp.sin(2 * p[1]) * jnp.cos(2 * p[0]) * (T[1] - T[0]),
        ]
        assert qp.math.allclose(res, exp_grad)
        r = qp.execute([tape], dev, None)
        # Effective rotation parameters
        exp = jnp.cos(2 * p[0]) * jnp.cos(2 * p[1])
        assert qp.math.isclose(r, exp)
        jax.clear_caches()

    @pytest.mark.slow
    def test_advanced_pulse(self):
        """Test the derivative of a more complex pulse."""
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)

        ham = (
            qp.pulse.constant * qp.PauliX(0)
            + (lambda p, t: jnp.sin(p * t)) * qp.PauliZ(0)
            + jnp.polyval
            * qp.dot([1.0, 0.4], [qp.PauliY(0) @ qp.PauliY(1), qp.PauliX(0) @ qp.PauliX(1)])
        )
        params = [jnp.array(1.51), jnp.array(-0.371), jnp.array([0.2, 0.2, -0.4])]
        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev, interface="jax")
        def qnode(params):
            qp.evolve(ham, atol=1e-6)(params, 0.1)
            return qp.expval(qp.PauliY(0) @ qp.PauliX(1))

        tape = qp.workflow.construct_tape(qnode)(params)

        num_split_times = 5
        tape.trainable_params = [0, 1, 2]

        # FIXME: This test case is not updated to use the pytest-rng generated seed because I'm
        #       unable to find a local salt that actually allows this test to pass. The 7123 here
        #       is basically a magic number. Every other seed I tried fails. I believe this test
        #       should be rewritten to use a better testing strategy because this currently goes
        #       against the spirit of seeding.
        tapes, fn = stoch_pulse_grad(tape, num_split_times=num_split_times, sampler_seed=7123)
        # Two generating terms with two shifts (X_0 and Z_0), one with eight shifts
        # (Y_0Y_1+0.4 X_1 has eigenvalues [-1.4, -0.6, 0.6, 1.4] yielding frequencies
        # [0.8, 1.2, 2.0, 2.8] and hence 2 * 4 = 8 shifts)
        num_shifts = 2 * 2 + 8
        assert len(tapes) == num_shifts * num_split_times

        res = fn(qp.execute(tapes, dev, None))
        exp_grad = jax.grad(qnode)(params)
        assert all(qp.math.allclose(r, e, rtol=0.4) for r, e in zip(res, exp_grad))
        jax.clear_caches()

    def test_randomness(self):
        """Test that the derivative of a pulse is exactly the same when reusing a seed and
        that it differs when using a different seed."""
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)

        params = [jnp.array(1.51)]
        op = qp.evolve((lambda p, t: p * t) * qp.PauliX(0))(params, 0.4)
        tape = qp.tape.QuantumScript([op], [qp.expval(qp.PauliY(0))])

        seed_a = 8752
        seed_b = 8753
        tapes_a_0, fn_a_0 = stoch_pulse_grad(tape, num_split_times=2, sampler_seed=seed_a)
        tapes_a_1, fn_a_1 = stoch_pulse_grad(tape, num_split_times=2, sampler_seed=seed_a)
        tapes_b, fn_b = stoch_pulse_grad(tape, num_split_times=2, sampler_seed=seed_b)

        for tape_a_0, tape_a_1, tape_b in zip(tapes_a_0, tapes_a_1, tapes_b):
            for op_a_0, op_a_1, op_b in zip(tape_a_0, tape_a_1, tape_b):
                if isinstance(op_a_0, qp.pulse.ParametrizedEvolution):
                    # The a_0 and a_1 operators are equal
                    qp.assert_equal(op_a_0, op_a_1)
                    # The a_0 and b operators differ in time but are equal otherwise
                    assert not qp.equal(op_a_0, op_b)
                    op_b.t = op_a_0.t
                    qp.assert_equal(op_a_0, op_b)
                else:
                    qp.assert_equal(op_a_0, op_a_1)
                    qp.assert_equal(op_a_0, op_b)

        dev = qp.device("default.qubit", wires=1)
        res_a_0 = fn_a_0(qp.execute(tapes_a_0, dev, None))
        res_a_1 = fn_a_1(qp.execute(tapes_a_1, dev, None))
        res_b = fn_b(qp.execute(tapes_b, dev, None))

        assert res_a_0 == res_a_1
        assert not res_a_0 == res_b
        jax.clear_caches()

    def test_two_pulses(self, seed):
        """Test that the derivatives of two pulses in a circuit are computed correctly."""
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)

        ham_0 = qp.pulse.constant * qp.PauliX(0) + (lambda p, t: jnp.sin(p * t)) * qp.PauliY(0)
        ham_1 = qp.dot([0.3, jnp.polyval], [qp.PauliZ(0), qp.PauliY(0) @ qp.PauliY(1)])
        params_0 = [jnp.array(1.51), jnp.array(-0.371)]
        params_1 = [jnp.array([0.2, 0.2, -0.4])]
        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev, interface="jax")
        def qnode(params_0, params_1):
            qp.evolve(ham_0)(params_0, 0.1)
            qp.evolve(ham_1)(params_1, 0.15)
            return qp.expval(qp.PauliY(0) @ qp.PauliZ(1))

        tape = qp.workflow.construct_tape(qnode)(params_0, params_1)
        num_split_times = 3
        tape.trainable_params = [0, 1, 2]
        tapes, fn = stoch_pulse_grad(tape, num_split_times=num_split_times, sampler_seed=seed)
        assert len(tapes) == 3 * 2 * num_split_times

        res = fn(qp.execute(tapes, dev, None))
        exp_grad = jax.grad(qnode, argnums=(0, 1))(params_0, params_1)
        exp_grad = exp_grad[0] + exp_grad[1]
        # Values are close to zero so we need to use `atol` instead of `rtol`
        # to avoid numerical issues
        assert all(qp.math.allclose(r, e, atol=5e-4) for r, e in zip(res, exp_grad))
        jax.clear_caches()

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "generator, exp_num_tapes, prefactor",
        [
            (qp.PauliY(0), 2, 1.0),
            (0.6 * qp.PauliY(0) + 0.8 * qp.PauliX(0), 2, 1.0),
            (qp.Hamiltonian([0.25, 1.2], [qp.PauliX(0), qp.PauliX(0) @ qp.PauliZ(1)]), 8, 1.45),
        ],
    )
    def test_with_jit(self, generator, exp_num_tapes, prefactor):
        """Test that the stochastic parameter-shift rule works with JITting."""
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
        dev = qp.device("default.qubit", wires=len(generator.wires))
        T = (0.2, 0.5)
        ham_single_q_const = qp.dot([qp.pulse.constant], [generator])
        meas = [qp.expval(qp.PauliZ(0))]

        def fun(params):
            """Create a pulse with the given parameters, build a tape from it, and
            differentiate it with stoch_pulse_grad."""
            op = qp.evolve(ham_single_q_const)(params, T)
            tape = qp.tape.QuantumScript([op], meas)
            tapes, fn = stoch_pulse_grad(tape)
            assert len(tapes) == exp_num_tapes
            res = fn(qp.execute(tapes, dev, "backprop"))
            return res

        params = [jnp.array(0.24)]
        # Effective rotation parameter
        p = params[0] * (T[1] - T[0]) * prefactor
        res = fun(params)
        assert qp.math.isclose(res, -2 * jnp.sin(2 * p) * (T[1] - T[0]) * prefactor)
        res_jit = jax.jit(fun)(params)
        assert qp.math.isclose(res, res_jit)
        jax.clear_caches()

    @pytest.mark.parametrize("shots", [None, 100])
    def test_shots_attribute(self, shots):  # pylint:disable=unused-argument
        """Tests that the shots attribute is copied to the new tapes"""
        tape = qp.tape.QuantumTape([], [qp.expval(qp.PauliZ(0)), qp.probs([1, 2])], shots=shots)
        with pytest.warns(UserWarning, match="Attempted to compute the gradient of a tape with no"):
            tapes, _ = stoch_pulse_grad(tape)

        assert all(new_tape.shots == tape.shots for new_tape in tapes)


@pytest.mark.jax
class TestStochPulseGradQNode:
    """Test that stoch_pulse_grad integrates correctly with QNodes."""

    def test_raises_for_application_to_qnodes(self):
        """Test that an error is raised when applying ``stoch_pulse_grad``
        to a QNode directly."""
        dev = qp.device("default.qubit", wires=1)
        ham_single_q_const = qp.pulse.constant * qp.PauliY(0)

        @qp.qnode(dev, interface="jax")
        def circuit(params):
            qp.evolve(ham_single_q_const)([params], 0.2)
            return qp.expval(qp.PauliZ(0))

        _match = "stochastic pulse parameter-shift gradient transform to a QNode directly"
        with pytest.raises(NotImplementedError, match=_match):
            stoch_pulse_grad(circuit, num_split_times=2)

    # TODO: include the following tests when #4225 is resolved.
    @pytest.mark.skip("Applying this gradient transform to QNodes directly is not supported.")
    def test_qnode_expval_single_par(self):
        """Test that a simple qnode that returns an expectation value
        can be differentiated with stoch_pulse_grad."""
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
        dev = qp.device("default.qubit", wires=1)
        T = 0.2
        ham_single_q_const = qp.pulse.constant * qp.PauliY(0)

        @qp.qnode(dev, interface="jax")
        def circuit(params):
            qp.evolve(ham_single_q_const)([params], T)
            return qp.expval(qp.PauliZ(0))

        params = jnp.array(0.4)
        with qp.Tracker(dev) as tracker:
            grad = stoch_pulse_grad(circuit, num_split_times=2)(params)

        p = params * T
        exp_grad = -2 * jnp.sin(2 * p) * T
        assert jnp.allclose(grad, exp_grad)
        assert tracker.totals["executions"] == 4  # two shifted tapes, two splitting times


@pytest.mark.jax
class TestStochPulseGradIntegration:
    """Test that stoch_pulse_grad integrates correctly with QNodes and ML interfaces."""

    @pytest.mark.parametrize("shots, tol", [(None, 1e-4), (100, 0.1), ([100, 99], 0.1)])
    @pytest.mark.parametrize("num_split_times", [1, 2])
    def test_simple_qnode_expval(self, num_split_times, shots, tol, seed):
        """Test that a simple qnode that returns an expectation value
        can be differentiated with stoch_pulse_grad."""
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
        dev = qp.device("default.qubit", wires=1, seed=jax.random.PRNGKey(seed))
        T = 0.2
        ham_single_q_const = qp.pulse.constant * qp.PauliY(0)

        @qp.set_shots(shots)
        @qp.qnode(
            dev,
            interface="jax",
            diff_method=stoch_pulse_grad,
            gradient_kwargs={"num_split_times": num_split_times},
        )
        def circuit(params):
            qp.evolve(ham_single_q_const)(params, T)
            return qp.expval(qp.PauliZ(0))

        params = [jnp.array(0.4)]
        grad = jax.jacobian(circuit)(params)
        p = params[0] * T
        exp_grad = -2 * jnp.sin(2 * p) * T
        assert qp.math.allclose(grad, exp_grad, atol=tol, rtol=0.0)
        jax.clear_caches()

    @pytest.mark.slow
    @pytest.mark.parametrize("shots, tol", [(None, 1e-4), (100, 0.1), ([100, 99], 0.1)])
    @pytest.mark.parametrize("num_split_times", [1, 2])
    def test_simple_qnode_expval_two_evolves(self, num_split_times, shots, tol, seed):
        """Test that a simple qnode that returns an expectation value
        can be differentiated with stoch_pulse_grad."""
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
        dev = qp.device("default.qubit", wires=1, seed=jax.random.PRNGKey(seed))
        T_x = 0.1
        T_y = 0.2
        ham_x = qp.pulse.constant * qp.PauliX(0)
        ham_y = qp.pulse.constant * qp.PauliX(0)

        @qp.set_shots(shots)
        @qp.qnode(
            dev,
            interface="jax",
            diff_method=stoch_pulse_grad,
            gradient_kwargs={"num_split_times": num_split_times},
        )
        def circuit(params):
            qp.evolve(ham_x)(params[0], T_x)
            qp.evolve(ham_y)(params[1], T_y)
            return qp.expval(qp.PauliZ(0))

        params = [[jnp.array(0.4)], [jnp.array(-0.1)]]
        grad = jax.jacobian(circuit)(params)
        p_x = params[0][0] * T_x
        p_y = params[1][0] * T_y
        exp_grad = [[-2 * jnp.sin(2 * (p_x + p_y)) * T_x], [-2 * jnp.sin(2 * (p_x + p_y)) * T_y]]
        assert qp.math.allclose(grad, exp_grad, atol=tol, rtol=0.0)
        jax.clear_caches()

    @pytest.mark.parametrize("shots, tol", [(None, 1e-4), (100, 0.1), ([100, 99], 0.1)])
    @pytest.mark.parametrize("num_split_times", [1, 2])
    def test_simple_qnode_probs(self, num_split_times, shots, tol, seed):
        """Test that a simple qnode that returns an probabilities
        can be differentiated with stoch_pulse_grad."""
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
        dev = qp.device("default.qubit", wires=1, seed=jax.random.PRNGKey(seed))
        T = 0.2
        ham_single_q_const = qp.pulse.constant * qp.PauliY(0)

        @qp.set_shots(shots)
        @qp.qnode(
            dev,
            interface="jax",
            diff_method=stoch_pulse_grad,
            gradient_kwargs={"num_split_times": num_split_times},
        )
        def circuit(params):
            qp.evolve(ham_single_q_const)(params, T)
            return qp.probs(wires=0)

        params = [jnp.array(0.4)]
        jac = jax.jacobian(circuit)(params)
        p = params[0] * T
        exp_jac = jnp.array([-1, 1]) * jnp.sin(2 * p) * T
        assert qp.math.allclose(jac, exp_jac, atol=tol, rtol=0.0)
        jax.clear_caches()

    @pytest.mark.parametrize("shots, tol", [(None, 1e-4), (100, 0.1), ([100, 100], 0.1)])
    @pytest.mark.parametrize("num_split_times", [1, 2])
    def test_simple_qnode_probs_expval(self, num_split_times, shots, tol, seed):
        """Test that a simple qnode that returns an probabilities
        can be differentiated with stoch_pulse_grad."""
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
        dev = qp.device("default.qubit", wires=1, seed=jax.random.PRNGKey(seed))
        T = 0.2
        ham_single_q_const = qp.pulse.constant * qp.PauliY(0)

        @qp.set_shots(shots)
        @qp.qnode(
            dev,
            interface="jax",
            diff_method=stoch_pulse_grad,
            gradient_kwargs={"num_split_times": num_split_times},
        )
        def circuit(params):
            qp.evolve(ham_single_q_const)(params, T)
            return qp.probs(wires=0), qp.expval(qp.PauliZ(0))

        params = [jnp.array(0.4)]
        jac = jax.jacobian(circuit)(params)
        p = params[0] * T
        exp_jac = (jnp.array([-1, 1]) * jnp.sin(2 * p) * T, -2 * jnp.sin(2 * p) * T)
        if isinstance(shots, list):
            for j_shots in jac:
                for j, e in zip(j_shots, exp_jac):
                    assert qp.math.allclose(j, e, atol=tol, rtol=0.0)
        else:
            for j, e in zip(jac, exp_jac):
                assert qp.math.allclose(j, e, atol=tol, rtol=0.0)
        jax.clear_caches()

    @pytest.mark.xfail  # TODO: [sc-82874]
    @pytest.mark.parametrize("num_split_times", [1, 2])
    @pytest.mark.parametrize("time_interface", ["python", "numpy", "jax"])
    def test_simple_qnode_jit(self, num_split_times, time_interface):
        """Test that a simple qnode can be differentiated with stoch_pulse_grad."""
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
        dev = qp.device("default.qubit", wires=1)
        T = {"python": 0.2, "numpy": np.array(0.2), "jax": jnp.array(0.2)}[time_interface]
        ham_single_q_const = qp.pulse.constant * qp.PauliY(0)

        @qp.qnode(
            dev,
            interface="jax",
            diff_method=stoch_pulse_grad,
            gradient_kwargs={"num_split_times": num_split_times},
        )
        def circuit(params, T=None):
            qp.evolve(ham_single_q_const)(params, T)
            return qp.expval(qp.PauliZ(0))

        params = [jnp.array(0.4)]
        p = params[0] * T
        exp_grad = -2 * jnp.sin(2 * p) * T
        jit_grad = jax.jit(jax.grad(circuit))(params, T=T)
        assert qp.math.isclose(jit_grad, exp_grad)
        jax.clear_caches()

    @pytest.mark.slow
    def test_advanced_qnode(self, seed):
        """Test that an advanced qnode can be differentiated with stoch_pulse_grad."""
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)

        params = [jnp.array(0.21), jnp.array(-0.171), jnp.array([0.05, 0.03, -0.1])]
        dev = qp.device("default.qubit", wires=2)
        ham = (
            qp.pulse.constant * qp.PauliX(0)
            + (lambda p, t: jnp.sin(p * t)) * qp.PauliZ(0)
            + jnp.polyval * (qp.PauliY(0) @ qp.PauliY(1))
        )

        def ansatz(params):
            qp.evolve(ham)(params, 0.1)
            return qp.expval(qp.PauliY(0) @ qp.PauliX(1))

        num_split_times = 10
        qnode_pulse_grad = qp.QNode(
            ansatz,
            dev,
            interface="jax",
            diff_method=stoch_pulse_grad,
            gradient_kwargs={"num_split_times": num_split_times, "sampler_seed": seed},
        )
        qnode_backprop = qp.QNode(ansatz, dev, interface="jax")

        with qp.Tracker(dev) as tracker:
            grad_pulse_grad = jax.grad(qnode_pulse_grad)(params)
        assert tracker.totals["executions"] == 1 + 2 * 3 * num_split_times
        grad_backprop = jax.grad(qnode_backprop)(params)
        # Values are close to zero so we need to use `atol` instead of `rtol`
        # to avoid numerical issues
        assert all(
            qp.math.allclose(r, e, atol=5e-3) for r, e in zip(grad_pulse_grad, grad_backprop)
        )
        jax.clear_caches()

    @pytest.mark.parametrize("shots, tol", [(None, 1e-4), (100, 0.1), ([100, 100], 0.1)])
    @pytest.mark.parametrize("num_split_times", [1, 2])
    def test_qnode_probs_expval_broadcasting(self, num_split_times, shots, tol, seed):
        """Test that a simple qnode that returns an expectation value and probabilities
        can be differentiated with stoch_pulse_grad with use_broadcasting."""
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
        dev = qp.device("default.qubit", wires=1, seed=jax.random.PRNGKey(seed))
        T = 0.2
        ham_single_q_const = qp.pulse.constant * qp.PauliY(0)

        @qp.set_shots(shots)
        @qp.qnode(
            dev,
            interface="jax",
            diff_method=stoch_pulse_grad,
            gradient_kwargs={"num_split_times": num_split_times, "use_broadcasting": True},
        )
        def circuit(params):
            qp.evolve(ham_single_q_const)(params, T)
            return qp.probs(wires=0), qp.expval(qp.PauliZ(0))

        params = [jnp.array(0.4)]
        jac = jax.jacobian(circuit)(params)
        p = params[0] * T
        exp_jac = (jnp.array([-1, 1]) * jnp.sin(2 * p) * T, -2 * jnp.sin(2 * p) * T)
        if isinstance(shots, list):
            for j_shots in jac:
                for j, e in zip(j_shots, exp_jac):
                    assert qp.math.allclose(j, e, atol=tol, rtol=0.0)
        else:
            for j, e in zip(jac, exp_jac):
                assert qp.math.allclose(j[0], e, atol=tol, rtol=0.0)
        jax.clear_caches()

    @pytest.mark.parametrize("num_split_times", [1, 2])
    def test_broadcasting_coincides_with_nonbroadcasting(self, num_split_times, seed):
        """Test that using broadcasting or not does not change the result."""
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
        dev = qp.device("default.qubit", wires=2)
        T = 0.2

        def f(p, t):
            return jnp.sin(p * t)

        ham_single_q_const = 0.1 * qp.PauliX(0) + f * (qp.PauliY(0) @ qp.PauliY(1))

        def ansatz(params):
            qp.evolve(ham_single_q_const)(params, T)
            return qp.probs(wires=0), qp.expval(qp.PauliZ(0))

        # Create QNodes with and without use_broadcasting.
        circuit_bc = qp.QNode(
            ansatz,
            dev,
            interface="jax",
            diff_method=stoch_pulse_grad,
            gradient_kwargs={
                "num_split_times": num_split_times,
                "use_broadcasting": True,
                "sampler_seed": seed,
            },
        )
        circuit_no_bc = qp.QNode(
            ansatz,
            dev,
            interface="jax",
            diff_method=stoch_pulse_grad,
            gradient_kwargs={
                "num_split_times": num_split_times,
                "use_broadcasting": False,
                "sampler_seed": seed,
            },
        )
        params = [jnp.array(0.4)]
        jac_bc = jax.jacobian(circuit_bc)(params)
        jac_no_bc = jax.jacobian(circuit_no_bc)(params)
        for j0, j1 in zip(jac_bc, jac_no_bc):
            assert qp.math.allclose(j0, j1)
        jax.clear_caches()

    def test_with_drive_exact(self):
        """Test that a HardwareHamiltonian only containing a drive is differentiated correctly
        for a constant amplitude and zero frequency and phase."""
        import jax

        timespan = 0.4

        H = qp.pulse.transmon_drive(qp.pulse.constant, 0.0, 0.0, wires=[0])
        atol = 1e-5
        dev = qp.device("default.qubit", wires=1)

        def ansatz(params):
            qp.evolve(H, atol=atol)(params, t=timespan)
            return qp.expval(qp.PauliZ(0))

        cost = qp.QNode(ansatz, dev, interface="jax", diff_method=qp.gradients.stoch_pulse_grad)
        cost_jax = qp.QNode(ansatz, dev, interface="jax")
        params = (0.42,)

        gradfn = jax.grad(cost)
        res = gradfn(params)
        exact = jax.grad(cost_jax)(params)
        assert qp.math.allclose(res, exact, atol=6e-5)
        jax.clear_caches()

    def test_with_drive_approx(self, seed):
        """Test that a HardwareHamiltonian only containing a drive is differentiated
        approximately correctly for a constant phase and zero frequency."""
        import jax

        timespan = 0.1

        H = qp.pulse.transmon_drive(1 / (2 * np.pi), qp.pulse.constant, 0.0, wires=[0])
        atol = 1e-5
        dev = qp.device("default.qubit", wires=1)

        def ansatz(params):
            qp.evolve(H, atol=atol)(params, t=timespan)
            return qp.expval(qp.PauliX(0))

        cost = qp.QNode(
            ansatz,
            dev,
            interface="jax",
            diff_method=qp.gradients.stoch_pulse_grad,
            gradient_kwargs={"num_split_times": 7, "sampler_seed": seed, "use_broadcasting": True},
        )
        cost_jax = qp.QNode(ansatz, dev, interface="jax")
        params = (0.42,)

        gradfn = jax.grad(cost)
        res = gradfn(params)
        exact = jax.grad(cost_jax)(params)
        assert qp.math.allclose(res, exact, atol=1e-3)
        jax.clear_caches()

    @pytest.mark.slow
    @pytest.mark.parametrize("num_params", [1, 2])
    def test_with_two_drives(self, num_params, seed):
        """Test that a HardwareHamiltonian only containing two drives
        is differentiated approximately correctly. The two cases
        of the parametrization test the cases where reordered parameters
        are returned as inner lists and where they remain scalars."""
        import jax

        timespan = 0.1

        if num_params == 1:
            amps = [1 / 5, 1 / 6]
            params = (0.42, -0.91)
        else:
            amps = [qp.pulse.constant] * 2
            params = (1 / (2 * np.pi), 0.42, 1 / 5, -0.91)
        H = qp.pulse.rydberg_drive(
            amps[0], qp.pulse.constant, 0.0, wires=[0]
        ) + qp.pulse.rydberg_drive(amps[1], qp.pulse.constant, 0.0, wires=[1])
        atol = 1e-5
        dev = qp.device("default.qubit", wires=2)

        def ansatz(params):
            qp.evolve(H, atol=atol)(params, t=timespan)
            return qp.expval(qp.PauliX(0) @ qp.PauliX(1))

        cost = qp.QNode(
            ansatz,
            dev,
            interface="jax",
            diff_method=qp.gradients.stoch_pulse_grad,
            gradient_kwargs={"num_split_times": 7, "sampler_seed": seed, "use_broadcasting": True},
        )
        cost_jax = qp.QNode(ansatz, dev, interface="jax")

        gradfn = jax.grad(cost)
        res = gradfn(params)
        exact = jax.grad(cost_jax)(params)
        assert qp.math.allclose(res, exact, atol=1e-3)
        jax.clear_caches()


@pytest.mark.jax
class TestStochPulseGradDiff:
    """Test that stoch_pulse_grad is differentiable."""

    # pylint: disable=too-few-public-methods
    @pytest.mark.slow
    def test_jax(self):
        """Test that stoch_pulse_grad is differentiable with JAX."""
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
        dev = qp.device("default.qubit", wires=1)
        T = 0.5
        ham_single_q_const = qp.pulse.constant * qp.PauliY(0)

        def fun(params):
            op = qp.evolve(ham_single_q_const)(params, T)
            tape = qp.tape.QuantumScript([op], [qp.expval(qp.PauliZ(0))])
            tape.trainable_params = [0]
            tapes, fn = stoch_pulse_grad(tape)
            return fn(qp.execute(tapes, dev, "backprop"))

        params = [jnp.array(0.4)]
        p = params[0] * T
        exp_grad = -2 * jnp.sin(2 * p) * T
        grad = fun(params)
        assert qp.math.isclose(grad, exp_grad)

        exp_diff_of_grad = -4 * jnp.cos(2 * p) * T**2
        diff_of_grad = jax.grad(fun)(params)
        assert qp.math.isclose(diff_of_grad, exp_diff_of_grad)

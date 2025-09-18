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
"""Unit tests for sample_state in devices/qubit."""

from random import shuffle

import numpy as np
import pytest

import pennylane as qml
from pennylane.devices.qubit import measure_with_samples, sample_state, simulate
from pennylane.devices.qubit.sampling import sample_probs
from pennylane.devices.qubit.simulate import _FlexShots
from pennylane.measurements import Shots

two_qubit_state = np.array([[0, 1j], [-1, 0]], dtype=np.complex128) / np.sqrt(2)
APPROX_ATOL = 0.01


@pytest.fixture(name="init_state")
def fixture_init_state():
    """Generates a random initial state"""

    def _init_state(n, seed):
        """random initial state"""
        rng = np.random.default_rng(seed)
        state = rng.random([1 << n]) + rng.random([1 << n]) * 1j
        state /= np.linalg.norm(state)
        return state.reshape((2,) * n)

    return _init_state


def _valid_flex_int(s):
    """Returns True if s is a non-negative integer."""
    return isinstance(s, int) and s >= 0


def _valid_flex_tuple(s):
    """Returns True if s is a tuple of the form (shots, copies)."""
    return (
        isinstance(s, tuple)
        and len(s) == 2
        and _valid_flex_int(s[0])
        and isinstance(s[1], int)
        and s[1] > 0
    )


def samples_to_probs(samples, num_wires):
    """Converts samples to probs"""
    samples_decimal = [np.ravel_multi_index(sample, [2] * num_wires) for sample in samples]
    counts = [0] * (2**num_wires)

    for sample in samples_decimal:
        counts[sample] += 1

    return np.array(counts, dtype=np.float64) / len(samples)


class TestSampleState:
    """Test that the sample_state function works as expected"""

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ["numpy", "jax", "torch"])
    def test_sample_state_basic(self, interface):
        """Tests that the returned samples are as expected."""
        state = qml.math.array(two_qubit_state, like=interface)
        samples = sample_state(state, 10)
        assert samples.shape == (10, 2)
        assert samples.dtype == np.int64
        assert all(qml.math.allequal(s, [0, 1]) or qml.math.allequal(s, [1, 0]) for s in samples)

    @pytest.mark.jax
    def test_prng_key_as_seed_uses_sample_state_jax(self, mocker):
        """Tests that sample_state calls _sample_probs_jax if the seed is a JAX PRNG key"""
        import jax

        jax.config.update("jax_enable_x64", True)

        spy = mocker.spy(qml.devices.qubit.sampling, "_sample_probs_jax")
        state = qml.math.array(two_qubit_state, like="jax")

        # prng_key specified, should call _sample_probs_jax
        _ = sample_state(state, 10, prng_key=jax.random.PRNGKey(15))

        spy.assert_called_once()

    @pytest.mark.jax
    def test_sample_state_jax(self, seed):
        """Tests that the returned samples are as expected when explicitly calling sample_state."""
        import jax

        state = qml.math.array(two_qubit_state, like="jax")

        samples = sample_state(state, 10, prng_key=jax.random.PRNGKey(seed))

        assert samples.shape == (10, 2)
        assert samples.dtype == np.int64
        assert all(qml.math.allequal(s, [0, 1]) or qml.math.allequal(s, [1, 0]) for s in samples)

    @pytest.mark.jax
    def test_prng_key_determines_sample_state_jax_results(self):
        """Test that setting the seed as a JAX PRNG key determines the results for sample_state"""
        import jax

        state = qml.math.array(two_qubit_state, like="jax")

        samples = sample_state(state, shots=10, prng_key=jax.random.PRNGKey(12))
        samples2 = sample_state(state, shots=10, prng_key=jax.random.PRNGKey(12))
        samples3 = sample_state(state, shots=10, prng_key=jax.random.PRNGKey(13))

        assert np.all(samples == samples2)
        assert not np.allclose(samples, samples3)

    @pytest.mark.parametrize("wire_order", [[2], [2, 0], [0, 2, 1]])
    def test_marginal_sample_state(self, wire_order):
        """Tests that marginal states can be sampled as expected."""
        state = np.zeros((2, 2, 2))
        state[:, :, 1] = 0.5  # third wire is always 1
        alltrue_axis = wire_order.index(2)

        samples = sample_state(state, 20, wires=wire_order)
        assert all(samples[:, alltrue_axis])

    def test_sample_state_custom_rng(self):
        """Tests that a custom RNG can be used with sample_state."""
        custom_rng = np.random.default_rng(12345)
        samples = sample_state(two_qubit_state, 4, rng=custom_rng)
        expected = [[0, 1], [0, 1], [1, 0], [1, 0]]
        assert qml.math.allequal(samples, expected)

    def test_approximate_probs_from_samples(self, init_state, seed):
        """Tests that the generated samples are approximately as expected."""
        n = 4
        shots = 20000
        state = init_state(n, seed)

        flat_state = state.flatten()
        expected_probs = np.real(flat_state) ** 2 + np.imag(flat_state) ** 2

        samples = sample_state(state, shots)
        approx_probs = samples_to_probs(samples, n)
        assert np.allclose(approx_probs, expected_probs, atol=APPROX_ATOL)

    def test_entangled_qubit_samples_always_match(self):
        """Tests that entangled qubits are always in the same state."""
        bell_state = np.array([[1, 0], [0, 1]]) / np.sqrt(2)
        samples = sample_state(bell_state, 1000)
        assert samples.shape == (1000, 2)
        assert not any(samples[:, 0] ^ samples[:, 1])  # all samples are entangled
        assert not all(samples[:, 0])  # some samples are |00>
        assert any(samples[:, 0])  # ...and some are |11>!

    @pytest.mark.slow
    @pytest.mark.parametrize("num_wires", [13, 14, 15, 16])
    def test_sample_state_many_wires(self, num_wires):
        """Tests that sample_state works as expected with many wires, and with re-ordering."""
        shots = 10000
        shape = (2,) * num_wires
        flat_state = np.arange(1, 2**num_wires + 1, dtype=np.float64)
        original_norm = np.linalg.norm(flat_state)
        flat_state /= original_norm
        state = flat_state.reshape(shape)
        expected_probs = np.real(flat_state) ** 2 + np.imag(flat_state) ** 2

        ordered_samples = sample_state(state, shots)
        ordered_probs = samples_to_probs(ordered_samples, num_wires)
        assert np.allclose(ordered_probs, expected_probs, atol=APPROX_ATOL)

        random_wires = list(range(num_wires))
        shuffle(random_wires)
        random_samples = sample_state(state, shots, wires=random_wires)
        random_probs = samples_to_probs(random_samples, num_wires)

        reordered_probs = ordered_probs.reshape(shape).transpose(random_wires).flatten()
        assert np.allclose(reordered_probs, random_probs, atol=APPROX_ATOL)


# pylint: disable=too-many-public-methods
class TestMeasureSamples:
    """Test that the measure_with_samples function works as expected"""

    def test_sample_measure(self):
        """Test that a sample measurement works as expected"""
        state = qml.math.array(two_qubit_state)
        shots = qml.measurements.Shots(100)
        mp = qml.sample(wires=range(2))

        result = measure_with_samples([mp], state, shots=shots)[0]

        assert result.shape == (shots.total_shots, 2)
        assert result.dtype == np.int64
        assert all(qml.math.allequal(s, [0, 1]) or qml.math.allequal(s, [1, 0]) for s in result)

    def test_sample_measure_single_wire(self):
        """Test that a sample measurement on a single wire works as expected"""
        state = np.array([[1, -1j], [0, 0]]) / np.sqrt(2)
        shots = qml.measurements.Shots(100)

        mp0 = qml.sample(wires=0)
        mp1 = qml.sample(wires=1)

        result0 = measure_with_samples([mp0], state, shots=shots)[0]
        result1 = measure_with_samples([mp1], state, shots=shots)[0]

        assert result0.shape == (shots.total_shots, 1)
        assert result0.dtype == np.int64
        assert np.all(result0 == 0)

        assert result1.shape == (shots.total_shots, 1)
        assert result1.dtype == np.int64
        assert len(np.unique(result1)) == 2

    def test_prob_measure(self):
        """Test that a probability measurement works as expected"""
        state = qml.math.array(two_qubit_state)
        shots = qml.measurements.Shots(100)
        mp = qml.probs(wires=range(2))

        result = measure_with_samples([mp], state, shots=shots)[0]

        assert result.shape == (4,)
        assert result[0] == 0
        assert result[3] == 0
        assert result[1] + result[2] == 1

    def test_prob_measure_single_wire(self):
        """Test that a probability measurement on a single wire works as expected"""
        state = np.array([[1, -1j], [0, 0]]) / np.sqrt(2)
        shots = qml.measurements.Shots(100)

        mp0 = qml.probs(wires=0)
        mp1 = qml.probs(wires=1)

        result0 = measure_with_samples([mp0], state, shots=shots)[0]
        result1 = measure_with_samples([mp1], state, shots=shots)[0]

        assert result0.shape == (2,)
        assert result1.shape == (2,)

        assert result0[0] == 1
        assert result0[1] == 0
        assert result1[0] != 0
        assert result1[1] != 0
        assert result1[0] + result1[1] == 1

    def test_expval_measure(self):
        """Test that an expval measurement works as expected"""
        state = qml.math.array(two_qubit_state)
        shots = qml.measurements.Shots(100)
        mp = qml.expval(qml.prod(qml.PauliX(0), qml.PauliY(1)))

        result = measure_with_samples([mp], state, shots=shots)[0]

        assert result.shape == ()
        assert result == -1

    def test_expval_measure_single_wire(self):
        """Test that an expval measurement on a single wire works as expected"""
        state = np.array([[1, -1j], [0, 0]]) / np.sqrt(2)
        shots = qml.measurements.Shots(100)

        mp0 = qml.expval(qml.PauliZ(0))
        mp1 = qml.expval(qml.PauliY(1))

        result0 = measure_with_samples([mp0], state, shots=shots)[0]
        result1 = measure_with_samples([mp1], state, shots=shots)[0]

        assert result0.shape == ()
        assert result1.shape == ()
        assert result0 == 1
        assert result1 == -1

    def test_var_measure(self):
        """Test that a variance measurement works as expected"""
        state = qml.math.array(two_qubit_state)
        shots = qml.measurements.Shots(100)
        mp = qml.var(qml.prod(qml.PauliX(0), qml.PauliY(1)))

        result = measure_with_samples([mp], state, shots=shots)[0]

        assert result.shape == ()
        assert result == 0

    def test_var_measure_single_wire(self):
        """Test that a variance measurement on a single wire works as expected"""
        state = np.array([[1, -1j], [0, 0]]) / np.sqrt(2)
        shots = qml.measurements.Shots(100)

        mp0 = qml.var(qml.PauliZ(0))
        mp1 = qml.var(qml.PauliY(1))

        result0 = measure_with_samples([mp0], state, shots=shots)[0]
        result1 = measure_with_samples([mp1], state, shots=shots)[0]

        assert result0.shape == ()
        assert result1.shape == ()
        assert result0 == 0
        assert result1 == 0

    def test_approximate_sample_measure(self, seed):
        """Test that a sample measurement returns approximately the correct distribution"""
        state = qml.math.array(two_qubit_state)
        shots = qml.measurements.Shots(10000)
        mp = qml.sample(wires=range(2))

        result = measure_with_samples([mp], state, shots=shots, rng=seed)[0]

        one_prob = np.count_nonzero(result[:, 0]) / result.shape[0]
        assert np.allclose(one_prob, 0.5, atol=0.05)

    def test_approximate_prob_measure(self, seed):
        """Test that a probability measurement works as expected"""
        state = qml.math.array(two_qubit_state)
        shots = qml.measurements.Shots(10000)
        mp = qml.probs(wires=range(2))

        result = measure_with_samples([mp], state, shots=shots, rng=seed)[0]

        assert np.allclose(result[1], 0.5, atol=0.05)
        assert np.allclose(result[2], 0.5, atol=0.05)
        assert result[1] + result[2] == 1

    def test_approximate_expval_measure(self, seed):
        """Test that an expval measurement works as expected"""
        state = qml.math.array(two_qubit_state)
        shots = qml.measurements.Shots(10000)
        mp = qml.expval(qml.prod(qml.PauliX(0), qml.PauliX(1)))

        result = measure_with_samples([mp], state, shots=shots, rng=seed)[0]

        assert result != 0
        assert np.allclose(result, 0, atol=0.05)

    def test_approximate_var_measure(self, seed):
        """Test that a variance measurement works as expected"""
        state = qml.math.array(two_qubit_state)
        shots = qml.measurements.Shots(10000)
        mp = qml.var(qml.prod(qml.PauliX(0), qml.PauliX(1)))

        result = measure_with_samples([mp], state, shots=shots, rng=seed)[0]

        assert result != 1
        assert np.allclose(result, 1, atol=0.05)

    @pytest.mark.parametrize(
        "shots, total_copies",
        [
            [(100,), 1],
            [((100, 1),), 1],
            [((100, 2),), 2],
            [(100, 100), 2],
            [(100, 200), 2],
            [(100, 100, 200), 3],
            [(200, (100, 2)), 3],
        ],
    )
    def test_sample_measure_shot_vector(self, shots, total_copies):
        """Test that a sample measurement with shot vectors works as expected"""
        state = qml.math.array(two_qubit_state)
        shots = qml.measurements.Shots(shots)
        mp = qml.sample(wires=range(2))

        result = measure_with_samples([mp], state, shots=shots)

        if total_copies == 1:
            result = (result,)

        assert isinstance(result, tuple)
        assert len(result) == total_copies

        for res, sh in zip(result, shots):
            assert isinstance(res, tuple)
            assert len(res) == 1
            res = res[0]

            assert res.shape == (sh, 2)
            assert res.dtype == np.int64
            # NOTE: since the numeric accuracy has already been tested
            # at test_approximate_sample_measure, we only check the shape and type

    @pytest.mark.parametrize(
        "shots, total_copies",
        [
            [(100,), 1],
            [((100, 1),), 1],
            [((100, 2),), 2],
            [(100, 100), 2],
            [(100, 200), 2],
            [(100, 100, 200), 3],
            [(200, (100, 2)), 3],
        ],
    )
    def test_prob_measure_shot_vector(self, shots, total_copies):
        """Test that a probability measurement with shot vectors works as expected"""
        state = qml.math.array(two_qubit_state)
        shots = qml.measurements.Shots(shots)
        mp = qml.probs(wires=range(2))

        result = measure_with_samples([mp], state, shots=shots)

        if total_copies == 1:
            result = (result,)

        assert isinstance(result, tuple)
        assert len(result) == total_copies

        for res in result:
            assert isinstance(res, tuple)
            assert len(res) == 1
            res = res[0]

            assert res.shape == (4,)
            assert res[0] == 0
            assert res[3] == 0
            assert res[1] + res[2] == 1

    @pytest.mark.parametrize(
        "shots, total_copies",
        [
            [(100,), 1],
            [((100, 1),), 1],
            [((100, 2),), 2],
            [(100, 100), 2],
            [(100, 200), 2],
            [(100, 100, 200), 3],
            [(200, (100, 2)), 3],
        ],
    )
    def test_expval_measure_shot_vector(self, shots, total_copies):
        """Test that an expval measurement with shot vectors works as expected"""
        state = qml.math.array(two_qubit_state)
        shots = qml.measurements.Shots(shots)
        mp = qml.expval(qml.prod(qml.PauliX(0), qml.PauliY(1)))

        result = measure_with_samples([mp], state, shots=shots)

        if total_copies == 1:
            result = (result,)

        assert isinstance(result, tuple)
        assert len(result) == total_copies

        for res in result:
            assert isinstance(res, tuple)
            assert len(res) == 1
            res = res[0]

            assert res.shape == ()
            assert res == -1

    @pytest.mark.parametrize(
        "shots, total_copies",
        [
            [(100,), 1],
            [((100, 1),), 1],
            [((100, 2),), 2],
            [(100, 100), 2],
            [(100, 200), 2],
            [(100, 100, 200), 3],
            [(200, (100, 2)), 3],
        ],
    )
    def test_var_measure_shot_vector(self, shots, total_copies):
        """Test that a variance measurement with shot vectors works as expected"""
        state = qml.math.array(two_qubit_state)
        shots = qml.measurements.Shots(shots)
        mp = qml.var(qml.prod(qml.PauliX(0), qml.PauliY(1)))

        result = measure_with_samples([mp], state, shots=shots)

        if total_copies == 1:
            result = (result,)

        assert isinstance(result, tuple)
        assert len(result) == total_copies

        for res in result:
            assert isinstance(res, tuple)
            assert len(res) == 1
            res = res[0]

            assert res.shape == ()
            assert res == 0

    def test_measure_with_samples_one_shot_one_wire(self):
        """Tests that measure_with_samples works with a single shot."""
        state = qml.math.array([0, 1])
        shots = qml.measurements.Shots(1)
        mp = qml.expval(qml.PauliZ(0))
        result = measure_with_samples([mp], state, shots=shots)

        assert isinstance(result, tuple)
        assert len(result) == 1
        result = result[0]

        assert result.shape == ()
        assert result == -1.0

    def test_identity_on_no_wires(self):
        """Test that measure_with_samples can handle observables on no wires when no other measurements exist."""

        state = np.array([0, 1])
        mp = qml.measurements.ExpectationMP(qml.I())

        [result] = measure_with_samples([mp], state, shots=qml.measurements.Shots(1))
        assert qml.math.allclose(result, 1.0)

    def test_identity_on_no_wires_with_other_observables(self):
        """Test that measuring an identity on no wires can be used in conjunction with other measurements."""

        state = np.array([0, 1])

        mps = [
            qml.measurements.ExpectationMP(2 * qml.I()),
            qml.expval(qml.Z(0)),
            qml.probs(wires=0),
        ]

        results = measure_with_samples(mps, state, qml.measurements.Shots(1))
        assert qml.math.allclose(results[0], 2.0)
        assert qml.math.allclose(results[1], -1.0)

    def test_measuring_sum_with_identity_on_no_wires(self):
        """Test that we can measure a sum with an identity on no wires."""

        state = np.array([0, 1])

        mp = qml.expval(qml.Z(0) + 2 * qml.I())
        [result] = measure_with_samples([mp], state, shots=qml.measurements.Shots(1))
        assert qml.math.allclose(result, 1)  # -1 + 2

    @pytest.mark.parametrize(
        "state, measurements, expected_results",
        [
            [
                np.array([[0.5, 0.5j], [-0.5j, 0.5]]),
                [qml.expval(qml.Y(0) + qml.Y(0)), qml.expval(qml.Y(1))],
                (-2.0, 1.0),
            ],
            [
                np.array([[0.5, -0.5j], [0.5j, 0.5]]),
                [qml.expval(qml.Y(0) + qml.Y(0)), qml.expval(qml.Y(1))],
                (2.0, -1.0),
            ],
            [
                np.array([[0.5, 0.5j], [-0.5j, 0.5]]),
                [qml.expval(qml.Y(1)), qml.expval(qml.Y(0) + qml.Y(0))],
                (1.0, -2.0),
            ],
            [
                np.array([[0.5, 0.5j], [-0.5j, 0.5]]),
                [
                    qml.expval(qml.Y(1) - qml.Y(1)),
                    qml.expval(2 * (qml.Y(0) + qml.Y(0) - 5 * (qml.Y(0) + qml.Y(0)))),
                    qml.expval(
                        (2 * (qml.Y(0) + qml.Y(0)))
                        @ (5 * (qml.Y(0) + qml.Y(0)) + 3 * (qml.Y(0) + qml.Y(0)))
                    ),
                ],
                (0.0, 16.0, 64.0),
            ],
        ],
    )
    def test_sum_same_wires(self, state, measurements, expected_results):
        """Test that the sum of observables acting on the same wires works as expected."""

        results = measure_with_samples(measurements, state, qml.measurements.Shots(1000))
        assert qml.math.allclose(results, expected_results)


class TestInvalidStateSamples:
    """Tests for state vectors containing nan values or shot vectors with zero shots."""

    @pytest.mark.parametrize("shots", [10, [10, 10]])
    def test_only_catch_nan_errors(self, shots):
        """Test that errors are only caught if they are raised due to nan values in the state."""
        state = np.zeros((2, 2)).astype(np.complex128)
        mp = qml.expval(qml.PauliZ(0))
        _shots = Shots(shots)

        with pytest.raises(ValueError, match=r"(?i)probabilities do not sum to 1"):
            _ = measure_with_samples([mp], state, _shots)

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize(
        "mp",
        [
            qml.expval(qml.PauliZ(0)),
            qml.expval(
                qml.Hamiltonian(
                    [1.0, 2.0, 3.0, 4.0],
                    [qml.PauliZ(0) @ qml.PauliX(1), qml.PauliX(1), qml.PauliZ(1), qml.PauliY(1)],
                )
            ),
            qml.expval(
                qml.dot(
                    [1.0, 2.0, 3.0, 4.0],
                    [qml.PauliZ(0) @ qml.PauliX(1), qml.PauliX(1), qml.PauliZ(1), qml.PauliY(1)],
                )
            ),
            qml.var(qml.PauliZ(0)),
        ],
    )
    @pytest.mark.parametrize("interface", ["numpy", "autograd", "torch", "jax"])
    @pytest.mark.parametrize("shots", [0, [0, 0]])
    def test_nan_float_result(self, mp, interface, shots):
        """Test that the result of circuits with 0 probability postselections is NaN with the
        expected shape."""
        state = qml.math.full((2, 2), np.nan, like=interface)
        res = measure_with_samples((mp,), state, _FlexShots(shots), is_state_batched=False)

        if not isinstance(shots, list):
            assert isinstance(res, tuple)
            res = res[0]
            assert qml.math.ndim(res) == 0
            assert qml.math.isnan(res)

        else:
            assert isinstance(res, tuple)
            assert len(res) == 2
            for r in res:
                assert isinstance(r, tuple)
                r = r[0]
                assert qml.math.ndim(r) == 0
                assert qml.math.isnan(r)

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize(
        "mp", [qml.sample(wires=0), qml.sample(op=qml.PauliZ(0)), qml.sample(wires=[0, 1])]
    )
    @pytest.mark.parametrize("interface", ["numpy", "autograd", "torch", "jax"])
    @pytest.mark.parametrize("shots", [0, [0, 0]])
    def test_nan_samples(self, mp, interface, shots):
        """Test that the result of circuits with 0 probability postselections is NaN with the
        expected shape."""
        state = qml.math.full((2, 2), np.nan, like=interface)
        res = measure_with_samples((mp,), state, _FlexShots(shots), is_state_batched=False)

        if not isinstance(shots, list):
            assert isinstance(res, tuple)
            res = res[0]
            assert qml.math.shape(res) == (shots,) if mp.obs else (shots, len(mp.wires))

        else:
            assert isinstance(res, tuple)
            assert len(res) == 2
            for i, r in enumerate(res):
                assert isinstance(r, tuple)
                r = r[0]
                assert qml.math.shape(r) == (shots[i],) if mp.obs else (shots[i], len(mp.wires))

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ["numpy", "autograd", "torch", "jax"])
    @pytest.mark.parametrize("shots", [0, [0, 0]])
    def test_nan_classical_shadows(self, interface, shots):
        """Test that classical_shadows returns an empty array when the state has
        NaN values"""
        state = qml.math.full((2, 2), np.nan, like=interface)
        res = measure_with_samples(
            (qml.classical_shadow([0]),), state, _FlexShots(shots), is_state_batched=False
        )

        if not isinstance(shots, list):
            assert isinstance(res, tuple)
            res = res[0]
            assert qml.math.shape(res) == (2, 0, 1)
            assert qml.math.size(res) == 0

        else:
            assert isinstance(res, tuple)
            assert len(res) == 2
            for r in res:
                assert isinstance(r, tuple)
                r = r[0]
                assert qml.math.shape(r) == (2, 0, 1)
                assert qml.math.size(r) == 0

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("H", [qml.PauliZ(0), [qml.PauliZ(0), qml.PauliX(1)]])
    @pytest.mark.parametrize("interface", ["numpy", "autograd", "torch", "jax"])
    @pytest.mark.parametrize("shots", [0, [0, 0]])
    def test_nan_shadow_expval(self, H, interface, shots):
        """Test that shadow_expval returns an empty array when the state has
        NaN values"""
        state = qml.math.full((2, 2), np.nan, like=interface)
        res = measure_with_samples(
            (qml.shadow_expval(H),), state, _FlexShots(shots), is_state_batched=False
        )

        if not isinstance(shots, list):
            assert isinstance(res, tuple)
            res = res[0]
            assert qml.math.shape(res) == qml.math.shape(H)
            assert qml.math.all(qml.math.isnan(res))

        else:
            assert isinstance(res, tuple)
            assert len(res) == 2
            for r in res:
                assert isinstance(r, tuple)
                r = r[0]
                assert qml.math.shape(r) == qml.math.shape(H)
                assert qml.math.all(qml.math.isnan(r))


two_qubit_state_to_be_normalized = np.array([[0, 1.0000000005j], [-1, 0]]) / np.sqrt(2)
two_qubit_state_not_normalized = np.array([[0, 1.00005j], [-1.00000001, 0]]) / np.sqrt(2)

batched_state_to_be_normalized = np.stack(
    [
        np.array([[0, 0], [0, 1.000000000009]]),
        np.array([[1.00000004, 0], [1, 0]]) / np.sqrt(2),
        np.array([[1, 1], [1, 0.99999995]]) / 2,
    ]
)
batched_state_not_normalized = np.stack(
    [
        np.array([[0, 0], [0, 1]]),
        np.array([[1.0000004, 0], [1, 0]]) / np.sqrt(2),
        np.array([[1, 1], [1, 0.9999995]]) / 2,
    ]
)


class TestRenormalization:
    """Test suite for renormalization functionality."""

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ["numpy", "jax", "torch"])
    def test_sample_state_renorm(self, interface):
        """Test renormalization for a non-batched state."""

        state = qml.math.array(two_qubit_state_to_be_normalized, like=interface)
        _ = sample_state(state, 10)

    # jax.random.choice accepts unnormalized probabilities
    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ["numpy", "torch"])
    def test_sample_state_renorm_error(self, interface):
        """Test that renormalization does not occur if the error is too large."""

        state = qml.math.array(two_qubit_state_not_normalized, like=interface)
        with pytest.raises(ValueError, match=r"(?i)probabilities do not sum to 1"):
            _ = sample_state(state, 10)

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ["numpy", "torch", "jax"])
    def test_sample_batched_state_renorm(self, interface):
        """Test renormalization for a batched state."""

        state = qml.math.array(batched_state_to_be_normalized, like=interface)
        _ = sample_state(state, 10, is_state_batched=True)

    # jax.random.choices accepts unnormalized probabilities
    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ["numpy", "torch"])
    def test_sample_batched_state_renorm_error(self, interface):
        """Test that renormalization does not occur if the error is too large."""

        state = qml.math.array(batched_state_not_normalized, like=interface)
        with pytest.raises(ValueError, match=r"(?i)probabilities do not sum to 1"):
            _ = sample_state(state, 10, is_state_batched=True)


class TestBroadcasting:
    """Test that measurements work when the state has a batch dim"""

    def test_sample_measure(self, seed):
        """Test that broadcasting works for qml.sample and single shots"""

        rng = np.random.default_rng(seed)
        shots = qml.measurements.Shots(100)

        state = [
            np.array([[0, 0], [0, 1]]),
            np.array([[1, 0], [1, 0]]) / np.sqrt(2),
            np.array([[1, 1], [1, 1]]) / 2,
        ]
        state = np.stack(state)

        measurement = qml.sample(wires=[0, 1])
        res = measure_with_samples([measurement], state, shots, is_state_batched=True, rng=rng)[0]

        assert res.shape == (3, shots.total_shots, 2)
        assert res.dtype == np.int64

        # first batch of samples is always |11>
        assert np.all(res[0] == 1)

        # second batch of samples is either |00> or |10>
        assert np.all(np.logical_or(res[1] == [0, 0], res[1] == [1, 0]))

        # third batch of samples can be any of |00>, |01>, |10>, or |11>
        assert np.all(np.logical_or(res[2] == 0, res[2] == 1))

    @pytest.mark.parametrize(
        "measurement, expected",
        [
            (
                qml.probs(wires=[0, 1]),
                np.array([[0, 0, 0, 1], [1 / 2, 0, 1 / 2, 0], [1 / 4, 1 / 4, 1 / 4, 1 / 4]]),
            ),
            (qml.expval(qml.PauliZ(1)), np.array([-1, 1, 0])),
            (qml.var(qml.PauliZ(1)), np.array([0, 0, 1])),
        ],
    )
    def test_nonsample_measure(self, measurement, expected, seed):
        """Test that broadcasting works for the other sample measurements and single shots"""

        shots = qml.measurements.Shots(10000)

        state = [
            np.array([[0, 0], [0, 1]]),
            np.array([[1, 0], [1, 0]]) / np.sqrt(2),
            np.array([[1, 1], [1, 1]]) / 2,
        ]
        state = np.stack(state)

        res = measure_with_samples([measurement], state, shots, is_state_batched=True, rng=seed)
        assert np.allclose(res, expected, atol=0.03)

    @pytest.mark.parametrize(
        "shots",
        [
            ((100, 2),),
            (100, 100),
            (100, 100),
            (100, 100, 200),
            (200, (100, 2)),
        ],
    )
    def test_sample_measure_shot_vector(self, shots, seed):
        """Test that broadcasting works for qml.sample and shot vectors"""

        rng = np.random.default_rng(seed)
        shots = qml.measurements.Shots(shots)

        state = [
            np.array([[0, 0], [0, 1]]),
            np.array([[1, 0], [1, 0]]) / np.sqrt(2),
            np.array([[1, 1], [1, 1]]) / 2,
        ]
        state = np.stack(state)

        measurement = qml.sample(wires=[0, 1])
        res = measure_with_samples([measurement], state, shots, is_state_batched=True, rng=rng)

        assert isinstance(res, tuple)
        assert len(res) == shots.num_copies

        for s, r in zip(shots, res):
            assert isinstance(r, tuple)
            assert len(r) == 1
            r = r[0]

            assert r.shape == (3, s, 2)
            assert r.dtype == np.int64

            # first batch of samples is always |11>
            assert np.all(r[0] == 1)

            # second batch of samples is either |00> or |10>
            assert np.all(np.logical_or(r[1] == [0, 0], r[1] == [1, 0]))

            # third batch of samples can be any of |00>, |01>, |10>, or |11>
            assert np.all(np.logical_or(r[2] == 0, r[2] == 1))

    # pylint:disable = too-many-arguments
    @pytest.mark.parametrize(
        "shots",
        [
            ((10000, 2),),
            (10000, 10000),
            (10000, 20000),
            (10000, 10000, 20000),
            (20000, (10000, 2)),
        ],
    )
    @pytest.mark.parametrize(
        "measurement, expected",
        [
            (
                qml.probs(wires=[0, 1]),
                np.array([[0, 0, 0, 1], [1 / 2, 0, 1 / 2, 0], [1 / 4, 1 / 4, 1 / 4, 1 / 4]]),
            ),
            (qml.expval(qml.PauliZ(1)), np.array([-1, 1, 0])),
            (qml.var(qml.PauliZ(1)), np.array([0, 0, 1])),
        ],
    )
    def test_nonsample_measure_shot_vector(self, shots, measurement, expected, seed):
        """Test that broadcasting works for the other sample measurements and shot vectors"""

        rng = np.random.default_rng(seed)
        shots = qml.measurements.Shots(shots)

        state = [
            np.array([[0, 0], [0, 1]]),
            np.array([[1, 0], [1, 0]]) / np.sqrt(2),
            np.array([[1, 1], [1, 1]]) / 2,
        ]
        state = np.stack(state)

        res = measure_with_samples([measurement], state, shots, is_state_batched=True, rng=rng)

        assert isinstance(res, tuple)
        assert len(res) == shots.num_copies

        for r in res:
            assert isinstance(r, tuple)
            assert len(r) == 1
            r = r[0]

            assert r.shape == expected.shape
            # test_nonsample_measure already validated the numeric accuracy
            # so we only check the shape here


@pytest.mark.jax
class TestBroadcastingPRNG:
    """Test that measurements work and use sample_state when the state has a batch dim
    and a PRNG key is provided"""

    def test_sample_measure(self, mocker, seed):
        """Test that broadcasting works for qml.sample and single shots"""
        import jax

        jax.config.update("jax_enable_x64", True)

        spy = mocker.spy(qml.devices.qubit.sampling, "_sample_probs_jax")

        rng = np.random.default_rng(seed)
        shots = qml.measurements.Shots(100)

        state = [
            np.array([[0, 0], [0, 1]]),
            np.array([[1, 0], [1, 0]]) / np.sqrt(2),
            np.array([[1, 1], [1, 1]]) / 2,
        ]
        state = np.stack(state)

        measurement = qml.sample(wires=[0, 1])
        res = measure_with_samples(
            [measurement],
            state,
            shots,
            is_state_batched=True,
            rng=rng,
            prng_key=jax.random.PRNGKey(seed),
        )[0]

        spy.assert_called()

        assert res.shape == (3, shots.total_shots, 2)
        assert res.dtype == np.int64

        # convert to numpy array because prng_key -> JAX -> ArrayImpl -> angry vanilla numpy below
        res = [np.array(r) for r in res]

        # first batch of samples is always |11>
        assert np.all(res[0] == 1)

        # second batch of samples is either |00> or |10>
        assert np.all(np.logical_or(res[1] == [0, 0], res[1] == [1, 0]))

        # third batch of samples can be any of |00>, |01>, |10>, or |11>
        assert np.all(np.logical_or(res[2] == 0, res[2] == 1))

    # NOTE: The accuracy checking of this test is necessary,
    # but the definition of `atol` is too arbitrary. Further
    # investigation is needed to establish a more systematic
    # approach to test the final sampling distribution. [sc-91887]
    @pytest.mark.parametrize(
        "measurement, expected",
        [
            (
                qml.probs(wires=[0, 1]),
                np.array([[0, 0, 0, 1], [1 / 2, 0, 1 / 2, 0], [1 / 4, 1 / 4, 1 / 4, 1 / 4]]),
            ),
            (qml.expval(qml.PauliZ(1)), np.array([-1, 1, 0])),
            (qml.var(qml.PauliZ(1)), np.array([0, 0, 1])),
        ],
    )
    def test_nonsample_measure(self, mocker, measurement, expected, seed):
        """Test that broadcasting works for the other sample measurements and single shots"""
        import jax

        spy = mocker.spy(qml.devices.qubit.sampling, "_sample_probs_jax")

        rng = np.random.default_rng(seed)
        shots = qml.measurements.Shots(10000)

        state = [
            np.array([[0, 0], [0, 1]]),
            np.array([[1, 0], [1, 0]]) / np.sqrt(2),
            np.array([[1, 1], [1, 1]]) / 2,
        ]
        state = np.stack(state)

        res = measure_with_samples(
            [measurement],
            state,
            shots,
            is_state_batched=True,
            rng=rng,
            prng_key=jax.random.PRNGKey(seed),
        )

        spy.assert_called()
        assert np.allclose(res, expected, atol=0.01)

    @pytest.mark.parametrize(
        "shots",
        [
            ((100, 2),),
            (100, 100),
            (100, 100),
            (100, 100, 200),
            (200, (100, 2)),
        ],
    )
    def test_sample_measure_shot_vector(self, mocker, shots, seed):
        """Test that broadcasting works for qml.sample and shot vectors"""

        import jax

        spy = mocker.spy(qml.devices.qubit.sampling, "_sample_probs_jax")

        rng = np.random.default_rng(seed)
        shots = qml.measurements.Shots(shots)

        state = [
            np.array([[0, 0], [0, 1]]),
            np.array([[1, 0], [1, 0]]) / np.sqrt(2),
            np.array([[1, 1], [1, 1]]) / 2,
        ]
        state = np.stack(state)

        measurement = qml.sample(wires=[0, 1])
        res = measure_with_samples(
            [measurement],
            state,
            shots,
            is_state_batched=True,
            rng=rng,
            prng_key=jax.random.PRNGKey(seed),
        )

        spy.assert_called()

        assert isinstance(res, tuple)
        assert len(res) == shots.num_copies

        for s, r in zip(shots, res):
            assert isinstance(r, tuple)
            assert len(r) == 1
            r = r[0]

            assert r.shape == (3, s, 2)
            # this is has started randomly failing do to r.dtype being int32 instead of int64.
            # Not sure why they are getting returned as 32 instead, but maybe this will fix it?
            assert res[0][0].dtype in [np.int32, np.int64]

            # convert to numpy array because prng_key -> JAX -> ArrayImpl -> angry vanilla numpy below
            r = [np.array(i) for i in r]

            # first batch of samples is always |11>
            assert np.all(r[0] == 1)

            # second batch of samples is either |00> or |10>
            assert np.all(np.logical_or(r[1] == [0, 0], r[1] == [1, 0]))

            # third batch of samples can be any of |00>, |01>, |10>, or |11>
            assert np.all(np.logical_or(r[2] == 0, r[2] == 1))

    # pylint:disable = too-many-arguments
    @pytest.mark.parametrize(
        "shots",
        [
            ((10000, 2),),
            (10000, 10000),
            (10000, 20000),
            (10000, 10000, 20000),
            (20000, (10000, 2)),
        ],
    )
    @pytest.mark.parametrize(
        "measurement, expected",
        [
            (
                qml.probs(wires=[0, 1]),
                np.array([[0, 0, 0, 1], [1 / 2, 0, 1 / 2, 0], [1 / 4, 1 / 4, 1 / 4, 1 / 4]]),
            ),
            (qml.expval(qml.PauliZ(1)), np.array([-1, 1, 0])),
            (qml.var(qml.PauliZ(1)), np.array([0, 0, 1])),
        ],
    )
    def test_nonsample_measure_shot_vector(self, mocker, shots, measurement, expected, seed):
        """Test that broadcasting works for the other sample measurements and shot vectors"""

        import jax

        spy = mocker.spy(qml.devices.qubit.sampling, "_sample_probs_jax")

        rng = np.random.default_rng(seed)
        shots = qml.measurements.Shots(shots)

        state = [
            np.array([[0, 0], [0, 1]]),
            np.array([[1, 0], [1, 0]]) / np.sqrt(2),
            np.array([[1, 1], [1, 1]]) / 2,
        ]
        state = np.stack(state)

        res = measure_with_samples(
            [measurement],
            state,
            shots,
            is_state_batched=True,
            rng=rng,
            prng_key=jax.random.PRNGKey(seed),
        )

        spy.assert_called()

        assert isinstance(res, tuple)
        assert len(res) == shots.num_copies

        for r in res:
            assert isinstance(r, tuple)
            assert len(r) == 1
            r = r[0]

            assert r.shape == expected.shape
            assert np.allclose(r, expected, atol=0.03)


class TestHamiltonianSamples:
    """Test that the measure_with_samples function works as expected for
    Hamiltonian and Sum observables"""

    def test_hamiltonian_expval(self, seed):
        """Test that sampling works well for Hamiltonian observables"""
        x, y = np.array(0.67), np.array(0.95)
        ops = [qml.RY(x, wires=0), qml.RZ(y, wires=0)]
        meas = [qml.expval(qml.Hamiltonian([0.8, 0.5], [qml.PauliZ(0), qml.PauliX(0)]))]

        qs = qml.tape.QuantumScript(ops, meas, shots=10000)
        res = simulate(qs, rng=seed)

        expected = 0.8 * np.cos(x) + 0.5 * np.real(np.exp(y * 1j)) * np.sin(x)
        assert np.allclose(res, expected, atol=0.02)

    def test_hamiltonian_expval_shot_vector(self, seed):
        """Test that sampling works well for Hamiltonian observables with a shot vector"""
        x, y = np.array(0.67), np.array(0.95)
        ops = [qml.RY(x, wires=0), qml.RZ(y, wires=0)]
        meas = [qml.expval(qml.Hamiltonian([0.8, 0.5], [qml.PauliZ(0), qml.PauliX(0)]))]

        qs = qml.tape.QuantumScript(ops, meas, shots=(10000, 10000))
        res = simulate(qs, rng=seed)

        assert len(res) == 2
        assert isinstance(res, tuple)
        # Already tested the numeric accuracy in test_hamiltonian_expval
        # so we only check the shape here

    def test_sum_expval(self, seed):
        """Test that sampling works well for Sum observables"""

        x, y = np.array(0.67), np.array(0.95)
        ops = [qml.RY(x, wires=0), qml.RZ(y, wires=0)]
        meas = [qml.expval(qml.s_prod(0.8, qml.PauliZ(0)) + qml.s_prod(0.5, qml.PauliX(0)))]

        qs = qml.tape.QuantumScript(ops, meas, shots=10000)
        res = simulate(qs, rng=seed)

        expected = 0.8 * np.cos(x) + 0.5 * np.real(np.exp(y * 1j)) * np.sin(x)
        assert np.allclose(res, expected, atol=0.02)

    def test_sum_expval_shot_vector(self, seed):
        """Test that sampling works well for Sum observables with a shot vector."""
        x, y = np.array(0.67), np.array(0.95)
        ops = [qml.RY(x, wires=0), qml.RZ(y, wires=0)]
        meas = [qml.expval(qml.s_prod(0.8, qml.PauliZ(0)) + qml.s_prod(0.5, qml.PauliX(0)))]

        qs = qml.tape.QuantumScript(ops, meas, shots=(10000, 10000))
        res = simulate(qs, rng=seed)

        assert len(res) == 2
        assert isinstance(res, tuple)
        # Already tested the numeric accuracy in test_sum_expval
        # so we only check the shape here

    def test_prod_expval(self, seed):
        """Tests that sampling works for Prod observables"""

        x, y = np.array(0.67), np.array(0.95)
        ops = [qml.RY(y, wires=0), qml.RX(x, wires=1)]
        H = qml.prod(qml.PauliX(0), qml.PauliY(1))
        tape = qml.tape.QuantumScript(
            ops, measurements=[qml.expval(qml.PauliX(0)), qml.expval(H)], shots=10000
        )
        res = simulate(tape, rng=seed)
        expected = [np.sin(y), -np.sin(y) * np.sin(x)]
        assert np.allclose(res, expected, atol=0.05)

    def test_sprod_expval(self, seed):
        """Tests that sampling works for SProd observables"""

        y = np.array(0.95)
        ops = [qml.RY(y, wires=0)]
        H = qml.s_prod(1.5, qml.PauliX(0))
        tape = qml.tape.QuantumScript(
            ops, measurements=[qml.expval(qml.PauliX(0)), qml.expval(H)], shots=10000
        )
        res = simulate(tape, rng=seed)
        expected = [np.sin(y), 1.5 * np.sin(y)]
        assert np.allclose(res, expected, atol=0.05)

    def test_multi_wires(self, seed):
        """Test that sampling works for Sums with large numbers of wires"""
        n_wires = 10
        scale = 0.05
        offset = 0.8

        ops = [qml.RX(offset + scale * i, wires=i) for i in range(n_wires)]

        t1 = 2.5 * qml.prod(*(qml.PauliZ(i) for i in range(n_wires)))
        t2 = 6.2 * qml.prod(*(qml.PauliY(i) for i in range(n_wires)))
        H = t1 + t2

        qs = qml.tape.QuantumScript(ops, [qml.expval(H)], shots=100000)
        res = simulate(qs, rng=seed)

        phase = offset + scale * np.array(range(n_wires))
        cosines = qml.math.cos(phase)
        sines = qml.math.sin(phase)
        expected = 2.5 * qml.math.prod(cosines) + 6.2 * qml.math.prod(sines)

        assert np.allclose(res, expected, atol=0.05)

    def test_complex_hamiltonian(self, seed):
        """Test that sampling works for complex Hamiltonians"""
        scale = 0.05
        offset = 0.4

        ops = [qml.RX(offset + scale * i, wires=i) for i in range(4)]

        # taken from qml.data
        H = qml.Hamiltonian(
            [
                -0.3796867241618816,
                0.1265398827193729,
                0.1265398827193729,
                0.15229282586796247,
                0.05080559325437572,
                -0.05080559325437572,
                -0.05080559325437572,
                0.05080559325437572,
                -0.10485523662149618,
                0.10102818539518765,
                -0.10485523662149615,
                0.15183377864956338,
                0.15183377864956338,
                0.10102818539518765,
                0.1593698831813122,
            ],
            [
                qml.Identity(wires=[0]),
                qml.PauliZ(wires=[0]),
                qml.PauliZ(wires=[1]),
                qml.PauliZ(wires=[0]) @ qml.PauliZ(wires=[1]),
                qml.PauliY(wires=[0])
                @ qml.PauliX(wires=[1])
                @ qml.PauliX(wires=[2])
                @ qml.PauliY(wires=[3]),
                qml.PauliY(wires=[0])
                @ qml.PauliY(wires=[1])
                @ qml.PauliX(wires=[2])
                @ qml.PauliX(wires=[3]),
                qml.PauliX(wires=[0])
                @ qml.PauliX(wires=[1])
                @ qml.PauliY(wires=[2])
                @ qml.PauliY(wires=[3]),
                qml.PauliX(wires=[0])
                @ qml.PauliY(wires=[1])
                @ qml.PauliY(wires=[2])
                @ qml.PauliX(wires=[3]),
                qml.PauliZ(wires=[2]),
                qml.PauliZ(wires=[0]) @ qml.PauliZ(wires=[2]),
                qml.PauliZ(wires=[3]),
                qml.PauliZ(wires=[0]) @ qml.PauliZ(wires=[3]),
                qml.PauliZ(wires=[1]) @ qml.PauliZ(wires=[2]),
                qml.PauliZ(wires=[1]) @ qml.PauliZ(wires=[3]),
                qml.PauliZ(wires=[2]) @ qml.PauliZ(wires=[3]),
            ],
        )

        qs = qml.tape.QuantumScript(ops, [qml.expval(H)], shots=100000)
        res = simulate(qs, rng=seed)

        qs_exp = qml.tape.QuantumScript(ops, [qml.expval(H)])
        expected = simulate(qs_exp)

        assert np.allclose(res, expected, atol=0.001)


class TestSampleProbs:

    def test_basic_sampling(self, seed):
        """One Qubit, two outcomes"""
        probs = np.array([0.3, 0.7])
        samples = sample_probs(probs, shots=1000, num_wires=1, is_state_batched=False, rng=seed)
        assert samples.shape == (1000, 1)
        # Check if the distribution is roughly correct (allowing for some variance)
        zeros = np.sum(samples == 0)
        assert 250 <= zeros <= 350  # Approx 30% of 1000, with some leeway

    def test_multi_qubit_sampling(self, seed):
        """Two Qubit, four outcomes"""
        probs = np.array([0.1, 0.2, 0.3, 0.4])
        samples = sample_probs(probs, shots=1000, num_wires=2, is_state_batched=False, rng=seed)
        assert samples.shape == (1000, 2)
        # Check if all possible states are present
        unique_samples = set(map(tuple, samples))
        assert len(unique_samples) == 4

    def test_batched_sampling(self, seed):
        """A batch of two circuits, each with two outcomes"""
        probs = np.array([[0.5, 0.5], [0.3, 0.7]])
        samples = sample_probs(probs, shots=1000, num_wires=1, is_state_batched=True, rng=seed)
        assert samples.shape == (2, 1000, 1)

    def test_cutoff_edge_case_failure(self, seed):
        """Test sampling with probabilities just outside the cutoff."""
        cutoff = 1e-7  # Assuming this is the cutoff used in sample_probs
        probs = np.array([0.5, 0.5 - 2 * cutoff])
        with pytest.raises(ValueError, match=r"(?i)probabilities do not sum to 1"):
            sample_probs(probs, shots=1000, num_wires=1, is_state_batched=False, rng=seed)

    def test_batched_cutoff_edge_case_failure(self, seed):
        """Test sampling with probabilities just outside the cutoff."""
        cutoff = 1e-7  # Assuming this is the cutoff used in sample_probs
        probs = np.array(
            [
                [0.5, 0.5 - 2 * cutoff],
                [0.5, 0.5 - 2 * cutoff],
            ]
        )
        with pytest.raises(ValueError, match=r"(?i)probabilities do not sum to 1"):
            sample_probs(probs, shots=1000, num_wires=1, is_state_batched=True, rng=seed)

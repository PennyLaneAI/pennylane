# Copyright 2024 Xanadu Quantum Technologies Inc.

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
Tests for the transform implementing the deferred measurement principle.
"""
# pylint: disable=too-few-public-methods, too-many-arguments

import numpy as np
import pytest

import pennylane as qml
from pennylane.measurements import (
    CountsMP,
    ExpectationMP,
    MeasurementValue,
    MidMeasureMP,
    ProbabilityMP,
    SampleMP,
)
from pennylane.transforms.dynamic_one_shot import parse_native_mid_circuit_measurements


@pytest.mark.parametrize(
    "measurement",
    [
        qml.state(),
        qml.density_matrix(0),
        qml.vn_entropy(0),
        qml.mutual_info(0, 1),
        qml.purity(0),
        qml.classical_shadow(0),
    ],
)
def test_parse_native_mid_circuit_measurements_unsupported_meas(measurement):
    circuit = qml.tape.QuantumScript([qml.RX(1.0, 0)], [measurement])
    with pytest.raises(TypeError, match="Native mid-circuit measurement mode does not support"):
        parse_native_mid_circuit_measurements(circuit, [circuit], [np.empty((0,))])


def test_postselection_error_with_wrong_device():
    """Test that an error is raised when a device does not support native execution."""
    dev = qml.device("default.mixed", wires=2)

    with pytest.raises(TypeError, match="does not support mid-circuit measurements natively"):

        @qml.dynamic_one_shot
        @qml.qnode(dev)
        def _():
            qml.measure(0, postselect=1)
            return qml.probs(wires=[0])


@pytest.mark.parametrize("postselect_mode", ["hw-like", "fill-shots"])
def test_postselect_mode(postselect_mode, mocker):
    """Test that invalid shots are discarded if requested"""
    shots = 100
    dev = qml.device("default.qubit", shots=shots)
    spy = mocker.spy(qml, "dynamic_one_shot")

    @qml.qnode(dev, postselect_mode=postselect_mode)
    def f(x):
        qml.RX(x, 0)
        _ = qml.measure(0, postselect=1)
        return qml.sample(wires=[0, 1])

    res = f(np.pi / 2)
    spy.assert_called_once()

    if postselect_mode == "hw-like":
        assert len(res) < shots
    else:
        assert len(res) == shots
    assert np.all(res != np.iinfo(np.int32).min)


@pytest.mark.jax
@pytest.mark.parametrize("use_jit", [True, False])
@pytest.mark.parametrize("diff_method", [None, "best"])
def test_hw_like_with_jax(use_jit, diff_method):
    """Test that invalid shots are replaced with INTEGER_MIN_VAL if
    postselect_mode="hw-like" with JAX"""
    import jax  # pylint: disable=import-outside-toplevel

    shots = 10
    dev = qml.device("default.qubit", shots=shots, seed=jax.random.PRNGKey(123))

    @qml.qnode(dev, postselect_mode="hw-like", diff_method=diff_method)
    def f(x):
        qml.RX(x, 0)
        _ = qml.measure(0, postselect=1)
        return qml.sample(wires=[0, 1])

    if use_jit:
        f = jax.jit(f)

    res = f(jax.numpy.array(np.pi / 2))

    assert len(res) == shots
    assert np.any(res == np.iinfo(np.int32).min)


def test_unsupported_measurements():
    """Test that using unsupported measurements raises an error."""
    tape = qml.tape.QuantumScript([MidMeasureMP(0)], [qml.state()])

    with pytest.raises(
        TypeError,
        match="Native mid-circuit measurement mode does not support StateMP measurements.",
    ):
        _, _ = qml.dynamic_one_shot(tape)


def test_unsupported_shots():
    """Test that using shots=None raises an error."""
    tape = qml.tape.QuantumScript([MidMeasureMP(0)], [qml.probs(wires=0)], shots=None)

    with pytest.raises(
        qml.QuantumFunctionError,
        match="dynamic_one_shot is only supported with finite shots.",
    ):
        _, _ = qml.dynamic_one_shot(tape)


@pytest.mark.parametrize("n_shots", range(1, 10))
def test_len_tapes(n_shots):
    """Test that the transform produces the correct number of tapes."""
    tape = qml.tape.QuantumScript([MidMeasureMP(0)], [qml.expval(qml.PauliZ(0))], shots=n_shots)
    tapes, _ = qml.dynamic_one_shot(tape)
    assert len(tapes) == 1


@pytest.mark.parametrize("n_batch", range(1, 4))
@pytest.mark.parametrize("n_shots", range(1, 4))
def test_len_tape_batched(n_batch, n_shots):
    """Test that the transform produces the correct number of tapes with batches."""
    params = np.random.rand(n_batch)
    tape = qml.tape.QuantumScript(
        [qml.RX(params, 0), MidMeasureMP(0, postselect=1), qml.CNOT([0, 1])],
        [qml.expval(qml.PauliZ(0))],
        shots=n_shots,
    )
    tapes, _ = qml.dynamic_one_shot(tape)
    assert len(tapes) == n_batch


@pytest.mark.parametrize(
    "measure, aux_measure, n_meas",
    (
        (qml.counts, CountsMP, 1),
        (qml.expval, ExpectationMP, 1),
        (qml.probs, ProbabilityMP, 1),
        (qml.sample, SampleMP, 1),
        (qml.var, SampleMP, 1),
    ),
)
def test_len_measurements_obs(measure, aux_measure, n_meas):
    """Test that the transform produces the correct number of measurements in tapes measuring observables."""
    n_shots = 10
    n_mcms = 1
    tape = qml.tape.QuantumScript(
        [qml.Hadamard(0)] + [MidMeasureMP(0)] * n_mcms, [measure(op=qml.PauliZ(0))], shots=n_shots
    )
    tapes, _ = qml.dynamic_one_shot(tape)
    assert len(tapes) == 1
    aux_tape = tapes[0]
    assert len(aux_tape.measurements) == n_meas + n_mcms
    assert isinstance(aux_tape.measurements[0], aux_measure)
    assert all(isinstance(m, SampleMP) for m in aux_tape.measurements[1:])


@pytest.mark.parametrize(
    "measure, aux_measure, n_meas",
    (
        (qml.counts, SampleMP, 0),
        (qml.expval, SampleMP, 0),
        (qml.probs, SampleMP, 0),
        (qml.sample, SampleMP, 0),
        (qml.var, SampleMP, 0),
    ),
)
def test_len_measurements_mcms(measure, aux_measure, n_meas):
    """Test that the transform produces the correct number of measurements in tapes measuring MCMs."""
    n_shots = 10
    n_mcms = 1
    tape = qml.tape.QuantumScript(
        [qml.Hadamard(0)] + [MidMeasureMP(0)] * n_mcms,
        [measure(op=MeasurementValue([MidMeasureMP(0)], lambda x: x))],
        shots=n_shots,
    )
    tapes, _ = qml.dynamic_one_shot(tape)
    assert len(tapes) == 1
    aux_tape = tapes[0]
    assert len(aux_tape.measurements) == n_meas + n_mcms
    assert isinstance(aux_tape.measurements[0], aux_measure)
    assert all(isinstance(m, SampleMP) for m in aux_tape.measurements[1:])


def assert_results(res, shots, n_mcms):
    """Helper to check that expected raw results of executing the transformed tape are correct"""
    assert len(res) == shots
    # One for the non-MeasurementValue MP, and the rest of the mid-circuit measurements
    assert all(len(r) == n_mcms + 1 for r in res)
    # Not validating distribution of results as device sampling unit tests already validate
    # that samples are generated correctly.


@pytest.mark.jax
@pytest.mark.parametrize("measure_f", (qml.expval, qml.probs, qml.sample, qml.var))
@pytest.mark.parametrize("shots", [20, [20, 21]])
@pytest.mark.parametrize("n_mcms", [1, 3])
def test_tape_results_jax(shots, n_mcms, measure_f):
    """Test that the simulation results of a tape are correct with jax parameters"""
    import jax

    dev = qml.device("default.qubit", wires=4, shots=shots, seed=jax.random.PRNGKey(123))
    param = jax.numpy.array(np.pi / 2)

    mv = qml.measure(0)
    mp = mv.measurements[0]

    tape = qml.tape.QuantumScript(
        [qml.RX(param, 0), mp] + [MidMeasureMP(0, id=str(i)) for i in range(n_mcms - 1)],
        [measure_f(op=qml.PauliZ(0)), measure_f(op=mv)],
        shots=shots,
    )

    tapes, _ = qml.dynamic_one_shot(tape)
    results = dev.execute(tapes)[0]

    # The transformed tape never has a shot vector
    if isinstance(shots, list):
        shots = sum(shots)

    assert_results(results, shots, n_mcms)


@pytest.mark.jax
@pytest.mark.parametrize(
    "measure_f, expected1, expected2",
    [
        (qml.expval, 1.0, 1.0),
        (qml.probs, [1, 0], [0, 1]),
        (qml.sample, 1, 1),
        (qml.var, 0.0, 0.0),
    ],
)
@pytest.mark.parametrize("shots", [20, [20, 21]])
@pytest.mark.parametrize("n_mcms", [1, 3])
def test_jax_results_processing(shots, n_mcms, measure_f, expected1, expected2):
    """Test that the results of tapes are processed correctly for tapes with jax parameters"""
    import jax.numpy as jnp

    mv = qml.measure(0)
    mp = mv.measurements[0]

    tape = qml.tape.QuantumScript(
        [qml.RX(1.5, 0), mp] + [MidMeasureMP(0)] * (n_mcms - 1),
        [measure_f(op=qml.PauliZ(0)), measure_f(op=mv)],
        shots=shots,
    )
    _, fn = qml.dynamic_one_shot(tape)
    all_shots = sum(shots) if isinstance(shots, list) else shots

    first_res = jnp.array([1.0, 0.0]) if measure_f == qml.probs else jnp.array(1.0)
    rest = jnp.array(1, dtype=int)
    single_shot_res = (first_res,) + (rest,) * n_mcms
    # Raw results for each shot are (sample_for_first_measurement,) + (sample for 1st MCM, sample for 2nd MCM, ...)
    raw_results = (single_shot_res,) * all_shots
    raw_results = (raw_results,)
    res = fn(raw_results)

    if measure_f is qml.sample:
        # All samples 1
        expected1 = (
            [[expected1] * s for s in shots] if isinstance(shots, list) else [expected1] * shots
        )
        expected2 = (
            [[expected2] * s for s in shots] if isinstance(shots, list) else [expected2] * shots
        )
    else:
        expected1 = [expected1 for _ in shots] if isinstance(shots, list) else expected1
        expected2 = [expected2 for _ in shots] if isinstance(shots, list) else expected2

    if isinstance(shots, list):
        assert len(res) == len(shots)
        for r, e1, e2 in zip(res, expected1, expected2):
            # Expected result is 2-list since we have two measurements in the tape
            assert qml.math.allclose(r, [e1, e2])
    else:
        # Expected result is 2-list since we have two measurements in the tape
        assert qml.math.allclose(res, [expected1, expected2])


@pytest.mark.jax
@pytest.mark.parametrize(
    "measure_f, expected1, expected2",
    [
        (qml.expval, 1.0, 1.0),
        (qml.probs, [1, 0], [0, 1]),
        (qml.sample, 1, 1),
        (qml.var, 0.0, 0.0),
    ],
)
@pytest.mark.parametrize("shots", [20, [20, 22]])
def test_jax_results_postselection_processing(shots, measure_f, expected1, expected2):
    """Test that the results of tapes are processed correctly for tapes with jax parameters
    when postselecting"""
    import jax.numpy as jnp

    param = jnp.array(np.pi / 2)
    fill_value = np.iinfo(np.int32).min
    mv = qml.measure(0, postselect=1)
    mp = mv.measurements[0]

    tape = qml.tape.QuantumScript(
        [qml.RX(param, 0), mp, MidMeasureMP(0)],
        [measure_f(op=qml.PauliZ(0)), measure_f(op=mv)],
        shots=shots,
    )
    _, fn = qml.dynamic_one_shot(tape)
    all_shots = sum(shots) if isinstance(shots, list) else shots

    # Alternating tuple. Only the values at odd indices are valid
    first_res_two_shot = (
        (jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0]))
        if measure_f == qml.probs
        else (jnp.array(1.0), jnp.array(0.0))
    )
    first_res = first_res_two_shot * (all_shots // 2)
    # Tuple of alternating 1s and 0s. Zero is invalid as postselecting on 1
    postselect_res = (jnp.array(1, dtype=int), jnp.array(0, dtype=int)) * (all_shots // 2)
    rest = (jnp.array(1, dtype=int),) * all_shots
    # Raw results for each shot are (sample_for_first_measurement, sample for 1st MCM, sample for 2nd MCM)
    raw_results = tuple(zip(first_res, postselect_res, rest))
    raw_results = (raw_results,)
    res = fn(raw_results)

    if measure_f is qml.sample:
        expected1 = (
            [[expected1, fill_value] * (s // 2) for s in shots]
            if isinstance(shots, list)
            else [expected1, fill_value] * (shots // 2)
        )
        expected2 = (
            [[expected2, fill_value] * (s // 2) for s in shots]
            if isinstance(shots, list)
            else [expected2, fill_value] * (shots // 2)
        )
    else:
        expected1 = [expected1 for _ in shots] if isinstance(shots, list) else expected1
        expected2 = [expected2 for _ in shots] if isinstance(shots, list) else expected2

    if isinstance(shots, list):
        assert len(res) == len(shots)
        for r, e1, e2 in zip(res, expected1, expected2):
            # Expected result is 2-list since we have two measurements in the tape
            assert qml.math.allclose(r, [e1, e2])
    else:
        # Expected result is 2-list since we have two measurements in the tape
        assert qml.math.allclose(res, [expected1, expected2])

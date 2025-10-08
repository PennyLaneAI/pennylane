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

import numpy as np
import pytest
from default_qubit_legacy import DefaultQubitLegacy

import pennylane as qml
from pennylane.exceptions import QuantumFunctionError
from pennylane.measurements import (
    CountsMP,
    ExpectationMP,
    MeasurementValue,
    MidMeasureMP,
    ProbabilityMP,
    SampleMP,
)
from pennylane.transforms.dynamic_one_shot import (
    _supports_one_shot,
    fill_in_value,
    gather_non_mcm,
    get_legacy_capabilities,
    parse_native_mid_circuit_measurements,
)

# pylint: disable=too-few-public-methods, too-many-arguments


def test_gather_non_mcm_unsupported_measurement():
    """Test that gather_non_mcm raises an error on supported measurements."""

    with pytest.raises(TypeError, match="does not support"):
        gather_non_mcm(qml.state(), np.array([0, 0]), np.array([True, True]))


def test_get_legacy_capability():
    dev = DefaultQubitLegacy(wires=[0], shots=1)
    with pytest.warns(
        qml.exceptions.PennyLaneDeprecationWarning, match="shots on device is deprecated"
    ):
        dev = qml.devices.LegacyDeviceFacade(dev)
    caps = get_legacy_capabilities(dev)
    assert caps["model"] == "qubit"
    assert not "supports_mid_measure" in caps
    assert not _supports_one_shot(dev)

    with pytest.warns(
        qml.exceptions.PennyLaneDeprecationWarning, match="shots on device is deprecated"
    ):
        dev2 = qml.devices.DefaultMixed(wires=[0], shots=1)
    assert not _supports_one_shot(dev2)


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
        parse_native_mid_circuit_measurements(circuit, [circuit], [np.empty((1, 1))])


def test_postselection_error_with_wrong_device():
    """Test that an error is raised when a device does not support native execution."""
    dev = qml.device("default.mixed", wires=2)

    with pytest.raises(
        TypeError,
        match="does not support mid-circuit measurements and/or one-shot execution mode natively",
    ):

        @qml.dynamic_one_shot
        @qml.qnode(dev)
        def _():
            qml.measure(0, postselect=1)
            return qml.probs(wires=[0])


def test_postselect_mode():
    """Test that invalid shots are discarded if requested"""
    shots = 100
    dev = qml.device("default.qubit")

    @qml.set_shots(shots)
    @qml.qnode(dev, postselect_mode="hw-like")
    def f(x):
        qml.RX(x, 0)
        _ = qml.measure(0, postselect=1)
        return qml.sample(wires=[0, 1])

    res = f(np.pi / 2)
    assert len(res) < shots
    assert np.all(res != np.iinfo(np.int32).min)


@pytest.mark.jax
@pytest.mark.parametrize("use_jit", [True, False])
@pytest.mark.parametrize("diff_method", [None, "best"])
def test_hw_like_with_jax(use_jit, diff_method, seed):
    """Test that invalid shots are replaced with INTEGER_MIN_VAL if
    postselect_mode="hw-like" with JAX"""
    import jax  # pylint: disable=import-outside-toplevel

    shots = 10
    dev = qml.device("default.qubit", seed=jax.random.PRNGKey(seed))

    @qml.set_shots(shots)
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
        QuantumFunctionError,
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


def generate_dummy_raw_results(measure_f, n_mcms, shots, postselect, interface):
    """Helper function for generating dummy raw results. Raw results are the output(s) of
    executing the transformed tape(s) that are given to the processing function. For
    ``dynamic_one_shot``, the first items in the measurements for the transformed tape will
    be all the measurements of the original tape that were applied to wires/observables,
    and the rest will be SampleMPs on all the mid-circuit measurements in the original tape.

    In this unit test suite, the original tape will have one measurement on wires/observables,
    so the transformed tape will have one measurement on wires/observables, and ``n_mcms``
    ``SampleMP`` measurements on MCMs.

    The raw results will be all 1s with the appropriate shape for tests without postselection.
    For tests with postselection, the results for the wires/observable measurement and first
    ``SampleMP(mcm)`` will be alternating with valid results at odd indices and invalid results
    at even indices. The results for the rest of the ``SampleMP(mcm)`` will be all 1s."""

    if postselect is None:
        # First raw result for a single shot, i.e, result of wires/obs measurement
        obs_res_single_shot = qml.math.array(
            [1.0, 0.0] if measure_f == qml.probs else [[1.0]], like=interface
        )
        # Result of SampleMP on mid-circuit measurements
        rest_single_shot = qml.math.array([[1]], like=interface)
        single_shot_res = (obs_res_single_shot,) + (rest_single_shot,) * n_mcms
        # Raw results for each shot are (sample_for_first_measurement,) + (sample for 1st MCM, sample for 2nd MCM, ...)
        raw_results = (single_shot_res,) * shots

    else:
        # When postselecting, we start by creating results for two shots as alternating indices
        # will have valid results.
        # Alternating tuple. Only the values at odd indices are valid
        if measure_f == qml.probs:
            obs_res_two_shot = (
                qml.math.array([1.0, 0.0], like=interface),
                qml.math.array([0.0, 1.0], like=interface),
            )
        elif measure_f == qml.sample:
            obs_res_two_shot = (
                qml.math.array([1.0], like=interface),
                qml.math.array([0.0], like=interface),
            )
        else:
            obs_res_two_shot = (
                qml.math.array(1.0, like=interface),
                qml.math.array(0.0, like=interface),
            )
        obs_res = obs_res_two_shot * (shots // 2)
        # Tuple of alternating 1s and 0s.
        postselect_res = (
            qml.math.array(int(postselect), like=interface),
            qml.math.array(int(not postselect), like=interface),
        ) * (shots // 2)
        rest = (qml.math.array(1, like=interface),) * shots
        # Raw results for each shot are (sample_for_first_measurement, sample for 1st MCM, sample for 2nd MCM)
        raw_results = tuple(zip(obs_res, postselect_res, rest))

    # Wrap in 1-tuple as there is a single transformed tape unless broadcasting
    return (raw_results,)


# pylint: disable=too-many-arguments, import-outside-toplevel
@pytest.mark.all_interfaces
@pytest.mark.parametrize("interface", ["autograd", "jax", "torch", "numpy", None])
@pytest.mark.parametrize("use_interface_for_results", [True, False])
class TestInterfaces:
    """Unit tests for ML interfaces with dynamic_one_shot"""

    @pytest.mark.parametrize("measure_f", (qml.expval, qml.probs, qml.sample, qml.var))
    @pytest.mark.parametrize("shots", [1, 20, [20, 21]])
    @pytest.mark.parametrize("n_mcms", [1, 3])
    def test_interface_tape_results(
        self, shots, n_mcms, measure_f, interface, use_interface_for_results, seed
    ):  # pylint: disable=unused-argument
        """Test that the simulation results of a tape are correct with interface parameters"""
        if interface == "jax":
            from jax.random import PRNGKey

            seed = PRNGKey(seed)

        dev = qml.device("default.qubit", wires=4, seed=seed)
        param = qml.math.array(np.pi / 2, like=interface)

        mv = qml.measure(0)
        mcms = [mv.measurements[0]] + [MidMeasureMP(0, id=str(i)) for i in range(n_mcms - 1)]

        tape = qml.tape.QuantumScript(
            [qml.RX(param, 0)] + mcms,
            [measure_f(op=qml.PauliZ(0)), measure_f(op=mv)],
            shots=shots,
        )

        tapes, _ = qml.dynamic_one_shot(tape)
        results = dev.execute(tapes)[0]

        # The transformed tape never has a shot vector
        if isinstance(shots, list):
            shots = sum(shots)

        assert_results(results, shots, n_mcms)

    @pytest.mark.parametrize(
        "measure_f, expected1, expected2",
        [
            (qml.expval, 1.0, 1.0),
            (qml.probs, [1, 0], [0, 1]),
            (
                qml.sample,
                # The expected results provided for qml.sample are
                # just the result of a single shot
                1,
                1,
            ),
            (qml.var, 0.0, 0.0),
        ],
    )
    @pytest.mark.parametrize("shots", [1, 20, [20, 21]])
    @pytest.mark.parametrize("n_mcms", [1, 3])
    def test_interface_results_processing(
        self, shots, n_mcms, measure_f, expected1, expected2, interface, use_interface_for_results
    ):
        """Test that the results of tapes are processed correctly for tapes with interface
        parameters"""
        param = qml.math.array(1.5, like=interface)
        mv = qml.measure(0)
        mcms = [mv.measurements[0]] + [MidMeasureMP(0)] * (n_mcms - 1)
        ops = [qml.RX(param, 0)] + mcms

        tape = qml.tape.QuantumScript(
            ops, [measure_f(op=qml.PauliZ(0)), measure_f(op=mv)], shots=shots
        )
        _, fn = qml.dynamic_one_shot(tape)
        total_shots = sum(shots) if isinstance(shots, list) else shots

        raw_results = generate_dummy_raw_results(
            measure_f=measure_f,
            n_mcms=n_mcms,
            shots=total_shots,
            postselect=None,
            interface=interface if use_interface_for_results else None,
        )
        processed_results = fn(raw_results)

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

        if use_interface_for_results:
            expected_interface = "numpy" if interface is None else interface
            assert qml.math.get_deep_interface(processed_results) == expected_interface
        else:
            assert qml.math.get_deep_interface(processed_results) == "numpy"

        if isinstance(shots, list):
            assert len(processed_results) == len(shots)
            for r, e1, e2 in zip(processed_results, expected1, expected2):
                # Expected result is 2-list since we have two measurements in the tape
                assert qml.math.allclose(r[0], e1)
                assert qml.math.allclose(r[1], e2)
        else:
            # Expected result is 2-list since we have two measurements in the tape
            assert qml.math.allclose(processed_results[0], expected1)
            assert qml.math.allclose(processed_results[1], expected2)

    @pytest.mark.parametrize(
        "measure_f, expected1, expected2",
        [
            (qml.expval, 1.0, 1.0),
            (qml.probs, [1, 0], [0, 1]),
            (
                qml.sample,
                # The expected results provided for qml.sample are
                # just the result of a single shot
                1,
                1,
            ),
            (qml.var, 0.0, 0.0),
        ],
    )
    @pytest.mark.parametrize("shots", [20, [20, 22]])
    def test_interface_results_postselection_processing(
        self, shots, measure_f, expected1, expected2, interface, use_interface_for_results
    ):
        """Test that the results of tapes are processed correctly for tapes with interface
        parameters when postselecting"""
        postselect = 1
        param = qml.math.array(np.pi / 2, like=interface)
        mv = qml.measure(0, postselect=postselect)
        mp = mv.measurements[0]

        tape = qml.tape.QuantumScript(
            [qml.RX(param, 0), mp, MidMeasureMP(0)],
            [measure_f(op=qml.PauliZ(0)), measure_f(op=mv)],
            shots=shots,
        )
        _, fn = qml.dynamic_one_shot(
            tape, postselect_mode="pad-invalid-samples" if interface == "jax" else None
        )
        total_shots = sum(shots) if isinstance(shots, list) else shots

        raw_results = generate_dummy_raw_results(
            measure_f=measure_f,
            n_mcms=2,
            shots=total_shots,
            postselect=postselect,
            interface=interface if use_interface_for_results else None,
        )
        processed_results = fn(raw_results)

        if measure_f is qml.sample:
            if interface == "jax":
                expected1 = [expected1, fill_in_value]
                expected2 = [expected2, fill_in_value]
            else:
                expected1 = [expected1]
                expected2 = [expected2]
            expected1 = (
                [expected1 * (s // 2) for s in shots]
                if isinstance(shots, list)
                else expected1 * (shots // 2)
            )
            expected2 = (
                [expected2 * (s // 2) for s in shots]
                if isinstance(shots, list)
                else expected2 * (shots // 2)
            )

        else:
            expected1 = [expected1 for _ in shots] if isinstance(shots, list) else expected1
            expected2 = [expected2 for _ in shots] if isinstance(shots, list) else expected2

        if use_interface_for_results:
            expected_interface = "numpy" if interface is None else interface
            assert qml.math.get_deep_interface(processed_results) == expected_interface
        else:
            assert qml.math.get_deep_interface(processed_results) == "numpy"

        if isinstance(shots, list):
            assert len(processed_results) == len(shots)
            for r, e1, e2 in zip(processed_results, expected1, expected2):
                # Expected result is 2-list since we have two measurements in the tape
                assert qml.math.allclose(qml.math.squeeze(r[0]), e1)
                assert qml.math.allclose(r[1], e2)
        else:
            # Expected result is 2-list since we have two measurements in the tape
            assert qml.math.allclose(qml.math.squeeze(processed_results[0]), expected1)
            assert qml.math.allclose(processed_results[1], expected2)

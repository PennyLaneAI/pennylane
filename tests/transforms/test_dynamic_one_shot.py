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
        parse_native_mid_circuit_measurements(circuit, [circuit], [[]])


def test_postselection_error_with_wrong_device():
    """Test that an error is raised when a device does not support native execution."""
    dev = qml.device("default.mixed", wires=2)

    with pytest.raises(TypeError, match="does not support mid-circuit measurements natively"):

        @qml.dynamic_one_shot
        @qml.qnode(dev)
        def _():
            qml.measure(0, postselect=1)
            return qml.probs(wires=[0])


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
    assert len(tapes) == n_shots


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
    assert len(tapes) == n_shots * n_batch


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
    assert len(tapes) == n_shots
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
    assert len(tapes) == n_shots
    aux_tape = tapes[0]
    assert len(aux_tape.measurements) == n_meas + n_mcms
    assert isinstance(aux_tape.measurements[0], aux_measure)
    assert all(isinstance(m, SampleMP) for m in aux_tape.measurements[1:])

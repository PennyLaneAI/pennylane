# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for default qubit preprocessing."""
from functools import reduce
from typing import Sequence

from flaky import flaky
import numpy as np
import pytest

import pennylane as qml
from pennylane.devices.qubit.apply_operation import apply_mid_measure, MidMeasureMP
from pennylane.transforms.dynamic_one_shot import (
    accumulate_native_mcm,
    parse_native_mid_circuit_measurements,
)


def validate_counts(shots, results1, results2):
    """Compares two counts.

    If the results are ``Sequence``s, loop over entries.

    Fails if a key of ``results1`` is not found in ``results2``.
    Passes if counts are too low, chosen as ``100``.
    Otherwise, fails if counts differ by more than ``20`` plus 20 percent.
    """
    if isinstance(results1, Sequence):
        assert isinstance(results2, Sequence)
        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2):
            validate_counts(shots, r1, r2)
        return
    for key1, val1 in results1.items():
        val2 = results2[key1]
        if abs(val1 + val2) > 100:
            assert np.allclose(val1, val2, rtol=20, atol=0.2)


def validate_samples(shots, results1, results2):
    """Compares two samples.

    If the results are ``Sequence``s, loop over entries.

    Fails if the results do not have the same shape, within ``20`` entries plus 20 percent.
    This is to handle cases when post-selection yields variable shapes.
    Otherwise, fails if the sums of samples differ by more than ``20`` plus 20 percent.
    """
    if isinstance(shots, Sequence):
        assert isinstance(results1, Sequence)
        assert isinstance(results2, Sequence)
        assert len(results1) == len(results2)
        for s, r1, r2 in zip(shots, results1, results2):
            validate_samples(s, r1, r2)
    else:
        sh1, sh2 = results1.shape[0], results2.shape[0]
        assert np.allclose(sh1, sh2, rtol=20, atol=0.2)
        assert results1.ndim == results2.ndim
        if results2.ndim > 1:
            assert results1.shape[1] == results2.shape[1]
        np.allclose(np.sum(results1), np.sum(results2), rtol=20, atol=0.2)


def validate_expval(shots, results1, results2):
    """Compares two expval, probs or var.

    If the results are ``Sequence``s, validate the average of items.

    If ``shots is None``, validate using ``np.allclose``'s default parameters.
    Otherwise, fails if the results do not match within ``0.01`` plus 20 percent.
    """
    if isinstance(results1, Sequence):
        assert isinstance(results2, Sequence)
        assert len(results1) == len(results2)
        results1 = reduce(lambda x, y: x + y, results1) / len(results1)
        results2 = reduce(lambda x, y: x + y, results2) / len(results2)
        validate_expval(shots, results1, results2)
        return
    if shots is None:
        assert np.allclose(results1, results2)
        return
    assert np.allclose(results1, results2, atol=0.01, rtol=0.2)


def validate_measurements(func, shots, results1, results2):
    """Calls the correct validation function based on measurement type."""
    if func is qml.counts:
        validate_counts(shots, results1, results2)
        return

    if func is qml.sample:
        validate_samples(shots, results1, results2)
        return

    validate_expval(shots, results1, results2)


def test_apply_mid_measure():
    with pytest.raises(ValueError, match="MidMeasureMP cannot be applied to batched states."):
        _ = apply_mid_measure(
            MidMeasureMP(0), np.zeros((2, 2)), is_state_batched=True, mid_measurements={}
        )
    m0 = MidMeasureMP(0, postselect=1)
    mid_measurements = {}
    state = apply_mid_measure(m0, np.zeros(2), mid_measurements=mid_measurements)
    assert mid_measurements[m0] == 0
    assert np.allclose(state, 0.0)
    state = apply_mid_measure(m0, np.array([1, 0]), mid_measurements=mid_measurements)
    assert mid_measurements[m0] == 0
    assert np.allclose(state, 0.0)


def test_accumulate_native_mcm_unsupported_error():
    with pytest.raises(
        TypeError,
        match=f"Native mid-circuit measurement mode does not support {type(qml.var(qml.PauliZ(0))).__name__}",
    ):
        accumulate_native_mcm(qml.tape.QuantumScript([], [qml.var(qml.PauliZ(0))]), [None], [None])


def test_all_invalid_shots_circuit():

    dev = qml.device("default.qubit")

    @qml.qnode(dev)
    def circuit_op():
        m = qml.measure(0, postselect=1)
        qml.cond(m, qml.PauliX)(1)
        return (
            qml.expval(op=qml.PauliZ(1)),
            qml.probs(op=qml.PauliY(0) @ qml.PauliZ(1)),
            qml.var(op=qml.PauliZ(1)),
        )

    res1 = circuit_op()
    res2 = circuit_op(shots=10)
    for r1, r2 in zip(res1, res2):
        if isinstance(r1, Sequence):
            assert len(r1) == len(r2)
        assert np.all(np.isnan(r1))
        assert np.all(np.isnan(r2))

    @qml.qnode(dev)
    def circuit_mcm():
        m = qml.measure(0, postselect=1)
        qml.cond(m, qml.PauliX)(1)
        return qml.expval(op=m), qml.probs(op=m), qml.var(op=m)

    res1 = circuit_mcm()
    res2 = circuit_mcm(shots=10)
    for r1, r2 in zip(res1, res2):
        if isinstance(r1, Sequence):
            assert len(r1) == len(r2)
        assert np.all(np.isnan(r1))
        assert np.all(np.isnan(r2))


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
    circuit = qml.tape.QuantumScript([qml.RX(1, 0)], [measurement])
    with pytest.raises(TypeError, match="Native mid-circuit measurement mode does not support"):
        parse_native_mid_circuit_measurements(circuit, None, None)


def test_unsupported_measurement():
    dev = qml.device("default.qubit", shots=1000)
    params = np.pi / 4 * np.ones(2)

    @qml.qnode(dev)
    def func(x, y):
        qml.RX(x, wires=0)
        m0 = qml.measure(0)
        qml.cond(m0, qml.RY)(y, wires=1)
        return qml.classical_shadow(wires=0)

    with pytest.raises(
        TypeError,
        match=f"Native mid-circuit measurement mode does not support {type(qml.classical_shadow(wires=0)).__name__}",
    ):
        func(*params)


@flaky(max_runs=5)
@pytest.mark.parametrize("shots", [None, 1000, [1000, 1001]])
@pytest.mark.parametrize("postselect", [None, 0, 1])
@pytest.mark.parametrize("reset", [False, True])
@pytest.mark.parametrize("measure_f", [qml.counts, qml.expval, qml.probs, qml.sample, qml.var])
def test_single_mcm_single_measure_mcm(shots, postselect, reset, measure_f):
    """Tests that DefaultQubit handles a circuit with a single mid-circuit measurement and a
    conditional gate. A single measurement of the mid-circuit measurement value is performed at
    the end."""

    dev = qml.device("default.qubit", shots=shots)
    params = np.pi / 4 * np.ones(2)

    @qml.qnode(dev)
    def func(x, y):
        qml.RX(x, wires=0)
        m0 = qml.measure(0, reset=reset, postselect=postselect)
        qml.cond(m0, qml.RY)(y, wires=1)
        return measure_f(op=m0)

    func1 = qml.dynamic_one_shot(func)
    func2 = qml.defer_measurements(func)

    if shots is None and measure_f in (qml.counts, qml.sample):
        return

    results1 = func1(*params)
    results2 = func2(*params)

    if postselect is None or measure_f in (qml.expval, qml.probs, qml.var):
        validate_measurements(measure_f, shots, results1, results2)


# pylint: disable=unused-argument
def obs_tape(x, y, z, reset=False, postselect=None):
    qml.RX(x, 0)
    qml.RZ(np.pi / 4, 0)
    m0 = qml.measure(0, reset=reset)
    qml.cond(m0 == 0, qml.RX)(np.pi / 4, 0)
    qml.cond(m0 == 0, qml.RZ)(np.pi / 4, 0)
    qml.cond(m0 == 1, qml.RX)(-np.pi / 4, 0)
    qml.cond(m0 == 1, qml.RZ)(-np.pi / 4, 0)
    qml.RX(y, 1)
    qml.RZ(np.pi / 4, 1)
    m1 = qml.measure(1, postselect=postselect)
    qml.cond(m1 == 0, qml.RX)(np.pi / 4, 1)
    qml.cond(m1 == 0, qml.RZ)(np.pi / 4, 1)
    qml.cond(m1 == 1, qml.RX)(-np.pi / 4, 1)
    qml.cond(m1 == 1, qml.RZ)(-np.pi / 4, 1)
    return m0, m1


@flaky(max_runs=5)
@pytest.mark.parametrize("shots", [None, 5000, [5000, 5001]])
@pytest.mark.parametrize("postselect", [None, 0, 1])
@pytest.mark.parametrize("reset", [False, True])
@pytest.mark.parametrize("measure_f", [qml.counts, qml.expval, qml.probs, qml.sample, qml.var])
@pytest.mark.parametrize("obs", [qml.PauliZ(0), qml.PauliY(1), qml.PauliZ(0) @ qml.PauliY(1)])
def test_single_mcm_single_measure_obs(shots, postselect, reset, measure_f, obs):
    """Tests that DefaultQubit handles a circuit with a single mid-circuit measurement and a
    conditional gate. A single measurement of a common observable is performed at the end."""

    dev = qml.device("default.qubit", shots=shots)
    params = [np.pi / 7, np.pi / 6, -np.pi / 5]

    @qml.qnode(dev)
    def func(x, y, z):
        obs_tape(x, y, z, reset=reset, postselect=postselect)
        return measure_f(op=obs)

    func1 = func
    func2 = qml.defer_measurements(func)

    if shots is None and measure_f in (qml.counts, qml.sample):
        return

    results1 = func1(*params)
    results2 = func2(*params)

    if postselect is None or measure_f in (qml.expval, qml.probs, qml.var):
        validate_measurements(measure_f, shots, results1, results2)


@flaky(max_runs=5)
@pytest.mark.parametrize("shots", [None, 3000, [3000, 3001]])
@pytest.mark.parametrize("postselect", [None, 0, 1])
@pytest.mark.parametrize("reset", [False, True])
@pytest.mark.parametrize("measure_f", [qml.counts, qml.probs, qml.sample])
@pytest.mark.parametrize("wires", [[0], [0, 1]])
def test_single_mcm_single_measure_wires(shots, postselect, reset, measure_f, wires):
    """Tests that DefaultQubit handles a circuit with a single mid-circuit measurement and a
    conditional gate. A single measurement of one or several wires is performed at the end."""

    dev = qml.device("default.qubit", shots=shots)
    params = np.pi / 4 * np.ones(2)

    @qml.qnode(dev)
    def func(x, y):
        qml.RX(x, wires=0)
        m0 = qml.measure(0, reset=reset, postselect=postselect)
        qml.cond(m0, qml.RY)(y, wires=1)
        return measure_f(wires=wires)

    func1 = func
    func2 = qml.defer_measurements(func)

    if shots is None and measure_f in (qml.counts, qml.sample):
        return

    results1 = func1(*params)
    results2 = func2(*params)

    if postselect is None or measure_f in (qml.expval, qml.probs, qml.var):
        validate_measurements(measure_f, shots, results1, results2)


@flaky(max_runs=5)
@pytest.mark.parametrize("shots", [5000])
@pytest.mark.parametrize("postselect", [None, 0, 1])
@pytest.mark.parametrize("reset", [False, True])
@pytest.mark.parametrize("measure_f", [qml.counts, qml.expval, qml.probs, qml.sample, qml.var])
def test_single_mcm_multiple_measurements(shots, postselect, reset, measure_f):
    """Tests that DefaultQubit handles a circuit with a single mid-circuit measurement with reset
    and a conditional gate. Multiple measurements of the mid-circuit measurement value are
    performed."""

    dev = qml.device("default.qubit", shots=shots)
    params = [np.pi / 7, np.pi / 6, -np.pi / 5]
    obs = qml.PauliY(1)

    @qml.qnode(dev)
    def func(x, y, z):
        mcms = obs_tape(x, y, z, reset=reset, postselect=postselect)
        return measure_f(op=obs), measure_f(op=mcms[0])

    func1 = func
    func2 = qml.defer_measurements(func)

    results1 = func1(*params)
    results2 = func2(*params)

    if postselect is None or measure_f in (qml.expval, qml.probs, qml.var):
        for r1, r2 in zip(results1, results2):
            validate_measurements(measure_f, shots, r1, r2)


@flaky(max_runs=5)
@pytest.mark.parametrize("shots", [None, 5000, [5000, 5001]])
@pytest.mark.parametrize("postselect", [None, 0, 1])
@pytest.mark.parametrize("reset", [False, True])
@pytest.mark.parametrize("measure_f", [qml.counts, qml.expval, qml.sample, qml.var])
def test_composite_mcm_measure_composite_mcm(shots, postselect, reset, measure_f):
    """Tests that DefaultQubit handles a circuit with a composite mid-circuit measurement and a
    conditional gate. A single measurement of a composite mid-circuit measurement is performed
    at the end."""

    dev = qml.device("default.qubit", shots=shots)
    param = np.pi / 3

    @qml.qnode(dev)
    def func(x):
        qml.RX(x, 0)
        m0 = qml.measure(0)
        qml.RX(0.5 * x, 1)
        m1 = qml.measure(1, reset=reset, postselect=postselect)
        qml.cond((m0 + m1) == 2, qml.RY)(2.0 * x, 0)
        m2 = qml.measure(0)
        return measure_f(op=(m0 - 2 * m1) * m2 + 7)

    func1 = func
    func2 = qml.defer_measurements(func)

    if shots is None and measure_f in (qml.counts, qml.sample):
        return

    results1 = func1(param)
    results2 = func2(param)

    if postselect is None or measure_f in (qml.expval, qml.probs, qml.var):
        validate_measurements(measure_f, shots, results1, results2)


@flaky(max_runs=5)
@pytest.mark.parametrize("shots", [None, 5000, [5000, 5001]])
@pytest.mark.parametrize("postselect", [None, 0, 1])
@pytest.mark.parametrize("reset", [False, True])
@pytest.mark.parametrize("measure_f", [qml.counts, qml.expval, qml.probs, qml.sample, qml.var])
def test_composite_mcm_single_measure_obs(shots, postselect, reset, measure_f):
    """Tests that DefaultQubit handles a circuit with a composite mid-circuit measurement and a
    conditional gate. A single measurement of a common observable is performed at the end."""

    dev = qml.device("default.qubit", shots=shots)
    params = [np.pi / 7, np.pi / 6, -np.pi / 5]
    obs = qml.PauliZ(0) @ qml.PauliY(1)

    @qml.qnode(dev)
    def func(x, y, z):
        mcms = obs_tape(x, y, z, reset=reset, postselect=postselect)
        qml.cond(mcms[0] != mcms[1], qml.RY)(z, wires=0)
        qml.cond(mcms[0] == mcms[1], qml.RY)(z, wires=1)
        return measure_f(op=obs)

    func1 = func
    func2 = qml.defer_measurements(func)

    if shots is None and measure_f in (qml.counts, qml.sample):
        return

    results1 = func1(*params)
    results2 = func2(*params)

    if postselect is None or measure_f in (qml.expval, qml.probs, qml.var):
        validate_measurements(measure_f, shots, results1, results2)


@flaky(max_runs=5)
@pytest.mark.parametrize("shots", [5000, [5000, 5001]])
@pytest.mark.parametrize("postselect", [None, 0, 1])
@pytest.mark.parametrize("reset", [False, True])
@pytest.mark.parametrize("measure_f", [qml.counts, qml.probs, qml.sample])
def test_composite_mcm_measure_value_list(shots, postselect, reset, measure_f):
    """Tests that DefaultQubit handles a circuit with a composite mid-circuit measurement and a
    conditional gate. A single measurement of a composite mid-circuit measurement is performed
    at the end."""

    dev = qml.device("default.qubit", shots=shots)
    param = np.pi / 3

    @qml.qnode(dev)
    def func(x):
        qml.RX(x, 0)
        m0 = qml.measure(0)
        qml.RX(0.5 * x, 1)
        m1 = qml.measure(1, reset=reset, postselect=postselect)
        qml.cond((m0 + m1) == 2, qml.RY)(2.0 * x, 0)
        m2 = qml.measure(0)
        return measure_f(op=[m0, m1, m2])

    func1 = func
    func2 = qml.defer_measurements(func)

    results1 = func1(param)
    results2 = func2(param)

    validate_measurements(measure_f, shots, results1, results2)


@flaky(max_runs=5)
@pytest.mark.parametrize("shots", [5000])
@pytest.mark.parametrize("postselect", [None, 0, 1])
@pytest.mark.parametrize("reset", [False, True])
@pytest.mark.parametrize("measure_f", [qml.expval])
def composite_mcm_gradient_measure_obs(shots, postselect, reset, measure_f):
    """Tests that DefaultQubit can differentiate a circuit with a composite mid-circuit
    measurement and a conditional gate. A single measurement of a common observable is
    performed at the end."""

    dev = qml.device("default.qubit", shots=shots)
    param = qml.numpy.array([np.pi / 3, np.pi / 6])
    obs = qml.PauliZ(0) @ qml.PauliZ(1)

    @qml.qnode(dev, diff_method="parameter-shift")
    def func(x, y):
        qml.RX(x, 0)
        m0 = qml.measure(0)
        qml.RX(y, 1)
        m1 = qml.measure(1, reset=reset, postselect=postselect)
        qml.cond((m0 + m1) == 2, qml.RY)(2 * np.pi / 3, 0)
        qml.cond((m0 + m1) > 0, qml.RY)(2 * np.pi / 3, 1)
        return measure_f(op=obs)

    func1 = func
    func2 = qml.defer_measurements(func)

    results1 = func1(*param)
    results2 = func2(*param)

    validate_measurements(measure_f, shots, results1, results2)

    grad1 = qml.grad(func)(*param)
    grad2 = qml.grad(func2)(*param)

    assert np.allclose(grad1, grad2, atol=0.01, rtol=0.3)

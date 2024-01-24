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

from flaky import flaky
import numpy as np
import pytest

import pennylane as qml


def validate_samples(shots, results1, results2):
    for res in [results1, results2]:
        if isinstance(shots, list):
            assert len(res) == len(shots)
            assert all(r.shape == (s,) for r, s in zip(res, shots))
            assert all(
                abs(sum(r1) - sum(r2)) < s // 10 for r1, r2, s in zip(results1, results2, shots)
            )
        else:
            assert res.shape == (shots,)
            assert abs(sum(results1) - sum(results2)) < shots // 10


def validate_expval(shots, results1, results2):
    if shots is None:
        assert np.allclose(results1, results2)
    assert np.allclose(results1, results2, atol=0, rtol=0.3)


@flaky(max_runs=5)
@pytest.mark.parametrize("shots", [None, 1000, [1000, 1001]])
@pytest.mark.parametrize("measure_f", [qml.expval, qml.probs, qml.sample, qml.counts, qml.var])
def test_single_mcm_single_measure_mcm(shots, measure_f):
    """Tests that DefaultQubit handles a circuit with a single mid-circuit measurement and a
    conditional gate. A single measurement of the mid-circuit measurement value is performed at
    the end."""
    dev = qml.device("default.qubit", shots=shots)
    params = np.pi / 4 * np.ones(2)

    @qml.qnode(dev)
    def func1(x, y):
        qml.RX(x, wires=0)
        m0 = qml.measure(0)
        qml.cond(m0, qml.RY)(y, wires=1)
        return measure_f(op=m0)

    @qml.qnode(dev)
    @qml.defer_measurements
    def func2(x, y):
        qml.RX(x, wires=0)
        m0 = qml.measure(0)
        qml.cond(m0, qml.RY)(y, wires=1)
        return measure_f(op=m0)

    if shots is None and measure_f in (qml.counts, qml.sample):
        return

    results1 = func1(*params)
    results2 = func2(*params)

    if measure_f is qml.counts:
        return

    if measure_f is qml.sample:
        validate_samples(shots, results1, results2)
        return

    validate_expval(shots, results1, results2)


@flaky(max_runs=5)
@pytest.mark.parametrize("shots", [None, 1000, [1000, 1001]])
@pytest.mark.parametrize("measure_f", [qml.expval, qml.probs, qml.sample, qml.counts, qml.var])
def test_single_mcm_single_measure_obs(shots, measure_f):
    """Tests that DefaultQubit handles a circuit with a single mid-circuit measurement and a
    conditional gate. A single measurement of a common observable is is performed at the end."""
    dev = qml.device("default.qubit", shots=shots)
    params = np.pi / 4 * np.ones(2)
    obs = qml.PauliZ(0)

    @qml.qnode(dev)
    def func1(x, y):
        qml.RX(x, wires=0)
        m0 = qml.measure(0)
        qml.cond(m0, qml.RY)(y, wires=1)
        return measure_f(op=obs)

    @qml.qnode(dev)
    @qml.defer_measurements
    def func2(x, y):
        qml.RX(x, wires=0)
        m0 = qml.measure(0)
        qml.cond(m0, qml.RY)(y, wires=1)
        return measure_f(op=obs)

    if shots is None and measure_f in (qml.counts, qml.sample):
        return

    results1 = func1(*params)
    results2 = func2(*params)

    if measure_f is qml.counts:
        return

    if measure_f is qml.sample:
        validate_samples(shots, results1, results2)
        return

    validate_expval(shots, results1, results2)


@flaky(max_runs=5)
@pytest.mark.parametrize("shots", [1000])
@pytest.mark.parametrize("measure_f", [qml.expval, qml.probs, qml.sample, qml.counts, qml.var])
def test_single_mcm_multiple_measurements(shots, measure_f):
    """Tests that DefaultQubit handles a circuit with a single mid-circuit measurement and a
    conditional gate. Multiple measurements of the mid-circuit measurement value are performed at
    the end."""
    dev = qml.device("default.qubit", shots=shots)
    params = np.pi / 4 * np.ones(2)
    obs = qml.PauliZ(0)

    @qml.qnode(dev)
    def func1(x, y):
        qml.RX(x, wires=0)
        m0 = qml.measure(0)
        qml.cond(m0, qml.RY)(y, wires=1)
        return measure_f(op=obs), measure_f(op=m0)

    @qml.qnode(dev)
    @qml.defer_measurements
    def func2(x, y):
        qml.RX(x, wires=0)
        m0 = qml.measure(0)
        qml.cond(m0, qml.RY)(y, wires=1)
        return measure_f(op=obs), measure_f(op=m0)

    results1 = func1(*params)
    results2 = func2(*params)

    for r1, r2 in zip(results1, results2):
        if measure_f is qml.counts:
            continue

        if measure_f is qml.sample:
            validate_samples(shots, r1, r2)
            continue

        validate_expval(shots, r1, r2)


@flaky(max_runs=5)
@pytest.mark.parametrize("shots", [1000])
@pytest.mark.parametrize("measure_f", [qml.expval, qml.probs, qml.sample, qml.counts, qml.var])
def test_single_mcm_reset(shots, measure_f):
    """Tests that DefaultQubit handles a circuit with a single mid-circuit measurement with reset
    and a conditional gate. Multiple measurements of the mid-circuit measurement value are
    performed."""
    dev = qml.device("default.qubit", shots=shots)
    params = np.pi / 4 * np.ones(2)
    obs = qml.PauliZ(0)

    @qml.qnode(dev)
    def func1(x, y):
        qml.RX(x, wires=0)
        m0 = qml.measure(0, reset=True)
        qml.cond(m0, qml.RY)(y, wires=1)
        return measure_f(op=obs), measure_f(op=m0)

    @qml.qnode(dev)
    @qml.defer_measurements
    def func2(x, y):
        qml.RX(x, wires=0)
        m0 = qml.measure(0, reset=True)
        qml.cond(m0, qml.RY)(y, wires=1)
        return measure_f(op=obs), measure_f(op=m0)

    results1 = func1(*params)
    results2 = func2(*params)

    print(results1)
    print(results2)

    for r1, r2 in zip(results1, results2):
        if measure_f is qml.counts:
            continue

        if measure_f is qml.sample:
            validate_samples(shots, r1, r2)
            continue

        validate_expval(shots, r1, r2)

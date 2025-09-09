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
from collections.abc import Sequence

import mcm_utils
import numpy as np
import pytest

import pennylane as qml
from pennylane.devices.qubit.apply_operation import MidMeasureMP
from pennylane.devices.qubit.simulate import combine_measurements_core, measurement_with_no_shots
from pennylane.transforms.dynamic_one_shot import fill_in_value

pytestmark = pytest.mark.slow


# pylint: disable=too-many-arguments


def test_combine_measurements_core():
    """Test that combine_measurements_core raises for unsupported measurements."""
    with pytest.raises(TypeError, match="Native mid-circuit measurement mode does not support"):
        _ = combine_measurements_core(qml.classical_shadow(0), None)


def test_measurement_with_no_shots():
    """Test that measurement_with_no_shots returns the correct NaNs."""
    assert np.isnan(measurement_with_no_shots(qml.expval(0)))
    probs = measurement_with_no_shots(qml.probs(wires=0))
    assert probs.shape == (2,)
    assert all(np.isnan(probs).tolist())
    probs = measurement_with_no_shots(qml.probs(wires=[0, 1]))
    assert probs.shape == (4,)
    assert all(np.isnan(probs).tolist())
    probs = measurement_with_no_shots(qml.probs(op=qml.PauliY(0)))
    assert probs.shape == (2,)
    assert all(np.isnan(probs).tolist())


@pytest.mark.parametrize("obs", ["mid-meas", "pauli"])
@pytest.mark.parametrize("mcm_method", ["one-shot", "tree-traversal"])
def test_all_invalid_shots_circuit(obs, mcm_method):
    """Test that circuits in which all shots mismatch with post-selection conditions return the same answer as ``defer_measurements``."""

    dev = qml.device("default.qubit")
    dev_shots = qml.device("default.qubit")

    def circuit_op():
        m = qml.measure(0, postselect=1)
        qml.cond(m, qml.PauliX)(1)
        return (
            (
                qml.expval(op=qml.PauliZ(1)),
                qml.probs(op=qml.PauliY(0) @ qml.PauliZ(1)),
                qml.var(op=qml.PauliZ(1)),
            )
            if obs == "pauli"
            else (qml.expval(op=m), qml.probs(op=m), qml.var(op=m))
        )

    res1 = qml.QNode(circuit_op, dev, mcm_method="deferred")()
    res2 = qml.set_shots(qml.QNode(circuit_op, dev_shots, mcm_method=mcm_method), shots=10)()
    for r1, r2 in zip(res1, res2):
        if isinstance(r1, Sequence):
            assert len(r1) == len(r2)
        assert np.all(np.isnan(r1))
        assert np.all(np.isnan(r2))


@pytest.mark.parametrize("mcm_method", ["one-shot", "tree-traversal"])
def test_unsupported_measurement(mcm_method):
    """Test that circuits with unsupported measurements raise the correct error."""

    dev = qml.device("default.qubit")
    params = np.pi / 4 * np.ones(2)

    @qml.set_shots(1000)
    @qml.qnode(dev, mcm_method=mcm_method)
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


@pytest.mark.parametrize("mcm_method", ["one-shot", "tree-traversal"])
@pytest.mark.parametrize("shots", [None, 5500, [5500, 5501]])
@pytest.mark.parametrize(
    "params",
    [
        [np.pi / 2.5, np.pi / 3, -np.pi / 3.5],
        [[np.pi / 2.5, -np.pi / 3.5], [np.pi / 4.5, np.pi / 3.2], [np.pi, -np.pi / 0.5]],
    ],
)
@pytest.mark.parametrize("postselect", [None, 0, 1])
@pytest.mark.parametrize("reset", [False, True])
def test_multiple_measurements_and_reset(mcm_method, shots, params, postselect, reset, seed):
    """Tests that DefaultQubit handles a circuit with a single mid-circuit measurement with reset
    and a conditional gate. Multiple measurements of the mid-circuit measurement value are
    performed. This function also tests `reset` parametrizing over the parameter."""

    if mcm_method == "one-shot" and shots is None:
        pytest.skip("`mcm_method='one-shot'` is incompatible with analytic mode (`shots=None`)")

    batch_size = len(params[0]) if isinstance(params[0], list) else None
    if batch_size is not None and shots is not None and postselect is not None:
        pytest.skip("Postselection with samples doesn't work with broadcasting")

    dev = qml.device("default.qubit", seed=seed)
    obs = qml.PauliY(1)
    state = qml.math.zeros((4,))
    state[0] = 1.0

    def func(x, y, z):
        qml.StatePrep(state, wires=[0, 1])
        mcms = obs_tape(x, y, z, reset=reset, postselect=postselect)
        return (
            (
                qml.expval(op=mcms[0]),
                qml.probs(op=obs),
                qml.var(op=obs),
                qml.expval(op=obs),
                qml.probs(op=obs),
                qml.var(op=mcms[0]),
            )
            if shots is None
            else (
                qml.counts(op=obs),
                qml.expval(op=mcms[0]),
                qml.probs(op=obs),
                qml.sample(op=mcms[0]),
                qml.var(op=obs),
            )
        )

    results0 = qml.set_shots(qml.QNode(func, dev, mcm_method=mcm_method), shots=shots)(*params)
    results1 = qml.set_shots(qml.QNode(func, dev, mcm_method="deferred"), shots=shots)(*params)

    measurements = (
        [qml.expval, qml.probs, qml.var, qml.expval, qml.probs, qml.var]
        if shots is None
        else [qml.counts, qml.expval, qml.probs, qml.sample, qml.var]
    )

    if not isinstance(shots, list):
        shots, results0, results1 = [shots], [results0], [results1]

    for shot, res1, res0 in zip(shots, results1, results0):
        for measure_f, r1, r0 in zip(measurements, res1, res0):
            if shots is None and measure_f in [qml.expval, qml.probs] and batch_size is not None:
                r0 = qml.math.squeeze(r0)
            mcm_utils.validate_measurements(measure_f, shot, r1, r0, batch_size=batch_size)


@pytest.mark.parametrize("mcm_method", ["one-shot", "tree-traversal"])
@pytest.mark.parametrize("shots", [None, 5000, [5000, 5001]])
@pytest.mark.parametrize(
    "mcm_name, mcm_func",
    [
        ("single", lambda x: x * -1),
        ("single", lambda x: x * 2),
        ("single", lambda x: 1 - x),
        ("single", lambda x: x & 3),
        ("mix", lambda x, y: x == y),
        ("mix", lambda x, y: 4 * x + 2 * y),
        ("all", lambda x, y, z: [x, y, z]),
        ("all", lambda x, y, z: (x - 2 * y) * z + 7),
    ],
)
@pytest.mark.parametrize("measure_f", [qml.counts, qml.expval, qml.probs, qml.sample, qml.var])
def test_composite_mcms(mcm_method, shots, mcm_name, mcm_func, measure_f, seed):
    """Tests that DefaultQubit handles a circuit with a composite mid-circuit measurement and a
    conditional gate. A single measurement of a composite mid-circuit measurement is performed
    at the end."""

    if mcm_method == "one-shot" and shots is None:
        pytest.skip("`mcm_method='one-shot'` is incompatible with analytic mode (`shots=None`)")

    if measure_f in (qml.counts, qml.sample) and shots is None:
        pytest.skip("Can't measure counts/sample in analytic mode (`shots=None`)")

    if measure_f in (qml.expval, qml.var) and mcm_name in ["mix", "all"]:
        pytest.skip(
            "expval/var does not support measuring sequences of measurements or observables."
        )

    if measure_f in (qml.probs,) and mcm_name in ["mix", "all"]:
        pytest.skip(
            "Cannot use qml.probs() when measuring multiple mid-circuit measurements collected using arithmetic operators."
        )

    dev = qml.device("default.qubit", seed=seed)
    param = qml.numpy.array([np.pi / 3, np.pi / 6])

    def func(x, y):
        qml.RX(x, 0)
        m0 = qml.measure(0)
        qml.RX(0.5 * x, 1)
        m1 = qml.measure(1)
        qml.cond((m0 + m1) == 2, qml.RY)(2.0 * x, 0)
        m2 = qml.measure(0)
        obs = (
            mcm_func(m2)
            if mcm_name == "single"
            else (mcm_func(m0, m1) if mcm_name == "mix" else mcm_func(m0, m1, m2))
        )
        return measure_f(op=obs)

    results0 = qml.set_shots(qml.QNode(func, dev, mcm_method=mcm_method), shots=shots)(*param)
    results1 = qml.set_shots(qml.QNode(func, dev, mcm_method="deferred"), shots=shots)(*param)

    mcm_utils.validate_measurements(measure_f, shots, results1, results0)


@pytest.mark.parametrize("shots", [5000])
@pytest.mark.parametrize("postselect", [None, 0, 1])
@pytest.mark.parametrize("reset", [False, True])
@pytest.mark.parametrize("measure_f", [qml.expval])
def composite_mcm_gradient_measure_obs(shots, postselect, reset, measure_f, seed):
    """Tests that DefaultQubit can differentiate a circuit with a composite mid-circuit
    measurement and a conditional gate. A single measurement of a common observable is
    performed at the end."""

    dev = qml.device("default.qubit", seed=seed)
    param = qml.numpy.array([np.pi / 3, np.pi / 6])
    obs = qml.PauliZ(0) @ qml.PauliZ(1)

    @qml.set_shots(shots)
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

    mcm_utils.validate_measurements(measure_f, shots, results1, results2)

    grad1 = qml.grad(func)(*param)
    grad2 = qml.grad(func2)(*param)

    assert np.allclose(grad1, grad2, atol=0.01, rtol=0.3)


@pytest.mark.parametrize("mcm_method", ["one-shot", "tree-traversal"])
def test_sample_with_broadcasting_and_postselection_error(mcm_method, seed):
    """Test that an error is raised if returning qml.sample if postselecting with broadcasting"""
    tape = qml.tape.QuantumScript(
        [qml.RX([0.1, 0.2], 0), MidMeasureMP(0, postselect=1)], [qml.sample(wires=0)], shots=10
    )
    with pytest.raises(ValueError, match="Returning qml.sample is not supported when"):
        qml.transforms.dynamic_one_shot(tape)

    dev = qml.device("default.qubit", seed=seed)

    @qml.set_shots(10)
    @qml.qnode(dev, mcm_method=mcm_method)
    def circuit(x):
        qml.RX(x, 0)
        qml.measure(0, postselect=1)
        return qml.sample(wires=0)

    with pytest.raises(ValueError, match="Returning qml.sample is not supported when"):
        _ = circuit([0.1, 0.2])


# pylint: disable=import-outside-toplevel, not-an-iterable
@pytest.mark.jax
class TestJaxIntegration:
    """Integration tests for dynamic_one_shot with jax"""

    @pytest.mark.parametrize("mcm_method", ["one-shot", "tree-traversal"])
    @pytest.mark.parametrize("shots", [100, [100, 101], [100, 100, 101]])
    @pytest.mark.parametrize("postselect", [None, 0, 1])
    def test_sample_with_prng_key(self, mcm_method, shots, postselect, seed):
        """Test that setting a PRNGKey gives the expected behaviour. With separate calls
        to DefaultQubit.execute, the same results are expected when using a PRNGKey"""
        # pylint: disable=import-outside-toplevel
        from jax.random import PRNGKey

        dev = qml.device("default.qubit", seed=PRNGKey(seed))
        params = [np.pi / 4, np.pi / 3]
        obs = qml.PauliZ(0) @ qml.PauliZ(1)

        def func(x, y):
            obs_tape(x, y, None, postselect=postselect)
            return qml.sample(op=obs)

        func0 = qml.set_shots(qml.QNode(func, dev, mcm_method=mcm_method), shots=shots)
        results0 = func0(*params)
        results1 = qml.set_shots(qml.QNode(func, dev, mcm_method="deferred"), shots=shots)(*params)

        mcm_utils.validate_measurements(qml.sample, shots, results1, results0, batch_size=None)

        evals = obs.eigvals()
        for eig in evals:
            # When comparing with the results from a circuit with deferred measurements
            # we're not always expected to have the functions used to sample are different
            if isinstance(shots, list):
                for r in results1:
                    assert not np.all(np.isclose(r, eig))
            else:
                assert not np.all(np.isclose(results1, eig))

        results0_2 = func0(*params)
        # Same result expected with multiple executions
        if isinstance(shots, list):
            for r0, r0_2 in zip(results0, results0_2):
                assert np.allclose(r0, r0_2)
        else:
            assert np.allclose(results0, results0_2)

    @pytest.mark.parametrize("diff_method", [None, "best"])
    @pytest.mark.parametrize("postselect", [None, 1])
    @pytest.mark.parametrize("reset", [False, True])
    def test_jax_jit(self, diff_method, postselect, reset, seed):
        """Tests that DefaultQubit handles a circuit with a single mid-circuit measurement and a
        conditional gate. A single measurement of a common observable is performed at the end."""
        import jax

        shots = 10

        dev = qml.device("default.qubit", seed=jax.random.PRNGKey(seed))
        params = [np.pi / 2.5, np.pi / 3, -np.pi / 3.5]
        obs = qml.PauliY(0)

        @qml.set_shots(shots)
        @qml.qnode(dev, diff_method=diff_method)
        def func(x, y, z):
            m0, m1 = obs_tape(x, y, z, reset=reset, postselect=postselect)
            return (
                qml.probs(wires=[1]),
                qml.probs(wires=[0, 1]),
                qml.sample(wires=[1]),
                qml.sample(wires=[0, 1]),
                qml.expval(obs),
                qml.probs(obs),
                qml.sample(obs),
                qml.var(obs),
                qml.expval(op=m0 + 2 * m1),
                qml.probs(op=m0),
                qml.sample(op=m0 + 2 * m1),
                qml.var(op=m0 + 2 * m1),
                qml.probs(op=[m0, m1]),
            )

        func1 = func
        results1 = func1(*params)

        jaxpr = str(jax.make_jaxpr(func)(*params))
        assert "pure_callback" not in jaxpr

        func2 = jax.jit(func)
        results2 = func2(*params)

        measures = [
            qml.probs,
            qml.probs,
            qml.sample,
            qml.sample,
            qml.expval,
            qml.probs,
            qml.sample,
            qml.var,
            qml.expval,
            qml.probs,
            qml.sample,
            qml.var,
            qml.probs,
        ]
        for measure_f, r1, r2 in zip(measures, results1, results2):
            r1, r2 = np.array(r1).ravel(), np.array(r2).ravel()
            if measure_f == qml.sample:
                r2 = r2[r2 != fill_in_value]
            np.allclose(r1, r2)

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

import pennylane as qp
from pennylane.devices.qubit.simulate import combine_measurements_core, measurement_with_no_shots
from pennylane.ops import MidMeasure
from pennylane.transforms.dynamic_one_shot import fill_in_value

pytestmark = pytest.mark.slow


# pylint: disable=too-many-arguments


def test_combine_measurements_core():
    """Test that combine_measurements_core raises for unsupported measurements."""
    with pytest.raises(TypeError, match="Native mid-circuit measurement mode does not support"):
        _ = combine_measurements_core(qp.classical_shadow(0), None)


def test_measurement_with_no_shots():
    """Test that measurement_with_no_shots returns the correct NaNs."""
    assert np.isnan(measurement_with_no_shots(qp.expval(0)))
    probs = measurement_with_no_shots(qp.probs(wires=0))
    assert probs.shape == (2,)
    assert all(np.isnan(probs).tolist())
    probs = measurement_with_no_shots(qp.probs(wires=[0, 1]))
    assert probs.shape == (4,)
    assert all(np.isnan(probs).tolist())
    probs = measurement_with_no_shots(qp.probs(op=qp.PauliY(0)))
    assert probs.shape == (2,)
    assert all(np.isnan(probs).tolist())


@pytest.mark.parametrize("obs", ["mid-meas", "pauli"])
@pytest.mark.parametrize("mcm_method", ["one-shot", "tree-traversal"])
def test_all_invalid_shots_circuit(obs, mcm_method):
    """Test that circuits in which all shots mismatch with post-selection conditions return the same answer as ``defer_measurements``."""

    dev = qp.device("default.qubit")
    dev_shots = qp.device("default.qubit")

    def circuit_op():
        m = qp.measure(0, postselect=1)
        qp.cond(m, qp.PauliX)(1)
        return (
            (
                qp.expval(op=qp.PauliZ(1)),
                qp.probs(op=qp.PauliY(0) @ qp.PauliZ(1)),
                qp.var(op=qp.PauliZ(1)),
            )
            if obs == "pauli"
            else (qp.expval(op=m), qp.probs(op=m), qp.var(op=m))
        )

    res1 = qp.QNode(circuit_op, dev, mcm_method="deferred")()
    res2 = qp.set_shots(qp.QNode(circuit_op, dev_shots, mcm_method=mcm_method), shots=10)()
    for r1, r2 in zip(res1, res2):
        if isinstance(r1, Sequence):
            assert len(r1) == len(r2)
        assert np.all(np.isnan(r1))
        assert np.all(np.isnan(r2))


@pytest.mark.parametrize("mcm_method", ["one-shot", "tree-traversal"])
def test_unsupported_measurement(mcm_method):
    """Test that circuits with unsupported measurements raise the correct error."""

    dev = qp.device("default.qubit")
    params = np.pi / 4 * np.ones(2)

    @qp.set_shots(1000)
    @qp.qnode(dev, mcm_method=mcm_method)
    def func(x, y):
        qp.RX(x, wires=0)
        m0 = qp.measure(0)
        qp.cond(m0, qp.RY)(y, wires=1)
        return qp.classical_shadow(wires=0)

    with pytest.raises(
        TypeError,
        match=f"Native mid-circuit measurement mode does not support {type(qp.classical_shadow(wires=0)).__name__}",
    ):
        func(*params)


# pylint: disable=unused-argument
def obs_tape(x, y, z, reset=False, postselect=None):
    qp.RX(x, 0)
    qp.RZ(np.pi / 4, 0)
    m0 = qp.measure(0, reset=reset)
    qp.cond(m0 == 0, qp.RX)(np.pi / 4, 0)
    qp.cond(m0 == 0, qp.RZ)(np.pi / 4, 0)
    qp.cond(m0 == 1, qp.RX)(-np.pi / 4, 0)
    qp.cond(m0 == 1, qp.RZ)(-np.pi / 4, 0)
    qp.RX(y, 1)
    qp.RZ(np.pi / 4, 1)
    m1 = qp.measure(1, postselect=postselect)
    qp.cond(m1 == 0, qp.RX)(np.pi / 4, 1)
    qp.cond(m1 == 0, qp.RZ)(np.pi / 4, 1)
    qp.cond(m1 == 1, qp.RX)(-np.pi / 4, 1)
    qp.cond(m1 == 1, qp.RZ)(-np.pi / 4, 1)
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

    dev = qp.device("default.qubit", seed=seed)
    obs = qp.PauliY(1)
    state = qp.math.zeros((4,))
    state[0] = 1.0

    def func(x, y, z):
        qp.StatePrep(state, wires=[0, 1])
        mcms = obs_tape(x, y, z, reset=reset, postselect=postselect)
        return (
            (
                qp.expval(op=mcms[0]),
                qp.probs(op=obs),
                qp.var(op=obs),
                qp.expval(op=obs),
                qp.probs(op=obs),
                qp.var(op=mcms[0]),
            )
            if shots is None
            else (
                qp.counts(op=obs),
                qp.expval(op=mcms[0]),
                qp.probs(op=obs),
                qp.sample(op=mcms[0]),
                qp.var(op=obs),
            )
        )

    results0 = qp.set_shots(qp.QNode(func, dev, mcm_method=mcm_method), shots=shots)(*params)
    results1 = qp.set_shots(qp.QNode(func, dev, mcm_method="deferred"), shots=shots)(*params)

    measurements = (
        [qp.expval, qp.probs, qp.var, qp.expval, qp.probs, qp.var]
        if shots is None
        else [qp.counts, qp.expval, qp.probs, qp.sample, qp.var]
    )

    if not isinstance(shots, list):
        shots, results0, results1 = [shots], [results0], [results1]

    for shot, res1, res0 in zip(shots, results1, results0):
        for measure_f, r1, r0 in zip(measurements, res1, res0):
            if shots is None and measure_f in [qp.expval, qp.probs] and batch_size is not None:
                r0 = qp.math.squeeze(r0)
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
@pytest.mark.parametrize("measure_f", [qp.counts, qp.expval, qp.probs, qp.sample, qp.var])
def test_composite_mcms(mcm_method, shots, mcm_name, mcm_func, measure_f, seed):
    """Tests that DefaultQubit handles a circuit with a composite mid-circuit measurement and a
    conditional gate. A single measurement of a composite mid-circuit measurement is performed
    at the end."""

    if mcm_method == "one-shot" and shots is None:
        pytest.skip("`mcm_method='one-shot'` is incompatible with analytic mode (`shots=None`)")

    if measure_f in (qp.counts, qp.sample) and shots is None:
        pytest.skip("Can't measure counts/sample in analytic mode (`shots=None`)")

    if measure_f in (qp.expval, qp.var) and mcm_name in ["mix", "all"]:
        pytest.skip(
            "expval/var does not support measuring sequences of measurements or observables."
        )

    if measure_f in (qp.probs,) and mcm_name in ["mix", "all"]:
        pytest.skip(
            "Cannot use qp.probs() when measuring multiple mid-circuit measurements collected using arithmetic operators."
        )

    dev = qp.device("default.qubit", seed=seed)
    param = qp.numpy.array([np.pi / 3, np.pi / 6])

    def func(x, y):
        qp.RX(x, 0)
        m0 = qp.measure(0)
        qp.RX(0.5 * x, 1)
        m1 = qp.measure(1)
        qp.cond((m0 + m1) == 2, qp.RY)(2.0 * x, 0)
        m2 = qp.measure(0)
        obs = (
            mcm_func(m2)
            if mcm_name == "single"
            else (mcm_func(m0, m1) if mcm_name == "mix" else mcm_func(m0, m1, m2))
        )
        return measure_f(op=obs)

    results0 = qp.set_shots(qp.QNode(func, dev, mcm_method=mcm_method), shots=shots)(*param)
    results1 = qp.set_shots(qp.QNode(func, dev, mcm_method="deferred"), shots=shots)(*param)

    mcm_utils.validate_measurements(measure_f, shots, results1, results0)


@pytest.mark.parametrize("shots", [5000])
@pytest.mark.parametrize("postselect", [None, 0, 1])
@pytest.mark.parametrize("reset", [False, True])
@pytest.mark.parametrize("measure_f", [qp.expval])
def composite_mcm_gradient_measure_obs(shots, postselect, reset, measure_f, seed):
    """Tests that DefaultQubit can differentiate a circuit with a composite mid-circuit
    measurement and a conditional gate. A single measurement of a common observable is
    performed at the end."""

    dev = qp.device("default.qubit", seed=seed)
    param = qp.numpy.array([np.pi / 3, np.pi / 6])
    obs = qp.PauliZ(0) @ qp.PauliZ(1)

    @qp.set_shots(shots)
    @qp.qnode(dev, diff_method="parameter-shift")
    def func(x, y):
        qp.RX(x, 0)
        m0 = qp.measure(0)
        qp.RX(y, 1)
        m1 = qp.measure(1, reset=reset, postselect=postselect)
        qp.cond((m0 + m1) == 2, qp.RY)(2 * np.pi / 3, 0)
        qp.cond((m0 + m1) > 0, qp.RY)(2 * np.pi / 3, 1)
        return measure_f(op=obs)

    func1 = func
    func2 = qp.defer_measurements(func)

    results1 = func1(*param)
    results2 = func2(*param)

    mcm_utils.validate_measurements(measure_f, shots, results1, results2)

    grad1 = qp.grad(func)(*param)
    grad2 = qp.grad(func2)(*param)

    assert np.allclose(grad1, grad2, atol=0.01, rtol=0.3)


@pytest.mark.parametrize("mcm_method", ["one-shot", "tree-traversal"])
def test_sample_with_broadcasting_and_postselection_error(mcm_method, seed):
    """Test that an error is raised if returning qp.sample if postselecting with broadcasting"""
    tape = qp.tape.QuantumScript(
        [qp.RX([0.1, 0.2], 0), MidMeasure(0, postselect=1)], [qp.sample(wires=0)], shots=10
    )
    with pytest.raises(ValueError, match="Returning qp.sample is not supported when"):
        qp.transforms.dynamic_one_shot(tape)

    dev = qp.device("default.qubit", seed=seed)

    @qp.set_shots(10)
    @qp.qnode(dev, mcm_method=mcm_method)
    def circuit(x):
        qp.RX(x, 0)
        qp.measure(0, postselect=1)
        return qp.sample(wires=0)

    with pytest.raises(ValueError, match="Returning qp.sample is not supported when"):
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

        dev = qp.device("default.qubit", seed=PRNGKey(seed))
        params = [np.pi / 4, np.pi / 3]
        obs = qp.PauliZ(0) @ qp.PauliZ(1)

        def func(x, y):
            obs_tape(x, y, None, postselect=postselect)
            return qp.sample(op=obs)

        func0 = qp.set_shots(qp.QNode(func, dev, mcm_method=mcm_method), shots=shots)
        results0 = func0(*params)
        results1 = qp.set_shots(qp.QNode(func, dev, mcm_method="deferred"), shots=shots)(*params)

        mcm_utils.validate_measurements(qp.sample, shots, results1, results0, batch_size=None)

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

        dev = qp.device("default.qubit", seed=jax.random.PRNGKey(seed))
        params = [np.pi / 2.5, np.pi / 3, -np.pi / 3.5]
        obs = qp.PauliY(0)

        @qp.set_shots(shots)
        @qp.qnode(dev, diff_method=diff_method)
        def func(x, y, z):
            m0, m1 = obs_tape(x, y, z, reset=reset, postselect=postselect)
            return (
                qp.probs(wires=[1]),
                qp.probs(wires=[0, 1]),
                qp.sample(wires=[1]),
                qp.sample(wires=[0, 1]),
                qp.expval(obs),
                qp.probs(obs),
                qp.sample(obs),
                qp.var(obs),
                qp.expval(op=m0 + 2 * m1),
                qp.probs(op=m0),
                qp.sample(op=m0 + 2 * m1),
                qp.var(op=m0 + 2 * m1),
                qp.probs(op=[m0, m1]),
            )

        func1 = func
        results1 = func1(*params)

        jaxpr = str(jax.make_jaxpr(func)(*params))
        assert "pure_callback" not in jaxpr

        func2 = jax.jit(func)
        results2 = func2(*params)

        measures = [
            qp.probs,
            qp.probs,
            qp.sample,
            qp.sample,
            qp.expval,
            qp.probs,
            qp.sample,
            qp.var,
            qp.expval,
            qp.probs,
            qp.sample,
            qp.var,
            qp.probs,
        ]
        for measure_f, r1, r2 in zip(measures, results1, results2):
            r1, r2 = np.array(r1).ravel(), np.array(r2).ravel()
            if measure_f == qp.sample:
                r2 = r2[r2 != fill_in_value]
            np.allclose(r1, r2)

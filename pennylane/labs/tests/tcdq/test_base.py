# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the TCDQSimulator base class and the estimator contract."""

from functools import partial

import numpy as np
import pytest

from pennylane.labs.tcdq import (
    Estimator,
    EstimatorSpec,
    IQPSimulator,
    MMDConfig,
    ObservableAlgebra,
    QuditIQPSimulator,
    QuditMMDConfig,
    TCDQSimulator,
    build_mmd_loss,
    build_qudit_mmd_loss,
    create_local_gates,
    estimator,
)

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

jax.config.update("jax_enable_x64", True)


def _iqp_simulator(n_qubits=4, n_samples=512):
    """Return a small qubit IQP simulator."""
    return IQPSimulator(
        gates=create_local_gates(n_qubits, max_weight=2),
        n_qubits=n_qubits,
        n_samples=n_samples,
        key=jax.random.PRNGKey(0),
    )


def _qudit_simulator(dims=3, n_qudits=2, n_samples=512):
    """Return a small qudit IQP simulator."""
    return QuditIQPSimulator(
        dims=dims,
        n_qudits=n_qudits,
        gates={0: [[1, 0]], 1: [[0, 1]]},
        n_samples=n_samples,
        key=jax.random.PRNGKey(0),
    )


class _ZOnlySimulator(TCDQSimulator):
    """A minimal external simulator that only measures diagonal observables."""

    def __init__(self, n_qubits):
        self._n_qubits = n_qubits

    @property
    def local_dims(self):
        return (2,) * self._n_qubits

    @estimator("z_expval", algebra=ObservableAlgebra.PAULI_Z)
    def _build_z_expval(self):
        n_qubits = self._n_qubits

        def z_expval(params, observables, *, key=None, n_samples=None, phase_params=None):
            # pylint: disable=unused-argument
            ops = jnp.asarray(observables)
            weights = jnp.sum(ops == 3, axis=1)
            values = jnp.cos(jnp.sum(params) * weights / n_qubits)
            return values, jnp.zeros_like(values)

        return z_expval


class _HWOnlySimulator(TCDQSimulator):
    """A minimal external simulator declaring the Heisenberg-Weyl algebra."""

    def __init__(self, dims):
        self._dims = tuple(dims)

    @property
    def local_dims(self):
        return self._dims

    @estimator("hw", algebra=ObservableAlgebra.HEISENBERG_WEYL)
    def _build_hw(self):
        def hw(params, observables, *, key=None, n_samples=None, phase_params=None):
            # pylint: disable=unused-argument
            l_vecs = jnp.asarray(observables[0])
            n_obs = l_vecs.shape[0]
            values = jnp.full((n_obs,), jnp.sum(params) * 0.0 + 0.5, dtype=jnp.complex128)
            return values, jnp.zeros((n_obs, 2, 2))

        return hw


class TestEstimatorSpec:
    """Tests for the estimator specification."""

    def test_n_wires_derived_from_local_dims(self):
        """n_wires is inferred from local_dims rather than stored separately."""
        spec = EstimatorSpec("e", ObservableAlgebra.PAULI, (2, 3, 4))
        assert spec.n_wires == 3

    def test_spec_is_hashable(self):
        """Specs are frozen and hashable."""
        spec = EstimatorSpec("e", ObservableAlgebra.PAULI, (2, 2))
        assert hash(spec) == hash(EstimatorSpec("e", ObservableAlgebra.PAULI, (2, 2)))


class TestSimulatorRegistration:
    """Tests for estimator registration on TCDQSimulator subclasses."""

    def test_available_estimators(self):
        """Decorated methods are discoverable by name."""
        assert _iqp_simulator().available_estimators() == ("pauli_expval",)
        assert _qudit_simulator().available_estimators() == ("hw_expval",)
        assert _ZOnlySimulator(3).available_estimators() == ("z_expval",)

    def test_registries_are_isolated_between_subclasses(self):
        """One subclass's estimators do not leak into another's registry."""
        assert "z_expval" not in _iqp_simulator().available_estimators()
        assert "pauli_expval" not in _ZOnlySimulator(3).available_estimators()

    def test_unknown_estimator_raises(self):
        """Requesting an unregistered estimator lists the available ones."""
        with pytest.raises(ValueError, match="has no estimator 'nope'"):
            _iqp_simulator().build_estimator("nope")

    def test_build_estimator_returns_spec(self):
        """The built estimator carries a spec describing what it measures."""
        est = _iqp_simulator(n_qubits=4).build_estimator("pauli_expval")
        assert isinstance(est, Estimator)
        assert est.spec.name == "pauli_expval"
        assert est.spec.algebra is ObservableAlgebra.PAULI
        assert est.spec.local_dims == (2, 2, 2, 2)
        assert est.spec.n_wires == 4

    def test_qudit_spec_reports_mixed_dims(self):
        """Mixed local dimensions are reported faithfully."""
        est = _qudit_simulator(dims=[2, 5]).build_estimator("hw_expval")
        assert est.spec.local_dims == (2, 5)
        assert est.spec.algebra is ObservableAlgebra.HEISENBERG_WEYL

    def test_n_wires_matches_local_dims(self):
        """The simulator's n_wires is derived from local_dims."""
        sim = _iqp_simulator(n_qubits=5)
        assert sim.n_wires == len(sim.local_dims) == 5

    def test_abstract_base_cannot_be_instantiated(self):
        """local_dims is abstract."""
        with pytest.raises(TypeError, match="abstract"):
            TCDQSimulator()  # pylint: disable=abstract-class-instantiated


class TestEstimatorIsJitStatic:
    """An Estimator must be usable as a static argument to jax.jit."""

    def test_estimator_is_hashable(self):
        """Frozen dataclass over a closure hashes by closure identity."""
        est = _iqp_simulator().build_estimator("pauli_expval")
        assert hash(est) is not None
        assert est == est  # pylint: disable=comparison-with-itself

    def test_rebuilt_estimators_are_distinct(self):
        """Separate builds wrap distinct closures and so compare unequal."""
        sim = _iqp_simulator()
        assert sim.build_estimator("pauli_expval") != sim.build_estimator("pauli_expval")

    def test_usable_as_static_argname(self):
        """A jitted function can take an estimator as a static argument."""
        sim = _iqp_simulator()
        est = sim.build_estimator("pauli_expval")
        obs = jnp.array([[3, 3, 0, 0], [0, 0, 3, 3]])

        @partial(jax.jit, static_argnames=["est"])
        def run(params, est):
            return est(params, obs)[0]

        values = run(jnp.zeros(len(sim.gates)), est)
        assert values.shape == (2,)


class TestCallTimeObservables:
    """Observables are always supplied at call time."""

    def test_qubit_estimator_requires_observables(self):
        """The estimator takes observables positionally."""
        sim = _iqp_simulator()
        est = sim.build_estimator("pauli_expval")
        with pytest.raises(TypeError):
            est(jnp.zeros(len(sim.gates)))

    def test_different_observables_per_call(self):
        """Successive calls may use different observable batches."""
        sim = _iqp_simulator()
        est = sim.build_estimator("pauli_expval")
        params = jnp.zeros(len(sim.gates))

        one, _ = est(params, jnp.array([[3, 3, 0, 0]]))
        two, _ = est(params, jnp.array([[3, 3, 0, 0], [0, 0, 3, 3]]))

        assert one.shape == (1,)
        assert two.shape == (2,)
        np.testing.assert_allclose(one[0], two[0], atol=1e-12)


class TestMMDLossCompatibilityChecks:
    """The MMD losses validate the estimator they are handed."""

    def test_accepts_pauli_estimator(self):
        """A PAULI estimator satisfies the Pauli-Z requirement."""
        sim = _iqp_simulator()
        loss_fn = build_mmd_loss(
            sim.build_estimator("pauli_expval"), MMDConfig(bandwidth=1.0, n_ops=8)
        )
        target = np.random.default_rng(0).binomial(1, 0.5, size=(20, 4))
        assert loss_fn(jnp.zeros(len(sim.gates)), target, jax.random.PRNGKey(1)).shape == ()

    def test_accepts_external_pauli_z_estimator(self):
        """An externally defined PAULI_Z simulator works with the same loss."""
        est = _ZOnlySimulator(4).build_estimator("z_expval")
        loss_fn = build_mmd_loss(est, MMDConfig(bandwidth=1.0, n_ops=8))
        target = np.random.default_rng(0).binomial(1, 0.5, size=(20, 4))
        assert loss_fn(jnp.array([0.3]), target, jax.random.PRNGKey(1)).shape == ()

    def test_rejects_heisenberg_weyl_estimator(self):
        """A Heisenberg-Weyl estimator is refused with an explanatory message."""
        est = _qudit_simulator().build_estimator("hw_expval")
        with pytest.raises(TypeError, match="must declare 'pauli_z' or 'pauli'"):
            build_mmd_loss(est, MMDConfig(bandwidth=1.0, n_ops=8))

    def test_rejects_non_estimator(self):
        """A bare callable is refused."""
        with pytest.raises(TypeError, match="expects a tcdq Estimator"):
            build_mmd_loss(lambda *a, **k: None, MMDConfig(bandwidth=1.0, n_ops=8))

    def test_rejects_non_qubit_local_dims(self):
        """A Pauli estimator over non-qubits is refused."""
        spec = EstimatorSpec("fake", ObservableAlgebra.PAULI_Z, (2, 3))
        est = Estimator(spec=spec, fn=lambda *a, **k: None)
        with pytest.raises(ValueError, match="defined over qubits"):
            build_mmd_loss(est, MMDConfig(bandwidth=1.0, n_ops=8))

    def test_qudit_loss_accepts_external_hw_estimator(self):
        """An externally defined Heisenberg-Weyl simulator works with the qudit loss."""
        est = _HWOnlySimulator((3, 3)).build_estimator("hw")
        loss_fn = build_qudit_mmd_loss(est, QuditMMDConfig(bandwidth=0.7, n_ops=8))
        target = jnp.array(np.random.default_rng(0).integers(0, 3, size=(20, 2)), dtype=jnp.int32)
        assert loss_fn(jnp.array([0.1]), target, jax.random.PRNGKey(1)).shape == ()

    def test_qudit_loss_rejects_pauli_estimator(self):
        """A Pauli estimator is refused by the qudit loss."""
        est = _iqp_simulator().build_estimator("pauli_expval")
        with pytest.raises(TypeError, match="must declare 'heisenberg_weyl'"):
            build_qudit_mmd_loss(est, QuditMMDConfig(bandwidth=0.7, n_ops=8))

    def test_qudit_loss_rejects_non_estimator(self):
        """A bare callable is refused by the qudit loss."""
        with pytest.raises(TypeError, match="expects a tcdq Estimator"):
            build_qudit_mmd_loss(lambda *a, **k: None, QuditMMDConfig(bandwidth=0.7, n_ops=8))


class TestNewApiMatchesLegacy:
    """The simulator API and the deprecated config API agree numerically."""

    def test_qubit_expval_matches(self):
        """IQPSimulator reproduces build_expval_func."""
        from pennylane.labs.tcdq import (  # pylint: disable=import-outside-toplevel
            CircuitConfig,
            build_expval_func,
        )

        gates = create_local_gates(4, max_weight=2)
        obs = jnp.array([[3, 3, 0, 0], [1, 0, 2, 3]])
        params = jnp.linspace(-0.5, 0.5, len(gates))
        config = CircuitConfig(
            gates=gates,
            n_samples=1024,
            key=jax.random.PRNGKey(3),
            n_qubits=4,
            observables=obs,
        )

        legacy, _ = build_expval_func(config)(params)
        new, _ = IQPSimulator(
            gates=gates, n_qubits=4, n_samples=1024, key=jax.random.PRNGKey(3)
        ).build_estimator("pauli_expval")(params, obs)

        np.testing.assert_array_equal(legacy, new)

    def test_qubit_mmd_matches(self):
        """build_mmd_loss reproduces the deprecated mmd_loss."""
        from pennylane.labs.tcdq import (  # pylint: disable=import-outside-toplevel
            CircuitConfig,
            mmd_loss,
        )

        gates = create_local_gates(4, max_weight=2)
        params = jnp.linspace(-0.5, 0.5, len(gates))
        target = np.random.default_rng(7).binomial(1, 0.4, size=(50, 4))
        config = CircuitConfig(gates=gates, n_samples=1024, key=jax.random.PRNGKey(3), n_qubits=4)
        mmd_config = MMDConfig(bandwidth=[0.8, 1.6], n_ops=16)

        legacy = mmd_loss(params, config, mmd_config, target, key=jax.random.PRNGKey(9))
        new = build_mmd_loss(
            IQPSimulator(
                gates=gates, n_qubits=4, n_samples=1024, key=jax.random.PRNGKey(3)
            ).build_estimator("pauli_expval"),
            mmd_config,
        )(params, target, jax.random.PRNGKey(9))

        np.testing.assert_array_equal(legacy, new)

    def test_qudit_mmd_matches(self):
        """The estimator-based qudit loss reproduces the config-based one."""
        from pennylane.labs.tcdq import (  # pylint: disable=import-outside-toplevel
            QuditCircuitConfig,
        )

        gates = {0: [[1, 0]], 1: [[0, 1]]}
        params = jnp.array([0.2, -0.1])
        target = jnp.array(np.random.default_rng(11).integers(0, 3, size=(40, 2)), dtype=jnp.int32)
        mmd_config = QuditMMDConfig(bandwidth=[0.4, 1.2], n_ops=12)

        config = QuditCircuitConfig(
            dims=3, n_qudits=2, gates=gates, n_samples=1024, key=jax.random.PRNGKey(3)
        )
        legacy = build_qudit_mmd_loss(config, mmd_config)(params, target, key=jax.random.PRNGKey(9))

        new = build_qudit_mmd_loss(
            QuditIQPSimulator(
                dims=3, n_qudits=2, gates=gates, n_samples=1024, key=jax.random.PRNGKey(3)
            ).build_estimator("hw_expval"),
            mmd_config,
        )(params, target, jax.random.PRNGKey(9))

        np.testing.assert_array_equal(legacy, new)


class TestSimulatorValidation:
    """Constructor-level validation."""

    @pytest.mark.parametrize("n_samples", [0, 1, -3])
    def test_qubit_rejects_small_n_samples(self, n_samples):
        """n_samples must exceed 1 for the variance estimate to be defined."""
        with pytest.raises(ValueError, match="n_samples must be greater than 1"):
            IQPSimulator(
                gates={0: [[0]]}, n_qubits=1, n_samples=n_samples, key=jax.random.PRNGKey(0)
            )

    @pytest.mark.parametrize("n_samples", [0, 1, -3])
    def test_qudit_rejects_small_n_samples(self, n_samples):
        """n_samples must exceed 1 for the qudit simulator too."""
        with pytest.raises(ValueError, match="n_samples must be greater than 1"):
            QuditIQPSimulator(
                dims=3,
                n_qudits=1,
                gates={0: [[1]]},
                n_samples=n_samples,
                key=jax.random.PRNGKey(0),
            )

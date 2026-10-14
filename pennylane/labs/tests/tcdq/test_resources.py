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
"""Tests for the resource estimation of one shot of a TCDQ circuit."""

import numpy as np
import pytest

import pennylane.labs.estimator_beta as qre
from pennylane.labs.estimator_beta.templates import tcdq_sampling as ts
from pennylane.labs.tcdq import (
    CircuitConfig,
    QuditCircuitConfig,
    create_local_gates,
    create_random_gates,
    shot_resources,
)
from pennylane.templates.state_preparations.sum_of_slaters import select_sos_rows

jax = pytest.importorskip("jax")


def _qubit_config(n, gates, **kwargs):
    return CircuitConfig(gates=gates, n_samples=10, key=jax.random.PRNGKey(0), n_qubits=n, **kwargs)


def _qudit_config(dims, n, gates, **kwargs):
    return QuditCircuitConfig(
        dims=dims, n_qudits=n, gates=gates, n_samples=10, key=jax.random.PRNGKey(0), **kwargs
    )


def _random_states(num_states, n, high, seed):
    """``num_states`` distinct dit-strings of length ``n`` with entries below ``high``."""
    rng = np.random.default_rng(seed)
    states = set()
    while len(states) < num_states:
        states.add(tuple(int(v) for v in rng.integers(0, high, size=n)))
    return np.array(sorted(states))


class TestQubitConfig:
    """CircuitConfig is priced as an IQPSamplingShot."""

    @pytest.mark.parametrize("n, w", [(6, 1), (10, 2), (8, 3)])
    def test_default_input_state(self, n, w):
        """No input state is |0>: a basis state, and the gates of create_local_gates are all
        gates up to weight w."""
        op = shot_resources(_qubit_config(n, create_local_gates(n, w)))
        expected = qre.IQPSamplingShot(n, ts.iqp_parity_weights_all_gates(n, w), 1, "arbitrary")
        assert op.resource_rep_from_op() == expected.resource_rep_from_op()

    def test_gates_merge_by_support(self):
        """Random gates are priced through their merged parity-weight histogram."""
        gates = create_random_gates(12, 40, max_weight=3)
        op = shot_resources(_qubit_config(12, gates))
        assert op.resource_params["parity_weights"] == qre.iqp_parity_weights_from_gates(gates)

    @pytest.mark.parametrize(
        "amps, init_state",
        [
            (np.ones(32), "uniform_real"),
            (np.full(32, 0.5 + 0.5j), "arbitrary"),
            (np.linspace(1, 2, 32), "arbitrary"),
        ],
    )
    def test_input_state(self, amps, init_state):
        """N and the amplitude type are read off the config, and the identification bits are
        computed exactly from the basis states."""
        elems = _random_states(32, 16, 2, seed=1)
        op = shot_resources(
            _qubit_config(
                16, create_local_gates(16, 1), init_state_elems=elems, init_state_amps=amps
            )
        )
        assert op.resource_params["num_init_states"] == 32
        assert op.resource_params["init_state"] == init_state
        assert op.resource_params["num_bits"] == len(select_sos_rows(elems.T)[0])

    def test_single_input_state(self):
        """One basis state needs no identification register."""
        op = shot_resources(
            _qubit_config(
                4, {0: [[0]]}, init_state_elems=np.array([[1, 0, 1, 0]]), init_state_amps=[1.0]
            )
        )
        assert op.resource_params["num_init_states"] == 1
        assert op.resource_params["num_bits"] is None

    def test_knobs(self):
        """select_swap_depth and precision reach the shot."""
        op = shot_resources(
            _qubit_config(6, create_local_gates(6, 2)), select_swap_depth=4, precision=1e-6
        )
        assert op.resource_params["select_swap_depth"] == 4
        assert op.resource_params["precision"] == 1e-6


class TestQuditConfig:
    """QuditCircuitConfig is priced as a QuditSamplingShot."""

    def test_mixed_dims(self):
        """Per-site dimensions and the merged Walsh parities are read off the config."""
        dims = [3, 4, 5, 4]
        gates = {0: [[1, 0, 0, 0]], 1: [[0, 2, 3, 0]], 2: [[2, 0, 0, 1]]}
        op = shot_resources(_qudit_config(dims, 4, gates))
        expected = qre.QuditSamplingShot(
            dims, qre.qudit_parity_weights_from_gates(gates, dims), 1, "arbitrary"
        )
        assert op.resource_rep_from_op() == expected.resource_rep_from_op()

    def test_input_state_binary_encoding(self):
        """The identification bits are computed on the binary encoding of the qudit labels."""
        elems = _random_states(20, 6, 4, seed=2)
        op = shot_resources(
            _qudit_config(
                4,
                6,
                {0: [[1, 0, 0, 0, 0, 0]]},
                init_state_elems=elems,
                init_state_amps=np.ones(20),
            )
        )
        bits = np.array([(elems[:, q] >> j) & 1 for q in range(6) for j in range(2)])
        assert op.resource_params["num_init_states"] == 20
        assert op.resource_params["init_state"] == "uniform_real"
        assert op.resource_params["num_bits"] == len(select_sos_rows(bits)[0])


class TestPhaseFn:
    """A phase_fn must be priced explicitly."""

    @pytest.mark.parametrize(
        "config",
        [
            _qubit_config(4, {0: [[0]]}, phase_fn=lambda p, z: p[0] * z.sum()),
            _qudit_config(3, 2, {0: [[1, 0]]}, phase_fn=lambda p, z: p[0] * z.sum()),
        ],
    )
    def test_phase_fn_without_phase_op_raises(self, config):
        """The cost of a Python phase function cannot be inferred."""
        with pytest.raises(ValueError, match="Pass the cost of the phase layer as phase_op"):
            shot_resources(config)

    def test_phase_op(self):
        """The phase layer passed as phase_op is part of the shot."""
        config = _qubit_config(4, {0: [[0]]}, phase_fn=lambda p, z: p[0] * z.sum())
        phase = qre.SumPhaseLayer(2, degree=1, num_qudits=4)
        op = shot_resources(config, phase_op=phase)
        assert op.resource_params["phase_cmpr_op"] == phase.resource_rep_from_op()


def test_invalid_config():
    """Only the two tcdq config classes are accepted."""
    with pytest.raises(TypeError, match="CircuitConfig or a QuditCircuitConfig"):
        shot_resources({"gates": {}})

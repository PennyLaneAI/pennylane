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
"""Compose fault-tolerant resource estimates for a TCDQ sampling shot."""

from collections import Counter

import pennylane.labs.estimator_beta as qre
from pennylane import math
from pennylane.labs.estimator_beta.templates import tcdq_sampling as sampling
from pennylane.templates.state_preparations.sum_of_slaters import select_sos_rows

from .expval_functions import CircuitConfig
from .qudit_expval_functions import QuditCircuitConfig, _dims_to_numpy, _normalize_sparse_gate


def _input_state(amps, bits, num_wires, select_swap_depth):
    """Prepare a sparse input, using the uniform model only for exactly equal real amplitudes."""
    if amps is None or bits is None:
        return qre.Identity()
    if len(amps) == 1:
        flips = int(math.sum(bits))
        return qre.Prod([(qre.X(), flips)]) if flips else qre.Identity()
    amps = math.to_numpy(amps)
    num_states = len(amps)
    uniform = bool(math.all(math.imag(amps) == 0) and math.all(amps == amps[0]))
    condensed = (
        qre.UniformStatePrep(num_states)
        if uniform
        else qre.QROMStatePreparation((num_states - 1).bit_length())
    )
    return qre.LabsSumOfSlatersPrep(
        num_states,
        num_wires,
        num_bits=len(select_sos_rows(bits)[0]),
        stateprep_op=condensed,
        select_swap_depth=select_swap_depth,
    )


def shot_resources(
    config: CircuitConfig | QuditCircuitConfig,
    phase_op=None,
    select_swap_depth=None,
    precision=None,
    params=None,
):
    """Build a resource-operator product for one computational-basis sampling shot.

    Args:
        config (CircuitConfig | QuditCircuitConfig): circuit structure and sparse input state
        phase_op (ResourceOperator | None): supplied cost of phase_fn; required when phase_fn is set
        select_swap_depth (int | None): input-preparation QROM select-swap depth
        precision (float | None): diagonal rotation precision; None uses configured MultiRZ/RZ
        params (TensorLike | None): trained angles, indexed as config.gates; None uses ordinary RZ

    Returns:
        ResourceOperator: a product accepted by pennylane.labs.estimator_beta.estimate

    Known angles use expected T counts from mixed synthesis. Custom RZ/MultiRZ
    decompositions override this model. Precision is per rotation, not a total error budget.
    See :mod:`pennylane.labs.estimator_beta.templates.tcdq_sampling` for model assumptions.

    >>> import jax
    >>> import pennylane.labs.estimator_beta as qre
    >>> from pennylane.labs.tcdq import CircuitConfig, shot_resources
    >>> config = CircuitConfig(gates={0: [[0]]}, n_samples=1, key=jax.random.PRNGKey(0),
    ...                        n_qubits=1)
    >>> qre.estimate(shot_resources(config, params=[1e-5], precision=1e-9)).gate_counts["T"]
    10
    """
    if not isinstance(config, (CircuitConfig, QuditCircuitConfig)):
        raise TypeError("config must be a CircuitConfig or QuditCircuitConfig.")
    if config.phase_fn is not None and phase_op is None:
        raise ValueError("Pass the cost of the phase layer as phase_op when phase_fn is set.")
    elems = config.init_state_elems
    if isinstance(config, QuditCircuitConfig):
        dims = [int(d) for d in _dims_to_numpy(config.dims, config.n_qudits)]
        widths = [(d - 1).bit_length() for d in dims]
        num_wires = sum(widths)
        elems = None if elems is None else math.to_numpy(elems).astype(int)
        bits = (
            None
            if elems is None
            else math.stack(
                [(elems[:, q] >> j) & 1 for q, width in enumerate(widths) for j in range(width)]
            )
        )
        gates = {
            k: [dict(zip(*_normalize_sparse_gate(g, config.n_qudits, dims))) for g in group]
            for k, group in config.gates.items()
        }
        parities = sampling.qudit_parities_from_gates(gates, dims, params)
        weights = Counter(sum(mask.bit_count() for _, mask in key) for key in parities)
        transforms = [(sampling.fourier_resources(d), count) for d, count in Counter(dims).items()]
        transforms += [
            (qre.Adjoint(sampling.fourier_resources(d)), count)
            for d, count in Counter(dims).items()
        ]
    else:
        num_wires = config.n_qubits
        bits = None if elems is None else math.to_numpy(elems).astype(int).T
        parities = sampling.iqp_parities_from_gates(config.gates, params, num_wires)
        weights = Counter(map(len, parities))
        transforms = [(qre.Hadamard(), 2 * num_wires)]
    state = _input_state(config.init_state_amps, bits, num_wires, select_swap_depth)
    layer = qre.ParityPhaseLayer(
        num_wires, weights, precision, None if params is None else list(parities.values())
    )
    factors = [state, layer, *transforms]
    if phase_op is not None:
        factors.append(phase_op)
    return qre.Prod(factors)

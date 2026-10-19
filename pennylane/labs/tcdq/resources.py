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
"""Resource estimation of one shot of a TCDQ circuit on a fault-tolerant quantum computer."""

import pennylane.labs.estimator_beta as qre
from pennylane import math
from pennylane.templates.state_preparations.sum_of_slaters import select_sos_rows

from .expval_functions import CircuitConfig
from .qudit_expval_functions import QuditCircuitConfig, _dims_to_numpy


def _input_state(elems, amps, bits):
    """``(num_init_states, init_state, num_bits)`` of the input state of a config."""
    if elems is None or amps is None:
        return 1, "arbitrary", None
    amps = math.to_numpy(amps)
    uniform = math.allclose(math.imag(amps), 0) and math.allclose(amps, amps[0])
    init_state = "uniform_real" if uniform else "arbitrary"
    num_bits = len(select_sos_rows(bits)[0]) if len(amps) > 1 else None
    return len(amps), init_state, num_bits


def shot_resources(
    config: CircuitConfig | QuditCircuitConfig,
    phase_op=None,
    select_swap_depth: int | None = None,
    precision: float | None = None,
    params=None,
):
    r"""Resource operator for one computational-basis shot of the circuit described by ``config``.

    The returned operator prices preparing the input state, applying the trained circuit once and
    measuring every wire, compiled to qubits. Pass it to
    :func:`~pennylane.labs.estimator_beta.estimate` to obtain the logical qubit and gate counts.

    Args:
        config (CircuitConfig | QuditCircuitConfig): the circuit. Its ``gates`` fix the diagonal
            layer and its ``init_state_elems`` and ``init_state_amps`` fix the input state
            (:math:`|0\rangle` if not given).
        phase_op (ResourceOperator | None): the cost of the trainable phase layer
            :math:`D'(\xi)`, e.g. a :class:`~pennylane.labs.estimator_beta.SumPhaseLayer`. It is
            required when ``config.phase_fn`` is set, since the cost of an arbitrary phase
            function cannot be inferred from it.
        select_swap_depth (int | None): select-swap depth of the QROM in the input-state
            preparation
        precision (float | None): synthesis precision per rotation of the diagonal layer.
            ``None`` uses the configured ``MultiRZ``/``RZ`` precision.
        params (TensorLike | None): the trained parameters, as passed to the expectation-value
            functions: ``params[k]`` is the angle of every gate in ``config.gates[k]``. If given,
            the merged angle of every rotation of the diagonal layer is passed to the shot as
            ``gates_params``, and rotations with small angles are priced with the angle-dependent
            synthesis of `Bothe et al., arXiv:2605.31544 <https://arxiv.org/abs/2605.31544>`__.

    Returns:
        ResourceOperator: an :class:`~pennylane.labs.estimator_beta.IQPSamplingShot` for a
        :class:`~.CircuitConfig` or a :class:`~pennylane.labs.estimator_beta.QuditSamplingShot`
        for a :class:`~.QuditCircuitConfig`

    Raises:
        ValueError: if ``config.phase_fn`` is set and ``phase_op`` is not given
        TypeError: if ``config`` is neither a ``CircuitConfig`` nor a ``QuditCircuitConfig``

    The input state is priced as ``"uniform_real"`` when all amplitudes are real and equal and as
    ``"arbitrary"`` otherwise. The number of qubits needed to tell its basis states apart is
    computed from ``init_state_elems``, as in :class:`~pennylane.SumOfSlatersPrep`.

    **Example**

    >>> import jax
    >>> import pennylane.labs.estimator_beta as qre
    >>> from pennylane.labs.tcdq import CircuitConfig, create_local_gates, shot_resources
    >>> config = CircuitConfig(
    ...     gates=create_local_gates(n_qubits=20, max_weight=2),
    ...     n_samples=1000,
    ...     key=jax.random.PRNGKey(0),
    ...     n_qubits=20,
    ... )
    >>> res = qre.estimate(shot_resources(config))
    >>> res.algo_wires, res.gate_counts["T"]
    (20, 9240)

    Trained parameters close to zero make the circuit cheaper:

    >>> params = jax.random.normal(jax.random.PRNGKey(1), (210,)) * 1e-5
    >>> qre.estimate(shot_resources(config, params=params)).gate_counts["T"]
    1718
    """
    if not isinstance(config, (CircuitConfig, QuditCircuitConfig)):
        raise TypeError(
            f"config must be a CircuitConfig or a QuditCircuitConfig, got {type(config).__name__}."
        )
    if config.phase_fn is not None and phase_op is None:
        raise ValueError(
            "config.phase_fn is set, but its cost cannot be inferred from a Python function. "
            "Pass the cost of the phase layer as phase_op, e.g. qre.SumPhaseLayer for a "
            "function of the digit sum, or remove phase_fn to price the circuit without it."
        )

    elems = config.init_state_elems
    if isinstance(config, QuditCircuitConfig):
        dims = [int(d) for d in _dims_to_numpy(config.dims, config.n_qudits)]
        bits = None
        if elems is not None:
            # binary encoding of the qudit labels: one row per encoding qubit
            elems = math.to_numpy(elems).astype(int)
            bits = math.stack(
                [
                    (elems[:, q] >> j) & 1
                    for q, d in enumerate(dims)
                    for j in range((d - 1).bit_length())
                ]
            )
        num_init_states, init_state, num_bits = _input_state(elems, config.init_state_amps, bits)
        gates_params = None
        if params is not None:
            gates_params = list(qre.qudit_parities_from_gates(config.gates, dims, params).values())
        return qre.QuditSamplingShot(
            dims,
            qre.qudit_parity_weights_from_gates(config.gates, dims),
            num_init_states,
            init_state,
            num_bits=num_bits,
            select_swap_depth=select_swap_depth,
            precision=precision,
            phase_op=phase_op,
            gates_params=gates_params,
        )

    bits = None if elems is None else math.to_numpy(elems).astype(int).T
    num_init_states, init_state, num_bits = _input_state(elems, config.init_state_amps, bits)
    gates_params = None
    if params is not None:
        gates_params = list(qre.iqp_parities_from_gates(config.gates, params).values())
    return qre.IQPSamplingShot(
        config.n_qubits,
        qre.iqp_parity_weights_from_gates(config.gates),
        num_init_states,
        init_state,
        num_bits=num_bits,
        select_swap_depth=select_swap_depth,
        precision=precision,
        phase_op=phase_op,
        gates_params=gates_params,
    )

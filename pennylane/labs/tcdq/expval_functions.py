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
"""Expectation-value estimator for qubit IQP circuits.

This module estimates Pauli expectation values for IQP circuits without
simulating the full quantum state.
"""

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike


@dataclass
class CircuitConfig:  # pylint: disable=too-many-instance-attributes
    """Description of a qubit IQP circuit for classical expectation-value estimation.

    This dataclass bundles all the information needed to build an expectation-value
    estimator via :func:`build_expval_func`: the gate structure, the observables to
    measure, sampling parameters, and an optional non-standard initial state.

    Args:
        gates (dict[int, list[list[int]]]): Circuit structure mapping each
            trainable parameter index to a list of gates. Each gate is itself a
            list of qubit indices that participate in a Pauli-Z tensor-product
            generator. For example, ``{0: [[0, 1]], 1: [[2]]}`` defines two
            parameters: parameter 0 drives a ZZ gate on qubits 0 and 1, and
            parameter 1 drives a Z gate on qubit 2. Use
            :func:`~pennylane.labs.tcdq.create_local_gates` or
            :func:`~pennylane.labs.tcdq.create_lattice_gates` to generate
            these automatically.
        n_samples (int): Number of random bitstrings drawn for the
            estimation.
        key (ArrayLike): JAX PRNG key for random bitstring generation.
        n_qubits (int): Total number of qubits in the circuit.
        observables (ArrayLike | None): Integer array of shape
            ``(n_observables, n_qubits)`` encoding Pauli operators (I=0, X=1,
            Y=2, Z=3). Each row is one observable. If ``None``, observables must
            be passed at call time to the function returned by
            :func:`build_expval_func`.
        init_state_elems (ArrayLike | None): Binary array of shape ``(N, n_qubits)``
            listing the computational-basis states with non-zero amplitude in a
            custom initial state. Use together with ``init_state_amps``. If
            ``None`` (default), the circuit starts in the uniform superposition
            state :math:`H^{\\otimes n}|0\\rangle`.
        init_state_amps (ArrayLike | None): Complex array of shape ``(N,)`` with
            the amplitudes corresponding to ``init_state_elems``.
        phase_fn (Callable | None): Optional custom phase function
            ``phase_fn(params, bitstring)`` applied as an extra diagonal layer.
            Defaults to ``None``.

    **Example**

    >>> import jax
    >>> from pennylane.labs.tcdq import CircuitConfig, create_local_gates
    >>> gates = create_local_gates(n_qubits=4, max_weight=2)
    >>> config = CircuitConfig(
    ...     gates=gates,
    ...     n_samples=2000,
    ...     key=jax.random.PRNGKey(42),
    ...     n_qubits=4,
    ...     observables=[[3, 3, 0, 0], [0, 0, 3, 3]],  # ZZ on (0,1) and ZZ on (2,3)
    ... )

    .. seealso::

        `arXiv:2501.04776 <https://arxiv.org/pdf/2501.04776>`_
    """

    #: Circuit structure mapping parameter indices to lists of gates.
    gates: dict[int, list[list[int]]] = None
    #: Number of random bitstrings drawn for the estimation.
    n_samples: int = None
    #: JAX PRNG key for random bitstring generation.
    key: ArrayLike = None
    #: Total number of qubits in the circuit.
    n_qubits: int = None
    #: Pauli observables encoded as an integer array, or ``None``.
    observables: ArrayLike | None = None
    #: Computational-basis states with non-zero amplitude, or ``None``.
    init_state_elems: ArrayLike | None = None
    #: Amplitudes for the custom initial state, or ``None``.
    init_state_amps: ArrayLike | None = None
    #: Optional custom phase function applied as an extra diagonal layer.
    phase_fn: Callable | None = None


def _parse_generator_dict(circuit_def: dict[int, list[list[int]]], n_qubits: int):
    """Convert a gate dictionary into a binary generator matrix and parameter map.

    Args:
        circuit_def (dict[int, list[list[int]]]): Dictionary mapping parameter indices to
            lists of qubit indices.
        n_qubits (int): Total number of qubits.

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]: Tuple containing:
            - Binary matrix of generators.
            - Integer array mapping each generator to its parameter index.
    """
    flat_gates = []
    param_indices = []

    for param_idx in sorted(circuit_def.keys()):
        gates_for_this_param = circuit_def[param_idx]
        for gate in gates_for_this_param:
            flat_gates.append(gate)
            param_indices.append(param_idx)

    n_gates = len(flat_gates)
    generators = np.zeros((n_gates, n_qubits), dtype=int)

    for i, qubits in enumerate(flat_gates):
        generators[i, qubits] = 1
    param_map = jnp.array(param_indices, dtype=int)
    return jnp.array(generators), param_map


def _compute_samples(key: ArrayLike, n_samples: int, n_qubits: int) -> jnp.ndarray:
    """Generate the random bitstrings used by the Monte Carlo estimator."""
    n_bytes = (n_qubits + 7) // 8
    random_bytes = jax.random.bits(key, shape=(n_samples, n_bytes), dtype=jnp.uint8)
    unpacked_bits = jnp.unpackbits(random_bytes, axis=-1)
    return unpacked_bits[:, :n_qubits]


def _prep_observables(observables_int: ArrayLike) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Precompute masks and phase factors for integer-encoded Pauli observables."""
    obs_arr = jnp.asarray(observables_int, dtype=jnp.int32)

    is_X = obs_arr == 1
    is_Y = obs_arr == 2
    is_Z = obs_arr == 3

    bitflips = jnp.array(is_Z | is_Y, dtype=jnp.int32)
    mask_XY = jnp.array(is_X | is_Y, dtype=jnp.int32)
    count_Y = jnp.array(is_Y.sum(axis=1), dtype=jnp.int32)

    y_phase = (-1j) ** count_Y[:, jnp.newaxis]

    return bitflips, mask_XY, y_phase


# pylint: disable=too-many-arguments
def _core_expval_execution(
    gates_params: ArrayLike,
    phase_fn_params: ArrayLike | None,
    samples: jnp.ndarray,
    obs_data: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    init_state_elems: ArrayLike | None,
    init_state_amps: ArrayLike | None,
    generators: jnp.ndarray,
    param_map: jnp.ndarray,
    vmapped_phase_func: Callable | None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate the Monte Carlo integrand and return means with standard errors."""
    bitflips, mask_XY, y_phase = obs_data

    s_f = samples.astype(jnp.float32)
    m_f = mask_XY.astype(jnp.float32)
    g_f = generators.astype(jnp.float32)
    b_f = bitflips.astype(jnp.float32)

    sign_flip = 1 - 2 * ((m_f @ s_f.T) % 2)
    phases = sign_flip * y_phase

    B = 1 - 2 * ((s_f @ g_f.T) % 2)
    C = 2 * ((b_f @ g_f.T) % 2)
    expanded_params = jnp.asarray(gates_params)[param_map]
    E = (C * expanded_params) @ B.T

    if vmapped_phase_func is not None:
        E += vmapped_phase_func(phase_fn_params, samples, bitflips)

    if init_state_elems is None or init_state_amps is None:
        expvals = jnp.real(phases) * jnp.cos(E) - jnp.imag(phases) * jnp.sin(E)
    else:
        M = phases * jnp.exp(1j * E)
        X = init_state_elems
        P = init_state_amps
        F = P[:, jnp.newaxis] * (1 - 2 * ((X @ samples.T) % 2))
        H1 = (1 - 2 * ((bitflips @ X.T) % 2)) @ F
        col_sums = jnp.sum(F.conj(), axis=0, keepdims=True)
        H = H1 * col_sums
        M = M * H
        expvals = jnp.real(M)

    std_err = jnp.std(expvals, axis=-1, ddof=1) / jnp.sqrt(samples.shape[0])

    return jnp.mean(expvals, axis=1), std_err


def build_expval_func(
    config: CircuitConfig,
) -> Callable:
    """Build an estimator for Pauli expectation values of a qubit IQP circuit.

    Returns a pure function that estimates the expectation value of each Pauli
    observable specified in ``config.observables`` or passed at call time.

    The returned function captures precomputed data from ``config`` (generator
    matrices, default samples, preprocessed observables) so that repeated
    evaluations with different parameters are fast.

    Args:
        config (CircuitConfig): Full circuit description including gate
            structure, observables, and sampling parameters. See
            :class:`CircuitConfig` for details on how to construct one.

    Returns:
        Callable: A function with signature::

            expval_fn(
                gates_params,
                phase_fn_params=None,
                observables=None,
                key=None,
                n_samples=None,
                init_state_elems=None,
                init_state_amps=None,
            ) -> (expvals, std_errs)

        where ``expvals`` is a real array of shape ``(n_observables,)`` and
        ``std_errs`` is the estimated standard error of each expectation value.

    **Example**

    >>> import jax
    >>> import jax.numpy as jnp
    >>> from pennylane.labs.tcdq import CircuitConfig, build_expval_func, create_local_gates
    >>> n_qubits = 4
    >>> gates = create_local_gates(n_qubits, max_weight=2)
    >>> config = CircuitConfig(
    ...     gates=gates,
    ...     n_samples=5000,
    ...     key=jax.random.PRNGKey(0),
    ...     n_qubits=n_qubits,
    ...     observables=[[3, 3, 0, 0], [0, 0, 3, 3]],  # ZZ on (0,1) and (2,3)
    ... )
    >>> expval_fn = jax.jit(build_expval_func(config))
    >>> params = jnp.zeros(len(gates))
    >>> expvals, std_errs = expval_fn(params)
    >>> expvals.shape
    (2,)

    .. seealso::

        :class:`~pennylane.labs.tcdq.CircuitConfig`,
        `Section 2, Classically Estimating Expectation Values <https://github.com/PennyLaneAI/pennylane/blob/port_tcdq_docs_pr/pennylane/labs/tcdq/notes.md#2-classically-estimating-expectation-values>`_
    """
    generators, param_map = _parse_generator_dict(config.gates, config.n_qubits)

    vmapped_phase_func = None
    if config.phase_fn is not None:

        def compute_phase(p_params, sample, b_flips):
            return config.phase_fn(p_params, sample) - config.phase_fn(
                p_params, (sample + b_flips) % 2
            )

        vmapped_phase_func = jax.vmap(
            jax.vmap(compute_phase, in_axes=(None, 0, None)), in_axes=(None, None, 0)
        )

    default_samples = _compute_samples(config.key, config.n_samples, config.n_qubits)
    default_obs_data = None if config.observables is None else _prep_observables(config.observables)

    # pylint: disable=too-many-arguments
    def expval_execution(
        gates_params: ArrayLike,
        phase_fn_params: ArrayLike | None = None,
        observables: ArrayLike | None = None,
        key: ArrayLike | None = None,
        n_samples: int | None = None,
        init_state_elems: ArrayLike | None = None,
        init_state_amps: ArrayLike | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Execute the estimator with optional runtime overrides.

        This closure captures the precomputed matrices and defaults from the
        CircuitConfig, while allowing dynamic injection of new parameters,
        observables, or sampling configurations at execution time.

        Args:
            gates_params (ArrayLike): Trainable parameters $\\theta$ for the circuit gates.
            phase_fn_params (ArrayLike | None, optional): Trainable parameters for the
                custom phase function. Defaults to None.
            observables (ArrayLike | None, optional): Runtime override for the Pauli
                observables (I=0, X=1, Y=2, Z=3). Defaults to None.
            key (ArrayLike | None, optional): Runtime override for the JAX PRNG key
                used for sampling. Defaults to None.
            n_samples (int | None, optional): Runtime override for the number of
                samples. Defaults to None.
            init_state_elems (ArrayLike | None, optional): Runtime override for the
                discrete elements of the initial state (X). Defaults to None.
            init_state_amps (ArrayLike | None, optional): Runtime override for the
                continuous amplitudes of the initial state (P). Defaults to None.

        Returns:
            tuple[jnp.ndarray, jnp.ndarray]: Estimated expectation values and
            their standard errors.
        """
        if key is not None or n_samples is not None:
            _key = key if key is not None else config.key
            _n = n_samples if n_samples is not None else config.n_samples
            samples = _compute_samples(_key, _n, config.n_qubits)
        else:
            samples = default_samples

        if observables is not None:
            obs_data = _prep_observables(observables)
        elif default_obs_data is not None:
            obs_data = default_obs_data
        else:
            raise ValueError(
                "No observables specified. Provide them in CircuitConfig "
                "or pass at call time via the observables argument."
            )

        state_elems = config.init_state_elems if init_state_elems is None else init_state_elems
        state_amps = config.init_state_amps if init_state_amps is None else init_state_amps

        return _core_expval_execution(
            gates_params,
            phase_fn_params,
            samples,
            obs_data,
            state_elems,
            state_amps,
            generators,
            param_map,
            vmapped_phase_func,
        )

    return expval_execution

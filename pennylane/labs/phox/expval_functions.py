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
"""
Pure function implementations for the expectation value functions.
"""

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike


@dataclass
class CircuitConfig:  # pylint: disable=too-many-instance-attributes
    """
    Configuration data for an IQP circuit simulation.

    Args:
        gates (dict[int, list[list[int]]]): Circuit structure mapping parameters to gates.
        observables (ArrayLike): List of Pauli observables mapped to integers (I=0, X=1, Y=2, Z=3).
        n_samples (int): Number of Monte Carlo samples for the estimation of the expectation value.
        key (ArrayLike): Random key for JAX.
        n_qubits (int): Number of qubits.
        init_state_elems (ArrayLike | None): Elements of the initial state (X) - fixed binary
            matrix of shape ``(N, n_qubits)``. Rows must be distinct (i.e. the standard
            ``np.unique``-deduplicated form); duplicates are not detected and produce
            silently incorrect results when ``control_variate`` is enabled.
        init_state_amps (ArrayLike | None): Amplitudes of the initial state (P) - continuous trainable params.
        phase_fn (Callable | None): Optional phase layer function.
        control_variate (bool): If ``True``, apply a closed-form control variate to
            reduce the Monte-Carlo variance of the expectation-value estimator.
            The control variate is the per-sample integrand evaluated at
            ``gates_params == 0`` (and ``phase_fn_params == 0``); its known mean
            is the input-state expectation value :math:`\\langle\\psi_{\\text{in}}|P_a|\\psi_{\\text{in}}\\rangle`,
            computable exactly from the sparse input state. The estimator is
            unbiased and asymptotically equivalent to the standard one, but its
            variance scales as :math:`\\mathcal{O}(\\|\\boldsymbol{\\theta}\\|^{2})`
            rather than :math:`\\Theta(1)` in the small-angle regime, which is
            particularly useful at initialization. Defaults to ``False``.

            .. warning::

                Only supported for batches of Z-type observables (Pauli labels
                in ``{I=0, Z=3}``) and for ``init_state_elems`` with distinct
                rows. Passing X or Y observables, or duplicated input-state
                support points, while ``control_variate`` is ``True`` yields
                silently incorrect offsets; the caller is responsible for
                enforcing both restrictions. They let the implementation
                reduce the offset to an :math:`O(l \\cdot N)` mat-vec and
                avoid the ``(l, N, N)`` matching tensor and ``(N, N)``
                row-equality mask, which otherwise dominate GPU memory for
                sparse input states with many support points.
    """

    gates: dict[int, list[list[int]]]
    observables: ArrayLike
    n_samples: int
    key: ArrayLike
    n_qubits: int
    init_state_elems: ArrayLike | None = None
    init_state_amps: ArrayLike | None = None
    phase_fn: Callable | None = None
    control_variate: bool = False


def bitflip_expval(
    generators: ArrayLike, params: ArrayLike, ops: ArrayLike
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute expectation value for the Bitflip noise model.

    Args:
        generators (ArrayLike): Binary matrix of shape ``(n_generators, n_qubits)``.
        params (ArrayLike): Error probabilities/parameters $\theta$.
        ops (ArrayLike): Binary matrix representing Pauli Z operators.

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]: A tuple containing:
            - Expectation values.
            - A zero array for standard error (since this is analytical).
    """
    probs = jnp.cos(2 * params)

    indicator = (ops @ generators.T) % 2
    X = probs * indicator

    result = jnp.prod(jnp.where(X == 0, 1.0, X), axis=1)

    return result, jnp.zeros(ops.shape[0])


def _parse_generator_dict(circuit_def: dict[int, list[list[int]]], n_qubits: int):
    """
    Converts dictionary circuit definition into matrices.

    Args:
        circuit_def (dict[int, list[list[int]]]): Dictionary mapping parameter indices to lists of qubit indices.
        n_qubits (int): Total number of qubits.

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]: Tuple containing:
            - Binary matrix of generators.
            - Integer array mapping parameters to generators.
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
    """Generates the stochastic sample matrix."""
    n_bytes = (n_qubits + 7) // 8
    random_bytes = jax.random.bits(key, shape=(n_samples, n_bytes), dtype=jnp.uint8)
    unpacked_bits = jnp.unpackbits(random_bytes, axis=-1)
    return unpacked_bits[:, :n_qubits]


def _prep_observables(observables_int: ArrayLike) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Converts integer observables (I=0, X=1, Y=2, Z=3)
    into precomputed bitmasks and y_phases.
    """
    obs_arr = jnp.asarray(observables_int, dtype=jnp.int32)

    is_X = obs_arr == 1
    is_Y = obs_arr == 2
    is_Z = obs_arr == 3

    bitflips = jnp.array(is_Z | is_Y, dtype=jnp.int32)
    mask_XY = jnp.array(is_X | is_Y, dtype=jnp.int32)
    count_Y = jnp.array(is_Y.sum(axis=1), dtype=jnp.int32)

    y_phase = (-1j) ** count_Y[:, jnp.newaxis]

    return bitflips, mask_XY, y_phase


def _cv_mean_z_only(
    bitflips: jnp.ndarray,
    X: jnp.ndarray,
    P: jnp.ndarray,
) -> jnp.ndarray:
    r"""Control-variate mean for all-Z observable batches on a sparse input state.

    With Z-only observables the constraint
    :math:`x_i \oplus x_j = m_{XY}(a) = 0` reduces to :math:`x_i = x_j`.
    Assuming the support points :math:`\{x_i\}` are unique (the contract on
    :class:`CircuitConfig.init_state_elems` when control-variate is enabled),
    the double sum further collapses to

    .. math::

        \langle P_a \rangle \;=\; \sum_i |\psi_{x_i}|^{2}\,(-1)^{b_a\cdot x_i},

    i.e.\ the basis-state probability moment of the Pauli string. The
    computation is :math:`O(l \cdot N)` and never materialises an
    ``(N, N)`` tensor, so it scales to the large sparse input states
    typical of MMD-style training.
    """
    weighted_psi = (P * jnp.conj(P)).real
    coef = 1 - 2 * ((bitflips @ X.T) % 2)
    return coef.astype(weighted_psi.dtype) @ weighted_psi


def _compute_control_variate_mean(
    bitflips: jnp.ndarray,
    init_state_elems: ArrayLike | None,
    init_state_amps: ArrayLike | None,
) -> jnp.ndarray:
    r"""Closed-form input-state expectation values used as control-variate offsets.

    Returns, for each Pauli observable :math:`P_a` in the batch, the exact
    expectation value :math:`\langle\psi_{\text{in}}|P_a|\psi_{\text{in}}\rangle`
    evaluated against the (sparse) input state. This is the mean of the
    per-sample integrand evaluated at ``gates_params = 0`` and
    ``phase_fn_params = 0``, and serves as the closed-form control-variate
    offset when :class:`CircuitConfig.control_variate` is enabled.

    .. note::

        Only supported for batches of Z-type observables (Pauli labels in
        ``{I=0, Z=3}``) and for input states whose support points are
        distinct (the standard ``np.unique``-deduplicated form). The caller
        is responsible for enforcing both restrictions; passing X or Y
        observables, or duplicated rows in ``init_state_elems``, yields
        silently incorrect offsets. These restrictions reduce the offset
        computation to an :math:`O(l \cdot N)` mat-vec and avoid the
        ``(l, N, N)`` matching tensor and ``(N, N)`` row-equality mask of
        the general path, which otherwise blow up memory and exceed the
        CUDA grid-size limit on GPU for sparse input states with many
        support points.

    For :math:`|0\dots0\rangle` input (``init_state_*`` is ``None``) the
    expectation value of any Z-only Pauli string is ``1``. For a sparse
    data state with unique support
    :math:`|\psi_{\text{in}}\rangle = \sum_{x\in\mathcal X}\psi_x|x\rangle`,
    the closed form is

    .. math::

        \langle P_a\rangle \;=\; \sum_{x\in\mathcal X}|\psi_x|^{2}\,(-1)^{b_a\cdot x},

    derived by averaging the per-sample integrand over the uniform
    distribution on :math:`\boldsymbol{z}\in\{0,1\}^n` and specialising
    to the all-Z, unique-support case.

    Args:
        bitflips: Integer ``(l, n)`` mask, ``1`` at sites where the Pauli is Z.
        init_state_elems: ``(N, n)`` sparse support of the input state with
            distinct rows, or ``None``.
        init_state_amps: ``(N,)`` amplitudes of the input state, or ``None``.

    Returns:
        Real array of shape ``(l,)`` with the exact input-state expectation
        value of each observable.
    """
    if init_state_elems is None or init_state_amps is None:
        return jnp.ones(bitflips.shape[0])

    X = jnp.asarray(init_state_elems)
    P = jnp.asarray(init_state_amps)
    return _cv_mean_z_only(bitflips, X, P)


# pylint: disable=too-many-arguments,too-many-locals
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
    use_control_variate: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """The pure mathematical core of the expectation value computation.

    When ``use_control_variate`` is ``True``, the per-sample integrand is
    centered by subtracting its closed-form value at zero parameters
    (``phases * H``); the exact mean of that subtraction is then added back as
    a constant offset. The result is an unbiased estimator with strictly lower
    variance whenever the centered integrand is positively correlated with
    the original, and asymptotically ``O(\\|theta\\|^2)`` variance instead of
    ``Theta(1)`` near zero parameters.
    """
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

    if use_control_variate:
        offset = _compute_control_variate_mean(bitflips, init_state_elems, init_state_amps)
    else:
        offset = jnp.zeros(bitflips.shape[0])

    if init_state_elems is None or init_state_amps is None:
        if use_control_variate:
            cos_term = jnp.cos(E) - 1
        else:
            cos_term = jnp.cos(E)
        expvals = jnp.real(phases) * cos_term - jnp.imag(phases) * jnp.sin(E)
    else:
        if use_control_variate:
            phase_factor = jnp.exp(1j * E) - 1
        else:
            phase_factor = jnp.exp(1j * E)
        M = phases * phase_factor
        X = init_state_elems
        P = init_state_amps
        F = P[:, jnp.newaxis] * (1 - 2 * ((X @ samples.T) % 2))
        H1 = (1 - 2 * ((bitflips @ X.T) % 2)) @ F
        col_sums = jnp.sum(F.conj(), axis=0, keepdims=True)
        H = H1 * col_sums
        M = M * H
        expvals = jnp.real(M)

    std_err = jnp.std(expvals, axis=-1, ddof=1) / jnp.sqrt(samples.shape[0])

    return jnp.mean(expvals, axis=1) + offset, std_err


def build_expval_func(
    config: CircuitConfig,
) -> Callable:
    """
    Factory that returns a flexible pure function for computing expectation values.
    The returned closure can optionally take runtime overrides for key, observables, etc.
    """
    generators, param_map = _parse_generator_dict(config.gates, config.n_qubits)
    use_control_variate = config.control_variate

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
    default_obs_data = _prep_observables(config.observables)

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
        """
        Executes the expectation value computation with optional runtime overrides.

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
                Monte Carlo samples. Defaults to None.
            init_state_elems (ArrayLike | None, optional): Runtime override for the
                discrete elements of the initial state (X). Defaults to None.
            init_state_amps (ArrayLike | None, optional): Runtime override for the
                continuous amplitudes of the initial state (P). Defaults to None.

        Returns:
            tuple[jnp.ndarray, jnp.ndarray]: A tuple containing:
                - Array of estimated expectation values.
                - Array of standard errors for the estimates.
        """
        if key is not None or n_samples is not None:
            _key = key if key is not None else config.key
            _n = n_samples if n_samples is not None else config.n_samples
            samples = _compute_samples(_key, _n, config.n_qubits)
        else:
            samples = default_samples

        obs_data = default_obs_data if observables is None else _prep_observables(observables)

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
            use_control_variate=use_control_variate,
        )

    return expval_execution

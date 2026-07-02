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
Implementations for the qubit and qudit expectation value functions
"""

import itertools
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import NamedTuple

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
        n_samples (int): Number of Monte Carlo samples for the estimation of the expectation value.
        key (ArrayLike): Random key for JAX.
        n_qubits (int): Number of qubits.
        observables (ArrayLike | None): List of Pauli observables mapped to integers
            (I=0, X=1, Y=2, Z=3). If ``None``, observables must be supplied at call time
            via the returned function's ``observables`` keyword argument.
        init_state_elems (ArrayLike | None): Elements of the initial state (X)
        init_state_amps (ArrayLike | None): Amplitudes of the initial state (P)
        phase_fn (Callable | None): Optional phase layer function.
    """

    gates: dict[int, list[list[int]]]
    n_samples: int
    key: ArrayLike
    n_qubits: int
    observables: ArrayLike | None = None
    init_state_elems: ArrayLike | None = None
    init_state_amps: ArrayLike | None = None
    phase_fn: Callable | None = None


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
        circuit_def (dict[int, list[list[int]]]): Dictionary mapping parameter indices to
            lists of qubit indices.
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
    """The pure mathematical core of the expectation value computation."""
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
    """
    Factory that returns a flexible pure function for computing expectation values.
    The returned closure can optionally take runtime overrides for key, observables, etc.
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


@dataclass
class QuditCircuitConfig:  # pylint: disable=too-many-instance-attributes
    """
    Configuration data for a qudit IQP circuit simulation.

    Args:
        d (int): Qudit dimension (e.g. 2 for qubits, 3 for qutrits).
        n_qudits (int): Number of qudits.
        gates (dict[int, list[list[int]]]): Circuit structure mapping parameter indices to lists
            of generator vectors. Each generator vector has length ``n_qudits`` with integer
            entries in ``{0, ..., d-1}``, representing the Z-power on each qudit.
        observables (tuple[ArrayLike, ArrayLike] | None): Pair ``(l_vecs, m_vecs)`` specifying
            the displacement operators ``D(l, m)`` to measure. Each array has shape
            ``(n_obs, n_qudits)`` with integer entries in ``{0, ..., d-1}``.  If ``None``,
            observables must be supplied when constructing the expectation-value function
            (e.g. via ``build_qudit_mmd_loss``, which generates its own observables).
        n_samples (int): Number of Monte Carlo samples for the estimation of the expectation value.
        key (ArrayLike): Random key for JAX.
        init_state_elems (ArrayLike | None): Support elements of the initial state. Array of
            shape ``(N, n_qudits)`` with integer entries in ``{0, ..., d-1}``, where ``N`` is
            the number of non-zero amplitudes. Each row is a dit-string in ``Z_d^n``.
        init_state_amps (ArrayLike | None): Complex amplitudes of the initial state. Array of
            shape ``(N,)`` corresponding to the support elements.
    """

    d: int
    n_qudits: int
    gates: dict[int, list[list[int]]]
    observables: tuple[ArrayLike, ArrayLike] | None = None
    n_samples: int = 10000
    key: ArrayLike = field(default_factory=lambda: jax.random.PRNGKey(0))
    init_state_elems: ArrayLike | None = None
    init_state_amps: ArrayLike | None = None


def _parse_qudit_generator_dict(circuit_def: dict[int, list[list[int]]], n_qudits: int):
    """
    Converts a qudit circuit definition dict into generator matrix and parameter map.

    Unlike the qubit version, generator vectors are provided explicitly (not as wire
    indices), so each inner list must already have length ``n_qudits`` with integer entries
    in ``{0, ..., d-1}``.

    Args:
        circuit_def (dict[int, list[list[int]]]): Maps parameter indices to lists of
            generator vectors of length ``n_qudits``.
        n_qudits (int): Number of qudits.

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]: Tuple containing:
            - Integer generator matrix of shape ``(n_gates, n_qudits)``.
            - Integer array mapping each gate to its parameter index.

    Raises:
        ValueError: If any generator vector has length != ``n_qudits``.
    """
    flat_gates = []
    param_indices = []

    for param_idx in sorted(circuit_def.keys()):
        for gate in circuit_def[param_idx]:
            if len(gate) != n_qudits:
                raise ValueError(f"Generator has length {len(gate)}, expected {n_qudits}.")
            flat_gates.append(gate)
            param_indices.append(param_idx)

    n_gates = len(flat_gates)
    if n_gates == 0:
        generators = np.zeros((0, n_qudits), dtype=int)
    else:
        generators = np.array(flat_gates, dtype=int)

    param_map = jnp.array(param_indices, dtype=int)
    return jnp.array(generators), param_map


def _compute_qudit_samples(key: ArrayLike, num_samples: int, n_qudits: int, d: int) -> jnp.ndarray:
    """Generates uniformly random dit-strings from Z_d^n."""
    return jax.random.randint(key, shape=(num_samples, n_qudits), minval=0, maxval=d)


class WeightGroupData(NamedTuple):
    """Precomputed static data for a group of gates sharing the same weight."""

    param_indices: jnp.ndarray
    samples_matrices: list[jnp.ndarray]
    obs_matrices: list[jnp.ndarray]


def _gather_support_values(
    vectors: ArrayLike, supports: np.ndarray, target_dim: int, n_gates: int, omega: int
) -> jnp.ndarray:
    """Gathers values at support positions and reshapes to (n_gates, omega, target_dim)."""
    flat_supports = supports.reshape(-1)
    return (
        jnp.array(vectors)[:, flat_supports].reshape(target_dim, n_gates, omega).transpose(1, 2, 0)
    )


def _compute_trigonometric_building_blocks(
    gate_vals: np.ndarray, z_at_support: jnp.ndarray, l_at_support: jnp.ndarray, d: int
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Returns (state_cos, state_sin, obs_cos, obs_sin) trig factors over the gate support."""
    g = jnp.array(gate_vals, dtype=jnp.float32)[:, :, jnp.newaxis]
    angle_z = 2 * jnp.pi * g * z_at_support.astype(jnp.float32) / d + jnp.pi / 4
    angle_l = 2 * jnp.pi * g * l_at_support.astype(jnp.float32) / d
    return (
        jnp.sqrt(2.0) * jnp.cos(angle_z),
        jnp.sqrt(2.0) * jnp.sin(angle_z),
        jnp.cos(angle_l),
        jnp.sin(angle_l),
    )


def _expand_angle_addition(
    state_cos: jnp.ndarray,
    state_sin: jnp.ndarray,
    obs_cos: jnp.ndarray,
    obs_sin: jnp.ndarray,
    n_gates: int,
    num_samples: int,
    n_obs: int,
    omega: int,
) -> tuple[list[jnp.ndarray], list[jnp.ndarray]]:
    """
    For each of the 2**omega ways to assign cos or sin to each of the omega active
    qudit positions, computes the corresponding gate-vs-sample matrix (samples_matrices entry)
    and gate-vs-observable matrix (obs_matrices entry).

    These terms come from expanding the product Φ_g(z - l) = Π_k T_k(z_k, l_k)
    using the angle-addition identity, where each factor T_k can independently
    contribute either a cos or sin term.

    The all-cos combination (every choice = cos) is always the first entry and
    equals Φ_g(z) itself.
    """
    state_factors = [state_cos, state_sin]
    obs_factors = [obs_cos, obs_sin]
    samples_list: list[jnp.ndarray] = []
    obs_list: list[jnp.ndarray] = []
    for sigma in itertools.product([0, 1], repeat=omega):
        B = jnp.ones((n_gates, num_samples), dtype=jnp.float32)
        C = jnp.ones((n_gates, n_obs), dtype=jnp.float32)
        for k, choice in enumerate(sigma):
            B *= state_factors[choice][:, k, :]
            C *= obs_factors[choice][:, k, :]
        samples_list.append(B)
        obs_list.append(C)
    return samples_list, obs_list


def _build_weight_group(
    generators_w: np.ndarray,
    param_indices: jnp.ndarray,
    samples: jnp.ndarray,
    l_vecs: jnp.ndarray,
    d: int,
    num_samples: int,
    n_obs: int,
    omega: int,
) -> WeightGroupData:
    """
    Precomputes the static samples_matrices and obs_matrices factor matrices for a group of gates
    that all act on exactly omega qudits.

    Steps:
      1. Find the active (non-zero) qudit positions for each gate.
      2. Restrict samples and observable vectors to those positions.
      3. Compute cos/sin factors at each active position.
      4. Enumerate all 2**omega cos/sin combinations to build samples_matrices and obs_matrices.
    """
    n_gates = len(generators_w)
    supports = np.array([np.where(g != 0)[0] for g in generators_w])  # (n_gates, omega)
    gate_vals = np.array([g[s] for g, s in zip(generators_w, supports)])  # (n_gates, omega)

    z_at_support = _gather_support_values(samples, supports, num_samples, n_gates, omega)
    l_at_support = _gather_support_values(l_vecs, supports, n_obs, n_gates, omega)

    state_cos, state_sin, obs_cos, obs_sin = _compute_trigonometric_building_blocks(
        gate_vals, z_at_support, l_at_support, d
    )
    samples_matrices, obs_matrices = _expand_angle_addition(
        state_cos, state_sin, obs_cos, obs_sin, n_gates, num_samples, n_obs, omega
    )
    return WeightGroupData(
        param_indices=param_indices, samples_matrices=samples_matrices, obs_matrices=obs_matrices
    )


def build_qudit_expval_func(  # pylint: disable=too-many-statements
    config: QuditCircuitConfig,
) -> Callable:
    """
    Factory that returns an expectation-value function for a qudit tcdq circuit.

    The returned function estimates ``<D(l, m)>`` for each displacement-operator
    observable specified in ``config.observables`` via a Monte Carlo method whose
    precision is controlled by ``config.n_samples``.

    Args:
        config (QuditCircuitConfig): Circuit configuration.

    Returns:
        Callable: ``qudit_expval_batched(gates_params, key=None, n_samples=None)``
        returning ``(expvals, cov)`` by default, where ``expvals`` has shape
        ``(n_obs,)`` and ``cov`` has shape ``(n_obs, 2, 2)``. For each
        observable, ``cov`` is the covariance matrix of the estimated expectation
        value: the diagonal holds the variances of the real and imaginary parts
        and the off-diagonal holds their covariance.

        When the returned callable is invoked with
        ``return_mean_y_sq=True``, it instead returns
        ``(expvals, cov, mean_y_sq)`` where ``mean_y_sq`` is the per-observable
        mean of ``|y_r|^2`` over the Monte Carlo samples. This is the quantity
        used by the qudit MMD loss to write the QQ U-statistic correction in the
        direct form from the notes.
    """
    generators, param_map = _parse_qudit_generator_dict(config.gates, config.n_qudits)

    d = config.d
    n = config.n_qudits

    default_samples = _compute_qudit_samples(config.key, config.n_samples, n, d)

    gen_np = np.array(generators)
    pm_np = np.array(param_map)
    gate_weights = np.sum(gen_np != 0, axis=1)

    def _obs_phase_matrix(samples: jnp.ndarray, m_f: jnp.ndarray, l_f: jnp.ndarray) -> jnp.ndarray:
        # [i, j] = exp(iπ/d · m_i · (2z_j − l_i))  — observable phase matrix.
        s_f = samples.astype(jnp.float32)
        return jnp.exp(
            (1j * jnp.pi / d) * (2 * m_f @ s_f.T - jnp.sum(m_f * l_f, axis=1, keepdims=True))
        )  # (|O|, |Z|)

    if config.observables is not None:
        default_l_vecs = jnp.array(config.observables[0], dtype=jnp.int32)
        default_m_vecs = jnp.array(config.observables[1], dtype=jnp.int32)
        default_n_obs = default_l_vecs.shape[0]
        default_l_f = default_l_vecs.astype(jnp.float32)
        default_m_f = default_m_vecs.astype(jnp.float32)

        # (eq:factored_E): decompose accumulated_phase_diffs = Σ_ω E_ω by gate weight ω = |S(g)|.
        # Because the shift is z − l (not z + l), the angle-addition expansion of
        # Φ_g(z − l) carries no alternating sign, so every σ-term enters accumulated_phase_diffs
        # with the same (negative) sign: E_ω = θ·B̃ − Σ_σ C_σᵀ diag(θ) B_σ.
        default_weight_data: list[WeightGroupData] = []
        for omega in sorted(set(gate_weights)):
            if omega == 0:  # identity gates contribute no phase shift
                continue
            gate_indices = np.where(gate_weights == omega)[0]
            default_weight_data.append(
                _build_weight_group(
                    generators_w=gen_np[gate_indices],
                    param_indices=jnp.array(pm_np[gate_indices]),
                    samples=default_samples,
                    l_vecs=default_l_vecs,
                    d=d,
                    num_samples=config.n_samples,
                    n_obs=default_n_obs,
                    omega=omega,
                )
            )
        default_obs_phase_matrix = _obs_phase_matrix(default_samples, default_m_f, default_l_f)
    else:
        default_l_vecs = None
        default_m_vecs = None
        default_n_obs = None
        default_l_f = None
        default_m_f = None
        default_weight_data = None
        default_obs_phase_matrix = None

    def qudit_expval_batched(  # pylint: disable=too-many-branches,too-many-statements
        gates_params: ArrayLike,
        key: ArrayLike | None = None,
        n_samples: int | None = None,
        observables: tuple[ArrayLike, ArrayLike] | None = None,
        init_state_elems: ArrayLike | None = None,
        init_state_amps: ArrayLike | None = None,
        return_mean_y_sq: bool = False,
    ) -> tuple[jnp.ndarray, jnp.ndarray] | tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Compute batched Monte Carlo expectation values for the qudit IQP circuit.

        Args:
            gates_params (ArrayLike): 1-D array of gate parameters.
            key (ArrayLike | None, optional): Runtime override for the JAX PRNG key
                used for sampling. Defaults to None.
            n_samples (int | None, optional): Runtime override for the number of
                Monte Carlo samples. Defaults to None.
            observables (tuple[ArrayLike, ArrayLike] | None, optional): Runtime override
                for the displacement-operator observables ``(l_vecs, m_vecs)``.
                Defaults to None.
            init_state_elems (ArrayLike | None, optional): Runtime override for the
                support elements of the initial state. Array of shape ``(N, n_qudits)``
                with integer entries in ``{0, ..., d-1}``. Defaults to None.
            init_state_amps (ArrayLike | None, optional): Runtime override for the
                complex amplitudes of the initial state. Array of shape ``(N,)``.
                Defaults to None.
            return_mean_y_sq (bool, optional): If ``True``, also return the
                per-observable Monte Carlo mean of ``|y_r|^2``. Defaults to ``False``.

        Returns:
            tuple[jnp.ndarray, jnp.ndarray] | tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            By default returns ``(expvals, cov)`` where ``expvals`` are the estimated
            complex expectation values, shape ``(n_obs,)``, and ``cov`` are the
            covariance matrices of the mean estimator, shape ``(n_obs, 2, 2)``.

            When ``return_mean_y_sq=True``, also returns ``mean_y_sq`` with shape
            ``(n_obs,)``. This equals 1 when the per-sample integrand has unit
            modulus (default input state, diagonal observables).
        """
        if observables is not None:
            l_vecs = jnp.array(observables[0], dtype=jnp.int32)
            m_vecs = jnp.array(observables[1], dtype=jnp.int32)
            n_obs = l_vecs.shape[0]
            l_f = l_vecs.astype(jnp.float32)
            m_f = m_vecs.astype(jnp.float32)
        else:
            if default_l_vecs is None:
                raise ValueError(
                    "No observables specified. Provide them in QuditCircuitConfig "
                    "or pass at call time via the observables argument."
                )
            l_vecs = default_l_vecs
            n_obs = default_n_obs
            l_f = default_l_f
            m_f = default_m_f

        if key is not None or n_samples is not None:
            _key = key if key is not None else config.key
            _n = n_samples if n_samples is not None else config.n_samples
            samples = _compute_qudit_samples(_key, _n, n, d)
        else:
            _n = config.n_samples
            samples = default_samples

        if key is not None or n_samples is not None or observables is not None:
            obs_phase_matrix = _obs_phase_matrix(samples, m_f, l_f)
            weight_data: list[WeightGroupData] = []
            for omega in sorted(set(gate_weights)):
                if omega == 0:
                    continue
                gate_indices = np.where(gate_weights == omega)[0]
                weight_data.append(
                    _build_weight_group(
                        generators_w=gen_np[gate_indices],
                        param_indices=jnp.array(pm_np[gate_indices]),
                        samples=samples,
                        l_vecs=l_vecs,
                        d=d,
                        num_samples=_n,
                        n_obs=n_obs,
                        omega=omega,
                    )
                )
        else:
            obs_phase_matrix = default_obs_phase_matrix
            weight_data = default_weight_data

        # Assemble E_ω = θ·B̃ − Σ_σ C_σᵀ diag(θ) B_σ   (eq:factored_E)
        accumulated_phase_diffs = jnp.zeros((n_obs, _n))

        for group in weight_data:
            theta_w = jnp.asarray(gates_params)[group.param_indices]
            accumulated_phase_diffs = (
                accumulated_phase_diffs + (theta_w @ group.samples_matrices[0])[jnp.newaxis, :]
            )  # B̃ = B_{σ=0}
            for B_sigma, C_sigma in zip(group.samples_matrices, group.obs_matrices):
                accumulated_phase_diffs = accumulated_phase_diffs - (C_sigma.T * theta_w) @ B_sigma

        state_elems = config.init_state_elems if init_state_elems is None else init_state_elems
        state_amps = config.init_state_amps if init_state_amps is None else init_state_amps

        if state_elems is None or state_amps is None:
            # {⟨O(l,m)⟩} = mean_1[ obs_phase_matrix ⊙ exp(i·accumulated_phase_diffs) ]
            integrand = obs_phase_matrix * jnp.exp(1j * accumulated_phase_diffs)
        else:
            s_f = samples.astype(jnp.float32)
            X_state = jnp.asarray(state_elems).astype(jnp.float32)  # (N, n)
            Psi = jnp.asarray(state_amps)  # (N,)

            # ω^{Z·X^T} where ω = exp(2πi/d) — shape (s, N)
            omega_ZX = jnp.exp(2j * jnp.pi * (s_f @ X_state.T) / d)

            # Ψ̃^(2) = ω^{Z·X^T} · Ψ — shape (s,)
            psi_tilde_2 = omega_ZX @ Psi

            # F = Ψ* · 1_{1×s} ⊙ ω^{-X·Z^T} — shape (N, s)
            F_mat = Psi.conj()[:, jnp.newaxis] * omega_ZX.conj().T

            # Ψ̃^(1) = ω^{L·X^T} · F — shape (l, s)
            omega_LX = jnp.exp(2j * jnp.pi * (l_f @ X_state.T) / d)  # (l, N)
            psi_tilde_1 = omega_LX @ F_mat

            # H = Ψ̃^(1) ⊙ (1_{l×1} · (Ψ̃^(2))^T) — shape (l, s)
            H = psi_tilde_1 * psi_tilde_2[jnp.newaxis, :]

            integrand = obs_phase_matrix * jnp.exp(1j * accumulated_phase_diffs) * H
        expvals = jnp.mean(integrand, axis=1)
        mean_y_sq = jnp.mean(jnp.abs(integrand) ** 2, axis=1)  # (n_obs,)

        # Covariance of the mean estimate: sample (co)variances of the per-sample real
        # and imaginary parts (ddof=1), divided by _n for the variance of the mean.
        re = jnp.real(integrand)
        im = jnp.imag(integrand)
        re_centered = re - jnp.mean(re, axis=1, keepdims=True)
        im_centered = im - jnp.mean(im, axis=1, keepdims=True)
        var_re = jnp.sum(re_centered**2, axis=1) / (_n - 1) / _n
        var_im = jnp.sum(im_centered**2, axis=1) / (_n - 1) / _n
        cov_re_im = jnp.sum(re_centered * im_centered, axis=1) / (_n - 1) / _n
        cov = jnp.stack(
            [
                jnp.stack([var_re, cov_re_im], axis=-1),
                jnp.stack([cov_re_im, var_im], axis=-1),
            ],
            axis=-2,
        )  # (n_obs, 2, 2)

        if return_mean_y_sq:
            return expvals, cov, mean_y_sq
        return expvals, cov

    return qudit_expval_batched

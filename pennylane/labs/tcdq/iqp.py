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

from .base import ObservableAlgebra, TCDQSimulator, estimator


@dataclass
class CircuitConfig:  # pylint: disable=too-many-instance-attributes
    """Description of a qubit IQP circuit for classical expectation-value estimation.

    This dataclass bundles all the information needed to build an expectation-value
    estimator via :func:`build_expval_func`: the gate structure, the observables to
    measure, sampling parameters, and an optional non-standard initial state.

    .. warning::

        ``CircuitConfig`` and :func:`build_expval_func` are superseded by
        :class:`IQPSimulator`, which exposes the same estimator through the
        :class:`~pennylane.labs.tcdq.TCDQSimulator` interface. They are kept
        for backwards compatibility and will be removed.

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

        `IQPopt: Fast optimization of instantaneous quantum polynomial circuits in JAX <https://arxiv.org/abs/2501.04776>`_
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
    """Evaluate the Monte Carlo integrand and return expectation values and variances."""
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
        integrand = jnp.real(phases) * jnp.cos(E) - jnp.imag(phases) * jnp.sin(E)
    else:
        M = phases * jnp.exp(1j * E)
        X = init_state_elems
        P = init_state_amps
        F = P[:, jnp.newaxis] * (1 - 2 * ((X @ samples.T) % 2))
        H1 = (1 - 2 * ((bitflips @ X.T) % 2)) @ F
        col_sums = jnp.sum(F.conj(), axis=0, keepdims=True)
        H = H1 * col_sums
        M = M * H
        integrand = jnp.real(M)

    expvals = jnp.mean(integrand, axis=1)
    variances = jnp.var(integrand, axis=-1, ddof=1) / samples.shape[0]

    return expvals, variances


class IQPSimulator(TCDQSimulator):
    r"""Qubit IQP circuit with classical Monte Carlo estimation of Pauli expectation values.

    An instantaneous quantum polynomial (IQP) circuit has the form
    :math:`U(\mathbf{\theta}) = H^{\otimes n} D(\mathbf{\theta}) H^{\otimes n}`,
    where :math:`D(\mathbf{\theta})` is a diagonal phase unitary built from
    Pauli-:math:`Z` tensor-product generators. This simulator estimates Pauli
    expectation values of such circuits without simulating the full quantum
    state, by averaging a trigonometric integrand over uniformly random
    bitstrings.

    Args:
        gates (dict[int, list[list[int]]]): Circuit structure mapping each
            trainable parameter index to a list of gates. Each gate is itself a
            list of qubit indices participating in a Pauli-:math:`Z`
            tensor-product generator. For example, ``{0: [[0, 1]], 1: [[2]]}``
            defines two parameters: parameter 0 drives a ZZ gate on qubits 0
            and 1, and parameter 1 drives a Z gate on qubit 2. Use
            :func:`~pennylane.labs.tcdq.create_local_gates` or
            :func:`~pennylane.labs.tcdq.create_lattice_gates` to generate these
            automatically.
        n_qubits (int): Total number of qubits in the circuit.
        n_samples (int): Default number of random bitstrings drawn per estimate.
        key (ArrayLike): Default JAX PRNG key for random bitstring generation.
        init_state (tuple[ArrayLike, ArrayLike] | None): Optional
            ``(elements, amplitudes)`` pair describing a custom initial state,
            where ``elements`` is a binary array of shape ``(N, n_qubits)``
            listing the computational-basis states with non-zero amplitude and
            ``amplitudes`` is a complex array of shape ``(N,)``. Defaults to
            ``None``, the uniform superposition
            :math:`H^{\otimes n}\vert 0 \rangle`.
        phase_fn (Callable | None): Optional diagonal phase layer
            ``phase_fn(params, bitstring)`` applied on top of the gates.
            Defaults to ``None``.

    Raises:
        ValueError: If ``n_samples`` is not greater than 1.

    **Example**

    >>> import jax
    >>> import jax.numpy as jnp
    >>> from pennylane.labs.tcdq import IQPSimulator, create_local_gates
    >>> gates = create_local_gates(n_qubits=4, max_weight=2)
    >>> sim = IQPSimulator(
    ...     gates=gates, n_qubits=4, n_samples=5000, key=jax.random.PRNGKey(0)
    ... )
    >>> expval = sim.build_estimator("pauli_expval")
    >>> observables = jnp.array([[3, 3, 0, 0], [0, 0, 3, 3]])  # ZZ on (0,1) and (2,3)
    >>> values, variances = expval(jnp.zeros(len(gates)), observables)
    >>> values.shape
    (2,)

    .. seealso::

        :class:`~pennylane.labs.tcdq.TCDQSimulator`,
        :func:`~pennylane.labs.tcdq.build_mmd_loss`,
        `IQPopt: Fast optimization of instantaneous quantum polynomial circuits in JAX <https://arxiv.org/abs/2501.04776>`_
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        gates: dict[int, list[list[int]]],
        n_qubits: int,
        n_samples: int,
        key: ArrayLike,
        init_state: tuple[ArrayLike, ArrayLike] | None = None,
        phase_fn: Callable | None = None,
    ):
        if n_samples <= 1:
            raise ValueError("n_samples must be greater than 1")

        self.gates = gates
        self.n_qubits = n_qubits
        self.n_samples = n_samples
        self.key = key
        self.init_state = init_state
        self.phase_fn = phase_fn

    @property
    def local_dims(self) -> tuple[int, ...]:
        """tuple[int, ...]: Local dimension of each wire, all equal to 2."""
        return (2,) * self.n_qubits

    @estimator("pauli_expval", algebra=ObservableAlgebra.PAULI)
    def _build_pauli_expval(self) -> Callable:
        """Build the Monte Carlo estimator for Pauli expectation values.

        Returns:
            Callable: A function with signature::

                expval(params, observables, *, key=None, n_samples=None,
                       phase_params=None) -> (values, variances)
        """
        generators, param_map = _parse_generator_dict(self.gates, self.n_qubits)
        default_samples = _compute_samples(self.key, self.n_samples, self.n_qubits)
        init_elems, init_amps = self.init_state if self.init_state is not None else (None, None)

        vmapped_phase_func = None
        if self.phase_fn is not None:
            phase_fn = self.phase_fn

            def compute_phase(p_params, sample, b_flips):
                return phase_fn(p_params, sample) - phase_fn(p_params, (sample + b_flips) % 2)

            vmapped_phase_func = jax.vmap(
                jax.vmap(compute_phase, in_axes=(None, 0, None)), in_axes=(None, None, 0)
            )

        # pylint: disable=too-many-arguments
        def pauli_expval(
            params: ArrayLike,
            observables: ArrayLike,
            *,
            key: ArrayLike | None = None,
            n_samples: int | None = None,
            phase_params: ArrayLike | None = None,
            init_state: tuple[ArrayLike, ArrayLike] | None = None,
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            """Estimate Pauli expectation values and the variance of each mean.

            Args:
                params (ArrayLike): Trainable gate parameters, shape ``(n_params,)``.
                observables (ArrayLike): Integer-encoded Pauli operators of shape
                    ``(n_obs, n_qubits)`` with ``I=0, X=1, Y=2, Z=3``.
                key (ArrayLike | None): Override for the bitstring sampling key.
                n_samples (int | None): Override for the number of bitstrings.
                phase_params (ArrayLike | None): Trainable parameters of the
                    phase layer, required when the simulator has a ``phase_fn``.
                init_state (tuple[ArrayLike, ArrayLike] | None): Override for the
                    simulator's ``init_state``.

            Returns:
                tuple[jnp.ndarray, jnp.ndarray]: ``(values, variances)``, both
                real arrays of shape ``(n_obs,)``. ``variances`` are the
                variances of the mean estimators.
            """
            if key is None and n_samples is None:
                samples = default_samples
            else:
                samples = _compute_samples(
                    self.key if key is None else key,
                    self.n_samples if n_samples is None else n_samples,
                    self.n_qubits,
                )

            elems, amps = (init_elems, init_amps) if init_state is None else init_state

            return _core_expval_execution(
                params,
                phase_params,
                samples,
                _prep_observables(observables),
                elems,
                amps,
                generators,
                param_map,
                vmapped_phase_func,
            )

        return pauli_expval


def _simulator_from_config(config: CircuitConfig) -> IQPSimulator:
    """Build an :class:`IQPSimulator` from a legacy :class:`CircuitConfig`."""
    elems, amps = config.init_state_elems, config.init_state_amps

    return IQPSimulator(
        gates=config.gates,
        n_qubits=config.n_qubits,
        n_samples=config.n_samples,
        key=config.key,
        init_state=None if elems is None or amps is None else (elems, amps),
        phase_fn=config.phase_fn,
    )


def build_expval_func(
    config: CircuitConfig,
) -> Callable:
    """Build an estimator for Pauli expectation values of a qubit IQP circuit.

    .. warning::

        This function is superseded by :class:`IQPSimulator`. Prefer
        ``IQPSimulator(...).build_estimator("pauli_expval")``, which returns an
        :class:`~pennylane.labs.tcdq.Estimator` carrying the metadata that loss
        functions use to check compatibility. This function is kept for
        backwards compatibility and will be removed.

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
            ) -> (expvals, variances)

        where ``expvals`` is a real array of shape ``(n_observables,)`` and
        ``variances`` contains the estimated variance of each expectation-value
        estimator.

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
    >>> expvals, variances = expval_fn(params)
    >>> expvals.shape
    (2,)

    .. seealso::

        :class:`~pennylane.labs.tcdq.IQPSimulator`,
        `IQPopt: Fast optimization of instantaneous quantum polynomial circuits in JAX <https://arxiv.org/abs/2501.04776>`_
    """
    base_estimator = _simulator_from_config(config).build_estimator("pauli_expval")

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

        Args:
            gates_params (ArrayLike): Trainable parameters for the circuit gates.
            phase_fn_params (ArrayLike | None, optional): Trainable parameters for the
                custom phase function. Defaults to None.
            observables (ArrayLike | None, optional): Runtime override for the Pauli
                observables (I=0, X=1, Y=2, Z=3). Defaults to None.
            key (ArrayLike | None, optional): Runtime override for the JAX PRNG key
                used for sampling. Defaults to None.
            n_samples (int | None, optional): Runtime override for the number of
                samples. Defaults to None.
            init_state_elems (ArrayLike | None, optional): Runtime override for the
                discrete elements of the initial state. Defaults to None.
            init_state_amps (ArrayLike | None, optional): Runtime override for the
                continuous amplitudes of the initial state. Defaults to None.

        Returns:
            tuple[jnp.ndarray, jnp.ndarray]: Estimated expectation values and
            the estimated variances of those estimators.

        Raises:
            ValueError: If no observables are available.
        """
        obs = config.observables if observables is None else observables
        if obs is None:
            raise ValueError(
                "No observables specified. Provide them in CircuitConfig "
                "or pass at call time via the observables argument."
            )

        elems = config.init_state_elems if init_state_elems is None else init_state_elems
        amps = config.init_state_amps if init_state_amps is None else init_state_amps

        return base_estimator(
            gates_params,
            obs,
            key=key,
            n_samples=n_samples,
            phase_params=phase_fn_params,
            init_state=None if elems is None or amps is None else (elems, amps),
        )

    return expval_execution

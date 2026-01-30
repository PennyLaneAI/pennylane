import math
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike


def _phase(pauli: str, qubit: int) -> complex:
    """
    For a Pauli P return the phase applied by Conjugation with Hadamard (HPH).
    Specifically, we have the relations:
    HXH = Z
    HYH = -Y
    HZH = X

    Args:
        pauli (str): The Pauli operator ("I", "X", "Y", "Z").
        qubit (int): The qubit index (0 or 1, although only checked for parity).

    Returns:
        complex: The phase factor.

    Raises:
        ValueError: If pauli is not one of "I", "X", "Y", "Z".
    """

    if pauli in ("I", "Z"):
        return 1

    if pauli == "Y":
        if qubit == 0:
            return -1j
        return 1j

    if pauli == "X":
        if qubit == 0:
            return 1
        return -1

    raise ValueError(f"Expected Pauli I, X, Y, or Z, got {pauli}.")


def _iqp_expval_core(
    generators: ArrayLike,
    ops: ArrayLike,
    n_samples: int,
    key: ArrayLike,
    init_state: tuple[ArrayLike, ArrayLike] | None = None,
    phase_layer: Callable = None,
):
    """
    Builds a function to compute expectation value of a batch of Pauli Z operators for an IQP circuit.

    Args:
        generators (ArrayLike): Binary matrix of shape ``(n_generators, n_qubits)`` representing the circuit generators $G$.
        ops (ArrayLike): Binary matrix representing Pauli Z operators $A$.
        n_samples (int): Number of stochastic samples to draw.
        key (ArrayLike): JAX PRNGKey for random number generation.
        init_state (tuple[ArrayLike, ArrayLike] | None): Optional tuple $(X, P)$ representing initial state stabilizers.
            If None, the initial state is $|0\rangle^{\otimes n}$.
        phase_layer (Callable | None): Optional function representing the alternating phase layer.

    Returns:
        Callable[[ArrayLike, ArrayLike], tuple[jnp.ndarray, jnp.ndarray]]: A function that takes parameters $\theta$ (and optional phase_params) and returns:
            - Mean expectation values $\langle Z_A \rangle$.
            - Standard error of the mean.
    """
    n_qubits = generators.shape[1]
    samples = jax.random.randint(key, (n_samples, n_qubits), 0, 2)

    bitflips = jnp.array([[1 if gate in {"Z", "Y"} else 0 for gate in op] for op in ops])
    phases = jnp.array(
        [
            [
                math.prod([_phase(gate, qubit) for gate, qubit in zip(op, sample)])
                for sample in samples
            ]
            for op in ops
        ]
    )

    def expval_func(
        params: ArrayLike, phase_params: ArrayLike = None
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        B = (-1) ** (samples @ generators.T)
        C = 1 - ((-1) ** (bitflips @ generators.T))
        E = C @ jnp.diag(params) @ B.T

        if phase_layer is not None:

            def compute_phase(phase_params, sample, bitflips):
                return phase_layer(phase_params, sample) - phase_layer(
                    phase_params, (sample + bitflips) % 2
                )

            phase_matrix = jax.vmap(compute_phase, in_axes=(None, 0, None))
            phase_matrix = jax.vmap(phase_matrix, in_axes=(None, None, 0))

            E += phase_matrix(phase_params, samples, bitflips)

        M = phases * jnp.exp(1j * E)

        if init_state is not None:
            X, P = init_state
            P = P[:, jnp.newaxis]

            F = jnp.broadcast_to(P, (P.shape[0], samples.shape[0])) * ((-1) ** (X @ samples.T))

            H1 = ((-1) ** (bitflips @ X.T)) @ F
            col_sums = jnp.sum(F.conj(), axis=0, keepdims=True)
            H2 = jnp.broadcast_to(col_sums, (bitflips.shape[0], samples.shape[0]))
            H = H1 * H2
            expvals = jnp.real(M * H)
        else:
            expvals = jnp.real(M)

        std_err = jnp.std(expvals, axis=-1, ddof=1) / jnp.sqrt(n_samples)
        return jnp.mean(expvals, axis=1), std_err

    return expval_func


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

    # Indicator = (A . G^T) % 2
    indicator = (ops @ generators.T) % 2
    X = probs * indicator

    # Use jnp.where to replace 0s with 1.0s before product to avoid zeroing out
    result = jnp.prod(jnp.where(X == 0, 1.0, X), axis=1)

    return result, jnp.zeros(ops.shape[0])


def _parse_iqp_dict(circuit_def: dict[int, list[list[int]]], n_qubits: int):
    """
    Converts dictionary circuit definition into JAX-ready matrices.

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


def iqp_expval(
    gates: dict[int, list[list[int]]],
    params: ArrayLike,
    ops: ArrayLike,
    n_samples: int,
    n_qubits: int,
    key: ArrayLike,
    init_state: tuple[ArrayLike, ArrayLike] | None = None,
    batch_size: int = 1000,
):
    """
    Computes IQP expectation values from a high-level circuit definition.
    Handles preprocessing and memory batching automatically.

    Args:
        gates (dict[int, list[list[int]]]): Dictionary mapping parameter indices to lists of qubit indices.
            Example: ``{0: [[0, 1], [1, 2]]}``.
        params (ArrayLike): Array of parameters matching the keys in ``gates``.
        ops (ArrayLike): Binary matrix of Pauli Z operators.
        n_samples (int): Number of stochastic shots per operator.
        n_qubits (int): Total number of qubits in the system.
        key (ArrayLike): JAX PRNGKey for random sampling.
        init_state (tuple[ArrayLike, ArrayLike] | None): Optional tuple $(X, P)$ representing initial state stabilizers.
        batch_size (int): Number of operators to process at once on the GPU. Defaults to 1000.

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]: A tuple containing:
            - Mean expectation values.
            - Standard deviation of the expectation values.
    """
    generators, param_map = _parse_iqp_dict(gates, n_qubits)

    params = jnp.asarray(params)
    expanded_params = params[param_map]

    n_ops = ops.shape[0]
    results_mean = []
    results_std = []

    for i in range(0, n_ops, batch_size):
        ops_chunk = ops[i : i + batch_size]

        expval_func = _iqp_expval_core(generators, ops_chunk, n_samples, key, init_state=init_state)
        chunk_mean, chunk_std = expval_func(expanded_params)

        results_mean.append(chunk_mean)
        results_std.append(chunk_std)

    return jnp.concatenate(results_mean), jnp.concatenate(results_std)

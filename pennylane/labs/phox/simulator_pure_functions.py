import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
import math


def _phase(pauli: str, qubit: int) -> complex:
    """For a Pauli P return the phase applied by HPH

    HXH = Z
    HYH = -Y
    HZH = X

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
    params: ArrayLike,
    ops: ArrayLike,
    n_samples: int,
    key: ArrayLike,
    init_state: tuple[ArrayLike, ArrayLike] | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute expectation value of a batch of Pauli Z operators for an IQP circuit.

    Args:
        generators (G): Binary matrix of shape (n_generators, n_qubits).
        params (Theta): Parameters for the diagonal operators.
        ops (A): Binary matrix representing Pauli Z operators.
        n_samples: Number of samples to draw.
        key: JAX PRNGKey.
        init_state: Optional tuple (X, P) representing initial state stabilizers.

    Returns:
        tuple: (Mean expectation values, Standard error)
    """
    n_qubits = generators.shape[1]

    if init_state is not None:
        return _iqp_expval_init_state(generators, params, ops, n_samples, key, init_state)

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

    B = (-1) ** (samples @ generators.T)
    C = 1 - ((-1) ** (bitflips @ generators.T))
    E = C @ jnp.diag(params) @ B.T
    M = phases * jnp.exp(1j * E)

    expvals = jnp.real(M)

    std_err = jnp.std(expvals, axis=-1, ddof=1) / jnp.sqrt(n_samples)
    return jnp.mean(expvals, axis=1), std_err


def _iqp_expval_init_state(
    generators: ArrayLike,
    params: ArrayLike,
    ops: ArrayLike,
    n_samples: int,
    key: ArrayLike,
    init_state: tuple[ArrayLike, ArrayLike],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Helper function for IQP expval with a specific initial state."""
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

    B = (-1) ** (samples @ generators.T)
    C = 1 - ((-1) ** (bitflips @ generators.T))
    E = C @ jnp.diag(params) @ B.T
    M = phases * jnp.exp(1j * E)
    X, P = init_state
    P = P[:, jnp.newaxis]

    F = jnp.broadcast_to(P, (P.shape[0], samples.shape[0])) * ((-1) ** (X @ samples.T))

    H1 = ((-1) ** (bitflips @ X.T)) @ F
    col_sums = jnp.sum(F.conj(), axis=0, keepdims=True)
    H2 = jnp.broadcast_to(col_sums, (bitflips.shape[0], samples.shape[0]))
    H = H1 * H2
    expvals = jnp.real(M * H)

    std_err = jnp.std(expvals, axis=-1, ddof=1) / jnp.sqrt(n_samples)
    return jnp.mean(expvals, axis=1), std_err


def bitflip_expval(
    generators: ArrayLike, params: ArrayLike, ops: ArrayLike
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute expectation value for the Bitflip noise model.

    Args:
        generators (G): Binary matrix of shape (n_generators, n_qubits).
        params (Theta): Error probabilities/parameters.
        ops (A): Binary matrix representing Pauli Z operators.

    Returns:
        tuple: (Expectation values, Zero array for std_err)
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
    batch_size: int = 1000,
):
    """
    Computes IQP expectation values from a high-level circuit definition.
    Handles preprocessing and memory batching automatically.

    Args:
        gates: Dictionary {param_idx: [[q0, q1], [q2]]}.
        params: Array of parameters matching the keys in 'gates'.
        ops: Binary matrix of Pauli Z operators.
        n_samples: Number of shots per operator.
        n_qubits: Total number of qubits (needed to build matrices).
        key: JAX PRNGKey.
        batch_size: Number of operators to process at once on the GPU.
    """
    generators, param_map = _parse_iqp_dict(gates, n_qubits)

    params = jnp.asarray(params)
    expanded_params = params[param_map]

    n_ops = ops.shape[0]
    results_mean = []
    results_std = []

    for i in range(0, n_ops, batch_size):
        ops_chunk = ops[i : i + batch_size]

        chunk_mean, chunk_std = _iqp_expval_core(
            generators, expanded_params, ops_chunk, n_samples, key
        )

        results_mean.append(chunk_mean)
        results_std.append(chunk_std)

    return jnp.concatenate(results_mean), jnp.concatenate(results_std)

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

def iqp_expval(
    generators: ArrayLike,
    params: ArrayLike,
    ops: ArrayLike,
    n_samples: int,
    key: ArrayLike,
    init_state: tuple[ArrayLike, ArrayLike] | None = None
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

    # Generate samples
    # Note: We do not split the key inside the function to maintain purity.
    # The user should manage key splitting externally.
    samples = jax.random.randint(key, (n_samples, n_qubits), 0, 2)

    # IQP Math
    # B = (-1)^(Z . G^T)
    B = (-1) ** (samples @ generators.T)
    
    # C = 1 - (-1)^(A . G^T)
    C = 1 - ((-1) ** (ops @ generators.T))
    
    # E = C . diag(Theta) . B^T
    E = C @ jnp.diag(params) @ B.T

    expvals = jnp.cos(E)

    std_err = jnp.std(expvals, axis=-1, ddof=1) / jnp.sqrt(n_samples)
    return jnp.mean(expvals, axis=1), std_err


def _iqp_expval_init_state(
    generators: ArrayLike,
    params: ArrayLike,
    ops: ArrayLike,
    n_samples: int,
    key: ArrayLike,
    init_state: tuple[ArrayLike, ArrayLike]
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Helper function for IQP expval with a specific initial state."""
    n_qubits = generators.shape[1]
    
    samples = jax.random.randint(key, (n_samples, n_qubits), 0, 2)

    B = (-1) ** (samples @ generators.T)
    C = 1 - ((-1) ** (ops @ generators.T))
    E = C @ jnp.diag(params) @ B.T
    
    X, P = init_state
    P = P[:, jnp.newaxis]

    # Calculate F factor based on initial state
    F = jnp.broadcast_to(P, (P.shape[0], n_samples)) * ((-1)**(X @ samples.T))

    H1 = ((-1) ** (ops @ X.T)) @ F
    col_sums = jnp.sum(F.conj(), axis=0, keepdims=True)
    H2 = jnp.broadcast_to(col_sums, (ops.shape[0], n_samples))
    H = H1 * H2

    expvals = jnp.cos(E) * jnp.real(H) - jnp.sin(E) * jnp.imag(H)

    std_err = jnp.std(expvals, axis=-1, ddof=1) / jnp.sqrt(n_samples)
    return jnp.mean(expvals, axis=1), std_err


def bitflip_expval(
    generators: ArrayLike,
    params: ArrayLike,
    ops: ArrayLike
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

def batched_iqp_expval(
    generators, 
    params, 
    ops, 
    n_samples, 
    key, 
    batch_size: int = 1000
):
    """
    Computes IQP expectation values in batches to save memory.
    """
    n_ops = ops.shape[0]
    results_mean = []
    results_std = []

    for i in range(0, n_ops, batch_size):
        ops_chunk = ops[i : i + batch_size]
        
        chunk_mean, chunk_std = iqp_expval(
            generators, params, ops_chunk, n_samples, key
        )
        
        results_mean.append(chunk_mean)
        results_std.append(chunk_std)

    return jnp.concatenate(results_mean), jnp.concatenate(results_std)

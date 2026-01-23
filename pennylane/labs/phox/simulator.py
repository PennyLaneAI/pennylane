import math

import jax
import jax.numpy as jnp
from numpy.typing import ArrayLike

class IqpSimulator:
    def __init__(self, generators: ArrayLike, trainable: list[list[int]], fixed: list[list[int]] = None):

        self.generators = generators
        self.trainable = trainable
        self.fixed = fixed
        self.n_generators, self.n_qubits = generators.shape

    def expval_gen(self, params: ArrayLike, ops: ArrayLike, n_samples: int, init_state: ArrayLike = None):
        if init_state is not None:
            return self._expval_gen_init_state(params, ops, n_samples, init_state=init_state)

        key = jax.random.PRNGKey(42)
        key, subkey = jax.random.split(key, 2)
        key = subkey

        samples = jax.random.randint(key, (n_samples, self.n_qubits), 0, 2)

        bitflips = jnp.array([[1 if gate in {"Z", "Y"} else 0 for gate in op] for op in ops])
        phases = jnp.array([[math.prod([_phase(gate, qubit) for gate, qubit in zip(op, sample)]) for sample in samples] for op in ops])

        expvals = _expval_gen_matrix(params, self.generators, bitflips, phases, samples)

        std_err = jnp.std(expvals, axis=-1, ddof=1) / jnp.sqrt(n_samples)
        return jnp.mean(expvals, axis=1), std_err

    def _expval_gen_init_state(self, params: ArrayLike, ops: list[list[int]], n_samples: int, init_state: ArrayLike = None):
        key = jax.random.PRNGKey(42)
        key, subkey = jax.random.split(key, 2)
        key = subkey

        samples = jax.random.randint(key, (n_samples, self.n_qubits), 0, 2)

        bitflips = jnp.array([[1 if gate in {"Z", "Y"} else 0 for gate in op] for op in ops])
        phases = jnp.array([[math.prod([_phase(gate, qubit) for gate, qubit in zip(op, sample)]) for sample in samples] for op in ops])

        expvals = _expval_gen_init_matrix(params, self.generators, bitflips, phases, samples, init_state)

        std_err = jnp.std(expvals, axis=-1, ddof=1) / jnp.sqrt(n_samples)
        return jnp.mean(expvals, axis=1), std_err

    def expval(self, params: ArrayLike, ops: list[list[int]], n_samples: int, init_state: ArrayLike = None):
        """Compute expval of a batch of Pauli Z operators

        ops - A
        params - Theta
        self.generators - G
        samples - Z
        init_state - X

        """

        if init_state is not None:
            return self._expval_init_state(params, ops, n_samples, init_state=init_state)

        key = jax.random.PRNGKey(42)
        key, subkey = jax.random.split(key, 2)
        key = subkey

        samples = jax.random.randint(key, (n_samples, self.n_qubits), 0, 2)

        B = (-1) ** (samples @ self.generators.T)
        C = 1 - ((-1) ** (ops @ self.generators.T))
        E = C @ jnp.diag(params) @ B.T

        expvals = jnp.cos(E)

        std_err = jnp.std(expvals, axis=-1, ddof=1) / jnp.sqrt(n_samples)
        return jnp.mean(expvals, axis=1), std_err

    def _expval_init_state(self, params: ArrayLike, ops: ArrayLike, n_samples: int, init_state: ArrayLike):
        """Compute expval of a batch of Pauli Z operators

        ops - A
        params - Theta
        self.generators - G
        samples - Z

        """

        key = jax.random.PRNGKey(42)
        key, subkey = jax.random.split(key, 2)
        key = subkey

        samples = jax.random.randint(key, (n_samples, self.n_qubits), 0, 2)

        B = (-1) ** (samples @ self.generators.T)
        C = 1 - ((-1) ** (ops @ self.generators.T))
        E = C @ jnp.diag(params) @ B.T
        X, P = init_state
        P = P[:, jnp.newaxis]

        F = jnp.broadcast_to(P, (P.shape[0], n_samples)) * ((-1)**(X @ samples.T))

        H1 = ((-1) ** (ops @ X.T)) @ F
        col_sums = jnp.sum(F.conj(), axis=0, keepdims=True)
        H2 = jnp.broadcast_to(col_sums, (ops.shape[0], n_samples))
        H = H1 * H2

        expvals = jnp.cos(E) * jnp.real(H) - jnp.sin(E) * jnp.imag(H)

        std_err = jnp.std(expvals, axis=-1, ddof=1) / jnp.sqrt(n_samples)
        return jnp.mean(expvals, axis=1), std_err

class BitflipSimulator:
    def __init__(self, generators: ArrayLike, trainable: list[list[int]], fixed: list[list[int]] = None):

        self.generators = generators
        self.trainable = trainable
        self.fixed = fixed
        self.n_generators, self.n_qubits = generators.shape

    def expval(self, params: ArrayLike, ops: ArrayLike, n_samples: int):
        """Compute expval of a batch of Pauli Z operators

        ops - A
        params - Theta
        self.generators - G
        samples - Z

        """

        probs = jnp.cos(2 * params)
        indicator = (ops @ self.generators.T) % 2
        X = probs * indicator

        return jnp.prod(jnp.where(X == 0, 1.0, X), axis=1), jnp.zeros(ops.shape[0])

def _phase(pauli: str, qubit: int) -> complex:
    """ For a Pauli P return the phase applied by HPH

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

def _expval_gen_matrix(params, generators, bitflips, phases, samples):
    B = (-1)**(samples @ generators.T)
    C = 1 - ((-1) ** (bitflips @ generators.T))
    E = C @ jnp.diag(params) @ B.T
    M = phases * jnp.exp(-1j * E)

    return jnp.real(M)

def _expval_gen_init_matrix(params, generators, bitflips, phases, samples, init_state):
    B = (-1) ** (samples @ generators.T)
    C = 1 - ((-1) ** (bitflips @ generators.T))
    E = C @ jnp.diag(params) @ B.T
    M = phases * jnp.exp(1j * E)
    X, P = init_state
    P = P[:, jnp.newaxis]

    F = jnp.broadcast_to(P, (P.shape[0], samples.shape[0])) * ((-1)**(X @ samples.T))

    H1 = ((-1) ** (bitflips @ X.T)) @ F
    col_sums = jnp.sum(F.conj(), axis=0, keepdims=True)
    H2 = jnp.broadcast_to(col_sums, (bitflips.shape[0], samples.shape[0]))
    H = H1 * H2

    return jnp.real(M * H)

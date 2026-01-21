import jax
import jax.numpy as jnp
from numpy.typing import ArrayLike


class IqpSimulator:
    def __init__(
        self, generators: ArrayLike, trainable: list[list[int]], fixed: list[list[int]] = None
    ):

        self.generators = generators
        self.trainable = trainable
        self.fixed = fixed
        self.n_generators, self.n_qubits = generators.shape

    def expval_gen_obs(self, params: ArrayLike, ops: ArrayLike, n_samples: int):
        key = jax.random.PRNGKey(42)
        key, subkey = jax.random.split(key, 2)
        key = subkey

        samples = jax.random.randint(key, (n_samples, self.n_qubits), 0, 2)

        bitflips = []
        phases = []

        for op in ops:
            generator = []
            for pauli in op:
                match pauli:
                    case "I":
                        generator.append(0)
                    case "X":
                        generator.append(1)
                    case "Y":
                        generator.append(1)
                    case "Z":
                        generator.append(0)
                    case _:
                        raise ValueError

                bitflips.append(generator)

    def expval(
        self, params: ArrayLike, ops: list[list[str]], n_samples: int, init_state: ArrayLike = None
    ):
        """Compute expval of a batch of Pauli Z operators

        ops - A
        params - Theta
        self.generators - G
        samples - Z
        init_state - X

        """

        if init_state is not None:
            return self._expval_init_state(params, ops, n_samples, init_state)

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

    def _expval_init_state(
        self, params: ArrayLike, ops: ArrayLike, n_samples: int, init_state: ArrayLike
    ):
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

        F = jnp.broadcast_to(P, (P.shape[0], n_samples)) * ((-1) ** (X @ samples.T))

        H1 = ((-1) ** (ops @ X.T)) @ F
        col_sums = jnp.sum(F.conj(), axis=0, keepdims=True)
        H2 = jnp.broadcast_to(col_sums, (ops.shape[0], n_samples))
        H = H1 * H2

        expvals = jnp.cos(E) * jnp.real(H) - jnp.sin(E) * jnp.imag(H)

        std_err = jnp.std(expvals, axis=-1, ddof=1) / jnp.sqrt(n_samples)
        return jnp.mean(expvals, axis=1), std_err


class BitflipSimulator:
    def __init__(
        self, generators: ArrayLike, trainable: list[list[int]], fixed: list[list[int]] = None
    ):

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

    if pauli in ("I", "X"):
        return 1

    if pauli == "Y":
        if qubit == 0:
            return 1j

        return -1j

    if pauli == "Z":
        if qubit == 0:
            return 1

        return -1

    raise ValueError(f"Expected Pauli I, X, Y, or Z, got {pauli}.")

import math
from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
from numpy.typing import ArrayLike

@dataclass
class CircuitConfig:
    n_samples: int
    n_qubits: int
    generators: list[list[int]]
    observables: list[list[str]]
    key: int
    init_state: ArrayLike = None
    phase_layer: Callable = None

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

def build_expval_func(config: CircuitConfig) -> Callable:
    samples = jax.random.randint(config.key, (config.n_samples, config.n_qubits), 0, 2)
    bitflips = jnp.array([[1 if gate in {"Z", "Y"} else 0 for gate in op] for op in config.observables])
    phases = jnp.array(
        [
            [
                math.prod([_phase(gate, qubit) for gate, qubit in zip(op, sample)])
                for sample in samples
            ]
            for op in config.observables
        ]
    )

    @jax.jit
    def expval(params, phase_params):
        B = (-1) ** (samples @ config.generators.T)
        C = 1 - ((-1) ** (bitflips @ config.generators.T))

        E = C @ jnp.diag(params) @ B.T

        if config.phase_layer is not None:
            def compute_phase(phase_params, sample, bitflips):
                return config.phase_layer(phase_params, sample) - config.phase_layer(phase_params, (sample + bitflips) % 2)

            phase_matrix = jax.vmap(compute_phase, in_axes=(None, 0, None))
            phase_matrix = jax.vmap(phase_matrix, in_axes=(None, None, 0))

            E += phase_matrix(phase_params, samples, bitflips)

        M = phases * jnp.exp(1j * E)

        if config.init_state is not None:
            X, P = config.init_state
            P = P[:, jnp.newaxis]
            F = jnp.broadcast_to(P, (P.shape[0], samples.shape[0])) * ((-1) ** (X @ samples.T))
            H1 = ((-1) ** (bitflips @ X.T)) @ F
            col_sums = jnp.sum(F.conj(), axis=0, keepdims=True)
            H2 = jnp.broadcast_to(col_sums, (bitflips.shape[0], samples.shape[0]))
            H = H1 * H2

            expvals = jnp.real(M * H)
        else:
            expvals = jnp.real(M)

        std_err = jnp.std(expvals, axis=-1, ddof=1) / jnp.sqrt(config.n_samples)
        return jnp.mean(expvals, axis=1), std_err

    return expval

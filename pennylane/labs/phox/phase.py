import jax
import jax.numpy as jnp
import numpy as np

import pennylane as qml
from pennylane.labs.phox.expval import CircuitConfig, build_expval_func

jax.config.update("jax_enable_x64", True)


def iqp_circuit_pl(generators, params, obs, init_state, phase_layer, phase_params):
    """Creates a PennyLane QNode for the IQP circuit."""
    n_qubits = len(obs)

    expval_ops = []
    for i, op in enumerate(obs):
        if op == "X":
            expval_ops.append(qml.X(i))
        elif op == "Y":
            expval_ops.append(qml.Y(i))
        elif op == "Z":
            expval_ops.append(qml.Z(i))
        elif op == "I":
            expval_ops.append(qml.Identity(i))

    expval_op = qml.prod(*expval_ops)

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit():
        # Start with specified initial state
        qml.StatePrep(np.array(init_state), wires=range(n_qubits))

        for i in range(n_qubits):
            qml.Hadamard(i)

        for param, gen in zip(params, generators):
            qml.MultiRZ(2 * -param, wires=gen)

        bitstrings = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        phases = jax.vmap(phase_layer, in_axes=(None, 0))(phase_params, bitstrings)
        diagonal = jnp.exp(1j * phases).flatten()

        qml.DiagonalQubitUnitary(diagonal, wires=[0, 1])

        for i in range(n_qubits):
            qml.Hadamard(i)

        return qml.expval(expval_op)

    return circuit


def compute_phase(params, z):
    hamming = jnp.mean(jnp.abs(z))
    hamming_powers = jnp.array([hamming**t for t in range(4)])
    return jnp.sum(params * hamming_powers)


if __name__ == "__main__":
    bitstrings = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    phase_params = jnp.array([0.11, 0.7, 3.0, 1.0])

    phases = jax.vmap(compute_phase, in_axes=(None, 0))(phase_params, bitstrings)
    diagonal = jnp.exp(1j * phases).flatten()

    obs = ["X", "Z"]
    gens = [[0], [1], [0, 1]]
    params = [0.37, 0.95, 0.73]
    state = [1, 0, 0, 0]

    config = CircuitConfig(
        n_qubits=2,
        generators=jnp.array([[1, 0], [0, 1], [1, 1]]),
        observables=[["X", "Z"]],
        init_state=(jnp.array([[0, 0]]), jnp.array([1])),
        phase_layer=compute_phase,
        n_samples=50000,
        key=jax.random.PRNGKey(42),
    )

    f = build_expval_func(config)
    expval, std = f(jnp.array(params), phase_params)

    print(expval)

    pl_circuit = iqp_circuit_pl(gens, params, obs, state, compute_phase, phase_params)

    print(pl_circuit())

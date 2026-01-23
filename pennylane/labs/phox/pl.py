import jax.numpy as jnp
import pennylane as qml
from simulator import IqpSimulator

def iqp_circuit(generators, params, obs, init_state):
    n_qubits = len(obs)

    expval_ops = []
    for i, op in enumerate(obs):
        match op:
            case "X":
                expval_ops.append(qml.X(i))
            case "Y":
                expval_ops.append(qml.Y(i))
            case "Z":
                expval_ops.append(qml.Z(i))

    expval_op = qml.prod(*expval_ops)

    dev = qml.device("default.qubit")
    @qml.qnode(dev)
    def circuit():
        qml.StatePrep(init_state, wires=[0, 1, 2])

        for i in range(n_qubits):
            qml.Hadamard(i)

        for param, gen in zip(params, generators):
            qml.MultiRZ(2* param, wires=gen)

        for i in range(n_qubits):
            qml.Hadamard(i)

        return qml.expval(expval_op)

    return circuit

if __name__ == '__main__':
    generators = [[0], [1], [0, 1, 2]]
    params = [0.37454012, 0.95071431, 0.73199394]
    obs = ["X", "Z", "X"]
    #state = [jnp.sqrt(2), 0, 0, 0, 0, 0, 0, jnp.sqrt(2)]
    state = [1] + [0]*7

    circuit = iqp_circuit(generators, params, obs, state)

    print(circuit())

    generators = jnp.array([[1, 0, 0], [0, 1, 0], [1, 1, 1]])
    params = jnp.array(params)
    obs = [obs]
    state = jnp.array([[0, 0, 0], [1, 1, 1]]), jnp.array([jnp.sqrt(2), jnp.sqrt(2)])

    simulator = IqpSimulator(generators, [])
    expval = simulator.expval_gen(params, obs, 10000)
    #expval = simulator.expval_gen(params, obs, 10000, init_state=state)

    print(expval)

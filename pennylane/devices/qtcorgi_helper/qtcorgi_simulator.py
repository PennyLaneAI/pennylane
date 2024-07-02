import functools
import jax
import jax.numpy as jnp
import pennylane as qml
from pennylane.tape import QuantumScript
from .apply_operations import apply_operation
from ..qutrit_mixed.simulate import measure_final_state
from ..qutrit_mixed.initialize_state import create_initial_state

op_types = []



def get_final_state_from_initial(operations, initial_state, qudit_dim):
    """
    TODO

    Args:
        circuit (.QuantumScript): The single circuit to simulate

    Returns:
        Tuple[TensorLike, bool]: A tuple containing the final state of the quantum script and
            whether the state has a batch dimension.

    """
    ops_type_index, ops_wires, ops_params = [], [[], []], [[], [], []]
    for op in operations:

        # op_index = None
        # for i, op_type in enumerate(op_types):
        #     if isinstance(op, op_type):
        #         op_index = i
        # if op_index is None:
        #     raise ValueError("This simulator only supports")

        # op_index = op_types.index(type(op))
        # wires = op.wires()
        # if len(wires) == 1:
        #     wires = [-1, wires[0]]
        # if len(wires) == 2:
        #     wires = list(wires)
        # params = op.parameters + ([0] * (3-op.num_params))
        # op_array.append([[op_index] + wires, params])

        ops_type_index.append(op_types.index(type(op)))
        wires = op.wires()
        if len(wires) == 1:
            wires = [-1, wires[0]]
        ops_wires[0].append(wires[0])
        ops_wires[1].append(wires[1])

        params = op.parameters + ([0] * (3 - op.num_params))
        ops_params[0].append(params[0])
        ops_params[1].append(params[1])
        ops_params[2].append(params[2])

    ops_info = {
        "type_index": jnp.array(ops_type_index),
        "wires": [jnp.array(ops_wires[0]), jnp.array(ops_wires[1])],
        "params": [jnp.array(ops_params[0]), jnp.array(ops_params[1]), jnp.array(ops_params[2])]
    }
    return jax.lax.scan(apply_operation, initial_state, qudit_dim, ops_info)[0]


def get_final_state(circuit):
    """
    TODO

    Args:
        circuit (.QuantumScript): The single circuit to simulate
        qudit_dim (): TODO
        interface (str): The machine learning interface to create the initial state with

    Returns:
        Tuple[TensorLike, bool]: A tuple containing the final state of the quantum script and
            whether the state has a batch dimension.

    """

    circuit = circuit.map_to_standard_wires()

    prep = None
    if len(circuit) > 0 and isinstance(circuit[0], qml.operation.StatePrepBase):
        prep = circuit[0]

    state = create_initial_state(sorted(circuit.op_wires), prep, like="jax")
    get_final_state_from_initial(circuit.operations[bool(prep):], state, 3)





def simulate(circuit: QuantumScript, rng=None, prng_key=None):
    """TODO

    Args:
        circuit (QuantumTape): The single circuit to simulate
        rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
            seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
            If no value is provided, a default RNG will be used.
        prng_key (Optional[jax.random.PRNGKey]): An optional ``jax.random.PRNGKey``. This is
            the key to the JAX pseudo random number generator. If None, a random key will be
            generated. Only for simulation using JAX.
        debugger (_Debugger): The debugger to use
        interface (str): The machine learning interface to create the initial state with

    Returns:
        tuple(TensorLike): The results of the simulation

    Note that this function can return measurements for non-commuting observables simultaneously.

    This function assumes that all operations provide matrices.

    >>> qs = qml.tape.QuantumScript([qml.TRX(1.2, wires=0)], [qml.expval(qml.GellMann(0, 3)), qml.probs(wires=(0,1))])
    >>> simulate(qs)
    (0.36235775447667357,
    tensor([0.68117888, 0.        , 0.        , 0.31882112, 0.        , 0.        ], requires_grad=True))

    """
    state, is_state_batched = get_final_state(circuit)
    return measure_final_state(circuit, state, is_state_batched, rng=rng, prng_key=prng_key)

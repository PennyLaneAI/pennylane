import jax
import jax.numpy as jnp
import pennylane as qml
from pennylane.tape import QuantumScript
from .apply_operations import qubit_branches, qutrit_branches
from ..qutrit_mixed.initialize_state import create_initial_state

op_types = []


def get_qubit_final_state_from_initial(operations, initial_state):
    """
    TODO

    Args:
        TODO

    Returns:
        Tuple[TensorLike, bool]: A tuple containing the final state of the quantum script and
            whether the state has a batch dimension.

    """
    ops_type_indices, ops_wires, ops_param = [[], []], [[], []], []
    for op in operations:

        wires = op.wires()

        if isinstance(op, qml.operation.Channel):
            ops_type_indices[0].append(2)
            ops_type_indices[1].append([].index(type(op)))
        elif len(wires) == 1:
            ops_type_indices[0].append(0)
            ops_type_indices[1].append([].index(type(op)))
        elif len(wires) == 1:
            ops_type_indices[0].append(0)
            ops_type_indices[1].append([].index(type(op)))

        if len(wires) == 1:
            wires = [wires[0], -1]
            params = op.parameters + ([0] * (3 - op.num_params))
        ops_wires[0].append(wires[0])
        ops_wires[1].append(wires[1])

        ops_param[0].append(params[0])

        op_index = op_types.index(type(op))
        ops_type_index.append(op_index)

    ops_info = {
        "type_index": jnp.array(ops_type_index),
        "wires": [jnp.array(ops_wires[0]), jnp.array(ops_wires[1])],
        "params": [jnp.array(ops_param)],
    }

    return jax.lax.scan(
        lambda state, op_info: (
            jax.lax.switch(op_info["branch"], qubit_branches, state, op_info),
            None,
        ),
        initial_state,
        ops_info,
    )[0]


def get_qutrit_final_state_from_initial(operations, initial_state):
    """
    TODO

    Args:
        TODO

    Returns:
        Tuple[TensorLike, bool]: A tuple containing the final state of the quantum script and
            whether the state has a batch dimension.

    """
    ops_type_index, ops_subspace, ops_wires, ops_params = [], [], [[], []], [[], [], []]
    for op in operations:
        op_index = op_types.index(type(op))
        ops_type_index.append(op_index)

        wires = op.wires()
        if len(wires) == 1:
            wires = [wires[0], -1]
            params = op.parameters + ([0] * (3 - op.num_params))

        ops_wires[0].append(wires[0])
        ops_wires[1].append(wires[1])

        ops_params[0].append(params[0])
        ops_params[1].append(params[1])
        ops_params[2].append(params[2])

        if op_index <= 2:
            ops_subspace.append([(0, 1), (0, 2), (1, 2)].index(op.subspace))
        else:
            ops_subspace.append(0)

    ops_info = {
        "type_index": jnp.array(ops_type_index),
        "wires": [jnp.array(ops_wires[0]), jnp.array(ops_wires[1])],
        "params": [jnp.array(ops_params[0]), jnp.array(ops_params[1]), jnp.array(ops_params[2])],
    }
    op_branch = jnp.nan

    return jax.lax.scan(
        lambda state, op_info: (jax.lax.switch(op_info["branch"], qutrit_branches, state, x), None),
        initial_state,
        ops_info,
    )[0]


def get_final_state_qutrit(circuit):
    """
    TODO

    Args:
        circuit (.QuantumScript): The single circuit to simulate

    Returns:
        Tuple[TensorLike, bool]: A tuple containing the final state of the quantum script and
            whether the state has a batch dimension.

    """

    circuit = circuit.map_to_standard_wires()

    prep = None
    if len(circuit) > 0 and isinstance(circuit[0], qml.operation.StatePrepBase):
        prep = circuit[0]

    state = create_initial_state(sorted(circuit.op_wires), prep, like="jax")
    return get_qutrit_final_state_from_initial(circuit.operations[bool(prep) :], state), False

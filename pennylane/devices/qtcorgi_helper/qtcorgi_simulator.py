import jax
import jax.numpy as jnp
import pennylane as qml
from pennylane.operation import Channel
from .apply_operations import qubit_branches, qutrit_branches

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

        if isinstance(op, Channel):
            ops_type_indices[0].append(2)
            ops_type_indices[1].append([].index(type(op)))
        elif len(wires) == 1:
            ops_type_indices[0].append(0)
            ops_type_indices[1].append([qml.RX, qml.RY, qml.RZ, qml.Hadamard].index(type(op)))
        elif len(wires) == 2:
            ops_type_indices[0].append(1)
            ops_type_indices[1].append(0)  # Assume always CNOT
        else:
            raise ValueError("TODO")

        if len(wires) == 1:
            wires = [wires[0], -1]
            params = op.parameters + ([0] * (3 - op.num_params))
        ops_wires[0].append(wires[0])
        ops_wires[1].append(wires[1])

        ops_param[0].append(params[0])

    ops_info = {
        "type_index": jnp.array(ops_type_indices),
        "wires": [jnp.array(ops_wires[0]), jnp.array(ops_wires[1])],
        "params": [jnp.array(ops_param)],
    }

    return jax.lax.scan(
        lambda state, op_info: (
            jax.lax.switch(op_info["type_indices"][0], qubit_branches, state, op_info),
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
    ops_type_indices, ops_subspace, ops_wires, ops_params = [[], []], [], [[], []], [[], [], []]
    for op in operations:

        wires = op.wires()

        if isinstance(op, Channel):
            ops_type_indices[0].append(2)
            ops_type_indices[1].append(
                [qml.QutritDepolarizingChannel, qml.QutritAmplitudeDamping, qml.TritFlip].index(
                    type(op)
                )
            )
            params = op.parameters + ([0] * (3 - op.num_params))
        elif len(wires) == 1:
            ops_type_indices[0].append(0)
            ops_type_indices[1].append([qml.TRX, qml.TRY, qml.TRZ, qml.THadamard].index(type(op)))
            if ops_type_indices[1][-1] == 3:
                params = [0] + list(op.subspace) if op.subspace is not None else [0, 0]
            else:
                params = list(op.params) + list(op.subspace)
        elif len(wires) == 2:
            ops_type_indices[0].append(1)
            ops_type_indices[1].append(0)  # Assume always TAdd
            params = [0, 0, 0]
        else:
            raise ValueError("TODO")
        ops_params[0].append(params[0])
        ops_params[1].append(params[1])
        ops_params[2].append(params[2])

        if len(wires) == 1:
            wires = [wires[0], -1]
        ops_wires[0].append(wires[0])
        ops_wires[1].append(wires[1])

    ops_info = {
        "type_indices": jnp.array(ops_type_indices),
        "wires": [jnp.array(ops_wires[0]), jnp.array(ops_wires[1])],
        "params": [jnp.array(ops_params[0]), jnp.array(ops_params[1]), jnp.array(ops_params[2])],
    }

    return jax.lax.scan(
        lambda state, op_info: (
            jax.lax.switch(op_info["type_indices"][0], qutrit_branches, state, op_info),
            None,
        ),
        initial_state,
        ops_info,
    )[0]

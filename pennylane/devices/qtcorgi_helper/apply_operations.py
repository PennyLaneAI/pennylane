import time
import jax
import jax.numpy as jnp
from jax.lax import scan

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")
import pennylane as qml
from string import ascii_letters as alphabet

import numpy as np
from functools import partial, reduce

alphabet_array = np.array(list(alphabet))


def get_einsum_mapping(wires, state):
    r"""Finds the indices for einsum to apply kraus operators to a mixed state

    Args:
        wires
        state (array[complex]): Input quantum state

    Returns:
        str: Indices mapping that defines the einsum
    """
    num_ch_wires = len(wires)
    num_wires = int(len(qml.math.shape(state)) / 2)
    rho_dim = 2 * num_wires

    # Tensor indices of the state. For each qutrit, need an index for rows *and* columns
    state_indices = alphabet[:rho_dim]

    # row indices of the quantum state affected by this operation
    row_wires_list = wires.tolist()
    row_indices = "".join(alphabet_array[row_wires_list].tolist())

    # column indices are shifted by the number of wires
    col_wires_list = [w + num_wires for w in row_wires_list]
    col_indices = "".join(alphabet_array[col_wires_list].tolist())

    # indices in einsum must be replaced with new ones
    new_row_indices = alphabet[rho_dim : rho_dim + num_ch_wires]
    new_col_indices = alphabet[rho_dim + num_ch_wires : rho_dim + 2 * num_ch_wires]

    # index for summation over Kraus operators
    kraus_index = alphabet[rho_dim + 2 * num_ch_wires : rho_dim + 2 * num_ch_wires + 1]

    # apply mapping function
    op_1_indices = f"{kraus_index}{new_row_indices}{row_indices}"
    op_2_indices = f"{kraus_index}{col_indices}{new_col_indices}"

    new_state_indices = get_new_state_einsum_indices(
        old_indices=col_indices + row_indices,
        new_indices=new_col_indices + new_row_indices,
        state_indices=state_indices,
    )
    # index mapping for einsum, e.g., '...iga,...abcdef,...idh->...gbchef'
    return f"...{op_1_indices},...{state_indices},...{op_2_indices}->...{new_state_indices}"


def get_new_state_einsum_indices(old_indices, new_indices, state_indices):
    """Retrieves the einsum indices string for the new state

    Args:
        old_indices (str): indices that are summed
        new_indices (str): indices that must be replaced with sums
        state_indices (str): indices of the original state

    Returns:
        str: The einsum indices of the new state
    """
    return reduce(
        lambda old_string, idx_pair: old_string.replace(idx_pair[0], idx_pair[1]),
        zip(old_indices, new_indices),
        state_indices,
    )


QUDIT_DIM = 3


def apply_operation_einsum(kraus, wires, state):
    r"""Apply a quantum channel specified by a list of Kraus operators to subsystems of the
    quantum state. For a unitary gate, there is a single Kraus operator.

    Args:
        kraus (??): TODO
        wires
        state (array[complex]): Input quantum state

    Returns:
        array[complex]: output_state
    """
    einsum_indices = get_einsum_mapping(wires, state)

    num_ch_wires = len(wires)

    # Shape kraus operators
    kraus_shape = [len(kraus)] + [QUDIT_DIM] * num_ch_wires * 2

    kraus = jnp.stack(kraus)
    kraus_transpose = jnp.stack(jnp.moveaxis(kraus, source=-1, destination=-2))
    # Torch throws error if math.conj is used before stack
    kraus_dagger = jnp.conj(kraus_transpose)

    kraus = jnp.reshape(kraus, kraus_shape)
    kraus_dagger = jnp.reshape(kraus_dagger, kraus_shape)

    return jnp.einsum(einsum_indices, kraus, state, kraus_dagger)


def get_two_qubit_unitary_matrix():
    pass


def get_CNOT_matrix(params):
    return jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])


single_qubit_ops = [qml.RX.compute_matrix, qml.RY.compute_matrix, qml.RZ.compute_matrix]
two_qubit_ops = [get_CNOT_matrix, get_two_qubit_unitary_matrix]
single_qubit_channels = [
    qml.DepolarizingChannel.compute_kraus_matrices,
    qml.AmplitudeDamping.compute_kraus_matrices,
    qml.BitFlip.compute_kraus_matrices,
]


def apply_single_qubit_unitary(state, op_info):
    wires, param = op_info["wires"][:0], op_info["params"][0]
    kraus_mat = jax.lax.switch(op_info["type_indices"][1], single_qubit_ops, param)
    return apply_operation_einsum(kraus_mat, wires, state)


def apply_two_qubit_unitary(state, op_info):
    wires, params = op_info["wires"], op_info["params"]
    kraus_mats = [jax.lax.switch(op_info["type_indices"][1], two_qubit_ops, params)]
    return apply_operation_einsum(kraus_mats, wires, state)


def apply_single_qubit_channel(state, op_info):
    wires, param = op_info["wires"][:0], op_info["params"][0]
    kraus_mats = [jax.lax.switch(op_info["type_indices"][1], single_qubit_channels, param)]
    return apply_operation_einsum(kraus_mats, wires, state)


qubit_branches = [apply_single_qubit_unitary, apply_two_qubit_unitary, apply_single_qubit_channel]


single_qutrit_ops = [
    qml.TRX.compute_matrix,
    qml.TRY.compute_matrix,
    qml.TRZ.compute_matrix,
    lambda params: (
        qml.THadamard.compute_matrix()
        if params[1] != 0
        else qml.THadamard.compute_matrix(subspace=params[1:])
    ),
]
single_qutrit_channels = [
    lambda params: qml.QutritDepolarizingChannel.compute_kraus_matrices(params[0]),
    lambda params: qml.QutritAmplitudeDamping.compute_kraus_matrices(*params),
    lambda params: qml.TritFlip.compute_kraus_matrices(*params),
]


def apply_single_qutrit_unitary(state, op_info):
    wires, param = op_info["wires"][:0], op_info["params"][0]
    kraus_mats = [jax.lax.switch(op_info["type_indices"][1], single_qutrit_ops, param)]
    return apply_operation_einsum(kraus_mats, wires, state)


def apply_two_qutrit_unitary(state, op_info):
    wires = op_info["wires"]
    kraus_mat = [
        jnp.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
            ]
        )
    ]
    return apply_operation_einsum(kraus_mat, wires, state)


def apply_single_qutrit_channel(state, op_info):
    wires, params = op_info["wires"][:0], op_info["params"]  # TODO qutrit channels take 3 params
    kraus_mats = [jax.lax.switch(op_info["type_indices"][1], single_qutrit_channels, *params)]
    return apply_operation_einsum(kraus_mats, wires, state)


qutrit_branches = [
    apply_single_qutrit_unitary,
    apply_two_qutrit_unitary,
    apply_single_qutrit_channel,
]

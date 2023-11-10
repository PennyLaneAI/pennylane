from typing import Iterable, Union
import pennylane as qml
from pennylane import (
    QutritBasisState,
)

qudit_dim = 3  # specifies qudit dimension


def create_initial_state(
    wires: Union[qml.wires.Wires, Iterable],
    prep_operation: qml.Operation = None,
    like: str = None,
):
    r"""
    Returns an initial state, defaulting to :math:`\ket{0}\bra{0}` if no state-prep operator is provided.

    Args:
        wires (Union[Wires, Iterable]): The wires to be present in the initial state
        prep_operation (Optional[StatePrepBase]): An operation to prepare the initial state
        like (Optional[str]): The machine learning interface used to create the initial state.
            Defaults to None

    Returns:
        array: The initial state of a circuit
    """
    if not prep_operation:
        num_wires = len(wires)
        rho = _create_basis_state(num_wires, 0)
    elif isinstance(prep_operation, QutritBasisState):
        rho = _apply_basis_state(prep_operation.parameters[0], wires)
    else:
        raise NotImplementedError(f"{prep_operation} has not been implemented for qutrit.mixed")

    return qml.math.asarray(rho, like=like)


def _apply_basis_state(state, wires):  # function is easy to abstract for qudit
    """Returns initial state for a specified computational basis state.

    Args:
        state (array[int]): computational basis state of shape ``(wires,)``
            consisting of 0s and 1s.
        wires (Wires): wires that the provided computational state should be initialized on

    """
    num_wires = len(wires)
    # length of basis state parameter
    n_basis_state = len(state)

    if not set(state).issubset({0, 1}):
        raise ValueError("BasisState parameter must consist of 0 or 1 integers.")

    if n_basis_state != len(wires):
        raise ValueError("BasisState parameter and wires must be of equal length.")

    # get computational basis state number
    basis_states = qudit_dim ** (num_wires - 1 - wires.toarray())
    num = int(qml.math.dot(state, basis_states))

    return _create_basis_state(num_wires, num)


def _create_basis_state(num_wires, index, dtype):  # function is easy to abstract for qudit
    """Return the density matrix representing a computational basis state over all wires.

    Args:
        index (int): integer representing the computational basis state.

    Returns:
        array[complex]: complex array of shape ``[3] * (2 * num_wires)``
        representing the density matrix of the basis state.
    """
    rho = qml.math.zeros((3**num_wires, 3**num_wires), dtype=dtype)
    rho[index, index] = 1
    return qml.math.reshape(rho, [qudit_dim] * (2 * num_wires))

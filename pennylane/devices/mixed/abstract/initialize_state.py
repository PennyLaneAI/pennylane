from typing import Iterable, Union
import pennylane as qml
import numpy as np


def apply_basis_state(state, wires, qudit_d=2):
    """Initialize the device in a specified computational basis state.

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
    basis_states = qudit_d ** (num_wires - 1 - wires.toarray())
    num = int(qml.math.dot(state, basis_states))

    return _create_basis_state(num_wires, num)


def create_basis_state(wires, index=0, qudit_d=2):
    """Return the density matrix representing a computational basis state over all wires.

    Args:
        index (int): integer representing the computational basis state.
        qudit_d (int): dimensions of qudit being used

    Returns:
        array[complex]: complex array of shape ``[qudit_d] * (2 * num_wires)``
        representing the density matrix of the basis state.
    """
    num_wires = len(wires)
    rho = np.zeros((qudit_d ** num_wires, qudit_d ** num_wires))
    rho[index, index] = 1
    return qml.math.reshape(rho, [qudit_d] * (2 * num_wires))

class CreateStateHelper:
    def __init__(self, qudit_d):
        self.qudit_d = qudit_d

    def apply_basis_state(self, state, wires):
        """Initialize the device in a specified computational basis state.

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
        basis_states = self.qudit_d ** (num_wires - 1 - wires.toarray())
        num = int(qml.math.dot(state, basis_states))

        return self.create_basis_state(num_wires, num)

    def create_basis_state(self, wires, index=0):
        """Return the density matrix representing a computational basis state over all wires.

        Args:
            index (int): integer representing the computational basis state.
            qudit_d (int): dimensions of qudit being used

        Returns:
            array[complex]: complex array of shape ``[qudit_d] * (2 * num_wires)``
            representing the density matrix of the basis state.
        """
        num_wires = len(wires)
        rho = np.zeros((self.qudit_d ** num_wires, self.qudit_d ** num_wires))
        rho[index, index] = 1
        return qml.math.reshape(rho, [self.qudit_d] * (2 * num_wires))
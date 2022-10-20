from functools import reduce
from string import ascii_letters as ABC
from pennylane.tape import QuantumScript

import numpy as np


ABC_ARRAY = np.array(list(ABC))


def _get_slice(index, axis, num_axes):
    idx = [slice(None)] * num_axes
    idx[axis] = index
    return tuple(idx)


class PlainNumpySimulator:
    """

    Current Restrictions:
    * No batching

    * No support for state preparation yet
    * No sampling yet
    * restricted measurement types

    Preprocessing restrictions:
    * Quantum Script wires must be adjacent integers starting from zero
    * All operations must have matrices



    """

    name = "PlainNumpy"

    def __init__(self):
        pass

    @classmethod
    def execute(cls, quantumscript: QuantumScript, dtype=np.complex128):

        num_indices = len(quantumscript.wires)
        state = cls.create_zeroes_state(num_indices, dtype=dtype)
        for op in quantumscript._ops:
            state = cls.apply_operation(state, op)

        return tuple(cls.measure(state, m) for m in quantumscript.measurements)

    @staticmethod
    def create_zeroes_state(num_indices, dtype=np.complex128):

        state = np.zeros(2**num_indices, dtype=dtype)
        state[0] = 1
        state.shape = [2] * num_indices
        return state

    @staticmethod
    def create_state_vector_state(num_indices, statevector, indices):
        if list(range(num_indices)) == indices:
            statevector.shape = [2] * num_indices
            return statevector
        raise NotImplementedError

    @classmethod
    def apply_operation(cls, state, operation):
        """ """
        matrix = operation.matrix()
        if len(operation.wires) < 3:
            return cls.apply_matrix_einsum(state, matrix, operation.wires)
        return cls.apply_matrix_tensordot(state, matrix, operation.wires)

    @classmethod
    def apply_matrix(cls, state, matrix, indices):
        if len(indices) < 3:
            return cls.apply_matrix_einsum(state, matrix, indices)
        return cls.apply_matrix_tensordot(state, matrix, indices)

    @staticmethod
    def apply_matrix_tensordot(state, matrix, indices):
        """ """
        total_indices = len(state.shape)
        num_indices = len(indices)
        reshaped_mat = np.reshape(matrix, [2] * (num_indices * 2))
        axes = (tuple(range(num_indices, 2 * num_indices)), indices)

        tdot = np.tensordot(reshaped_mat, state, axes=axes)

        unused_idxs = [i for i in range(total_indices) if i not in indices]
        perm = list(indices) + unused_idxs
        inv_perm = np.argsort(perm)

        return np.transpose(tdot, inv_perm)

    @staticmethod
    def apply_matrix_einsum(state, matrix, indices):
        """
        Args:
            state (array[complex]): input state
            mat (array): matrix to multiply
            indices (Iterable[integer]): indices to apply the matrix on

        Returns:
            array[complex]: output_state
        """
        total_indices = len(state.shape)
        num_indices = len(indices)

        state_indices = ABC[:total_indices]
        affected_indices = "".join(ABC[i] for i in indices)

        new_indices = ABC[total_indices : total_indices + num_indices]

        new_state_indices = state_indices
        for old, new in zip(affected_indices, new_indices):
            new_state_indices = new_state_indices.replace(old, new)

        einsum_indices = f"{new_indices}{affected_indices},{state_indices}->{new_state_indices}"

        reshaped_mat = np.reshape(matrix, [2] * (num_indices * 2))

        return np.einsum(einsum_indices, reshaped_mat, state)

    @classmethod
    def measure(cls, state, measurementprocess):
        mp_type = measurementprocess.return_type.value

        mp_map = {"probs": cls.probability, "expval": cls.expval, "state": lambda state, mp: state}
        if mp_type in mp_map:
            return mp_map[mp_type](state, measurementprocess)
        return state

    @classmethod
    def expval(cls, state, measurementprocess):

        mat = measurementprocess.obs.matrix()
        new_state = cls.apply_matrix(state, mat, measurementprocess.obs.wires)
        return np.vdot(state, new_state)

    @staticmethod
    def probability(state, measurementprocess):
        """
        Args:
            state

        """
        indices = measurementprocess.wires
        num_indices = len(indices)
        total_indices = len(state.shape)
        full_probs = np.absolute(state) ** 2

        if indices is None or list(indices) == list(range(total_indices)):
            return full_probs

        inactive_indices = tuple(i for i in range(total_indices) if i not in indices)
        marginal_probs = np.sum(full_probs, axis=inactive_indices)

        shrinking_map = {idx: new_idx for new_idx, idx in enumerate(sorted(indices))}
        new_indices = tuple(shrinking_map[i] for i in indices)
        old_indices = tuple(range(num_indices))
        return np.moveaxis(marginal_probs, source=old_indices, destination=new_indices)

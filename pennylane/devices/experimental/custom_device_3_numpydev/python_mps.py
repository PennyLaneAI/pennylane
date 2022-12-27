from functools import reduce
from string import ascii_letters as ABC
from pennylane.tape import QuantumScript

import numpy as np

import pennylane as qml


def _get_slice(index, axis, num_axes):
    idx = [slice(None)] * num_axes
    idx[axis] = index
    return tuple(idx)


class NumpyMPSSimulator:
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
    def execute(cls, qs: QuantumScript, dtype=np.complex128):
        num_indices = len(qs.wires)
        state = cls.create_zeroes_state(num_indices, dtype=dtype)
        for op in qs._ops:
            state = cls.apply_operation(state, op)

        measurements = tuple(cls.measure_state(state, m) for m in qs.measurements)
        return measurements[0] if len(measurements) == 1 else measurements

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
    def measure_state(cls, state, measurementprocess):
        if isinstance(measurementprocess, qml.measurements.StateMeasurement):
            total_indices = len(state.shape)
            wires = qml.wires.Wires(range(total_indices))
            if (
                measurementprocess.obs is not None
                and measurementprocess.obs.has_diagonalizing_gates
            ):
                for op in measurementprocess.obs.diagonalizing_gates():
                    state = cls.apply_operation(state, op)
            return measurementprocess.process_state(state.flatten(), wires)
        return state

    @classmethod
    def generate_samples(cls, state, rng, shots=1):
        total_indices = len(state.shape)
        probs = np.real(state) ** 2 + np.imag(state) ** 2
        basis_states = np.arange(2**total_indices)
        samples = rng.choice(basis_states, shots, p=probs.flatten())

        powers_of_two = 1 << np.arange(total_indices, dtype=np.int64)
        # `samples` typically is one-dimensional, but can be two-dimensional with broadcasting.
        # In any case we want to append a new axis at the *end* of the shape.
        states_sampled_base_ten = samples[..., None] & powers_of_two
        # `states_sampled_base_ten` can be two- or three-dimensional. We revert the *last* axis.
        return (states_sampled_base_ten > 0).astype(np.int64)[..., ::-1]



"""Toy code implementing a matrix product state."""
# Copyright 2018-2021 TeNPy Developers, GNU GPLv3

import numpy as np
from scipy.linalg import svd
# if you get an error message "LinAlgError: SVD did not converge",
# uncomment the following line. (This requires TeNPy to be installed.)
#  from tenpy.linalg.svd_robust import svd  # (works like scipy.linalg.svd)

import warnings


class SimpleMPS:
    """Simple class for a matrix product state.

    We index sites with `i` from 0 to L-1; bond `i` is left of site `i`.
    We *assume* that the state is in right-canonical form.

    Parameters
    ----------
    Bs, Ss, bc:
        Same as attributes.

    Attributes
    ----------
    Bs : list of np.Array[ndim=3]
        The 'matrices', in right-canonical form, one for each physical site
        (within the unit-cell for an infinite MPS).
        Each `B[i]` has legs (virtual left, physical, virtual right), in short ``vL i vR``.
    Ss : list of np.Array[ndim=1]
        The Schmidt values at each of the bonds, ``Ss[i]`` is left of ``Bs[i]``.
    bc : 'infinite', 'finite'
        Boundary conditions.
    L : int
        Number of sites (in the unit-cell for an infinite MPS).
    nbonds : int
        Number of (non-trivial) bonds: L-1 for 'finite' boundary conditions, L for 'infinite'.
    """
    def __init__(self, Bs, Ss, bc='finite'):
        assert bc in ['finite', 'infinite']
        self.Bs = Bs
        self.Ss = Ss
        self.bc = bc
        self.L = len(Bs)
        self.nbonds = self.L - 1 if self.bc == 'finite' else self.L

    def copy(self):
        return SimpleMPS([B.copy() for B in self.Bs], [S.copy() for S in self.Ss], self.bc)

    def get_theta1(self, i):
        """Calculate effective single-site wave function on sites i in mixed canonical form.

        The returned array has legs ``vL, i, vR`` (as one of the Bs).
        """
        return np.tensordot(np.diag(self.Ss[i]), self.Bs[i], [1, 0])  # vL [vL'], [vL] i vR

    def get_theta2(self, i):
        """Calculate effective two-site wave function on sites i,j=(i+1) in mixed canonical form.

        The returned array has legs ``vL, i, j, vR``.
        """
        j = (i + 1) % self.L
        return np.tensordot(self.get_theta1(i), self.Bs[j], [2, 0])  # vL i [vR], [vL] j vR

    def get_chi(self):
        """Return bond dimensions."""
        return [self.Bs[i].shape[2] for i in range(self.nbonds)]

    def site_expectation_value(self, op):
        """Calculate expectation values of a local operator at each site."""
        result = []
        for i in range(self.L):
            theta = self.get_theta1(i)  # vL i vR
            op_theta = np.tensordot(op, theta, axes=(1, 1))  # i [i*], vL [i] vR
            result.append(np.tensordot(theta.conj(), op_theta, [[0, 1, 2], [1, 0, 2]]))
            # [vL*] [i*] [vR*], [i] [vL] [vR]
        return np.real_if_close(result)

    def bond_expectation_value(self, op):
        """Calculate expectation values of a local operator at each bond."""
        result = []
        for i in range(self.nbonds):
            theta = self.get_theta2(i)  # vL i j vR
            op_theta = np.tensordot(op[i], theta, axes=([2, 3], [1, 2]))
            # i j [i*] [j*], vL [i] [j] vR
            result.append(np.tensordot(theta.conj(), op_theta, [[0, 1, 2, 3], [2, 0, 1, 3]]))
            # [vL*] [i*] [j*] [vR*], [i] [j] [vL] [vR]
        return np.real_if_close(result)

    def entanglement_entropy(self):
        """Return the (von-Neumann) entanglement entropy for a bipartition at any of the bonds."""
        bonds = range(1, self.L) if self.bc == 'finite' else range(0, self.L)
        result = []
        for i in bonds:
            S = self.Ss[i].copy()
            S[S < 1.e-20] = 0.  # 0*log(0) should give 0; avoid warning or NaN.
            S2 = S * S
            assert abs(np.linalg.norm(S) - 1.) < 1.e-13
            result.append(-np.sum(S2 * np.log(S2)))
        return np.array(result)

    def correlation_length(self):
        """Diagonalize transfer matrix to obtain the correlation length."""
        from scipy.sparse.linalg import eigs
        if self.get_chi()[0] > 100:
            warnings.warn("Skip calculating correlation_length() for large chi: could take long")
            return -1.
        assert self.bc == 'infinite'  # works only in the infinite case
        B = self.Bs[0]  # vL i vR
        chi = B.shape[0]
        T = np.tensordot(B, np.conj(B), axes=(1, 1))  # vL [i] vR, vL* [i*] vR*
        T = np.transpose(T, [0, 2, 1, 3])  # vL vL* vR vR*
        for i in range(1, self.L):
            B = self.Bs[i]
            T = np.tensordot(T, B, axes=(2, 0))  # vL vL* [vR] vR*, [vL] i vR
            T = np.tensordot(T, np.conj(B), axes=([2, 3], [0, 1]))
            # vL vL* [vR*] [i] vR, [vL*] [i*] vR*
        T = np.reshape(T, (chi**2, chi**2))
        # Obtain the 2nd largest eigenvalue
        eta = eigs(T, k=2, which='LM', return_eigenvectors=False, ncv=20)
        xi =  -self.L / np.log(np.min(np.abs(eta)))
        if xi > 1000.:
            return np.inf
        return xi

    def correlation_function(self, op_i, i, op_j, j):
        """Correlation function between two distant operators on sites i < j.

        Note: calling this function in a loop over `j` is inefficient for large j >> i.
        The optimization is left as an exercise to the user.
        Hint: Re-use the partial contractions up to but excluding site `j`.
        """
        assert i < j
        theta = self.get_theta1(i) # vL i vR
        C = np.tensordot(op_i, theta, axes=(1, 1)) # i [i*], vL [i] vR
        C = np.tensordot(theta.conj(), C, axes=([0, 1], [1, 0]))  # [vL*] [i*] vR*, [i] [vL] vR
        for k in range(i + 1, j):
            k = k % self.L
            B = self.Bs[k]  # vL k vR
            C = np.tensordot(C, B, axes=(1, 0)) # vR* [vR], [vL] k vR
            C = np.tensordot(B.conj(), C, axes=([0, 1], [0, 1])) # [vL*] [k*] vR*, [vR*] [k] vR
        j = j % self.L
        B = self.Bs[j]  # vL k vR
        C = np.tensordot(C, B, axes=(1, 0)) # vR* [vR], [vL] j vR
        C = np.tensordot(op_j, C, axes=(1, 1))  # j [j*], vR* [j] vR
        C = np.tensordot(B.conj(), C, axes=([0, 1, 2], [1, 0, 2])) # [vL*] [j*] [vR*], [j] [vR*] [vR]
        return C

def split_truncate_theta(theta, chi_max, eps):
    """Split and truncate a two-site wave function in mixed canonical form.

    Split a two-site wave function as follows::
          vL --(theta)-- vR     =>    vL --(A)--diag(S)--(B)-- vR
                |   |                       |             |
                i   j                       i             j

    Afterwards, truncate in the new leg (labeled ``vC``).

    Parameters
    ----------
    theta : np.Array[ndim=4]
        Two-site wave function in mixed canonical form, with legs ``vL, i, j, vR``.
    chi_max : int
        Maximum number of singular values to keep
    eps : float
        Discard any singular values smaller than that.

    Returns
    -------
    A : np.Array[ndim=3]
        Left-canonical matrix on site i, with legs ``vL, i, vC``
    S : np.Array[ndim=1]
        Singular/Schmidt values.
    B : np.Array[ndim=3]
        Right-canonical matrix on site j, with legs ``vC, j, vR``
    """
    chivL, dL, dR, chivR = theta.shape
    theta = np.reshape(theta, [chivL * dL, dR * chivR])
    X, Y, Z = svd(theta, full_matrices=False)
    # truncate
    chivC = min(chi_max, np.sum(Y > eps))
    assert chivC >= 1
    piv = np.argsort(Y)[::-1][:chivC]  # keep the largest `chivC` singular values
    X, Y, Z = X[:, piv], Y[piv], Z[piv, :]
    # renormalize
    S = Y / np.linalg.norm(Y)  # == Y/sqrt(sum(Y**2))
    # split legs of X and Z
    A = np.reshape(X, [chivL, dL, chivC])
    B = np.reshape(Z, [chivC, dR, chivR])
    return A, S, B

def update_bond(psi, i, U_bond, chi_max, eps):
    """Apply `U_bond` acting on i,j=(i+1) to `psi`."""
    j = (i + 1) % psi.L
    # construct theta matrix
    theta = psi.get_theta2(i)  # vL i j vR
    # apply U
    Utheta = np.tensordot(U_bond, theta, axes=([2, 3], [1, 2]))  # i j [i*] [j*], vL [i] [j] vR
    Utheta = np.transpose(Utheta, [2, 0, 1, 3])  # vL i j vR
    # split and truncate
    Ai, Sj, Bj = split_truncate_theta(Utheta, chi_max, eps)
    # put back into MPS
    Gi = np.tensordot(np.diag(psi.Ss[i]**(-1)), Ai, axes=(1, 0))  # vL [vL*], [vL] i vC
    psi.Bs[i] = np.tensordot(Gi, np.diag(Sj), axes=(2, 0))  # vL i [vC], [vC] vC
    psi.Ss[j] = Sj  # vC
    psi.Bs[j] = Bj  # vC j vR
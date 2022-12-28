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
    * (geometrically) local operations

    Preprocessing restrictions:
    * Quantum Script wires must be adjacent integers starting from zero
    * All operations must have matrices



    """

    name = "PlainMPS"

    def __init__(self, chi_max=100, eps=1e-10):
        self.chi_max = chi_max
        self.eps = eps

    @classmethod
    def execute(cls, qs: QuantumScript, dtype=np.complex128):
        num_indices = len(qs.wires)
        state = cls.init_MPS(num_indices)
        for i,op in enumerate(qs._ops):
            #print(i, op)
            state = cls.apply_operation(state, op)

        measurements = tuple(cls.measure_state(state, m) for m in qs.measurements)
        return measurements[0] if len(measurements) == 1 else measurements

    @staticmethod
    def init_MPS(L, up=0, dtype=np.complex128):
        """Initialize a product MPS of all up (0) or all down (1)"""
        B_init = np.zeros((1, 2, 1), dtype=dtype)
        B_init[:, up, :] = 1.
        Bs = [B_init for _ in range(L)] #.copy() copy is taken care of in SimpleMPS
        Ss = [np.array([1.]) for _ in range(L)]
        return SimpleMPS(Bs, Ss)

    @staticmethod
    def apply_operation(state, operation, chi_max=100, eps=1e-10):
        # TODO: figure out how to handle accessing chi_max and eps from self
        wires = operation.wires
        matrix = qml.matrix(operation)
        if len(wires) == 1:
            return update_site(state, i=operation.wires[0], U_site=matrix, chi_max=chi_max, eps=eps)
        if len(wires) == 2:
            U_bond = matrix.reshape((2, 2, 2, 2))
            return update_bond(state, i=operation.wires[0], U_bond=U_bond, chi_max=chi_max, eps=eps)
        raise NotImplementedError

    @classmethod
    def measure_state(cls, state, measurementprocess):
        wires = measurementprocess.wires
        i = wires[0]
        op = qml.matrix(measurementprocess.obs)
        if len(wires) == 1:
            return state.site_expectation_value(op, i)
        elif len(wires) == 2:
            op = op.reshape((2, 2, 2, 2))
            return state.bond_expectation_value(op, i)
        raise NotImplementedError



"""Toy code implementing a matrix product state."""
# Copyright 2018-2021 TeNPy Developers, GNU GPLv3
# The following is a Zombie of own code and toy codes from tenpy at 
# https://tenpy.readthedocs.io/en/latest/examples.html#toycodes
# TODO: make sure copyrights are not violated

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

    def site_expectation_value(self, op, i):
        """Calculate expectation values of a local operator at each site."""
        theta = self.get_theta1(i)  # vL i vR
        op_theta = np.tensordot(op, theta, axes=(1, 1))  # i [i*], vL [i] vR
        result = np.tensordot(theta.conj(), op_theta, [[0, 1, 2], [1, 0, 2]])
        # [vL*] [i*] [vR*], [i] [vL] [vR]
        return np.real_if_close(result)

    def bond_expectation_value(self, op, i):
        """Calculate expectation values of a local operator at each bond."""
        theta = self.get_theta2(i)  # vL i j vR
        #print(f"op {op.shape} theta {theta.shape}")
        op_theta = np.tensordot(op, theta, axes=([2, 3], [1, 2]))
        # i j [i*] [j*], vL [i] [j] vR
        result = np.tensordot(theta.conj(), op_theta, [[0, 1, 2, 3], [2, 0, 1, 3]])
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

    # def correlation_length(self):
    #     """Diagonalize transfer matrix to obtain the correlation length."""
    #     from scipy.sparse.linalg import eigs
    #     if self.get_chi()[0] > 100:
    #         warnings.warn("Skip calculating correlation_length() for large chi: could take long")
    #         return -1.
    #     assert self.bc == 'infinite'  # works only in the infinite case
    #     B = self.Bs[0]  # vL i vR
    #     chi = B.shape[0]
    #     T = np.tensordot(B, np.conj(B), axes=(1, 1))  # vL [i] vR, vL* [i*] vR*
    #     T = np.transpose(T, [0, 2, 1, 3])  # vL vL* vR vR*
    #     for i in range(1, self.L):
    #         B = self.Bs[i]
    #         T = np.tensordot(T, B, axes=(2, 0))  # vL vL* [vR] vR*, [vL] i vR
    #         T = np.tensordot(T, np.conj(B), axes=([2, 3], [0, 1]))
    #         # vL vL* [vR*] [i] vR, [vL*] [i*] vR*
    #     T = np.reshape(T, (chi**2, chi**2))
    #     # Obtain the 2nd largest eigenvalue
    #     eta = eigs(T, k=2, which='LM', return_eigenvectors=False, ncv=20)
    #     xi =  -self.L / np.log(np.min(np.abs(eta)))
    #     if xi > 1000.:
    #         return np.inf
    #     return xi

    # def correlation_function(self, op_i, i, op_j, j):
    #     """Correlation function between two distant operators on sites i < j.

    #     Note: calling this function in a loop over `j` is inefficient for large j >> i.
    #     The optimization is left as an exercise to the user.
    #     Hint: Re-use the partial contractions up to but excluding site `j`.
    #     """
    #     assert i < j
    #     theta = self.get_theta1(i) # vL i vR
    #     C = np.tensordot(op_i, theta, axes=(1, 1)) # i [i*], vL [i] vR
    #     C = np.tensordot(theta.conj(), C, axes=([0, 1], [1, 0]))  # [vL*] [i*] vR*, [i] [vL] vR
    #     for k in range(i + 1, j):
    #         k = k % self.L
    #         B = self.Bs[k]  # vL k vR
    #         C = np.tensordot(C, B, axes=(1, 0)) # vR* [vR], [vL] k vR
    #         C = np.tensordot(B.conj(), C, axes=([0, 1], [0, 1])) # [vL*] [k*] vR*, [vR*] [k] vR
    #     j = j % self.L
    #     B = self.Bs[j]  # vL k vR
    #     C = np.tensordot(C, B, axes=(1, 0)) # vR* [vR], [vL] j vR
    #     C = np.tensordot(op_j, C, axes=(1, 1))  # j [j*], vR* [j] vR
    #     C = np.tensordot(B.conj(), C, axes=([0, 1, 2], [1, 0, 2])) # [vL*] [j*] [vR*], [j] [vR*] [vR]
    #     return C

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
    X, Y, Z = svd(theta, full_matrices=False) #TODO switch to qml.math.svd
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
    return psi

def update_site(psi, i, U_site, chi_max, eps):
    """Apply `U_site` acting on site i to `psi`"""
    # Conctract Theta and U
    # restore B-form by left-multiplying inverse singular values
    theta = psi.get_theta1(i)
    Utheta = np.tensordot(U_site, theta, axes=([1], [1]))  # i [i*], vL [i] vR
    Utheta = np.transpose(Utheta, [1, 0, 2])  # i vL vR >> vL i vR

    # no truncation (unsure atm)

    # put back into B form
    Gi = np.tensordot(np.diag(psi.Ss[i]**(-1)), Utheta, axes=(1, 0))  # vL [vL*], [vL] i vC
    psi.Bs[i] = np.tensordot(Gi, np.diag(psi.Ss[(i+1) %psi.L]), axes=(2, 0))  # vL i [vC], [vC] vC
    return psi
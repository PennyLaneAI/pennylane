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

    def __init__(self):
        pass

    @classmethod
    def execute(cls, qs: QuantumScript, dtype=np.complex128, chi_max=100, eps=1e-10):
        num_indices = np.max(qs.wires)+1
        state = cls.init_MPS(num_indices)
        for i,op in enumerate(qs._ops):
            #print(i, op)
            #print(op, [_.shape for _ in state.Bs])
            state = cls.apply_operation(state, op, chi_max, eps)

        #bond_dim = np.max(state.get_chi())
        #print(f"bond dimension = {bond_dim}")
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
    def apply_operation(state, operation, chi_max, eps):
        wires = operation.wires
        print(operation, wires)
        matrix = qml.matrix(operation)
        if len(wires) == 1:
            return update_site(state, i=operation.wires[0], U_site=matrix, chi_max=chi_max, eps=eps)
        if len(wires) == 2:
            if qml.math.diff(wires) == 1:
                U_bond = matrix.reshape((2, 2, 2, 2))
                return update_bond(state, i=operation.wires[0], U_bond=U_bond, chi_max=chi_max, eps=eps)
            else:
                return contract_MPO_MPS(operation, state, chi_max, eps)
        raise NotImplementedError

    @classmethod
    def measure_state(cls, state, measurementprocess):
        # hacky solution to return state
        if not measurementprocess.return_type == qml.measurements.State:
            wires = measurementprocess.wires
            i = wires[0]
            obs = measurementprocess.obs
            if isinstance(obs, qml.Hamiltonian):
                return state.expval(obs)
            elif len(wires) == 1:
                op = qml.matrix(obs)
                return state.site_expectation_value(op, i)
            elif len(wires) == 2:
                op = qml.matrix(obs)
                op = op.reshape((2, 2, 2, 2))
                return state.bond_expectation_value(op, i)
            elif len(wires) > 2:
                raise NotImplementedError
        return state



"""Toy code implementing a matrix product state."""
# Copyright 2018-2021 TeNPy Developers, GNU GPLv3
# The following is a Zombie of own code and toy codes from tenpy at 
# https://tenpy.readthedocs.io/en/latest/examples.html#toycodes
# TODO: make sure copyrights are not violated

import numpy as np
# from scipy.linalg import svd
# if you get an error message "LinAlgError: SVD did not converge",
# uncomment the following line. (This requires TeNPy to be installed.)
#  from tenpy.linalg.svd_robust import svd  # (works like scipy.linalg.svd)
from tenpy.linalg.svd_robust import svd
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
    
    def expval_string(self, pstring):
        """Compute expectation value of Pauli string"""
        wires = pstring.wires
        res = []
        for pick, i in enumerate(wires):
            res.append(self.site_expectation_value(qml.matrix(pstring.obs[pick]), i))
        return np.prod(res)
    
    def expval(self, Hs):
        """Compute expval of either a list of operators or a Hamiltonian"""
        if isinstance(Hs, list):
            res = []
            for ob in Hs:
                if isinstance(ob, qml.operation.Tensor):
                    r = self.expval_string(ob)
                elif len(ob.wires.tolist()) == 1:
                    r = self.site_expectation_value(qml.matrix(ob), ob.wires.tolist()[0])
                else:
                    raise NotImplementedError()
                res.append(r)
            return res
        if isinstance(Hs, qml.Hamiltonian):
            res = self.expval(Hs.ops)
            result = np.dot(Hs.coeffs, res)
            return result

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

    def right_normalize(self,test=False):
        """
        returns a right_normalized version of the random_MPS() object
        
        new version using QR
        """
        #Pseudo Code
        #get M[L]
        #reshape M[L] to Psi
        #SVD of PSI
        #reshape V^h to B[L], save S[L], multiply M[L-1] U S = Mt[L-1]
        #repeat procedure
        Ms = self.Ms
        L,d,chi = self.L,self.d,self.chi
        Bs = []
        for i in range(L-1,-1,-1):
            chi1,d,chi2 = Ms[i].shape # a_i-1, sigma_i, s_i
            m = Ms[i].reshape(chi1,d*chi2)
            Q,R = qr(m.conjugate().transpose(), mode='reduced') #in numpy.linalg.qr 'reduced' is standard whilst not in scipy version
            B = Q.conjugate().transpose().reshape(min(m.shape),d,chi2) #  s_i-1, sigma_i , s_i
            Bs.append(B)
            # problem gefunden, ich speicher ja nirgends B ab
            
            # update next M matrix to M-tilde = M U S
            # in order not to overwrite the first matrix again when hitting i=0 use
            if i>0: 
                Ms[i-1] = np.tensordot(Ms[i-1],R.conjugate().transpose(),1) #  a_L-2 sigma_L-1 [a_L-1] , [a_L-1] s_L-1
            if test:
                # check if right-normalization is fulfilled, should give identity matrices
                print(np.real_if_close(np.tensordot(B,B.conjugate(),axes=([2,1],[2,1])))) # s_i-1 [sigma]  [a_i], s_i-1 [sigma] [a_i]
        #return Ms
        # or update
        self.Ms = Bs[::-1]
        self.normstat = 'right-norm'

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
    # print(f"singular values before truncating: {Y}")
    # truncate
    chivC = min(chi_max, np.sum(Y > eps))
    assert chivC >= 1
    piv = np.argsort(Y)[::-1][:chivC]  # keep the largest `chivC` singular values
    X, Y, Z = X[:, piv], Y[piv], Z[piv, :]
    # renormalize
    S = Y / np.linalg.norm(Y)  # == Y/sqrt(sum(Y**2))
    # print(f"singular values after truncating: {S}")
    # split legs of X and Z
    A = np.reshape(X, [chivL, dL, chivC])
    B = np.reshape(Z, [chivC, dR, chivR])
    return A, S, B

def update_bond(state, i, U_bond, chi_max, eps):
    """Apply `U_bond` acting on i,j=(i+1) to `psi`."""
    psi = state.copy() # This is currently necessary for gradients but probably overkill
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

def update_site(state, i, U_site, chi_max, eps):
    """Apply `U_site` acting on site i to `psi`"""
    psi = state.copy()    # This is currently necessary for gradients but probably overkill!
    
    # Conctract Theta and U
    # restore B-form by left-multiplying inverse singular values
    theta = psi.get_theta1(i)
    Utheta = np.tensordot(U_site, theta, axes=([1], [1]))  # i [i*], vL [i] vR
    Utheta = np.transpose(Utheta, [1, 0, 2])  # i vL vR >> vL i vR

    # making this effectively a two-site update because I am only sure here how to do it
    j = (i+1) % psi.L
    Utheta = np.tensordot(Utheta, psi.Bs[j], [2, 0])  # vL i [vR], [vL] j vR

    # split and truncate
    Ai, Sj, Bj = split_truncate_theta(Utheta, chi_max, eps)
    # put back into MPS
    Gi = np.tensordot(np.diag(psi.Ss[i]**(-1)), Ai, axes=(1, 0))  # vL [vL*], [vL] i vC
    psi.Bs[i] = np.tensordot(Gi, np.diag(Sj), axes=(2, 0))  # vL i [vC], [vC] vC
    psi.Ss[j] = Sj  # vC
    psi.Bs[j] = Bj  # vC j vR
    return psi

def construct_MPO(op):
    """MPO convention b1 s'2 s2 b2 (virtual left, physical out, physical in, virtual right)"""
    max_i, min_i = qml.math.max(op.wires.tolist()), qml.math.min(op.wires.tolist())
    if len(op.wires) != 2:
        raise ValueError(f"Can only use true 2-site operators. Recevied op with wires {op.wires}")
    # construct 2-site decomposition
    M2 = qml.matrix(op)
    M2 = M2.reshape((2, 2, 2, 2)) # s'1 s'2 s1 s2
    M2 = M2.transpose([0,2,1,3])  # s'1 s1 s'2 s2
    M2 = M2.reshape((4, 4))
    Q, R = np.linalg.qr(M2) # (s'1 s1) b1 |b1 (s'2 s2)

    Q = Q.reshape((2, 2, 4))   # s'1 s1 b1
    Q = Q[np.newaxis]          # 1 s'1 s1 b1   # trivial axis but to stick to convention

    R = R.reshape((4, 2, 2))   # b1 s'2 s2
    R = R[:, :, :, np.newaxis]  # b1 s'2 s2 1  # trivial axis but to stick to convention

    # construct multi-site MPO with idendities in the middle
    ID = np.eye(4*2).reshape((4, 2, 4, 2)) # b1 s'2 b2 s2
    ID = ID.transpose([0, 1, 3, 2])        # b1 s'2 s2 b2
    Ws = [Q] + [ID] * (max_i - min_i - 1) + [R]
    return Ws

def contract_MPO_MPS(op, psi, chi_max, eps):
    """Contract and MPO and MPS to obtain a new MPS in canonical form
    
    Only contracting on non-trivial sites (and all sites in between).
    """
    # Code consists of 3 steps 
    # 1. Construct MPO
    # 2. update all non-trivial sites with the four-legged W operator 
    # 3. restore canonical form by right-normalizing and keeping singular values#
    Ws = construct_MPO(op)
    sub_wires = op.wires.tolist()
    max_i, min_i = qml.math.max(sub_wires), qml.math.min(sub_wires)
    sub_wires_full = np.arange(min_i, max_i+1)

    Bs = psi.Bs.copy()
    Ss = psi.Ss.copy()

    # contract physical indices and obtain transfer matrices
    sub_Bs = Bs[min_i:max_i+1]

    Ws = construct_MPO(op)

    # 2. update sites between min_i and max_i site
    # contract physical indices
    # Note that the Ws in-between are identities, so there might be room for improvement
    for B, W, i in zip(sub_Bs, Ws, sub_wires_full):
        DL, _, _,  DR = W.shape # b1 s'2 s2 b2
        chiL, _ , chiR = B.shape # a1 s2 a2
        newM = np.tensordot(B, W, axes = [[1], [2]]) # a1 [s2] a2 | b1 s'2 [s2] b2 >> a1 a2 b1 s'2 b2
        newM = newM.transpose([0, 2, 3, 1, 4]) # a1 b1 s'2 a2 b2

        # combine two virtual indices into one
        # I.e. there currently is no approximation
        newM = newM.reshape((DL*chiL, 2, DR*chiR)) # a1 s2 a2
        Bs[i] = newM

    # 3. Bring back into right-canonical form and truncate bonds
    # I'm not sure this is actually necessary
    # let alone efficiently done here
    for i in range(max_i, min_i-1, -1):
        #print(i, len(Bs), psi.L)
        theta = np.tensordot(Bs[i-1], Bs[i], axes=[[-1],[0]]) # newM | A_from_before (naming might be confusing)
        A, S, B = split_truncate_theta(theta, chi_max, eps)             # >> A S B
        Bs[i] = B
        Ss[i] = S
        if i != min_i:
            Bs[i-1] = np.tensordot(A, np.diag(S), axes=1)

    return SimpleMPS(Bs, Ss)
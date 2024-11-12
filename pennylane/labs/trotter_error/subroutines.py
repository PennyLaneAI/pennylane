from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from pyblock2.algebra.io import MPSTools, MPOTools

import numpy as np
from scipy.linalg import expm
import itertools

class PTerror():
    def __init__(self, H):
        self.H = H
        self.eigenstates = None

    def get_eigenstates(self):
        r""" Computes self.eigenstates."""
        raise NotImplementedError

    def matrix_element(self, bra, op, ket):
        raise NotImplementedError


class PTerrorTensor(PTerror):
    def __init__(self, H, driver, name, cdf_loc, bond_dim):
        super().__init__(H)
        if driver is None:
            self.driver = DMRGDriver(scratch=f"./{name}_tmp", \
                                    symm_type=SymmetryTypes.SZ, \
                                        n_threads=4, stack_mem=6*1024**3)
        else:
            self.driver = driver

        self.load_cdf_coeffs(cdf_loc)
        self.bond_dim = bond_dim

        h1e = self.h1e
        g2e = self.eri
        

        # force spin = 0 and no orbital symmetry -- neither are available in PL for now anyway
        #assert self.H.spin == 0
        #assert self.H.orb_sym == [0] * self.H.ncas
        # Initialize DMRG driver
        self.driver.initialize_system(n_sites=self.H.ncas, n_elec=self.H.nelec, 
                                spin=self.H.spin, orb_sym=self.H.orb_sym)

        self.load_cdf_coeffs(cdf_loc)


    def load_cdf_coeffs(self, cdf_loc):
        r"""CDF coefficients are loaded from a location on disk"""
        # load the one-electron CDF
        U0, Z0 = np.load(f'{cdf_loc}/CDF_onebody.npy', allow_pickle=True)
        # load the two-electron CDF
        X, Z = np.load(f'{cdf_loc}/CDF_twobody.npy', allow_pickle=True)
        U = expm(X)
        # set core constant to zero for simplicity, as it does not affect PT results
        self.ecore = 0
        self.h1e = np.einsum('pq,qr,rs', U0, Z0, U0)
        self.cdf_eri = np.einsum('tpk,tqk,tkl,trl,tsl->tpqrs', U, U, Z, U, U)
        self.eri = np.sum(self.cdf_eri, axis=0)

    def get_eigenstates(self, bond_dims, nroots, noises, thrds):
        r"""Gets the MPS corresponding to the eigenstates of the Hamiltonian."""    

        # Compute MPO for the Hamiltonian
        mpo = self.driver.get_qc_mpo(h1e=self.h1e, g2e=self.eri, ecore=0, iprint=1)

        # Select the number of eigenstates to compute
        ket = self.driver.get_random_mps(tag="eigenstates", bond_dim=250, nroots=nroots)

        # Compute the eigenstates
        _ = self.driver.dmrg(mpo, ket, n_sweeps=20, bond_dims=bond_dims, noises=noises,
                            thrds=thrds, iprint=1)

        # Store the eigenstates
        self.eigenstates = ket

    def get_mf_mps(self):
        r""" Finds some of the simplest possible spin-restricted mean-field states, as a test 
        for now. Total number of such states is combinatorially large: could use something 
        like delta_SCF to zero in on key candidates, or simply use CIS / CISD. """

        hf_state = [2] * (self.H.nelec // 2) + [0] * (self.H.ncas - self.H.nelec // 2)
        mf_states = list(set(itertools.permutations(hf_state, len(hf_state))))

        self.mf_mps = {}
        for state in mf_states:
            ket = self.driver.get_random_mps(tag=str(state), bond_dim=self.bond_dim, \
                                        occs=state, dot=1)
            ket = self.driver.adjust_mps(ket, dot=1)[0]
            pyket = MPSTools.from_block2(ket)
            self.mf_mps[state] = pyket
        return self.mf_mps

    @staticmethod
    def matrix_element(pybra, pyop, pyket):
        r"""
        Computes matrix element of an operator represented as an MPO.
        All Block2 objects must be Python-based entities.

        Arguments:
            bra (MPS): bra state
            pyop (MPO): operator
            ket (MPS): ket state
        """
        return pybra @ pyop @ pyket

    @staticmethod
    def commutator(mpoa, mpob):
        r"""Computes the commutator of two MPOs."""
        return mpoa @ mpob - mpob @ mpoa

    def compute_error(self):
        self.get_mf_mps()

        error_per_state = {}
        for jj in range(1, len(self.cdf_eri)+1):

            # create mpo for sum_{i<j} H_i
            if jj == 1:
                h1e = self.h1e
                g2e = self.cdf_eri*0
                A_mpo = self.driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=self.ecore, \
                                        iprint=0, add_ident=False)
            else:
                h1e = self.h1e
                g2e = np.sum(self.cdf_eri[:jj], axis=0)
                A_mpo = self.driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=self.ecore, \
                                        iprint=0, add_ident=False)
            A_mpo_py = MPOTools.from_block2(A_mpo.prim_mpo)

            # create mpo for the counterpart, the H_j fragment
            h1e = self.h1e * 0
            g2e = self.cdf_eri[jj]
            Hj_mpo = self.driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=self.ecore, \
                                    iprint=0, add_ident=False)
            Hj_mpo_py = MPOTools.from_block2(Hj_mpo.prim_mpo)

            # evaluate all the six commutator expressions
            for label, mps in self.mf_mps.items(): #todo: make it more general
                Y3_j = self.matrix_element(mps, self.commutator(self.commutator(A_mpo_py, Hj_mpo_py), A_mpo_py), mps) / 12 \
                        + self.matrix_element(mps, self.commutator(self.commutator(A_mpo_py, Hj_mpo_py), Hj_mpo_py), mps) / 24

                if not label in error_per_state.keys():
                    error_per_state[label] = Y3_j
                else:
                    error_per_state[label] += Y3_j            


        return error_per_state



class H():
    def __init__(self, ncas, nelec, spin, orb_sym, h1e, eri, ecore):
        self.ncas = ncas
        self.nelec = nelec
        self.spin = spin
        self.orb_sym = orb_sym
        self.h1e = h1e
        self.eri = eri
        self.ecore = ecore


def _load_cdf(folder, verbose = False):
    """
    Helper function to load the pre-computed cdf one 
    and two electron integrals.
    """
    # load the one-electron integrals
    UZ0 = np.load(f'{folder}/CDF_onebody.npy', allow_pickle=True)
    U0, Z0 = UZ0[0], UZ0[1]

    # load the two-electron integrals
    opt_XZ = np.load(f'{folder}/CDF_twobody.npy', allow_pickle=True)
    X, Z = opt_XZ[0], opt_XZ[1]

    # load the core constant
    core_const = np.load(f'{folder}/CDF_core.npy', allow_pickle=True)

    #check whether X is antisymmetric and Z is symmetric
    if verbose:
        print(f'Is X antisymmetric', np.isclose(X, -np.transpose(X, (0,2,1))).all())
        print(f'Is Z symmetric', np.isclose(Z, np.transpose(Z, (0,2,1))).all())

    return X, Z, U0, Z0, core_const

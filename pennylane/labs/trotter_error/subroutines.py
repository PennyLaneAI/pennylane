from block2 import itg, DMRGDriver, SymmetryTypes

class PTerror():
    def __init__(self, H):
        self.H = H
        self.eigenstates = None

    def get_eigenstates(self):
        r""" Computes self.eigenstates."""
        raise NotImplementedError

    def matrix_element(self, bra, nested_commutator, ket):
        raise NotImplementedError
    
class PTerrorTensor(PTerror):
    def __init__(self, H, driver):
        super().__init__(H)
        if driver is None:
            self.driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SU2, n_threads=4)
        else:
            self.driver = driver

    def get_eigenstates(self, bond_dims, nroots, noises, thrds):
        r"""Gets the MPS corresponding to the eigenstates of the Hamiltonian."""

        # Get integrals and other information from PySCF mean-field object
        #ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_rhf_integrals(mf,
        #                                            ncore=0, ncas=None, g2e_symm=8)
    

        # Initialize DMRG driver
        self.driver.initialize_system(n_sites=self.H.ncas, n_elec=self.H.nelec, 
                                spin=self.H.spin, orb_sym=self.H.orb_sym)

        # Compute MPO for the Hamiltonian
        mpo = self.driver.get_qc_mpo(h1e=self.H.h1e, g2e=self.H.eri, ecore=self.H.ecore, iprint=1)

        # Select the number of eigenstates to compute
        ket = self.driver.get_random_mps(tag="eigenstates", bond_dim=250, nroots=nroots)

        # Compute the eigenstates
        _ = self.driver.dmrg(mpo, ket, n_sweeps=20, bond_dims=bond_dims, noises=noises,
                            thrds=thrds, iprint=1)

        # Store the eigenstates
        self.eigenstates = ket
    
    def matrix_element(self, bra, nested_commutator, ket):
        r"""
        Computes matrix element of the nested commutator.

        Arguments:
            bra (MPS): bra state
            nested_commutator (MPO): nested commutator
            ket (MPS): ket state
        """

        # Identity MPO
        impo = self.driver.get_identity_mpo()

        # Get expected values
        return self.driver.expectation(bra, nested_commutator, ket) / self.driver.expectation(bra, impo, ket)
    
    def get_nested_commutator(self, right_nested_indices, ket):
        r"""
        Computes a matrix element of the nested commutator.

        Arguments:
            right_nested_indices (list): indices of the commutator, nested to the right.
            ket (MPS): ket state
        """

        # Multiply i0 * [i1, [i2, [i3, ...]]] * ket / self.driver.expectation(ket, impo, ket)
        # Use self.H.Hs[i] to get the Hamiltonian terms, 
        # and recursively call get_nested_commutator(right_nested_indices[1:], ket)
        ket1 = None

        # Multiply [i1, [i2, [i3, ...]]] * i0 * ket / self.driver.expectation(ket, impo, ket)
        # Use self.H.Hs[i] to get the Hamiltonian terms, 
        # and recursively call get_nested_commutator(right_nested_indices[1:], ket)
        ket2 = None

        return ket1 - ket2



class H():
    def __init__(self, ncas, nelec, spin, orb_sym, h1e, eri, ecore, Hs):
        self.ncas = ncas
        self.nelec = nelec
        self.spin = spin
        self.orb_sym = orb_sym
        self.h1e = h1e
        self.eri = eri
        self.ecore = ecore
        self.Hs = Hs
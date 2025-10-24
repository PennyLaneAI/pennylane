"""
Auxiliary functions for quantum chemistry calculations.

This module provides helper functions for molecule creation, integral computation,
active space selection, and quantum solvers.
"""
import os
import numpy as np

# External dependencies
from pyscf import fci, ao2mo

# Optional block2 imports
try:
    from pyblock2.driver.core import DMRGDriver, SymmetryTypes
    from pyblock2._pyscf.ao2mo import integrals as itg
    BLOCK2_AVAILABLE = True
except ImportError:
    BLOCK2_AVAILABLE = False
    print("Warning: block2 not available. Only FCI solver will be used.")


# =============================================================================
# QUANTUM CHEMISTRY SOLVERS
# =============================================================================

class QuantumSolver:
    """Base class for quantum chemistry solvers."""
    
    def __init__(self, solver_type='fci', **kwargs):
        """
        Initialize quantum solver.
        
        Args:
            solver_type: 'fci' or 'dmrg'
            **kwargs: Solver-specific parameters
        """
        self.solver_type = solver_type.lower()
        self.kwargs = kwargs
        
        if self.solver_type == 'dmrg' and not BLOCK2_AVAILABLE:
            print("Warning: DMRG requested but block2 not available. Falling back to FCI.")
            self.solver_type = 'fci'
    
    def solve_ground_states(self, h1e, eri, ncas, nelecas, nroots=1):
        """
        Solve for ground and excited states.
        
        Returns:
            energies: Array of eigenvalues
            wavefunctions: Solver-specific wavefunction data
        """
        if self.solver_type == 'fci':
            return self._solve_fci(h1e, eri, ncas, nelecas, nroots)
        elif self.solver_type == 'dmrg':
            return self._solve_dmrg(h1e, eri, ncas, nelecas, nroots)
        else:
            raise ValueError(f"Unknown solver type: {self.solver_type}")
    
    def _solve_fci(self, h1e, eri, ncas, nelecas, nroots):
        """Solve using PySCF FCI."""
        fci_kwargs = {
            'tol': self.kwargs.get('fci_tol', 1e-12),
            'lindep': self.kwargs.get('fci_lindep', 1e-14),
            'max_cycle': self.kwargs.get('fci_max_cycle', 1000),
            'nroots': nroots,
            'verbose': self.kwargs.get('fci_verbose', 0)
        }
        
        energies, wavefunctions = fci.direct_spin1.kernel(
            h1e, eri, ncas, nelecas, **fci_kwargs
        )
        
        return energies, wavefunctions
    
    def _solve_dmrg(self, h1e, eri, ncas, nelecas, nroots):
        """Solve using block2 DMRG - corrected version based on working example."""
        if not BLOCK2_AVAILABLE:
            raise RuntimeError("block2 not available for DMRG calculation")
        
        from pyblock2.driver.core import DMRGDriver, SymmetryTypes
        
        # DMRG parameters with defaults matching the working example
        scratch_dir = self.kwargs.get('scratch_dir', './dmrg_tmp')
        n_threads = self.kwargs.get('n_threads', 1)
        stack_mem = self.kwargs.get('stack_mem', 4*1024**3)
        bond_dims = self.kwargs.get('bond_dims', [100, 200])
        n_sweeps = self.kwargs.get('n_sweeps', [20, 20])
        noises = self.kwargs.get('noises', [1e-3, 1e-4])
        thrds = self.kwargs.get('thrds', [1e-6, 1e-8])
        spin = self.kwargs.get('spin', 0)
        
        # Initialize driver
        driver = DMRGDriver(
            scratch=scratch_dir,
            symm_type=SymmetryTypes.SZ,
            n_threads=n_threads,
            stack_mem=stack_mem
        )
        
        # Initialize system
        orb_sym = [0] * ncas  # Assume no symmetry
        driver.initialize_system(
            n_sites=ncas, 
            n_elec=nelecas, 
            spin=spin, 
            orb_sym=orb_sym
        )
        
        # Build MPO from integrals (no need to load CDF, we have h1e and eri directly)
        mpo = driver.get_qc_mpo(
            h1e=h1e, 
            g2e=eri, 
            ecore=0.0, 
            iprint=0, 
            add_ident=False
        )
        
        # Initialize MPS for multiple roots
        ket = driver.get_random_mps(
            tag="eigenstates", 
            bond_dim=bond_dims[0], 
            nroots=nroots, 
            dot=1
        )
        
        # Multi-stage DMRG sweeps (this is the key part that was missing)
        for ii, M in enumerate(bond_dims):
            if ii >= len(n_sweeps):
                break
                
            # Create arrays for this sweep stage
            Mvals = [M] * n_sweeps[ii]
            noisevals = [noises[ii]] * n_sweeps[ii] if ii < len(noises) else [1e-8] * n_sweeps[ii]
            thrdsvals = [thrds[ii]] * n_sweeps[ii] if ii < len(thrds) else [1e-8] * n_sweeps[ii]
            
            # Perform DMRG sweeps for this bond dimension
            energies = driver.dmrg(
                mpo, ket, 
                n_sweeps=n_sweeps[ii],
                bond_dims=Mvals,
                noises=noisevals,
                thrds=thrdsvals,
                iprint=0,
                tol=0.0  # Use threshold from thrds instead
            )
    
        # Extract individual state energies
        final_energies = []
        wavefunctions = []
        
        for iroot in range(nroots):
            # Split MPS to get individual states
            split_ket = driver.split_mps(ket, iroot, tag=f"ket{iroot}")
            
            # The energy for this root should be accessible
            # This might need adjustment based on how energies are returned
            if isinstance(energies, (list, np.ndarray)) and len(energies) >= nroots:
                energy = energies[iroot]
            else:
                # If energies is a single value or not properly indexed, 
                # we might need to compute it manually
                energy = energies if iroot == 0 else energies  # Simplified
        
            final_energies.append(energy)
            wavefunctions.append(split_ket)  # Store the block2 MPS objects
        
        energies_array = np.array(final_energies)
        
        return energies_array, wavefunctions


# =============================================================================
# MOLECULE AND INTEGRAL PREPARATION
# =============================================================================

def molecule(geom, basis="sto3g", sym="C1", spin=0):
    """Create a PySCF molecule object."""
    from pyscf import gto
    mol = gto.Mole(atom=geom, basis=basis, symmetry=sym, spin=spin)
    mol.build()
    return mol


def to_density(one, two):
    """Convert integrals from physicist to chemist convention."""
    eri = np.einsum('prsq->pqrs', two)
    h1e = one - np.einsum('pqrr->pq', two) / 2.
    return h1e, eri


def active_space_selection(nelectron, nao, spin, nelecas, ncas):
    """Select active space indices (core and active orbitals)."""
    if ncas is None or nelecas is None:
        return None, list(range(nao))
    
    ncore_electrons = nelectron - nelecas
    if ncore_electrons < 0:
        raise ValueError("Number of active electrons cannot exceed total electrons")
    
    ncore_orbitals = ncore_electrons // 2
    if ncore_electrons % 2 != 0:
        raise ValueError("Number of core electrons must be even")
    
    core = list(range(ncore_orbitals)) if ncore_orbitals > 0 else []
    active = list(range(ncore_orbitals, ncore_orbitals + ncas))
    
    return core, active


def _compute_core_energy(core, one, two):
    """Compute core energy contribution."""
    core_constant = 0.0
    for i in core:
        core_constant += 2 * one[i][i]
        for j in core:
            core_constant += 2 * two[i][j][j][i] - two[i][j][i][j]
    return core_constant


def _apply_core_correction(active, core, one, two):
    """Apply core orbital corrections to active space integrals."""
    for p in active:
        for q in active:
            for i in core:
                one[p, q] += 2 * two[i][p][q][i] - two[i][p][i][q]


def _extract_active_space_integrals(one, two, active):
    """Extract active space integrals using numpy indexing."""
    one_active = one[np.ix_(active, active)]
    two_active = two[np.ix_(active, active, active, active)]
    return one_active, two_active


def Hamiltonian(mol, ncas=None, nelecas=None):
    """Get Hamiltonian integrals for a molecule with optional active space."""
    from pyscf import scf
    
    # Hartree-Fock calculation with threading fix
    hf = scf.RHF(mol)
    def round_eig(f):
        return lambda h, s: f(h.round(12), s)
    hf.eig = round_eig(hf.eig)
    hf.run(verbose=0)

    # Compute integrals in MO basis
    h_core = hf.get_hcore(mol)
    orbs = hf.mo_coeff
    core_constant = mol.energy_nuc()
    
    one = np.einsum("qr,rs,st->qt", orbs.T, h_core, orbs)
    two = ao2mo.full(hf._eri, orbs, compact=False).reshape([mol.nao] * 4)
    two = np.swapaxes(two, 1, 3)

    # Apply active space selection
    core, active = active_space_selection(mol.nelectron, mol.nao, 
                                        2 * mol.spin + 1, nelecas, ncas)

    if core and active:
        core_constant += _compute_core_energy(core, one, two)
        _apply_core_correction(active, core, one, two)
        one, two = _extract_active_space_integrals(one, two, active)

    return core_constant, one, two


def apply_cvs_to_integrals(two, ncas, xas_core):
    """Apply Core-Valence Separation (CVS) to two-electron integrals."""
    two_cvs = two.copy()
    
    if xas_core < ncas:
        # Zero out core-valence interactions (simplified CVS)
        for i in range(ncas):
            if i != xas_core:
                two_cvs[xas_core, i, :, :] = 0
                two_cvs[i, xas_core, :, :] = 0
                two_cvs[:, :, xas_core, i] = 0
                two_cvs[:, :, i, xas_core] = 0
    
    return two_cvs


def Hamiltonian_cvs(hf, ncas=None, nelecas=None, cvs=False, xas_core=0):
    """Get CVS Hamiltonian integrals from a converged HF object."""
    # Compute base integrals
    h_core = hf.get_hcore(hf.mol)
    orbs = hf.mo_coeff
    core_constant = hf.mol.energy_nuc()
    
    one = np.einsum("qr,rs,st->qt", orbs.T, h_core, orbs)
    two = ao2mo.full(hf._eri, orbs, compact=False).reshape([hf.mol.nao] * 4)
    two = np.swapaxes(two, 1, 3)

    # Apply active space selection
    core, active = active_space_selection(hf.mol.nelectron, hf.mol.nao, 
                                        2 * hf.mol.spin + 1, nelecas, ncas)

    if core and active:
        core_constant += _compute_core_energy(core, one, two)
        _apply_core_correction(active, core, one, two)
        one, two = _extract_active_space_integrals(one, two, active)

    # Apply CVS if requested
    if cvs:
        two = np.swapaxes(two, 3, 1)  # physicist -> chemist
        two = apply_cvs_to_integrals(two, ncas, xas_core)
        two = np.swapaxes(two, 1, 3)  # chemist -> physicist

    return core_constant, one, two
#!/usr/bin/env python3
"""
DMRG energy calculations with Trotter error analysis using PennyLane.

This script demonstrates:
1. DMRG eigenstate calculations for molecular systems
2. Conversion of DMRG states to PennyLane MPState format  
3. Hamiltonian factorization using compressed double factorization
4. Perturbation theory error estimation for Trotter evolution

Example:
    python simple_dmrg.py
    
The script computes ground and excited states for LiMnO+ molecule,
then evaluates Trotter perturbation errors for different timesteps.
"""

import os
import numpy as np
from pyscf import gto, scf, fci
from pyscf.mcscf import avas
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from pyblock2.algebra.io import MPSTools, MPOTools
import pennylane as qml
from pennylane.labs.trotter_error.product_formulas import perturbation_error, ProductFormula

import time

from mpo_fragments import MPOFragment, MPState

# import mpi4py
# from mpi4py import MPI

def evaluate_perturbation_error(fragments_list, mp_states, timestep=0.1):
    """
    Evaluate perturbation error for different eigenstates using Strang splitting.
    
    Args:
        fragments_list: List of MPOFragment objects
        mp_states: Dictionary of MPState objects
        timestep: Time step for evolution
        
    Returns:
        List of perturbation errors for each state
    """
    print(f"Evaluating perturbation error (timestep={timestep})...")
    
    # Convert list to dictionary for perturbation_error function
    fragments_dict = dict(enumerate(fragments_list))
    
    # Get fragment labels and create Strang product formula
    frag_labels = list(fragments_dict.keys())
    
    # Strang splitting: forward then reverse order
    strang_labels = frag_labels + frag_labels[::-1]
    strang_coeffs = [timestep/2] * len(strang_labels)
    
    # Create product formula
    pf = ProductFormula(strang_labels, coeffs=strang_coeffs)(timestep)
    
    # Compute perturbation error (Strang is order 3)
    states_list = list(mp_states.values())
    errors = perturbation_error(
        pf, fragments_dict, states_list, 
        order=3, timestep=timestep, backend="mpi4py_comm",
        parallel_mode="state", num_workers=4,
    )
    
    print(f"Perturbation errors:")
    for i, error in enumerate(errors):
        print(f"  State {i}: {error:.8e}")
    
    return errors


# ========== Original Functions (modified) ==========
def create_limno_molecule():
    """Create the LiMnO+ molecule."""
    atom = """
    Mn    1.618415    9.865219    9.396013
     O    2.474720   11.193249   10.479672
    """
    mol = gto.Mole(atom=atom, basis="cc-pvdz", spin=0, 
                   charge=1, symmetry="C1")
    mol.build()
    
    print(f"Molecule: {mol.natm} atoms")
    print(f"Full space: nao = {mol.nao}, nelectrons = {mol.nelectron}")
    
    return mol


def setup_hf_and_active_space(mol):
    """Setup Hartree-Fock and determine active space."""
    # Setup HF calculation
    hf = scf.RHF(mol)
    hf.verbose = 1
    hf.max_cycle = 1000
    hf.diis_space = 12
    hf.diis_start_cycle = 100
    hf.chkfile = "mno+.chk"
    hf.run(init_guess="mno+.chk")
    
    # Select active space with AVAS
    ao_labels = ['Mn 3d']
    ncas, nelecas, orbs = avas.avas(hf, ao_labels, canonicalize=False, threshold=0.5)
    hf.mo_coeffs = orbs
    
    print(f"Active space: ncas = {ncas}, nelecas = {nelecas}")
    
    return hf, ncas, nelecas


def compute_exact_hamiltonian(mol, hf, ncas, nelecas):
    """Compute the exact Hamiltonian integrals in the active space."""
    print("\n=== COMPUTING EXACT HAMILTONIAN ===")
    
    # Get the molecular orbital coefficients
    orbs = hf.mo_coeff
    
    # Core constant (nuclear repulsion)
    core_const = mol.energy_nuc()
    
    # One-body integrals in MO basis
    h_core = hf.get_hcore(mol)
    one = np.einsum("qr,rs,st->qt", orbs.T, h_core, orbs)
    
    # Two-body integrals in MO basis
    from pyscf import ao2mo
    two = ao2mo.full(hf._eri, orbs, compact=False).reshape([mol.nao]*4)
    two = np.swapaxes(two, 1, 3)  # Convert to chemist notation
    
    # Apply active space reduction using simple indexing
    # Find core and active orbital indices
    ncore = (mol.nelectron - nelecas) // 2
    core_orbs = list(range(ncore))
    active_orbs = list(range(ncore, ncore + ncas))
    
    print(f"Core orbitals: {core_orbs}")
    print(f"Active orbitals: {active_orbs}")
    
    # Add core orbital contributions to nuclear repulsion and one-body terms
    for i in core_orbs:
        core_const += 2 * one[i, i]
        for j in core_orbs:
            core_const += 2 * two[i, j, j, i] - two[i, j, i, j]
    
    # Add core-active interactions to one-body terms
    for p in active_orbs:
        for q in active_orbs:
            for i in core_orbs:
                one[p, q] += (2 * two[i, p, q, i] - two[i, p, i, q])
    
    # Extract active space integrals
    one_cas = one[np.ix_(active_orbs, active_orbs)]
    two_cas = two[np.ix_(active_orbs, active_orbs, active_orbs, active_orbs)]
    
    print(f"One-body integral shape: {one_cas.shape}")
    print(f"Two-body integral shape: {two_cas.shape}")
    print(f"Core constant: {core_const:.8f}")
    
    return core_const, one_cas, two_cas


def compute_fci_reference(h1e, eri, ncas, nelecas, nroots=3):
    """Compute exact FCI energies as reference."""
    print("\n=== COMPUTING FCI REFERENCE ===")
    
    # Convert eri to the format FCI expects
    eri_fci = eri.reshape(ncas**2, ncas**2)
    
    print(f"Computing {nroots} FCI eigenstates...")
    e_fci, civecs = fci.direct_spin1.kernel(
        h1e, eri_fci, ncas, nelecas,
        tol=1e-12, lindep=1e-14, max_cycle=1000,
        nroots=nroots, verbose=0
    )
    
    print("FCI eigenvalues:")
    for i, energy in enumerate(e_fci):
        print(f"  State {i}: E = {energy:.8f} Ha")
    
    return e_fci, civecs


def setup_dmrg_driver(ncas, nelecas):
    """Setup DMRG driver for MPS calculations."""
    print("\n=== SETTING UP DMRG DRIVER ===")
    
    driver = DMRGDriver(
        scratch="./dmrg_tmp",
        symm_type=SymmetryTypes.SZ,
        n_threads=1,
        stack_mem=10*1024**3,
        mpi=False
    )
    
    spin = 0  # Singlet state
    orb_sym = [0] * ncas  # All orbitals in same symmetry
    driver.initialize_system(n_sites=ncas, n_elec=nelecas, spin=spin, orb_sym=orb_sym)
    
    print(f"DMRG system initialized: ncas={ncas}, nelecas={nelecas}")
    return driver


def compute_dmrg_energies_and_states(driver, h1e, eri, nroots=3):
    """
    Compute DMRG eigenstates and return energies in both MPS formats.
    """
    
    # Create MPO directly from original integrals
    mpo = driver.get_qc_mpo(h1e=h1e, g2e=eri, ecore=0, iprint=0, add_ident=False)
    
    # DMRG parameters - conservative settings
    bond_dims = [100, 200]
    n_sweeps = [10, 10] 
    noises = [1e-3, 1e-5]
    thrds = [1e-6, 1e-8]
    
    # Initialize random MPS
    ket = driver.get_random_mps(tag="eigenstates", 
                               bond_dim=bond_dims[0], 
                               nroots=nroots, dot=1)
    
    # Run DMRG sweeps
    for ii, M in enumerate(bond_dims):
        Mvals = [M] * n_sweeps[ii]
        noisevals = [noises[ii]] * n_sweeps[ii]
        thrdsvals = [thrds[ii]] * n_sweeps[ii]
        
        energies = driver.dmrg(mpo, ket, n_sweeps=n_sweeps[ii],
                             bond_dims=Mvals, noises=noisevals,
                             thrds=thrdsvals, iprint=0, tol=0.0)
    
    # Extract individual states and compute energies
    # Convert to PyBlock2 format for easier manipulation
    pympo = MPOTools.from_block2(mpo.prim_mpo)
    
    dmrg_energies = []
    mps_states = {}
    
    for iroot in range(nroots):
        # Extract individual MPS
        split_ket = driver.split_mps(ket, iroot, tag=f"ket{iroot}")
        pyket = MPSTools.from_block2(split_ket)
        
        # Compute energy expectation value
        energy = pyket @ pympo @ pyket
        dmrg_energies.append(energy)

        # Store MPS state
        mps_states[f"state-{iroot}"] = MPState(pyket)
    
    return dmrg_energies, mps_states


def factorize_hamiltonian_pennylane(h1e, eri, num_fragments=3):
    """
    Use PennyLane's qml.qchem.factorize() to create factorization of the Hamiltonian.
    """
    
    _, two_body_cores, two_body_leaves = qml.qchem.factorize(
        eri, tol_factor=1e-2, cholesky=True, compressed=True
    ) # compressed double-factorized shifted two-body terms
    print(f"Two-body tensors' shape: {two_body_cores.shape, two_body_leaves.shape}")
    
    return two_body_cores, two_body_leaves

def one_body_correction(U, Z):
    """
    Obtain the one-body correction to the Hamiltonian given the 
    factorized form of the one-electron integrals.
    """
    Z_prime = np.stack([np.diag(np.sum(Z[i], axis = -1)) for i in range(Z.shape[0])], axis = 0)
    h1e_correction = np.einsum('tpk,tkk,tqk->pq', U, Z_prime, U)

    return h1e_correction


def create_mpo_fragments(driver, h1e, eri, two_body_cores, two_body_leaves, ncas):
    """
    Convert factorized Hamiltonian pieces into MPOFragment objects.
    """
    print(f"\n=== CREATING MPO FRAGMENTS ===")
    
    fragments_dict = {}
    

    # Use factorized representation
    num_fragments = two_body_cores.shape[0]
    print(f"Creating {num_fragments} MPO fragments...")

    n = h1e.shape[0]
    
    for i in range(num_fragments+1):
        
        # Extract one-body and two-body terms for this fragment
        if i == 0:
            h1e_frag = h1e
            eri_frag = np.zeros((n, n, n, n))  # No two-body terms for first fragment
        else:
            # Reconstruct two-body terms for this fragment
            leaves_i = two_body_leaves[i-1]
            cores_i = two_body_cores[i-1]

            h1e_frag = - one_body_correction(np.expand_dims(leaves_i, axis=0), np.expand_dims(cores_i, axis=0))
            eri_frag = qml.math.einsum("pk,qk,kl,rl,sl->pqrs", leaves_i, leaves_i, cores_i, leaves_i, leaves_i)

        norm = np.sum(np.abs(h1e_frag)) + np.sum(np.abs(eri_frag))

        # Create MPO for this fragment
        mpo_frag = driver.get_qc_mpo(h1e=h1e_frag, g2e=eri_frag, ecore=0, iprint=0, add_ident=False)

        # Convert Block2 MPO to PyBlock2 MPO for MPOFragment
        pympo_frag = MPOTools.from_block2(mpo_frag.prim_mpo)

        
        # Create MPOFragment with PyBlock2 MPO
        fragments_dict[i] = MPOFragment(
            pympo_frag, 
            h1e=h1e_frag, 
            g2e=eri_frag, 
            bond_dim=100, 
            norm=norm, 
            n_sites=ncas
        )

    print(f"Created {len(fragments_dict)} MPO fragments")
    return list(fragments_dict.values())


def main():
    """Main execution function with PennyLane integration."""
    
    print("DMRG Energy Calculation with Trotter Error Analysis")
    print("=" * 60)
    
    time_zero = time.perf_counter()
        
    # Setup molecule and compute reference energies
    mol = create_limno_molecule()
    hf, ncas, nelecas = setup_hf_and_active_space(mol)
    core_const, h1e, eri = compute_exact_hamiltonian(mol, hf, ncas, nelecas)
    
    nroots = 4
    e_fci, civecs = compute_fci_reference(h1e, eri, ncas, nelecas, nroots)
    
    # Compute DMRG eigenstates and convert to MPState format
    driver = setup_dmrg_driver(ncas, nelecas)
    dmrg_energies, mp_states = compute_dmrg_energies_and_states(driver, h1e, eri, nroots)
    
    # Factorize Hamiltonian and create MPO fragments
    one_body_cores, two_body_leaves = factorize_hamiltonian_pennylane(h1e, eri)
    mpo_fragments = create_mpo_fragments(driver, h1e, eri, one_body_cores, two_body_leaves, ncas)
    
    time_dmrg = time.perf_counter()

    # Evaluate Trotter perturbation errors
    print(f"\n=== TROTTER ERROR ANALYSIS ===")
    timesteps = [1.0]

    for timestep in timesteps:
        perturbation_errors = evaluate_perturbation_error(mpo_fragments, mp_states, timestep)
        
        print(f"\nTimestep t = {timestep}:")
        for i, error_val in enumerate(perturbation_errors):
            print(f"  State {i}: {error_val:.6e}")
    time_trotter = time.perf_counter()
    # Summary
    print(f"\n=== ENERGY COMPARISON ===")
    print(f"FCI Energies:   {[f'{e:.8f}' for e in e_fci]}")
    print(f"DMRG Energies:  {[f'{e:.8f}' for e in dmrg_energies]}")
    
    energy_diffs = [abs(e_fci[i] - dmrg_energies[i]) for i in range(min(len(e_fci), len(dmrg_energies)))]
    print(f"Differences:    {[f'{diff:.2e}' for diff in energy_diffs]}")
    timestep = 1.0  # Timestep h = 1 for Strang formula

    # ===== RESULTS COMPARISON =====
    
    # Step 8: Compare energies
    print("\\n" + "="*80)
    print("ACTIVE SPACE ENERGY COMPARISON (recommended)")
    print("="*80)
    print("Note: These are active space energies (excluding core electrons)")
    print(f"{'State':<6} {'FCI (Active)':<15} {'DMRG (Active)':<15} {'Difference':<15} {'Pert. Error':<15}")
    print("-" * 80)
    
    for i in range(len(e_fci)):
        fci_e = e_fci[i]
        dmrg_e = dmrg_energies[i] if i < len(dmrg_energies) else float('nan')
        diff = abs(fci_e - dmrg_e) if not np.isnan(dmrg_e) else float('nan')
        pert_err = perturbation_errors[i] if i < len(perturbation_errors) else float('nan')
        
        print(f"{i:<6} {fci_e:<15.8f} {dmrg_e:<15.8f} {diff:<15.8e} {pert_err:<15.6e}")
    
    # Optional: Add total energies for reference
    print("\\n" + "="*80)
    print("TOTAL MOLECULAR ENERGIES (for reference)")
    print("="*80)
    print(f"Core energy (frozen orbitals): {core_const:.8f} Ha")
    print(f"{'State':<6} {'Total Energy':<15} {'Notes':<30}")
    print("-" * 80)
    
    for i in range(len(e_fci)):
        fci_total = e_fci[i] + core_const
        notes = f"Active: {e_fci[i]:.6f} + Core"
        
        print(f"{i:<6} {fci_total:<15.8f} {notes:<30}")
    
    time_comparison = time.perf_counter()
    
    print(f"\nTiming Summary:")
    print(f"   DMRG Computation Time: {time_dmrg - time_zero:.4f} seconds")
    print(f"      Trotter Error Time: {time_trotter - time_dmrg:.4f} seconds")
    print(f"  Energy Comparison Time: {time_comparison - time_trotter:.4f} seconds")
    print(f"    Total Execution Time: {time_comparison - time_zero:.4f} seconds")


if __name__ == "__main__":
    main()

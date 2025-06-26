# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains functions to calculate electronic energies with different electronic structure methods."""

from typing import List, Optional, Union

import numpy as np

try:
    from pyscf import cc, gto, scf, tdscf, mcscf, fci
except ImportError as e:
    raise ImportError("This feature requires pyscf.") from e


def _setup_molecule(
    symbols: List[str],
    coords: np.ndarray,
    basis: str = "6-31g*",
    spin: int = 0,
    point_group: Optional[str] = None,
) -> gto.Mole:
    """
    Set up a PySCF Mole object with the given parameters.

    Args:
        symbols: List of atomic symbols
        coords: Atomic coordinates in Angstroms
        basis: Basis set
        spin: Spin multiplicity (0 for singlet, 1 for doublet, etc.)
        point_group: Point group symmetry (optional)

    Returns:
        PySCF Mole object
    """
    coords_list = [[symbol, tuple(np.array(coords)[i])] for i, symbol in enumerate(symbols)]

    mol = gto.Mole()
    mol.atom = coords_list
    mol.unit = "Angstrom"
    mol.basis = basis
    mol.spin = spin
    if point_group:
        mol.symmetry = point_group.lower()
        mol.symmetry_subgroup = point_group.lower()

    mol.build()
    return mol


def _run_scf(
    mol: gto.Mole, method_type: str = "hf", conv_tol: float = 1e-7, max_cycle: int = 50, **kwargs
) -> Union[scf.RHF, scf.UHF, scf.RKS, scf.UKS]:
    """
    Run SCF calculation with the given parameters.

    Args:
        mol: PySCF Mole object
        method_type: Type of SCF method ('hf' or 'dft')
        conv_tol: Convergence tolerance
        max_cycle: Maximum number of iterations
        **kwargs: Additional parameters for DFT calculations (e.g., functional)

    Returns:
        SCF object with converged results
    """
    # Select appropriate SCF method based on spin and method_type
    if method_type.lower() == "dft":
        if mol.spin == 0:
            mf = scf.RKS(mol)
        else:
            mf = scf.UKS(mol)

        # Set DFT functional if provided
        if "functional" in kwargs:
            mf.xc = kwargs["functional"]
    else:  # Default to HF
        if mol.spin == 0:
            mf = scf.RHF(mol)
        else:
            mf = scf.UHF(mol)

    mf.conv_tol = conv_tol
    mf.max_cycle = max_cycle
    mf.kernel()

    if not mf.converged:
        raise ValueError("SCF calculation did not converge")

    return mf


def _run_tddft(
    symbols: List[str],
    coords: np.ndarray,
    basis: str = "6-31g*",
    functional: str = "cam-b3lyp",
    nroots: int = 2,
    conv_tol: float = 1e-7,
    max_cycle: int = 50,
    restrict_to_singlet: bool = True,
    spin: int = 0,
    point_group: Optional[str] = None,
) -> List[float]:
    """
    Run TDDFT calculation and return energies.

    Args:
        symbols: List of atomic symbols
        coords: Atomic coordinates in Angstroms
        basis: Basis set
        functional: Exchange-correlation functional
        nroots: Number of excited states to compute
        conv_tol: Convergence tolerance
        max_cycle: Maximum number of iterations
        restrict_to_singlet: If True, only compute singlet states
        spin: Spin multiplicity (0 for singlet, 1 for doublet, etc.)
        point_group: Point group symmetry (optional)

    Returns:
        List of TDDFT energies for each root
    """
    # Set up molecule
    mol = _setup_molecule(symbols, coords, basis, spin, point_group)

    # Run SCF
    mf = _run_scf(
        mol, method_type="dft", conv_tol=conv_tol, max_cycle=max_cycle, functional=functional
    )

    # Run TDDFT
    td = tdscf.TDA(mf)
    td.nstates = nroots

    if restrict_to_singlet and mol.spin == 0:
        td.singlet = True
        td.triplet = False

    e_td = td.kernel()[0]

    # Return ground state energy + excitation energies
    energies = [mf.e_tot]
    for exc_energy in e_td:
        energies.append(mf.e_tot + exc_energy)

    return energies


def _run_eom_ccsd(
    symbols: List[str],
    coords: np.ndarray,
    basis: str = "6-31g*",
    nroots: int = 2,
    conv_tol: float = 1e-7,
    conv_tol_eom: float = 1e-6,
    max_cycle: int = 50,
    max_cycle_eom: int = 20,
    restrict_to_singlet: bool = True,
    spin: int = 0,
    frozen: Optional[int] = None,
    point_group: Optional[str] = None,
) -> List[float]:
    """
    Run EOM-CCSD calculation and return energies.

    Args:
        symbols: List of atomic symbols
        coords: Atomic coordinates in Angstroms
        basis: Basis set
        nroots: Number of excited states to compute
        conv_tol: Convergence tolerance for CCSD
        conv_tol_eom: Convergence tolerance for EOM
        max_cycle: Maximum number of iterations for CCSD
        max_cycle_eom: Maximum number of iterations for EOM
        restrict_to_singlet: If True, only compute singlet states
        spin: Spin multiplicity (0 for singlet, 1 for doublet, etc.)
        frozen: Number of frozen orbitals (optional)
        point_group: Point group symmetry (optional)

    Returns:
        List of EOM-CCSD energies for each root
    """
    # Set up molecule
    mol = _setup_molecule(symbols, coords, basis, spin, point_group)

    # Run SCF
    mf = _run_scf(mol, method_type="hf", conv_tol=conv_tol, max_cycle=max_cycle)

    # Run CCSD
    mycc = cc.CCSD(mf)
    mycc.conv_tol = conv_tol
    mycc.max_cycle = max_cycle
    mycc.frozen = frozen
    mycc.kernel()

    if not mycc.converged:
        raise ValueError("CCSD calculation did not converge")

    # Run EOM-CCSD
    if mol.spin == 0 and restrict_to_singlet:
        energies_eomcc, _ = mycc.eomee_ccsd_singlet(nroots=nroots)
    else:
        raise NotImplementedError("EOM-CCSD for triplet states is not yet supported.")

    # Return ground state energy + excitation energies
    energies = [mycc.e_tot]

    # Handle the case where nroots=1 (returns a single float)
    if nroots == 1 or isinstance(energies_eomcc, (float, np.float64)):
        energies.append(mycc.e_tot + energies_eomcc)
    else:
        # Handle the case where nroots>1 (returns an array of floats)
        for exc_energy in energies_eomcc:
            energies.append(mycc.e_tot + exc_energy)

    return energies


def _run_casscf(
    symbols: List[str],
    coords: np.ndarray,
    ncas: int,
    nelecas: int,
    basis: str = "6-31g*",
    max_cycle: int = 50,
    conv_tol: float = 1e-7,
    nroots: int = 2,
    restrict_to_singlet: bool = True,
    spin: int = 0,
    point_group: Optional[str] = None,
) -> List[float]:
    """
    Run CASSCF calculation and return energies.

    Args:
        symbols: List of atomic symbols
        coords: Atomic coordinates in Angstroms
        ncas: Number of active space orbitals
        nelecas: Number of active space electrons
        basis: Basis set
        max_cycle: Maximum number of iterations
        conv_tol: Convergence tolerance
        nroots: Number of states to compute
        restrict_to_singlet: If True, only compute singlet states
        spin: Spin multiplicity (0 for singlet, 1 for doublet, etc.)
        point_group: Point group symmetry (optional)

    Returns:
        List of CASSCF energies for each root
    """
    # Set up molecule
    mol = _setup_molecule(symbols, coords, basis, spin, point_group)
    
    # Run SCF
    mf = _run_scf(mol, method_type="hf", conv_tol=conv_tol, max_cycle=max_cycle)
    
    # Run CASSCF
    mc = mcscf.CASSCF(mf, ncas, nelecas)
    mc.conv_tol = conv_tol
    mc.max_cycle = max_cycle
    
    # Set spin restriction if requested
    if restrict_to_singlet:
        if mol.spin != 0:
            raise ValueError("Cannot restrict to singlet for open-shell system")
        mc.fcisolver = fci.direct_spin0.FCISolver(mol)
        mc.fcisolver.spin = 0
        mc.fix_spin_(shift=0.2, ss=0)
    else:
        # For open-shell systems, use appropriate FCI solver
        if mol.spin != 0:
            from pyscf.fci import direct_spin1
            mc.fcisolver = direct_spin1.FCISolver(mol)
            mc.fcisolver.spin = mol.spin
    
    # Compute multiple roots if requested
    if nroots > 1:
        weights = [1.0 / nroots] * nroots
        mc.fcisolver.nroots = nroots
        mc = mc.state_average_(weights)
        mc.verbose = 3
        mc.kernel()
        
        if not mc.converged:
            raise ValueError("The CASSCF did NOT converge.")
        
        # CASSCF energies for all states
        energies = mc.e_states
    else:
        energies = [mc.kernel()[0]]
    
    return energies


def run_electronic_method(
    method: str,
    symbols: List[str],
    coords: np.ndarray,
    **kwargs
) -> List[float]:
    """
    Dispatcher function to run different electronic structure methods.
    
    Args:
        method: Method to use ('casscf', 'tddft', or 'eom_ccsd')
        symbols: List of atomic symbols
        coords: Atomic coordinates in Angstroms
        **kwargs: Additional parameters for the specific method
        
    Returns:
        List of energies for each state
        
    Raises:
        ValueError: If required parameters are missing or method is not supported
    """
    # Common parameters for all methods
    common_params = {
        'symbols': symbols,
        'coords': coords,
    }
    
    # Method-specific required parameters
    required_params = {
        'casscf': ['ncas', 'nelecas'],
        'tddft': ['functional'],
        'eom_ccsd': []
    }
    
    # Check if method is supported
    if method.lower() not in required_params:
        raise ValueError(f"Unsupported method: {method}. Supported methods: 'casscf', 'tddft', 'eom_ccsd'")
    
    # Check for required parameters
    for param in required_params[method.lower()]:
        if param not in kwargs:
            raise ValueError(f"Missing required parameter '{param}' for method '{method}'")
    
    # Prepare method-specific parameters by combining common params with kwargs
    method_params = {**common_params, **kwargs}
    
    # Dispatch to the appropriate method
    if method.lower() == 'casscf':
        return _run_casscf(**method_params)
    elif method.lower() == 'tddft':
        return _run_tddft(**method_params)
    elif method.lower() == 'eom_ccsd':
        return _run_eom_ccsd(**method_params)

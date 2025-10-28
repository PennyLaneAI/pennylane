# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains functions and classes to generate data needed to construct a
vibrational Hamiltonian for a given molecule."""

from dataclasses import dataclass

import numpy as np

from pennylane.qchem.openfermion_pyscf import _import_pyscf
from pennylane import qchem

# pylint: disable=import-outside-toplevel, unused-variable, too-many-instance-attributes, too-many-arguments


@dataclass
class QMMVibrationalPES:
    r"""Data class to store information needed to construct a vibrational Hamiltonian for a molecule.

    Args:
        freqs (array[float]): normal-mode frequencies in atomic units
        grid (array[float]): grid points to compute potential energy surface data.
            Should be the sample points of the Gauss-Hermite quadrature.
        gauss_weights (array[float]): weights associate with each point in ``grid``.
            Should be the weights of the Gauss-Hermite quadrature.
        uloc (TensorLike[float]): normal mode localization matrix with shape ``(m, m)`` where
            ``m = len(freqs)``
        pes_data (list[TensorLike[float]]): list of one-mode, two-mode and three-mode potential
            energy surface data, with shapes ``(m, l)``, ``(m, m, l, l)`` ``(m, m, m, l, l, l)``,
            respectively, where ``m = len(freqs)`` and ``l > 0``
        dipole_data (list[TensorLike[float]]): list of  one-mode, two-mode and three-mode dipole
            moment data, with shapes ``(m, l, 3)``, ``(m, m, l, l, 3)`` ``(m, m, m, l, l, l, 3)``,
            respectively, where ``m = len(freqs)`` and ``l > 0``
        localized (bool): Whether the potential energy surface data correspond to localized normal
            modes. Default is ``True``.
        dipole_level (int): The level up to which dipole moment data are to be calculated. Input
            values can be ``1``, ``2``, or ``3`` for up to one-mode dipole, two-mode dipole and
            three-mode dipole, respectively. Default value is ``1``.
        qmm_info: variable containing all information related for MM environment

    **Example**

    This example shows how to construct the :class:`~.qchem.vibrational.VibrationalPES` object for a
    linear diatomic molecule, e.g., :math:`H_2`, with only one vibrational normal mode. The one-mode
    potential energy surface data is obtained by sampling ``9`` points along the normal mode, with
    grid points and weights that correspond to a Gauss-Hermite quadrature.

    >>> freqs = np.array([0.01885397])
    >>> grid, weights = np.polynomial.hermite.hermgauss(9)
    >>> pes_onemode = [[0.05235573, 0.03093067, 0.01501878, 0.00420778, 0.0,
    ...                 0.00584504, 0.02881817, 0.08483433, 0.22025702]]
    >>> vib_pes = qml.qchem.VibrationalPES(freqs=freqs, grid=grid,
    ...           gauss_weights=weights, pes_data=[pes_onemode])
    >>> vib_pes.freqs
    array([0.01885397])

    The following example shows how to construct the :class:`~.qchem.vibrational.VibrationalPES`
    object for a nonlinear triatomic molecule, e.g., :math:`H_3^+`, with three vibrational
    normal modes. We assume that the potential energy surface and dipole data are obtained by
    sampling ``5`` points along the normal mode, with grid points and weights that correspond to a
    Gauss-Hermite quadrature.

    >>> freqs = np.array([0.00978463, 0.00978489, 0.01663723])
    >>> grid, weights = np.polynomial.hermite.hermgauss(5)
    >>>
    >>> uloc = np.array([[-0.99098585,  0.13396657,  0.],
    ...                  [-0.13396657, -0.99098585,  0.],
    ...                  [ 0.        ,  0.        ,  1.]])
    >>>
    >>> pes_onemode = np.random.rand(3, 5)
    >>> pes_twomode = np.random.rand(3, 3, 5, 5)
    >>> pes_threemode = np.random.rand(3, 3, 3, 5, 5, 5)
    >>>
    >>> dipole_onemode = np.random.rand(3, 5, 3)
    >>> dipole_twomode = np.random.rand(3, 3, 5, 5, 3)
    >>> dipole_threemode = np.random.rand(3, 3, 3, 5, 5, 5, 3)
    >>>
    >>> localized = True
    >>> dipole_level = 3
    >>>
    >>> vib_obj = qml.qchem.VibrationalPES(freqs=freqs, grid=grid, gauss_weights=weights,
    ...           uloc=uloc, pes_data=[pes_onemode, pes_twomode, pes_threemode],
    ...           dipole_data=[dipole_onemode, dipole_twomode, dipole_threemode],
    ...           localized=True, dipole_level=3)
    >>> print(vib_obj.dipole_threemode.shape)
    (3, 3, 3, 5, 5, 5, 3)
    """

    def __init__(
        self,
        freqs=None,
        grid=None,
        gauss_weights=None,
        uloc=None,
        pes_data=None,
        dipole_data=None,
        localized=True,
        dipole_level=1,
        qmm_info=None
    ):
        self.freqs = freqs
        self.grid = grid
        self.gauss_weights = gauss_weights
        self.uloc = uloc
        self.pes_onemode = pes_data[0] if pes_data else None
        self.pes_twomode = pes_data[1] if pes_data and len(pes_data) > 1 else None
        self.pes_threemode = pes_data[2] if pes_data and len(pes_data) > 2 else None
        self.dipole_onemode = dipole_data[0] if dipole_data else None
        self.dipole_twomode = dipole_data[1] if dipole_level >= 2 else None
        self.dipole_threemode = dipole_data[2] if dipole_level >= 3 else None
        self.localized = localized
        self.dipole_level = dipole_level
        self.qmm_info = qmm_info


def _harmonic_analysis(scf_result, method="rhf"):
    r"""Performs harmonic analysis by evaluating the Hessian using PySCF routines.

    Args:
        scf_result (pyscf.scf object): pyscf QM/MM object from electronic structure calculations
        method (str): Electronic structure method that can be either restricted and unrestricted
            Hartree-Fock,  ``'rhf'`` and ``'uhf'``, respectively. Default is ``'rhf'``.

    Returns:
        tuple: A tuple containing the following:
         - list[float]: normal mode frequencies in ``cm^-1``
         - TensorLike[float]: corresponding displacement vectors for each normal mode

    """
    pyscf = _import_pyscf()
    from pyscf.hessian import thermo

    method = method.strip().lower()
    if method not in ["rhf", "uhf"]:
        raise ValueError(f"Specified electronic structure method, {method} is not available.")

    qmmm_hess = scf_result.Hessian().kernel()
    mol_qm = scf_result.mol

    harmonic_res = thermo.harmonic_analysis(mol_qm, qmmm_hess)
    print("\n\n" + 80*"#" + "\n"+ 80*"#")
    print(f"Found frequencies of {harmonic_res['freq_wavenumber']}")
    print(80*"#" + "\n"+ 80*"#" + "\n\n")

    return harmonic_res["freq_wavenumber"], harmonic_res["norm_mode"]

def _single_point(molecule, qmm_info, method="rhf"):
    r"""Runs electronic structure calculation.

    Args:
        molecule (:func:`~pennylane.qchem.molecule.Molecule`): Molecule object.
        method (str): Electronic structure method that can be either restricted and unrestricted
            Hartree-Fock,  ``'rhf'`` and ``'uhf'``, respectively. Default is ``'rhf'``.

    Returns:
        pyscf.scf object from electronic structure calculation

    """
    pyscf = _import_pyscf()
    from pyscf import qmmm

    method = method.strip().lower()
    if method not in ["rhf", "uhf"]:
        raise ValueError(f"Specified electronic structure method, {method}, is not available.")

    geom = [
        [symbol, tuple(np.array(molecule.coordinates)[i])]
        for i, symbol in enumerate(molecule.symbols)
    ]
    spin = int((molecule.mult - 1) / 2)
    mol = pyscf.gto.Mole(atom=geom, symmetry="C1", spin=spin, charge=molecule.charge, unit="Bohr")
    mol.basis = molecule.basis_name
    mol.build()
    if method == "rhf":
        scf_obj = pyscf.scf.RHF(mol).run(verbose=0)
    else:
        scf_obj = pyscf.scf.UHF(mol).run(verbose=0)

    qmm_scf = qmmm.mm_charge(scf_obj, qmm_info["mm_coords"], qmm_info["mm_charges"])
    qmm_scf.kernel()

    return qmm_scf


def _import_geometric():
    """Import geometric."""
    try:
        import geometric
    except ImportError as Error:
        raise ImportError(
            "This feature requires geometric. It can be installed with: pip install geometric."
        ) from Error

    return geometric


def recursive_optimize_geometry(molecule, qmm_info, method="rhf", maxsteps=500, check_for_imag=True, max_checks=10, curr_check=1, imag_tol=1e-4):
    r"""Computes the equilibrium geometry of a molecule.
    Args:
        molecule (~qchem.molecule.Molecule): the molecule object
        qmm_info (dict): QM/MM information with 'mm_coords' and 'mm_charges'
        method (str): Electronic structure method used to perform geometry optimization.
            Available options are ``"rhf"`` and ``"uhf"`` for restricted and unrestricted
            Hartree-Fock, respectively. Default is ``"rhf"``.
        maxsteps (int): Maximum number of optimization steps. Default is 100.
        check_for_imag (bool): Whether to check for imaginary frequencies. Default is True.
        max_checks (int): Maximum number of times to check and re-optimize. Default is 10.
        curr_check (int): Current check iteration (used internally for recursion). Default is 0.
    Returns:
        array[array[float]]: optimized atomic positions in Cartesian coordinates
    **Example**
    >>> symbols  = ['H', 'F']
    >>> geometry = np.array([[0.0, 0.0, 0.0],
    ...                      [0.0, 0.0, 1.0]])
    >>> mol = qml.qchem.Molecule(symbols, geometry)
    >>> qmm_info = {'mm_coords': [...], 'mm_charges': [...]}
    >>> eq_geom = optimize_geometry(mol, qmm_info)
    """
    pyscf = _import_pyscf()
    geometric = _import_geometric()
    from pyscf.geomopt.geometric_solver import optimize
    from pyscf import qmmm
    import numpy as np
    
    # print(f"\n=== Geometry optimization attempt {curr_check}/{max_checks} ===")
    
    scf_res = _single_point(molecule, qmm_info, method)
    geom_eq = optimize(scf_res, maxsteps=maxsteps)
    
    if not check_for_imag or curr_check >= max_checks:
        # Return final geometry without checking for imaginary frequencies
        if molecule.unit == "angstrom":
            return geom_eq.atom_coords(unit="A")
        return geom_eq.atom_coords(unit="B")
    
    # Check for imaginary frequencies
    # Get coordinates in the correct unit for creating new molecule
    if molecule.unit == "angstrom":
        geom_coords = geom_eq.atom_coords(unit="A")
    else:
        geom_coords = geom_eq.atom_coords(unit="B")
    
    mol_eq = qchem.Molecule(
        molecule.symbols,
        geom_coords,
        unit=molecule.unit,
        basis_name=molecule.basis_name,
        charge=molecule.charge,
        mult=molecule.mult,
        load_data=molecule.load_data,
    )
    
    scf_eq = _single_point(mol_eq, qmm_info, method)
    freqs, modes = _harmonic_analysis(scf_eq, method)
    
    # Check for imaginary frequencies (negative values)
    num_imag = np.sum(np.abs(np.imag(freqs)) > imag_tol)
    
    if num_imag == 0:
        # print(f"True minimum found (no imaginary frequencies)")
        if molecule.unit == "angstrom":
            return geom_eq.atom_coords(unit="A")
        return geom_eq.atom_coords(unit="B")
    
    # Found imaginary frequencies - displace along the mode and re-optimize
    print(f"Current geometry is {geom_eq.atom_coords(unit='B')}")
    print(f"Found {num_imag} imaginary frequency(ies): {freqs[np.abs(np.imag(freqs)) > imag_tol]}")
    print(f"Following imaginary mode downhill...")
    
    # Get the most negative (most imaginary) mode
    imag_idx = np.argmax(np.abs(np.imag(freqs)))
    imag_mode = modes[imag_idx]  # Shape: (natm, 3)
    
    # Displace geometry along imaginary mode
    # Step size in same units as geometry (Angstrom or Bohr)
    step_size = 0.1 if molecule.unit == "angstrom" else 0.2
    displaced_coords = geom_coords + step_size * imag_mode
    
    # Create new molecule with displaced geometry
    mol_displaced = qchem.Molecule(
        molecule.symbols,
        displaced_coords,
        unit=molecule.unit,
        basis_name=molecule.basis_name,
        charge=molecule.charge,
        mult=molecule.mult,
        load_data=molecule.load_data,
    )
    
    # Recursively optimize from displaced geometry
    return recursive_optimize_geometry(
        mol_displaced, 
        qmm_info, 
        method=method, 
        maxsteps=maxsteps, 
        check_for_imag=check_for_imag, 
        max_checks=max_checks, 
        curr_check=curr_check + 1
    )


def qmmm_geometry_optimization(mol, mm_coords, mm_charges, 
                                method, 
                                basis,
                                optimizer='L-BFGS-B',
                                gtol=1e-4,
                                max_iter=200,
                                max_step=0.3,  # Maximum step size in Bohr
                                energy_threshold=100.0):  # Energy change threshold
    """
    Perform QM/MM geometry optimization with stability checks.
    
    Parameters:
    -----------
    mol : pyscf.gto.Mole
        The QM molecule (solute)
    mm_coords : np.ndarray, shape (n_mm, 3)
        Coordinates of MM atoms in Angstrom
    mm_charges : np.ndarray, shape (n_mm,)
        Charges of MM atoms in atomic units
    method : str
        QM method ('RHF', 'UHF', 'RKS', etc.)
    basis : str
        Basis set
    optimizer : str
        Optimization algorithm ('L-BFGS-B', 'BFGS', 'CG')
    gtol : float
        Gradient convergence threshold (Hartree/Bohr)
    max_iter : int
        Maximum optimization steps
    max_step : float
        Maximum step size in Bohr (prevents catastrophic moves)
    energy_threshold : float
        Maximum allowed energy change per step (Hartree)
    
    Returns:
    --------
    opt_coords : np.ndarray
        Optimized QM coordinates in Angstrom
    energy : float
        Final QM/MM energy in Hartree
    gradients : np.ndarray
        Final gradients in Hartree/Bohr
    success : bool
        Whether optimization succeeded
    """
    pyscf = _import_pyscf()
    from pyscf import gto, scf
    from pyscf.qmmm import itrf
    from scipy.optimize import minimize
    
    # Store atom symbols
    atom_symbols = [atom[0] for atom in mol._atom]
    
    # Track previous energy for stability checks
    prev_energy = [None]
    scf_failures = [0]
    
    def check_geometry_sanity(qm_coords):
        """Check if geometry is physically reasonable"""
        # Check QM-QM distances
        for i in range(len(qm_coords)):
            for j in range(i+1, len(qm_coords)):
                dist = np.linalg.norm(qm_coords[i] - qm_coords[j])
                if dist < 0.5:  # Less than 0.5 Bohr (~0.26 Angstrom)
                    return False, f"QM atoms {i} and {j} too close: {dist:.3f} Bohr"
        
        # Check QM-MM distances
        for i, qm_coord in enumerate(qm_coords):
            for j, mm_coord in enumerate(mm_coords):
                dist = np.linalg.norm(qm_coord - mm_coord)
                if dist < 1.0:  # Less than 1.0 Bohr (~0.52 Angstrom)
                    return False, f"QM atom {i} and MM atom {j} too close: {dist:.3f} Bohr"
        
        return True, "OK"
    
    def energy_and_gradient(qm_coords_flat):
        """Calculate QM/MM energy and gradients with safety checks"""
        qm_coords = qm_coords_flat.reshape(-1, 3)
        
        # Check geometry sanity
        is_sane, msg = check_geometry_sanity(qm_coords)
        if not is_sane:
            print(f"Warning: Geometry issue - {msg}")
            # Return high energy and zero gradient to reject this step
            return 1e10, np.zeros_like(qm_coords_flat)
        
        # Build atom list
        atom_coords = []
        for i, symbol in enumerate(atom_symbols):
            atom_coords.append([symbol, qm_coords[i].tolist()])
        
        # Create new molecule
        mol_new = gto.Mole()
        mol_new.atom = atom_coords
        mol_new.basis = basis
        mol_new.charge = mol.charge
        mol_new.spin = mol.spin
        mol_new.unit = 'Bohr'
        mol_new.verbose = 0
        mol_new.build()
        
        # Set up QM calculation
        if method.upper() == 'RHF':
            mf = scf.RHF(mol_new)
        elif method.upper() == 'UHF':
            mf = scf.UHF(mol_new)
        elif method.upper() == 'RKS':
            mf = scf.RKS(mol_new)
            mf.xc = 'B3LYP'
        elif method.upper() == 'UKS':
            mf = scf.UKS(mol_new)
            mf.xc = 'B3LYP'
        else:
            raise ValueError(f"Unknown method: {method}")
        
        mf.verbose = 0
        mf.max_cycle = 100
        mf.conv_tol = 1e-8
        
        # Add MM charges
        mf = itrf.add_mm_charges(mf, mm_coords, mm_charges, unit='Bohr')
        
        # Run SCF with error handling
        try:
            energy_qm = mf.kernel()
        except Exception as e:
            print(f"SCF failed with exception: {e}")
            scf_failures[0] += 1
            return 1e10, np.zeros_like(qm_coords_flat)
        
        if not mf.converged:
            scf_failures[0] += 1
            if scf_failures[0] > 5:
                print("Too many SCF failures - stopping optimization")
                return 1e10, np.zeros_like(qm_coords_flat)
            print("Warning: SCF not converged")
            # Try to continue but be cautious
        else:
            scf_failures[0] = max(0, scf_failures[0] - 1)  # Reduce failure count on success
        
        energy_qmmm = energy_qm
        
        # Check for unreasonable energy changes
        if prev_energy[0] is not None:
            energy_change = abs(energy_qmmm - prev_energy[0])
            if energy_change > energy_threshold:
                print(f"Warning: Large energy change: {energy_change:.2f} Hartree")
                # Don't reject, but warn
        
        prev_energy[0] = energy_qmmm
        
        # Calculate gradients
        try:
            grad_qm = mf.nuc_grad_method()
            gradients = grad_qm.kernel()
        except Exception as e:
            print(f"Gradient calculation failed: {e}")
            return 1e10, np.zeros_like(qm_coords_flat)
        
        grad_flat = gradients.flatten()
        
        # Check for unreasonable gradients
        grad_norm = np.linalg.norm(grad_flat)
        if grad_norm > 10.0:  # Hartree/Bohr
            print(f"Warning: Very large gradient norm: {grad_norm:.2f}")
        
        return energy_qmmm, grad_flat
    
    # Initial QM coordinates in Bohr
    qm_coords_init = mol.atom_coords()
    qm_coords_flat = qm_coords_init.flatten()
    
    # Check initial geometry
    is_sane, msg = check_geometry_sanity(qm_coords_init)
    if not is_sane:
        raise ValueError(f"Initial geometry has issues: {msg}")
    
    # Storage for monitoring
    energies = []
    grad_norms = []
    iteration = [0]
    
    def callback(xk):
        """Callback function to monitor optimization"""
        iteration[0] += 1
        e, g = energy_and_gradient(xk)
        if e < 1e9:  # Only record reasonable energies
            energies.append(e)
            grad_norms.append(np.linalg.norm(g))
            print(f"Step {iteration[0]:3d}: E = {e:.8f} Hartree, "
                  f"|grad| = {np.linalg.norm(g):.6f} Hartree/Bohr")
    
    print("Starting QM/MM geometry optimization...")
    print(f"Method: {method}/{basis}")
    print(f"QM atoms: {mol.natm}, MM atoms: {len(mm_charges)}")
    print(f"Optimizer: {optimizer}, gtol: {gtol:.2e}")
    print(f"Max step size: {max_step:.2f} Bohr ({max_step*0.529:.2f} Angstrom)\n")
    
    # Set bounds to limit maximum displacement per step
    # This prevents catastrophic jumps
    bounds = []
    for coord in qm_coords_flat:
        bounds.append((coord - max_step*10, coord + max_step*10))  # Loose bounds
    
    # Optimize geometry with trust region constraints
    result = minimize(
        fun=lambda x: energy_and_gradient(x),
        x0=qm_coords_flat,
        method=optimizer,
        jac=True,
        callback=callback,
        bounds=bounds,
        options={
            'gtol': gtol,
            'maxiter': max_iter,
            'maxls': 20,  # Max line search steps
            'ftol': 1e-9,  # Function tolerance
        }
    )
    
    # Extract results
    opt_coords_bohr = result.x.reshape(-1, 3)
    opt_coords_ang = opt_coords_bohr * 0.529177249
    
    final_energy = result.fun
    final_grad = result.jac.reshape(-1, 3)
    
    # Check if result is reasonable
    success = result.success and final_energy < 0 and final_energy > -1000
    
    print("\n" + "="*60)
    print("QM/MM geometry optimization completed!")
    print(f"Success: {success}")
    print(f"Message: {result.message}")
    print(f"Final energy: {final_energy:.8f} Hartree")
    print(f"Final |gradient|: {np.linalg.norm(final_grad):.6f} Hartree/Bohr")
    print(f"Max gradient component: {np.max(np.abs(final_grad)):.6f} Hartree/Bohr")
    print(f"Number of iterations: {result.nit}")
    print(f"SCF failures: {scf_failures[0]}")
    print("="*60)
    
    return opt_coords_ang, success

def optimize_geometry(molecule, qmm_info, method="rhf", maxsteps=100, check_for_imag=True, max_checks=10, curr_check=0):
    r"""Computes the equilibrium geometry of a molecule.

    Args:
        molecule (~qchem.molecule.Molecule): the molecule object
        method (str): Electronic structure method used to perform geometry optimization.
            Available options are ``"rhf"`` and ``"uhf"`` for restricted and unrestricted
            Hartree-Fock, respectively. Default is ``"rhf"``.

    Returns:
        array[array[float]]: optimized atomic positions in Cartesian coordinates

    **Example**

    >>> symbols  = ['H', 'F']
    >>> geometry = np.array([[0.0, 0.0, 0.0],
    ...                      [0.0, 0.0, 1.0]])
    >>> mol = qml.qchem.Molecule(symbols, geometry)
    >>> eq_geom = qml.qchem.optimize_geometry(mol)
    >>> eq_geom
    array([[ 0.        ,  0.        , -0.40277116],
           [ 0.        ,  0.        ,  1.40277116]])

    """
    pyscf = _import_pyscf()
    geometric = _import_geometric()
    from pyscf.geomopt.geometric_solver import optimize, as_pyscf_method
    from pyscf import qmmm

    scf_res = _single_point(molecule, qmm_info, method)
    geom_eq = optimize(scf_res, maxsteps=maxsteps)

    if molecule.unit == "angstrom":
        return geom_eq.atom_coords(unit="A")
    return geom_eq.atom_coords(unit="B")


def _get_dipole(scf_result, method):
    r"""Evaluate the dipole moment for a Hartree-Fock state.

    Args:
        scf_result (pyscf.scf object): pyscf object from electronic structure calculations
        method (str): Electronic structure method that can be either restricted and unrestricted
            Hartree-Fock,  ``'rhf'`` and ``'uhf'``, respectively. Default is ``'rhf'``.

    Returns:
        TensorLike[float]: dipole moment

    """
    pyscf = _import_pyscf()
    from pyscf.scf import hf

    is_qmmm = hasattr(scf_result, 'mm_mol')
    if not is_qmmm:
        raise ValueError(f"Trying to obtain dipole moments inside QM/MM workflow but input SCF object is not QM/MM, input type was {type(scf_result)}")

    coords = scf_result.mol.atom_coords()
    masses = scf_result.mol.atom_mass_list()
    nuc_mass_center = np.einsum("z,zx->x", masses, coords) / masses.sum()

    dm = scf_result.make_rdm1()
    dip_qm = hf.dip_moment(scf_result.mol, dm, unit="au", origin=nuc_mass_center, verbose=0)

    mm_charges = scf_result.mm_mol.atom_charges()
    mm_coords = scf_result.mm_mol.atom_coords()
    mm_coords_centered = mm_coords - nuc_mass_center
    dip_mm = np.einsum("z,zx->x", mm_charges, mm_coords_centered)

    return dip_qm + dip_mm
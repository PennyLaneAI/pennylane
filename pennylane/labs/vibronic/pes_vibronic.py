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
"""This module contains functions to calculate potential energy surfaces
per normal modes on a grid."""

import numpy as np
import scipy as sp

import pyscf
from pyscf import scf, gto
from pyscf.hessian import thermo

from posym import SymmetryNormalModes

from scipy.spatial.transform import Rotation

from itertools import combinations

from pyscf import gto, scf, mcscf, tdscf, cc
from pyscf.fci import direct_spin0
from pyscf.symm.geom import SymmSys

from pennylane import concurrency, qchem

from pennylane.qchem.vibrational.vibrational_class import (
    VibrationalPES,
    optimize_geometry,
)


# constants
# TODO: Make this code work in atomic units only.
HBAR = (
    sp.constants.hbar * (1000 * sp.constants.Avogadro) * (10**20)
)  # kg*(m^2/s) to (amu)*(angstrom^2/s)
BOHR_TO_ANG = (
    sp.constants.physical_constants["Bohr radius"][0] * 1e10
)  # factor to convert bohr to angstrom
CM_TO_AU = 100 / sp.constants.physical_constants["hartree-inverse meter relationship"][0]  # m to cm


def _calculate_normal_modes(mol, method="RHF", functional=None):
    """
    Calculate normal modes and frequencies for a molecule.

    Args:
        mol: PySCF Mole object

    Returns:
        Dictionary containing vibrational analysis results

    Raises:
        AssertionError: If any frequencies are imaginary, indicating the geometry
                      is not at a minimum
    """

    # Create mean-field object based on method
    if method == "DFT":
        if functional is None:
            raise ValueError("Functional must be specified for DFT calculations")
        if mol.spin == 0:
            mf = scf.RKS(mol)
        else:
            mf = scf.UKS(mol)
        mf.xc = functional
    elif method == "RHF":
        mf = scf.RHF(mol)
    elif method == "ROHF":
        mf = scf.UHF(mol)
    else:
        raise ValueError(f"Unsupported method: {method}")

    mf.kernel()

    # Calculate Hessian
    if method == "DFT":
        if mol.spin == 0:
            from pyscf.hessian import rks

            hess = rks.Hessian(mf).kernel()
        else:
            from pyscf.hessian import uks

            hess = uks.Hessian(mf).kernel()
    elif method == "RHF":
        from pyscf.hessian import rhf

        hess = rhf.Hessian(mf).kernel()
    elif method == "ROHF":
        from pyscf.hessian import uhf

        hess = uhf.Hessian(mf).kernel()
    else:
        raise ValueError(f"Unsupported method: {method} for Hessian calculation")

    # Calculate normal modes using the Hessian
    vib_results = thermo.harmonic_analysis(
        mol, hess, exclude_trans=True, exclude_rot=True, imaginary_freq=True
    )

    # Check for imaginary frequencies
    frequencies = vib_results["freq_wavenumber"]

    if np.any(frequencies.imag != 0):
        raise ValueError(f"Found imaginary frequencies.")

    if mol.topgroup == "Coov":  # use maximal abelian subgroup
        mol.topgroup = "C2v"
        warnings.warn(
            f"Non-abelian point group identified, falling back to abelian subgroup {mol.topgroup}."
        )

    elif mol.topgroup == "Dooh":  # use maximal abelian subgroup
        mol.topgroup = "d2h"
        warnings.warn(
            f"Non-abelian point group identified, falling back to abelian subgroup {mol.topgroup}."
        )

    # Analyze symmetry of normal modes
    sym_modes_gs = SymmetryNormalModes(
        group=mol.topgroup,
        coordinates=[atom[1] for atom in mol._atom],
        modes=vib_results["norm_mode"],
        symbols=[atom[0] for atom in mol._atom],
    )
    mode_irreps = []
    for i in range(len(frequencies)):
        mode_vals = sym_modes_gs.get_state_mode(i).get_ir_representation().values
        irrep_idx = np.argmax(mode_vals)
        mode_irrep = sym_modes_gs.get_state_mode(i).get_ir_representation().index[irrep_idx]
        mode_irreps.append(mode_irrep)

    vib_results["irreps"] = mode_irreps

    return vib_results


def _get_rotation_matrix_to_align_with_z(p_current):
    """
    Calculates the rotation matrix to align a given vector p_current with the Z-axis [0,0,1].

    Args:
        p_current (numpy.ndarray): A 3D vector representing the current principal axis.
                                   Example: np.array([1.0, 2.0, 3.0])

    Returns:
        numpy.ndarray: The 3x3 rotation matrix.
                       Returns None if p_current is a zero vector.
    """
    p_current = np.asarray(p_current, dtype=float)
    if np.allclose(p_current, [0, 0, 0]):
        print("Error: The current principal axis vector cannot be a zero vector.")
        return None

    # Normalize the current principal axis vector (it must be a direction)
    p_current_normalized = p_current / np.linalg.norm(p_current)

    # Define the target vector (Z-axis)
    p_target = np.array([0.0, 0.0, 1.0])

    # Calculate the rotation object.
    # Rotation.align_vectors([b], [a]) finds rotation R such that R @ a aligns with b.
    # We want R @ p_current_normalized to align with p_target.
    rot, _ = Rotation.align_vectors([p_target], [p_current_normalized])

    # Get the rotation matrix
    rotation_matrix = rot.as_matrix()

    return rotation_matrix


def _rotate_molecule(mol_eq):
    r"""Rotate molecule so normal modes are aligned with symmetry operators.

    mol_eq: PySCF Molecule object

    """

    mol_eq.symmetry = True

    sym = SymmSys(mol_eq.atom)

    try:
        axes_of_rot, n_vals = zip(*sym.search_possible_rotations())
    except ValueError as e:
        print("Something went wrong with SymmSys. Defaulting principal axis.")
        axes_of_rot = [[0, 0, 1]]
        n_vals = [1]

    principal_axis_idx = np.argmax(n_vals)
    principal_axis_vec = axes_of_rot[principal_axis_idx]
    rotation_matrix = _get_rotation_matrix_to_align_with_z(principal_axis_vec)
    coords = mol_eq.atom_coords(unit="Bohr")

    rotated_coords = [coord @ rotation_matrix.T for coord in coords]

    mol_eq.atom = list(zip(mol_eq.elements, rotated_coords))
    mol_eq.build()

    return mol_eq


def _harmonic_analysis(mol_eq, rotate=True):
    r"""

    mol_eq: PySCF Molecule object
    """
    if rotate:
        mol_eq = _rotate_molecule(mol_eq)

    # Calculate normal modes
    vib_results = _calculate_normal_modes(mol_eq)

    # Extract results
    frequencies = vib_results["freq_wavenumber"]
    normal_modes = vib_results["norm_mode"]

    # TODO: remove if not needed
    reduced_masses = vib_results["reduced_mass"]
    irreps = vib_results["irreps"]

    return frequencies, normal_modes


def _grid_points(grid_type, grid_range, n_points):
    r"""Generate grid points in one dimension"""

    if grid_type == "uniform":
        return np.linspace(-grid_range / 2, grid_range / 2, n_points)

    elif grid_type == "gaussian":
        x = np.linspace(-grid_range / 2, grid_range / 2, n_points)
        weights = np.exp(-(x**2) / (grid_range / 4) ** 2)
        weights = weights / np.sum(weights)
        return np.cumsum(weights) * grid_range - grid_range / 2

    elif grid_type == "chebyshev":
        k = np.arange(n_points)
        return grid_range / 2 * np.cos(np.pi * (2 * k + 1) / (2 * n_points))

    else:
        raise ValueError(f"Unknown grid type: {grid_type}")


def _displace_geometry(eq_geometry, normal_modes, frequencies, mode_indices, displacements):
    """Displace geometry along one or more normal modes."""
    n_atoms = len(eq_geometry)

    displaced_coords = np.array([atom[0:] for atom in eq_geometry]).reshape(n_atoms, 3)

    if isinstance(mode_indices, int):
        mode_indices = [mode_indices]
        displacements = [displacements]

    for mode_idx, q_val in zip(mode_indices, displacements):
        vec = normal_modes[mode_idx].reshape(n_atoms, 3)
        scaling_factor = np.sqrt(HBAR / (2 * np.pi * frequencies[mode_idx] * 100 * sp.constants.c))
        displaced_coords += q_val * vec * scaling_factor

    return displaced_coords


def _generate_1d_grid(frequencies, normal_modes, eq_geometry, displacements):
    """Generate 1D grid points (d=1)"""
    grid_points = []
    n_modes = len(frequencies)

    displaced_coords = _displace_geometry(eq_geometry, normal_modes, frequencies, [], [])

    grid_points.append({"mode_indices": [], "displacements": [], "coordinates": displaced_coords})

    for mode_idx in range(n_modes):
        displacements = [d for d in displacements if abs(d) > 1e-10]
        for disp in displacements:
            displaced_coords = _displace_geometry(
                eq_geometry, normal_modes, frequencies, [mode_idx], [disp]
            )
            grid_points.append(
                {
                    "mode_indices": [mode_idx],
                    "displacements": [disp],
                    "coordinates": displaced_coords,
                }
            )

    return grid_points


def _generate_2d_grid(frequencies, normal_modes, eq_geometry, displacements):
    """Generate 2D grid points (d=2)"""
    grid_points = []
    n_modes = len(frequencies)

    modes_to_use = range(n_modes)

    for i, j in combinations(modes_to_use, 2):
        for di in displacements:
            for dj in displacements:
                if abs(di) < 1e-10 and abs(dj) < 1e-10:
                    continue
                displaced_coords = _displace_geometry(
                    eq_geometry, normal_modes, frequencies, [i, j], [di, dj]
                )
                grid_points.append(
                    {
                        "mode_indices": [i, j],
                        "displacements": [di, dj],
                        "coordinates": displaced_coords,
                    }
                )
    return grid_points


def _generate_3d_grid(frequencies, normal_modes, eq_geometry, displacements):
    """Generate 3D grid points (d=3)"""
    grid_points = []
    n_modes = len(frequencies)

    modes_to_use = range(n_modes)

    for i, j, k in combinations(modes_to_use, 3):
        for di in displacements:
            for dj in displacements:
                for dk in displacements:
                    if abs(di) < 1e-10 and abs(dj) < 1e-10 and abs(dk) < 1e-10:
                        continue
                    displaced_coords = _displace_geometry(
                        eq_geometry, normal_modes, frequencies, [i, j, k], [di, dj, dk]
                    )
                    grid_points.append(
                        {
                            "mode_indices": [i, j, k],
                            "displacements": [di, dj, dk],
                            "coordinates": displaced_coords,
                        }
                    )
    return grid_points


def _run_casscf(
    symbols,
    coords,
    ncas,
    nelecas,
    basis="6-31g*",
    max_macro=50,
    max_micro=20,
    conv_tol=1e-7,
    nroots=2,
    restrict_to_singlet=True,
    spin=0,
    point_group=None,
):
    """
    Run CASSCF calculation and return energies.

    Args:


    Returns:
        List of CASSCF energies for each root
    """

    coords = [[symbol, tuple(np.array(coords)[i])] for i, symbol in enumerate(symbols)]

    mol = gto.Mole()
    mol.atom = coords
    mol.unit = "Angstrom"
    mol.basis = basis
    mol.spin = spin
    if point_group:
        mol.symmetry = point_group.lower()
        mol.symmetry_subgroup = point_group.lower()

    mol.build()

    # Run RHF/UHF based on spin
    if mol.spin == 0:
        mf = scf.RHF(mol)
    else:
        mf = scf.UHF(mol)
    mf.kernel()

    # Run CASSCF
    mc = mcscf.CASSCF(mf, ncas, nelecas)
    mc.conv_tol = conv_tol

    # Set spin restriction if requested
    if restrict_to_singlet:
        if mol.spin != 0:
            raise ValueError("Cannot restrict to singlet for open-shell system")
        mc.fcisolver = direct_spin0.FCISolver(mol)
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
        # CASSCF energies for all states:
        energies = mc.e_states
    else:
        energies = [mc.kernel()[0]]

    return energies


def vibronic_pes(
    molecule,
    n_points=5,
    grid_range=2.0,
    grid_type="uniform",
    method="rhf",
    optimize=True,
    rotate=True,
    pes_level=1,
    num_workers=1,
    backend="serial",
):
    r"""Computes potential energy surfaces along vibrational normal modes.

    Args:
        molecule (~qchem.molecule.Molecule): the molecule object
        n_points (int): number of points for computing the potential energy surface. Default value is ``5``.
        grid_range (float): maximum range of points for computing the potential energy surface. Default value is ``2``.
        grid_type (str): method to generate the grid points. Options are ``'uniform'``, ``'gaussian'`` and ``'chebyshev'``.
            Default is ``'uniform'``.
        method (str): Electronic structure method used to perform geometry optimization.
            Available options are ``"rhf"`` and ``"uhf"`` for restricted and unrestricted
            Hartree-Fock, respectively. Default is ``"rhf"``.
        optimize (bool): if ``True`` perform geometry optimization. Default is ``True``.
        rotate(bool): if ``True`` rotate the molecule to align normal modes with symmetry operators
        pes_level (int): the number of coupled modes for generating the potential energy surface data.
            Available options are ``1``, ``2``, ``3`` for one-mode, two-mode and three-mode levels,
            respctively.
        num_workers (int): the number of concurrent units used for the computation. Default value is
            set to 1.
        backend (string): the executor backend from the list of supported backends. Available
            options are ``mp_pool``, ``cf_procpool``, ``cf_threadpool``, ``serial``,
            ``mpi4py_pool``, ``mpi4py_comm``. Default value is set to ``serial``. See Usage Details
            for more information.

    Returns:
       VibrationalPES: the VibrationalPES object

    **Example**

    >>> symbols  = ['H', 'F']
    >>> geometry = np.array([[0.0, 0.0, -0.40277116], [0.0, 0.0, 1.40277116]])
    >>> mol = qml.qchem.Molecule(symbols, geometry)
    >>> pes = qml.qchem.vibronic_pes(mol, optimize=False)
    >>> print(pes.freqs)
    [0.02038828]

    .. details::
        :title: Usage Details

        The ``backend`` options allow to run calculations using multiple threads or multiple
        processes.

        - ``serial``: This executor wraps Python standard library calls without support for
            multithreaded or multiprocess execution. Any calls to external libraries that utilize
            threads, such as BLAS through numpy, can still use multithreaded calls at that layer.

        - ``mp_pool``: This executor wraps Python standard library `multiprocessing.Pool <https://docs.python.org/3/library/multiprocessing.html#module-multiprocessing.pool>`_
            interface, and provides support for execution using multiple processes.

        - ``cf_procpool``: This executor wraps Python standard library `concurrent.futures.ProcessPoolExecutor <https://docs.python.org/3/library/concurrent.futures.html#processpoolexecutor>`_
            interface, and provides support for execution using multiple processes.

        - ``cf_threadpool``: This executor wraps Python standard library `concurrent.futures.ThreadPoolExecutor <https://docs.python.org/3/library/concurrent.futures.html#threadpoolexecutor>`_
            interface, and provides support for execution using multiple threads. The threading
            executor may not provide execution speed-ups for tasks when using a GIL-enabled Python.

        - ``mpi4py_pool``: This executor wraps the `mpi4py.futures.MPIPoolExecutor <https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html#mpipoolexecutor>`_
            class, and provides support for execution using multiple processes launched using MPI.

        - ``mpi4py_comm``: This executor wraps the `mpi4py.futures.MPICommExecutor <https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html#mpicommexecutor>`_
            class, and provides support for execution using multiple processes launched using MPI.
    """
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)

        if n_points < 1:
            raise ValueError("Number of sample points cannot be less than 1.")

        if optimize:
            geom_eq = optimize_geometry(molecule, method)
            mol_eq = qchem.Molecule(
                molecule.symbols,
                geom_eq,
                unit=molecule.unit,
                basis_name=molecule.basis_name,
                charge=molecule.charge,
                mult=molecule.mult,
                load_data=molecule.load_data,
            )
        else:
            mol_eq = molecule

        scf_result = _single_point(mol_eq, method)

        geom = [
            [symbol, tuple(np.array(mol_eq.coordinates)[i])]
            for i, symbol in enumerate(mol_eq.symbols)
        ]
        spin = int((mol_eq.mult - 1) / 2)
        mol = pyscf.gto.Mole(atom=geom, symmetry="C1", spin=spin, charge=mol_eq.charge, unit="Bohr")
        mol.basis = mol_eq.basis_name
        mol.build()

        freqs, vectors = _harmonic_analysis(mol, rotate)

        grid = _grid_points(grid_type, grid_range, n_points)

        eq_geometry = mol_eq.coordinates * BOHR_TO_ANG

        geometry_1d = _generate_1d_grid(freqs, vectors, eq_geometry, grid)

        geometry_2d = _generate_2d_grid(freqs, vectors, eq_geometry, grid)

        geometry_3d = _generate_3d_grid(freqs, vectors, eq_geometry, grid)

        energy_1 = []
        for geometry_point in geometry_1d:
            new_coords = geometry_point["coordinates"]
            energy_1.append(_run_casscf(mol_eq.symbols, new_coords, ncas=2, nelecas=2))

        energy_2 = []
        for geometry_point in geometry_2d:
            new_coords = geometry_point["coordinates"]
            energy_2.append(_run_casscf(mol_eq.symbols, new_coords, ncas=2, nelecas=2))

        energy_3 = []
        for geometry_point in geometry_3d:
            new_coords = geometry_point["coordinates"]
            energy_3.append(_run_casscf(mol_eq.symbols, new_coords, ncas=2, nelecas=2))

        freqs = freqs * CM_TO_AU

        return VibrationalPES(
            freqs,
            grid,
            pes_data=[energy_1, energy_2, energy_3],
        )

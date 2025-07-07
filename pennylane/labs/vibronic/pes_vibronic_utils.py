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

from pathlib import Path
from tempfile import TemporaryDirectory

import pyscf
from pyscf.hessian import thermo

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


def _import_posym():
    """Import sklearn."""
    try:
        import posym
    except ImportError as Error:
        raise ImportError("This feature requires posym.") from Error

    return posym


# constants
# TODO: Make this code work in atomic units only.
HBAR = (
    sp.constants.hbar * (1000 * sp.constants.Avogadro) * (10**20)
)  # kg*(m^2/s) to (amu)*(angstrom^2/s)
BOHR_TO_ANG = (
    sp.constants.physical_constants["Bohr radius"][0] * 1e10
)  # factor to convert bohr to angstrom
CM_TO_AU = 100 / sp.constants.physical_constants["hartree-inverse meter relationship"][0]  # m to cm


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


def _harmonic_analysis(
    molecule,
    rotate=True,
    method="rhf",
    functional="b3lyp",
):
    r"""Computes the harmonic vibrational normal modes and frequencies of a molecule.

    Args:
        molecule (:func:`~pennylane.qchem.molecule.Molecule`): Molecule object at equilibrium geometry
        method (str): Electronic structure method that can be either restricted and unrestricted
            Hartree-Fock,  ``'rhf'`` and ``'uhf'``, respectively or 'dft'. Default is ``'rhf'``.
        functional (str): The exchange-correlation functional for if DFT method is selected. Default
            is ``b3lyp``.
        rotate (bool): If ``True``, the molecule will be rotated such that the normal modes are
            aligned with symmetry operators. Default is ``True``.

    Returns:
        tuple[array[float]]: vibrational frequencies and normal modes

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

    if isinstance(molecule, qchem.Molecule):

        geom = [
            [symbol, tuple(np.array(molecule.coordinates)[i])]
            for i, symbol in enumerate(molecule.symbols)
        ]
        spin = int((molecule.mult - 1) / 2)
        charge = molecule.charge
        basis = molecule.basis_name
        molecule = pyscf.gto.Mole(
            atom=geom, symmetry="C1", spin=spin, charge=charge, unit="Bohr", basis=basis
        )
        molecule.build()

    if rotate:
        molecule = _rotate_molecule(molecule)

    if method == "dft":
        if molecule.spin == 0:
            from pyscf.hessian import rks

            mf = scf.RKS(molecule)
            mf.xc = functional
            mf.kernel()
            hess = rks.Hessian(mf).kernel()
        else:
            from pyscf.hessian import uks

            mf = scf.UKS(molecule)
            mf.xc = functional
            mf.kernel()
            hess = uks.Hessian(mf).kernel()
    elif method == "rhf":
        from pyscf.hessian import rhf

        mf = scf.RHF(molecule)
        mf.kernel()
        hess = rhf.Hessian(mf).kernel()
    elif method == "uhf":
        from pyscf.hessian import uhf

        mf = scf.UHF(molecule)
        mf.kernel()
        hess = uhf.Hessian(mf).kernel()
    else:
        raise ValueError(f"Unsupported method: {method}.")

    vib_results = thermo.harmonic_analysis(
        molecule, hess, exclude_trans=True, exclude_rot=True, imaginary_freq=True
    )

    frequencies = vib_results["freq_wavenumber"]
    normal_modes = vib_results["norm_mode"]

    if np.any(frequencies.imag != 0):
        raise ValueError(f"Found imaginary frequencies.")

    return frequencies, normal_modes


def _grid_points(grid_type, grid_range, n_points):
    r"""Generate grid points in one dimension.

    Args:
        grid_type (str): method for generating grid points. The currently supported types are
            'uniform', 'gaussian', 'chebyshev'.
        grid_range (float): the range of the generating grid points
        n_points (int): the number of the generating grid points

    Returns:
        array[float]: array of the grid points

    **Example**

    >>>
    """

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
    """Displace geometry along one or more normal modes.

     Args:
        eq_geometry (tensorlike[float]): the equilibrium geometry
        normal_modes (tensorlike[float]): the vibrational normal modes
        frequencies (array[float]): the vibrational frequencies
        mode_indices (list(int)): the indices of the normal modes used for displacement
        displacements (array(float)): the grid points

    Returns:
        tensorlike[float]: the displaced coordinates

    **Example**

    >>>
    """
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
    r"""Generates one-mode displaced coordinates.

     Args:
        frequencies (array[float]): the vibrational frequencies
        normal_modes (tensorlike[float]): the vibrational normal modes
        eq_geometry (tensorlike[float]): the equilibrium geometry
        displacements (array(float)): the grid points

    Returns:
        tensorlike[float]: the mode indices, grid points, and displaced coordinates

    **Example**

    >>>
    """
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
    r"""Generates two-mode displaced coordinates.

     Args:
        frequencies (array[float]): the vibrational frequencies
        normal_modes (tensorlike[float]): the vibrational normal modes
        eq_geometry (tensorlike[float]): the equilibrium geometry
        displacements (array(float)): the grid points

    Returns:
        tensorlike[float]: the mode indices, grid points, and displaced coordinates

    **Example**

    >>>
    """
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
    r"""Generates three-mode displaced coordinates.

     Args:
        frequencies (array[float]): the vibrational frequencies
        normal_modes (tensorlike[float]): the vibrational normal modes
        eq_geometry (tensorlike[float]): the equilibrium geometry
        displacements (array(float)): the grid points

    Returns:
        tensorlike[float]: the mode indices, grid points, and displaced coordinates

    **Example**

    >>>
    """
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

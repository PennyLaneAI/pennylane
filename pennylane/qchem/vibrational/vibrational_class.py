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
"""This module contains functions and classes to generate a pes object.
This object stores all the necessary information to construct
vibrational Hamiltonian for a given molecule."""

from dataclasses import dataclass

import numpy as np

import pennylane as qml

from ..openfermion_pyscf import _import_pyscf

# pylint: disable=import-outside-toplevel, unused-variable, too-many-instance-attributes, too-many-arguments

BOHR_TO_ANG = 0.5291772106  # Factor to convert Bohr to Angstrom


@dataclass
class VibrationalPES:
    r"""Data class to save the PES information to an object.

    Args:
       freqs: normal-mode frequencies
       gauss_grid: 1-D array containing the sample points on the Gauss-Hermite quadrature grid.
       gauss_weights: 1-D array containing the weights on the Gauss-Hermite quadrature grid.
       uloc: Localization matrix indicating the relationship between original and localized modes.
       pes_arr: Tuple containing one-mode, two-mode and three-mode PES.
       dipole_arr: Tuple containing one-mode, two-mode and three-mode dipole.
       localized: Whether the localization of modes was used to generate PES and dipole. Defauls is True.
       dipole_level: Defines the level upto which dipole matrix elements are to be calculated. Input values can be
                     1, 2, or 3 for upto one-mode dipole, two-mode dipole and three-mode dipole respectively. Default
                     value is 2.

    """

    def __init__(
        self,
        freqs,
        gauss_grid,
        gauss_weights,
        uloc,
        pes_arr,
        dipole_arr,
        localized=True,
        dipole_level=2,
    ):
        self.freqs = freqs
        self.gauss_grid = gauss_grid
        self.gauss_weights = gauss_weights
        self.uloc = uloc
        self.pes_onemode = pes_arr[0]
        self.pes_twomode = pes_arr[1]
        self.pes_threemode = pes_arr[2] if len(pes_arr) > 2 else None
        self.dipole_onemode = dipole_arr[0]
        self.dipole_twomode = dipole_arr[1] if dipole_level >= 2 else None
        self.dipole_threemode = dipole_arr[2] if dipole_level >= 3 else None
        self.localized = localized
        self.dipole_level = dipole_level


def harmonic_analysis(scf_result, method="rhf"):
    r"""Performs harmonic analysis by evaluating the Hessian using PySCF routines.

    Args:
       scf_result: pyscf object from electronic structure calculations
       method: Electronic structure method to define the level of theory
            for harmonic analysis. Default is restricted Hartree-Fock ``'rhf'``.

    Returns:
       a tuple of frequencies of normal modes and their corresponding displacement vectors
    """
    pyscf = _import_pyscf()
    from pyscf.hessian import thermo

    method = method.strip().lower()
    if method not in ["rhf", "uhf"]:
        raise ValueError(f"Specified electronic structure method, {method} is not available.")

    hess = getattr(pyscf.hessian, method).Hessian(scf_result).kernel()
    harmonic_res = thermo.harmonic_analysis(scf_result.mol, hess)

    return harmonic_res["freq_wavenumber"], harmonic_res["norm_mode"]


def _single_point(molecule, method="rhf"):
    r"""Runs electronic structure calculation.

    Args:
      molecule: Molecule object.
      method: Electronic structure method to define the level of theory.
              Default is restricted Hartree-Fock 'rhf'.

    Returns:
      pyscf object from electronic structure calculation
    """
    pyscf = _import_pyscf()

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
    return scf_obj


def _import_geometric():
    """Import geometric."""
    try:
        import geometric
    except ImportError as Error:
        raise ImportError(
            "This feature requires geometric. It can be installed with: pip install geometric."
        ) from Error

    return geometric


def optimize_geometry(molecule, method="rhf"):
    r"""Obtains equilibrium geometry for the molecule.

    Args:
      molecule: Molecule object.
      method: Electronic structure method to define the level of theory.
              Default is restricted Hartree-Fock ``'rhf'``.

    Returns:
      molecule object with optimized geometry

    """
    pyscf = _import_pyscf()
    geometric = _import_geometric()
    from pyscf.geomopt.geometric_solver import optimize

    scf_res = _single_point(molecule, method)
    geom_eq = optimize(scf_res, maxsteps=100)

    mol_eq = qml.qchem.Molecule(
        molecule.symbols,
        geom_eq.atom_coords(unit="B"),
        unit="Bohr",
        basis_name=molecule.basis_name,
        charge=molecule.charge,
        mult=molecule.mult,
        load_data=molecule.load_data,
    )

    scf_result = _single_point(mol_eq, method)
    return mol_eq, scf_result


def _get_rhf_dipole(scf_result):
    """
    Given an restricted Hartree-Fock object, evaluate the dipole moment
    in the restricted Hartree-Fock state.

    Args:
       scf_result: pyscf object from electronic structure calculations

    Returns:
       dipole moment
    """

    charges = scf_result.mol.atom_charges()
    coords = scf_result.mol.atom_coords()
    masses = scf_result.mol.atom_mass_list(isotope_avg=True)
    nuc_mass_center = np.einsum("z,zx->x", masses, coords) / masses.sum()
    scf_result.mol.set_common_orig_(nuc_mass_center)
    dip_ints = scf_result.mol.intor("int1e_r", comp=3)

    t_dm1 = scf_result.make_rdm1()
    if len(t_dm1.shape) == 3:
        dipole_e_alpha = np.einsum("xij,ji->x", dip_ints, t_dm1[0, ::])
        dipole_e_beta = np.einsum("xij,ji->x", dip_ints, t_dm1[1, ::])
        dipole_e = dipole_e_alpha + dipole_e_beta
    else:
        dipole_e = np.einsum("xij,ji->x", dip_ints, t_dm1)

    centered_coords = np.copy(coords)
    for num_atom in range(len(charges)):
        centered_coords[num_atom, :] -= nuc_mass_center
    dipole_n = np.einsum("z,zx->x", charges, centered_coords)

    dipole = -dipole_e + dipole_n
    return dipole


def _get_uhf_dipole(scf_result):
    """
    Given an unrestricted Hartree-Fock object, evaluate the dipole moment
    in the unrestricted Hartree-Fock state.

    Args:
       scf_result: pyscf object from electronic structure calculations

    Returns:
       dipole moment

    """

    charges = scf_result.mol.atom_charges()
    coords = scf_result.mol.atom_coords()
    masses = scf_result.mol.atom_mass_list(isotope_avg=True)
    nuc_mass_center = np.einsum("z,zx->x", masses, coords) / masses.sum()
    scf_result.mol.set_common_orig_(nuc_mass_center)

    t_dm1_alpha, t_dm1_beta = scf_result.make_rdm1()

    dip_ints = scf_result.mol.intor("int1e_r", comp=3)
    dipole_e_alpha = np.einsum("xij,ji->x", dip_ints, t_dm1_alpha)
    dipole_e_beta = np.einsum("xij,ji->x", dip_ints, t_dm1_beta)
    dipole_e = dipole_e_alpha + dipole_e_beta

    centered_coords = np.copy(coords)
    for num_atom in range(len(charges)):
        centered_coords[num_atom, :] -= nuc_mass_center
    dipole_n = np.einsum("z,zx->x", charges, centered_coords)

    dipole = -dipole_e + dipole_n
    return dipole


def get_dipole(scf_result, method):
    r"""Evaluate the dipole moment for a Hartree-Fock state.

    Args:
       scf_result: pyscf object from electronic structure calculations
       method: Electronic structure method to define the level of theory
            for dipole moment calculation. Input values can be ``'rhf'`` or ``'uhf'``.
            Default is restricted Hartree-Fock ``'rhf'``.
    Returns:
       dipole moment

    """
    method = method.strip().lower()
    if method == "rhf":
        return _get_rhf_dipole(scf_result)

    return _get_uhf_dipole(scf_result)
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

import numpy as np

import pennylane as qml

from ..openfermion_pyscf import _import_pyscf

# pylint: disable=import-outside-toplevel, unused-variable

BOHR_TO_ANG = 0.529177


def harmonic_analysis(scf_result, method="rhf"):
    r"""Performs harmonic analysis by evaluating the Hessian using PySCF routines.
    Args:
       scf_result: pyscf object from electronic structure calculations
       method: Electronic structure method to define the level of theory
            for harmonic analysis. Default is restricted Hartree-Fock 'rhf'.

    Returns:
       pyscf object containing information about the harmonic analysis
    """
    pyscf = _import_pyscf()
    from pyscf.hessian import thermo

    method = method.strip().lower()
    if method not in ["rhf", "uhf"]:
        raise ValueError(f"Specified electronic structure method, {method} is not available.")

    hess = getattr(pyscf.hessian, method).Hessian(scf_result).kernel()

    harmonic_res = thermo.harmonic_analysis(scf_result.mol, hess)
    return harmonic_res


def single_point(molecule, method="rhf"):
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
        [symbol, tuple(np.array(molecule.coordinates)[i] * BOHR_TO_ANG)]
        for i, symbol in enumerate(molecule.symbols)
    ]

    spin = int((molecule.mult - 1) / 2)
    mol = pyscf.gto.Mole(
        atom=geom, symmetry="C1", spin=spin, charge=molecule.charge, unit="Angstrom"
    )
    mol.basis = molecule.basis_name
    mol.build()
    if method == "rhf":
        scf_obj = pyscf.scf.RHF(mol).run(verbose=0)
    else:
        scf_obj = pyscf.scf.UHF(mol).run(verbose=0)
    return scf_obj


def optimize_geometry(molecule, method="rhf"):
    r"""Obtains equilibrium geometry for the molecule.

    Args:
      molecule: Molecule object.
      method: Electronic structure method to define the level of theory.
              Default is restricted Hartree-Fock 'rhf'.

    Returns:
      molecule object with optimized geometry

    """
    pyscf = _import_pyscf()
    from pyscf.geomopt.geometric_solver import optimize

    scf_res = single_point(molecule, method)
    geom_eq = optimize(scf_res, maxsteps=100)

    mol_eq = qml.qchem.Molecule(
        molecule.symbols,
        geom_eq.atom_coords(unit="B"),
        unit="bohr",
        basis_name=molecule.basis_name,
        charge=molecule.charge,
        mult=molecule.mult,
        load_data=molecule.load_data,
    )

    scf_result = single_point(mol_eq, method)

    return mol_eq, scf_result

def _get_rhf_dipole(disp_hf):
    """
    Given an restricted Hartree-Fock object, evaluate the dipole moment 
    in the restricted Hartree-Fock state.
    """

    charges = disp_hf.mol.atom_charges()
    coords = disp_hf.mol.atom_coords()
    masses = disp_hf.mol.atom_mass_list(isotope_avg=True)
    nuc_mass_center = np.einsum('z,zx->x', masses, coords)\
        / masses.sum()
    disp_hf.mol.set_common_orig_(nuc_mass_center)
    dip_ints = disp_hf.mol.intor('int1e_r', comp=3)

    t_dm1 = disp_hf.make_rdm1()
    if len(t_dm1.shape) == 3:
        dipole_e_alpha = np.einsum('xij,ji->x', dip_ints, t_dm1[0,::])
        dipole_e_beta = np.einsum('xij,ji->x', dip_ints, t_dm1[1,::])
        dipole_e = dipole_e_alpha + dipole_e_beta
    else:
        dipole_e = np.einsum('xij,ji->x', dip_ints, t_dm1)

    centered_coords = np.copy(coords)
    for num_atom in range(len(charges)):
        centered_coords[num_atom,:] -= nuc_mass_center
    dipole_n = np.einsum('z,zx->x', charges, centered_coords)

    dipole = -dipole_e + dipole_n
    return dipole

def _get_uhf_dipole(disp_hf):
    """
    Given an unrestricted Hartree-Fock object, evaluate the dipole moment 
    in the unrestricted Hartree-Fock state.
    """

    charges = disp_hf.mol.atom_charges()
    coords = disp_hf.mol.atom_coords()
    masses = disp_hf.mol.atom_mass_list(isotope_avg=True)
    nuc_mass_center = np.einsum('z,zx->x', masses, coords)\
        / masses.sum()
    disp_hf.mol.set_common_orig_(nuc_mass_center)

    t_dm1_alpha, t_dm1_beta = disp_hf.make_rdm1()

    dip_ints = disp_hf.mol.intor('int1e_r', comp=3)
    dipole_e_alpha = np.einsum('xij,ji->x', dip_ints, t_dm1_alpha)
    dipole_e_beta = np.einsum('xij,ji->x', dip_ints, t_dm1_beta)
    dipole_e = dipole_e_alpha + dipole_e_beta

    centered_coords = np.copy(coords)
    for num_atom in range(len(charges)):
        centered_coords[num_atom,:] -= nuc_mass_center
    dipole_n = np.einsum('z,zx->x', charges, centered_coords)

    dipole = -dipole_e + dipole_n
    return dipole

def get_dipole(hf, method):
    r"""Evaluate the dipole moment for a Hartree-Fock state."""
    method = method.strip().lower()
    if method == 'rhf':
        return _get_rhf_dipole(hf)

    return _get_uhf_dipole(hf)

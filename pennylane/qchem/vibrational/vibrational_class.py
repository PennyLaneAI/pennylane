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
            for dipole moment calculation. Input values cal be ``'rhf'`` or ``'uhf'``.
            Default is restricted Hartree-Fock ``'rhf'``.
    Returns:
       dipole moment

    """
    method = method.strip().lower()
    if method == "rhf":
        return _get_rhf_dipole(scf_result)

    return _get_uhf_dipole(scf_result)

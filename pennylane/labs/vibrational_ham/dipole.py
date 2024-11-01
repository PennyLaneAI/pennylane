import numpy as np
import matplotlib.pyplot as plt

from pyscf import gto, scf
from pyscf import lib
from pyscf import hessian
import h5py

hbar = 6.022 * 1.055e12  # (amu)*(angstrom^2/s)
c_light = 3 * 10**8  # m/s
au_to_cm = 219475


def _get_rhf_dipole(disp_hf):
    """
    Given an restricted Hartree-Fock object, evaluate the dipole moment
    in the restricted Hartree-Fock state.
    """
    # dipole matrix elements
    charges = disp_hf.mol.atom_charges()
    coords = disp_hf.mol.atom_coords()
    masses = disp_hf.mol.atom_mass_list(isotope_avg=True)
    nuc_mass_center = (
        np.einsum("z,zx->x", masses, coords) / masses.sum()
    )  # if you switch masses -> charges, you get rid of nuclear dipole!
    disp_hf.mol.set_common_orig_(nuc_mass_center)
    dip_ints = disp_hf.mol.intor("int1e_r", comp=3)
    # obtain ground state density matrix in MO representation
    t_dm1 = disp_hf.make_rdm1()
    # compute total electronic transition dipole
    if len(t_dm1.shape) == 3:
        # calculations come from spin != 0, separate alpha and beta sectors and sum
        dipole_e_alpha = np.einsum("xij,ji->x", dip_ints, t_dm1[0, ::])
        dipole_e_beta = np.einsum("xij,ji->x", dip_ints, t_dm1[1, ::])
        dipole_e = dipole_e_alpha + dipole_e_beta
    else:
        dipole_e = np.einsum("xij,ji->x", dip_ints, t_dm1)
    # add on the nuclear dipole
    centered_coords = np.copy(coords)
    for num_atom in range(len(charges)):
        centered_coords[num_atom, :] -= nuc_mass_center
    dipole_n = np.einsum("z,zx->x", charges, centered_coords)
    # total dipole
    dipole = -dipole_e + dipole_n
    return dipole


def _get_uhf_dipole(disp_hf):
    """
    Given an unrestricted Hartree-Fock object, evaluate the dipole moment
    in the unrestricted Hartree-Fock state.
    """
    # dipole matrix elements
    charges = disp_hf.mol.atom_charges()
    coords = disp_hf.mol.atom_coords()
    masses = disp_hf.mol.atom_mass_list(isotope_avg=True)
    nuc_mass_center = (
        np.einsum("z,zx->x", masses, coords) / masses.sum()
    )  # if you switch masses -> charges, you get rid of nuclear dipole!
    disp_hf.mol.set_common_orig_(nuc_mass_center)

    # Obtain UHF density matrix in MO representation
    t_dm1_alpha, t_dm1_beta = disp_hf.make_rdm1()

    # Compute total electronic transition dipole for alpha and beta spins
    dip_ints = disp_hf.mol.intor("int1e_r", comp=3)
    dipole_e_alpha = np.einsum("xij,ji->x", dip_ints, t_dm1_alpha)
    dipole_e_beta = np.einsum("xij,ji->x", dip_ints, t_dm1_beta)
    dipole_e = dipole_e_alpha + dipole_e_beta

    # add on the nuclear dipole
    centered_coords = np.copy(coords)
    for num_atom in range(len(charges)):
        centered_coords[num_atom, :] -= nuc_mass_center
    dipole_n = np.einsum("z,zx->x", charges, centered_coords)
    # total dipole
    dipole = -dipole_e + dipole_n
    return dipole


def get_dipole(hf, method):
    if method == "rhf":
        return _get_rhf_dipole(hf)
    elif method == "uhf":
        return _get_uhf_dipole(hf)

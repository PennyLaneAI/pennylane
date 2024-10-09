import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from pyscf import gto, scf
from pyscf import lib
from pyscf import hessian
import h5py

hbar = 6.022*1.055e12 # (amu)*(angstrom^2/s)
c_light = 3*10**8 # m/s
au_to_cm = 219475

def load_pes_threebody(num_pieces, nmodes, ngridpoints):
    """
    Loader to combine results from multiple ranks.
    """
    final_shape = (nmodes, nmodes, nmodes, \
                             ngridpoints, ngridpoints, ngridpoints)
    nmode_combos = nmodes**3

    pes_threebody = np.zeros(np.prod(final_shape))
    r0 = 0
    r1 = 0
    for mode_combo in range(nmode_combos):
        local_chunk = np.zeros(ngridpoints**3)

        l0 = 0
        l1 = 0
        for rank in range(num_pieces):
            f = h5py.File("v3data" + f"_{rank}" + '.hdf5', 'r+')
            local_pes_threebody = f['V3_PES'][()]
            chunk = np.array_split(local_pes_threebody, nmode_combos)[mode_combo]
            l1 += len(chunk)
            local_chunk[l0:l1] = chunk
            l0 += len(chunk)

        r1 += len(local_chunk)
        pes_threebody[r0:r1] = local_chunk
        r0 += len(local_chunk)

    pes_threebody = pes_threebody.reshape(final_shape)

    return pes_threebody


def load_dipole_threebody(num_pieces, nmodes, ngridpoints):
    """
    Loader to combine results from multiple ranks.
    """
    final_shape = (nmodes, nmodes, nmodes, \
                             ngridpoints, ngridpoints, ngridpoints)
    nmode_combos = nmodes**3

    dipole_threebody = np.zeros((np.prod(final_shape), 3))
    r0 = 0
    r1 = 0
    for mode_combo in range(nmode_combos):
        local_chunk = np.zeros((ngridpoints**3,3))

        l0 = 0
        l1 = 0
        for rank in range(num_pieces):
            f = h5py.File("v3data" + f"_{rank}" + '.hdf5', 'r+')
            local_dipole_threebody = f['D3_DMS'][()]
            chunk = np.array_split(local_dipole_threebody, nmode_combos, \
                                                        axis=0)[mode_combo]
            l1 += chunk.shape[0]
            local_chunk[l0:l1,:] = chunk
            l0 += chunk.shape[0]

        r1 += local_chunk.shape[0]
        dipole_threebody[r0:r1,:] = local_chunk
        r0 += local_chunk.shape[0]

    dipole_threebody = dipole_threebody.reshape(final_shape + (3,))

    return dipole_threebody


def get_rhf_dipole(disp_hf):
    """
    Given an restricted Hartree-Fock object, evaluate the dipole moment 
    in the restricted Hartree-Fock state.
    """
    # dipole matrix elements
    charges = disp_hf.mol.atom_charges()
    coords = disp_hf.mol.atom_coords()
    masses = disp_hf.mol.atom_mass_list(isotope_avg=True)
    nuc_mass_center = np.einsum('z,zx->x', masses, coords)\
        / masses.sum() # if you switch masses -> charges, you get rid of nuclear dipole!
    disp_hf.mol.set_common_orig_(nuc_mass_center)
    dip_ints = disp_hf.mol.intor('int1e_r', comp=3)
    # obtain ground state density matrix in MO representation
    t_dm1 = disp_hf.make_rdm1()
    # compute total electronic transition dipole
    if len(t_dm1.shape) == 3:
        #calculations come from spin != 0, separate alpha and beta sectors and sum
        dipole_e_alpha = np.einsum('xij,ji->x', dip_ints, t_dm1[0,::])
        dipole_e_beta = np.einsum('xij,ji->x', dip_ints, t_dm1[1,::])
        dipole_e = dipole_e_alpha + dipole_e_beta
    else:
        dipole_e = np.einsum('xij,ji->x', dip_ints, t_dm1)
    # add on the nuclear dipole
    centered_coords = np.copy(coords)
    for num_atom in range(len(charges)):
        centered_coords[num_atom,:] -= nuc_mass_center
    dipole_n = np.einsum('z,zx->x', charges, centered_coords)
    # total dipole
    dipole = -dipole_e + dipole_n
    return dipole

def get_uhf_dipole(disp_hf):
    """
    Given an unrestricted Hartree-Fock object, evaluate the dipole moment 
    in the unrestricted Hartree-Fock state.
    """
    # dipole matrix elements
    charges = disp_hf.mol.atom_charges()
    coords = disp_hf.mol.atom_coords()
    masses = disp_hf.mol.atom_mass_list(isotope_avg=True)
    nuc_mass_center = np.einsum('z,zx->x', masses, coords)\
        / masses.sum() # if you switch masses -> charges, you get rid of nuclear dipole!
    disp_hf.mol.set_common_orig_(nuc_mass_center)

    # Obtain UHF density matrix in MO representation
    t_dm1_alpha, t_dm1_beta = disp_hf.make_rdm1()

    # Compute total electronic transition dipole for alpha and beta spins
    dip_ints = disp_hf.mol.intor('int1e_r', comp=3)
    dipole_e_alpha = np.einsum('xij,ji->x', dip_ints, t_dm1_alpha)
    dipole_e_beta = np.einsum('xij,ji->x', dip_ints, t_dm1_beta)
    dipole_e = dipole_e_alpha + dipole_e_beta

    # add on the nuclear dipole
    centered_coords = np.copy(coords)
    for num_atom in range(len(charges)):
        centered_coords[num_atom,:] -= nuc_mass_center
    dipole_n = np.einsum('z,zx->x', charges, centered_coords)
    # total dipole
    dipole = -dipole_e + dipole_n
    return dipole

def get_dipole(hf, method):
    if method == 'rhf':
        return get_rhf_dipole(hf)
    elif method == 'uhf':
        return get_uhf_dipole(hf)



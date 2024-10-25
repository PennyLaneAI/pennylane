import numpy as np
import pennylane as qml
import pyscf
from pyscf import scf
from pyscf.geomopt.geometric_solver import optimize
from utils import *
import h5py
import sys, os, subprocess
from localize_modes import localize_normal_modes
from time import time
from mpi4py import MPI 
from vibrational_class import *

#constants
au_to_cm = 219475
bohr_to_ang = 0.529177
orig_stdout = sys.stdout

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
    
def pes_onemode(molecule, scf_result, freqs_au, displ_vecs, gauss_grid, method="rhf", do_dipole=False):
    r"""Computes the one-mode potential energy surface on a grid in real space, along the normal coordinate directions (or any directions set by the displ_vecs).
    Simultaneously, can compute the dipole one-body elements."""

    freqs = freqs_au * au_to_cm
    quad_order = len(gauss_grid)
    nmodes = len(freqs)
    init_geom = scf_result.mol.atom

    pes_onebody = np.zeros((nmodes, quad_order), dtype=float)
    harmonic_pes = np.zeros_like(pes_onebody)

    if do_dipole:
        dipole_onebody = np.zeros((nmodes, quad_order, 3), dtype=float)
        ref_dipole = get_dipole(scf_result, method)

    local_pes_onebody = np.zeros((quad_order), dtype=float)
    local_harmonic_pes = np.zeros_like(local_pes_onebody)
    if do_dipole:
        local_dipole_onebody = np.zeros((quad_order, 3), dtype=float)
        ref_dipole = get_dipole(scf_result, method)
    
    for ii in tqdm(range(len(displ_vecs)), desc='Loop one-body pes'):
        displ_vec = displ_vecs[ii]
        # imaginary frequency check
        if (freqs[ii].imag) > 1e-6:
            continue

        jobs_on_rank = np.array_split(range(quad_order), size)[rank]

        for jj in jobs_on_rank:
            pt = gauss_grid[jj]
            # numerical scaling out front to shrink region
            scaling = np.sqrt( hbar / (2 * np.pi * freqs[ii] * 100 * c_light))
            positions = np.array([np.array(init_geom[ll][1]) + \
                                  scaling * pt * displ_vec[ll,:] \
                            for ll in range(len(molecule.symbols))])

            disp_mol = qml.qchem.Molecule(molecule.symbols,
                                          positions,
                                          basis_name=molecule.basis_name,
                                          charge=molecule.charge,
                                          mult=molecule.mult,
                                          unit="angstrom",
                                          load_data=True)
            
            disp_hf = single_point(disp_mol, method=method)
            
            omega = freqs_au[ii]
            ho_const = omega / 2
            local_harmonic_pes[jj] = ho_const * (pt**2)

            local_pes_onebody[jj] = disp_hf.e_tot - scf_result.e_tot  
            if do_dipole:
                local_dipole_onebody[jj,:] = get_dipole(disp_hf, method) - ref_dipole

        # gather the results on head process only
        pes_onebody[ii,:] = np.sum(np.array(comm.gather(local_pes_onebody)), axis=0)
        harmonic_pes[ii,:] = np.sum(np.array(comm.gather(local_harmonic_pes)), axis=0)
        if do_dipole:
            dipole_onebody[ii,:] = np.sum(np.array(comm.gather(local_dipole_onebody)), axis=0)

    # broadcast the result to everybody
    final_pes_onebody = np.array(comm.bcast(pes_onebody, root=0))
    if do_dipole:
        final_dipole_onebody = np.array(comm.bcast(dipole_onebody, root=0))

    comm.Barrier()
    if do_dipole:
        return final_pes_onebody, final_dipole_onebody
    else:
        return final_pes_onebody, None

def pes_twomode(molecule, scf_result, freqs_au, displ_vecs, gauss_grid, pes_onebody, dipole_onebody, method="rhf", do_dipole=False):
    r"""Computes the two-mode potential energy surface on a grid in real space,
    along the normal coordinate directions (or any directions set by the 
    displ_vecs)."""
    
    freqs = freqs_au * au_to_cm
    quad_order = len(gauss_grid)
    nmodes = len(freqs)
    init_geom = scf_result.mol.atom

    pes_twobody = np.zeros((nmodes, nmodes, quad_order, quad_order), dtype=float)
    local_pes_twobody = np.zeros((quad_order, quad_order), dtype=float)
    gridx, gridy = np.meshgrid(gauss_grid, gauss_grid)

    if do_dipole:
        dipole_twobody = np.zeros((nmodes, nmodes, quad_order, quad_order, 3), dtype=float)
        local_dipole_twobody = np.zeros((quad_order, quad_order, 3), dtype=float)
        ref_dipole = get_dipole(scf_result, method)

    for aa in tqdm(range(len(displ_vecs)), desc='Outer Loop two-body pes'):
        displ_vec_a = displ_vecs[aa]
        scaling_a = np.sqrt( hbar / (2 * np.pi * freqs[aa] * 100 * c_light))

        for bb in tqdm(range(len(displ_vecs)), desc='Inner Loop two-body pes'):
            # skip the pieces that are not part of the Hamiltonian
            if bb >= aa:
                continue

            displ_vec_b = displ_vecs[bb]
            # imaginary frequency check
            if (freqs[aa].imag) > 1e-6 or (freqs[bb].imag) > 1e-6:
                continue

            scaling_b = np.sqrt( hbar / (2 * np.pi * freqs[bb] * 100 * c_light))

            all_jobs = []
            for ii, pt1 in enumerate(gauss_grid):
                for jj, pt2 in enumerate(gauss_grid):
                        all_jobs.append([ii, pt1, jj, pt2])

            jobs_on_rank = np.array_split(all_jobs, size)[rank]

            for [ii, pt1, jj, pt2] in jobs_on_rank:

                ii, jj = int(ii), int(jj)
                positions = np.array([np.array(init_geom[ll][1]) + \
                                      scaling_a * pt1 * displ_vec_a[ll,:] + \
                                      scaling_b * pt2 * displ_vec_b[ll,:] \
                                      for ll in range(len(molecule.symbols))])
                disp_mol = qml.qchem.Molecule(molecule.symbols, positions, basis_name=molecule.basis_name, charge=molecule.charge, mult=molecule.mult, unit="angstrom", load_data=True)
                disp_hf = single_point(disp_mol, method=method)
                local_pes_twobody[ii, jj] = disp_hf.e_tot - pes_onebody[aa, ii] - pes_onebody[bb, jj] - scf_result.e_tot
                    
                if do_dipole:
                    local_dipole_twobody[ii,jj,:] = get_dipole(disp_hf, method) - dipole_onebody[aa,ii,:] -  dipole_onebody[bb,jj,:] - ref_dipole

            # gather the results on head process only
            pes_twobody[aa,bb,:,:] = np.sum(np.array(comm.gather(local_pes_twobody, root=0)), axis=0)
            if do_dipole:
                dipole_twobody[aa,bb,:,:] = np.sum(np.array(comm.gather(local_dipole_twobody, root=0)), axis=0)

    # broadcast the result to everybody
    final_pes_twobody = np.array(comm.bcast(pes_twobody, root=0))
    if do_dipole:
        final_dipole_twobody = np.array(comm.bcast(dipole_twobody, root=0))

    comm.Barrier()
    if do_dipole:
        return final_pes_twobody, final_dipole_twobody
    else:
        return final_pes_twobody, None


def _local_pes_threemode(molecule, scf_result, freqs_au, displ_vecs, gauss_grid, pes_onebody, pes_twobody, dipole_onebody, dipole_twobody, method="rhf", do_dipole=False):
    r"""
    Computes the three-mode potential energy surface on a grid in real space,
    along the normal coordinate directions (or any directions set by the 
    displ_vecs).
    """
    freqs = freqs_au * au_to_cm
    quad_order = len(gauss_grid)
    init_geom = scf_result.mol.atom
    nmodes = len(freqs)

    all_mode_combos = []
    for aa in range(len(displ_vecs)):
        for bb in range(len(displ_vecs)):
            for cc in range(len(displ_vecs)):
                all_mode_combos.append([aa, bb, cc])

    all_bos_combos = []
    for ii, pt1 in enumerate(gauss_grid):
        for jj, pt2 in enumerate(gauss_grid):
            for kk, pt3 in enumerate(gauss_grid):
                all_bos_combos.append([ii, pt1, jj, pt2, kk, pt3])

    boscombos_on_rank = np.array_split(all_bos_combos, size)[rank]
    local_pes_threebody = np.zeros(len(all_mode_combos)*len(boscombos_on_rank))
    if do_dipole:
        local_dipole_threebody = np.zeros((len(all_mode_combos)*\
                                        len(boscombos_on_rank), 3), dtype=float)
        ref_dipole = get_dipole(scf_result, method)

    ll = 0
    for [aa, bb, cc] in tqdm(all_mode_combos, desc = "Outer loop three-body pes"):

        aa, bb, cc = int(aa), int(bb), int(cc)
        # skip the ones that are not needed
        if bb >= aa or cc >= bb:
            ll += 1
            continue

        # imaginary frequency check
        if (freqs[aa].imag) > 1e-6 or (freqs[bb].imag) > 1e-6 or (freqs[cc].imag) > 1e-6:
            ll += 1
            continue

        displ_vec_a = displ_vecs[aa]
        scaling_a = np.sqrt( hbar / (2 * np.pi * freqs[aa] * 100 * c_light))
            
        displ_vec_b = displ_vecs[bb]
        scaling_b = np.sqrt( hbar / (2 * np.pi * freqs[bb] * 100 * c_light))

        displ_vec_c = displ_vecs[cc]
        scaling_c = np.sqrt( hbar / (2 * np.pi * freqs[cc] * 100 * c_light))

        mm = 0
        for [ii, pt1, jj, pt2, kk, pt3] in tqdm(boscombos_on_rank, desc="Inner loop three-body pes"):

            ii, jj, kk = int(ii), int(jj), int(kk)

            positions = np.array([ np.array(init_geom[ll][1]) + \
                            scaling_a * pt1 * displ_vec_a[ll,:] + \
                            scaling_b * pt2 * displ_vec_b[ll,:] + \
                            scaling_c * pt3 * displ_vec_c[ll,:]
                                   for ll in range(scf_result.mol.natm)])
            disp_mol = qml.qchem.Molecule(molecule.symbols, positions, basis_name=molecule.basis_name, charge=molecule.charge, mult=molecule.mult, unit="angstrom", load_data=True)
            disp_hf = single_point(disp_mol, method=method)

            ind = ll*len(boscombos_on_rank) + mm
            local_pes_threebody[ind] = disp_hf.e_tot - pes_twobody[aa,bb,ii,jj] - pes_twobody[aa,cc,ii,kk] -\
                pes_twobody[bb,cc,jj,kk] - pes_onebody[aa, ii] - pes_onebody[bb, jj] - pes_onebody[cc,kk] - scf_result.e_tot
            if do_dipole:
                local_dipole_threebody[ind,:] = get_dipole(disp_hf, method) - dipole_twobody[aa,bb,ii,jj,:] - dipole_twobody[aa,cc,ii,kk,:] - \
                    dipole_twobody[bb,cc,jj,kk,:] - dipole_onebody[aa,ii,:] -  dipole_onebody[bb,jj,:] - dipole_onebody[cc,kk,:] - ref_dipole
            mm += 1
            
        ll += 1

    comm.Barrier()
    if do_dipole:
        return local_pes_threebody, local_dipole_threebody
    else:
        return local_pes_threebody, None

def _load_pes_threemode(num_pieces, nmodes, ngridpoints):
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


def _load_dipole_threemode(num_pieces, nmodes, ngridpoints):
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
    
def pes_threemode(molecule, scf_result, freqs_au, displ_vecs, gauss_grid, pes_onebody, pes_twobody, dipole_onebody, dipole_twobody, method="rhf", do_dipole=False):
    r"""Function for calculating threebody PES."""
    
    local_pes_threebody, local_dipole_threebody = _local_pes_threemode(molecule, scf_result, freqs_au, displ_vecs, gauss_grid, pes_onebody, pes_twobody, dipole_onebody, dipole_twobody, method=method, do_dipole=do_dipole)
    comm.Barrier()

    f = h5py.File("v3data" + f"_{rank}" + '.hdf5', 'w')
    f.create_dataset('V3_PES',data=local_pes_threebody)
    if do_dipole:
        dipole_threebody = None
        f.create_dataset('D3_DMS',data=local_dipole_threebody)
    f.close()
    comm.Barrier()

    pes_threebody = None
    if rank==0:
        pes_threebody = _load_pes_threemode(comm.Get_size(), len(freqs_au), len(gauss_grid))
        if do_dipole:
            dipole_threebody = _load_dipole_threemode(comm.Get_size(), len(freqs_au), len(gauss_grid))
                
        process = subprocess.Popen('rm ' + 'v3data*', stdout=subprocess.PIPE, shell=True)
        output, error = process.communicate()
    comm.Barrier()
    pes_threebody = comm.bcast(pes_threebody, root=0)
    if do_dipole:
        dipole_threebody = comm.bcast(dipole_threebody, root=0)
        return pes_threebody, dipole_threebody
    else:
        return pes_threebody, None
    
    
def vibrational_pes(molecule, quad_order=9, method="rhf", localize=True, loc_freqs=[2600], do_cubic=True, get_anh_dipole=2):

    r"""Builds potential energy surfaces over the normal modes.

    Args:
       molecule: molecule object.
       quad_order: order for Gauss-Hermite quadratures
       method: electronic structure method. Default is restricted Hartree-Fock 'RHF'.
       localize: perform normal mode localization.
       loc_freqs: array of frequencies (in cm-1) where separation happens for mode localization.
       do_cubic: use True to include three-mode couplings
       get_anh_dipole: True for also obtaining anharmonic matrix elements for molecular dipole, takes considerable time. If integer then gets up to that degree of anharmonic dipole

    Returns:
       VibrationalPES object.
    """
    molecule, scf_result = optimize_geometry(molecule, method)
    
    harmonic_res = None
    loc_res = None
    uloc = None
    displ_vecs = None
    if rank == 0:
        harmonic_res = harmonic_analysis(scf_result, method)   
        displ_vecs = harmonic_res["norm_mode"]
        if localize:
            loc_res, uloc, displ_vecs = localize_normal_modes(harmonic_res, freq_separation=loc_freqs)

    # Broadcast data to all threads
    harmonic_res = comm.bcast(harmonic_res, root=0)
    displ_vecs = np.array(comm.bcast(displ_vecs, root=0))
    loc_res = comm.bcast(loc_res, root=0)
    uloc = np.array(comm.bcast(uloc, root=0))
            
    comm.Barrier()

    freqs_au = loc_res['freq_wavenumber'] / au_to_cm    
    gauss_grid, gauss_weights = np.polynomial.hermite.hermgauss(quad_order)

    pes_onebody, dipole_onebody = pes_onemode(molecule, scf_result, freqs_au, displ_vecs, gauss_grid, method=method, do_dipole=True)
    comm.Barrier()

    # build PES -- two-body
    if get_anh_dipole < 2 or get_anh_dipole is False:
        do_dip_2 = False
    elif get_anh_dipole > 1 or get_anh_dipole is True:
        do_dip_2 = True

    pes_twobody, dipole_twobody = pes_twomode(molecule, scf_result, freqs_au, displ_vecs, gauss_grid, pes_onebody, dipole_onebody, method=method, do_dipole=do_dip_2)
    comm.Barrier()

    pes_arr = [pes_onebody, pes_twobody]
    dipole_arr = [dipole_onebody, dipole_twobody]
    
    if do_cubic:
        if get_anh_dipole < 3 or get_anh_dipole is False:
            do_dip_3 = False
        elif get_anh_dipole > 2 or get_anh_dipole is True:
            do_dip_3 = True

        pes_threebody, dipole_threebody = pes_threemode(molecule, scf_result, freqs_au, displ_vecs, gauss_grid, pes_onebody, pes_twobody, dipole_onebody, dipole_twobody, method=method, do_dipole=do_dip_3)
        comm.Barrier()
        pes_arr = [pes_onebody, pes_twobody, pes_threebody]
        dipole_arr = [dipole_onebody, dipole_twobody, dipole_threebody]

    return VibrationalPES(freqs_au, gauss_grid, gauss_weights, uloc, pes_arr, dipole_arr, localize, get_anh_dipole)

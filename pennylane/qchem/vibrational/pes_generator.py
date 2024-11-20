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
"""This module contains functions to calculate potential energy surfaces
per normal modes on a grid."""

import subprocess

import h5py
import numpy as np
from mpi4py import MPI

import pennylane as qml

from .vibrational_class import single_point, get_dipole

#constants
HBAR = 6.022*1.055e12 # (amu)*(angstrom^2/s)
C_LIGHT = 3*10**8 # m/s
AU_TO_CM = 219475
BOHR_TO_ANG = 0.529177

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def pes_onemode(molecule, scf_result, freqs_au, displ_vecs, gauss_grid, method="rhf", do_dipole=False):
    r"""Computes the one-mode potential energy surface on a grid in real space, along the normal coordinate directions (or any directions set by the displ_vecs).
    Simultaneously, can compute the dipole one-body elements."""

    local_pes_onebody, local_dipole_onebody = _local_pes_onemode(molecule, scf_result, freqs_au, displ_vecs, gauss_grid, method=method, do_dipole=do_dipole)

    f = h5py.File("v1data" + f"_{rank}" + '.hdf5', 'w')
    f.create_dataset('V1_PES',data=local_pes_onebody)
    if do_dipole:
        f.create_dataset('D1_DMS',data=local_dipole_onebody)
    f.close()
    comm.Barrier()
    pes_onebody = None
    dipole_onebody = None
    if rank == 0:
        pes_onebody, dipole_onebody = _load_pes_onemode(comm.Get_size(),len(freqs_au), len(gauss_grid), do_dipole=do_dipole)
        subprocess.run(['rm', 'v1data*'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True, check=True)

    comm.Barrier()
    pes_onebody = comm.bcast(pes_onebody, root=0)

    if do_dipole:
        dipole_onebody = comm.bcast(dipole_onebody, root=0)
        return pes_onebody, dipole_onebody

    return pes_onebody, None

def _local_pes_onemode(molecule, scf_result, freqs_au, displ_vecs, gauss_grid, method="rhf", do_dipole=False):
    r"""Computes the one-mode potential energy surface on a grid in real space, along the normal coordinate directions (or any directions set by the displ_vecs).
    Simultaneously, can compute the dipole one-body elements."""

    freqs = freqs_au * AU_TO_CM
    quad_order = len(gauss_grid)
    nmodes = len(freqs)
    init_geom = scf_result.mol.atom

    jobs_on_rank = np.array_split(range(quad_order), size)[rank]
    local_pes_onebody = np.zeros((nmodes, len(jobs_on_rank)), dtype=float)
    local_harmonic_pes = np.zeros_like(local_pes_onebody)
    if do_dipole:
        local_dipole_onebody = np.zeros((nmodes, len(jobs_on_rank), 3), dtype=float)
        ref_dipole = get_dipole(scf_result, method)
    for ii in range(nmodes):
        displ_vec = displ_vecs[ii]
        if (freqs[ii].imag) > 1e-6:
            continue

        idx = 0
        for jj in jobs_on_rank:
            pt = gauss_grid[jj]
            # numerical scaling out front to shrink region
            scaling = np.sqrt( HBAR / (2 * np.pi * freqs[ii] * 100 * C_LIGHT))
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

            local_harmonic_pes[ii][idx] = ho_const * (pt**2)

            local_pes_onebody[ii][idx] = disp_hf.e_tot - scf_result.e_tot
            if do_dipole:
                local_dipole_onebody[ii,idx,:] = get_dipole(disp_hf, method) - ref_dipole
            idx += 1

    if do_dipole:
        return local_pes_onebody, local_dipole_onebody
    return local_pes_onebody, None


def _load_pes_onemode(num_pieces, nmodes, quad_order, do_dipole=False):
    """
    Loader to combine pes_onebody from multiple ranks.
    """
    
    pes_onebody = np.zeros((nmodes, quad_order), dtype=float)

    if do_dipole:
        dipole_onebody = np.zeros((nmodes, quad_order, 3), dtype=float)

    for ii in range(nmodes):
        init_chunk = 0
        for piece in range(num_pieces):
            f = h5py.File("v1data" + f"_{piece}" + '.hdf5', 'r+')
            local_pes_onebody = f['V1_PES'][()]
            local_dipole_onebody = f['D1_DMS'][()]

            end_chunk = local_pes_onebody.shape[1]
            pes_onebody[ii][init_chunk:init_chunk+end_chunk] = local_pes_onebody[ii]
            if do_dipole:
                dipole_onebody[ii][init_chunk:init_chunk+end_chunk] = local_dipole_onebody[ii]
            init_chunk += end_chunk

    if do_dipole:
        return pes_onebody, dipole_onebody
    return pes_onebody, None

def pes_twomode(molecule, scf_result, freqs_au, displ_vecs, gauss_grid, pes_onebody, dipole_onebody, method="rhf", do_dipole=False):
    r"""Computes the two-mode potential energy surface on a grid in real space, along the normal coordinate directions (or any directions set by the displ_vecs).
    Simultaneously, can compute the dipole one-body elements."""

    local_pes_twobody, local_dipole_twobody = _local_pes_twomode(molecule, scf_result, freqs_au, displ_vecs, gauss_grid, pes_onebody, dipole_onebody, method=method, do_dipole=do_dipole)

    f = h5py.File("v2data" + f"_{rank}" + '.hdf5', 'w')
    f.create_dataset('V2_PES',data=local_pes_twobody)
    if do_dipole:
        f.create_dataset('D2_DMS',data=local_dipole_twobody)
    f.close()
    comm.Barrier()
    pes_twobody = None
    dipole_twobody = None
    if rank == 0:
        pes_twobody, dipole_twobody = _load_pes_twomode(comm.Get_size(),len(freqs_au), len(gauss_grid), do_dipole=do_dipole)
        subprocess.run(['rm', 'v2data*'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True, check=True)

    comm.Barrier()
    pes_twobody = comm.bcast(pes_twobody, root=0)

    if do_dipole:
        dipole_twobody = comm.bcast(dipole_twobody, root=0)
        return pes_twobody, dipole_twobody
    return pes_twobody, None


def _local_pes_twomode(molecule, scf_result, freqs_au, displ_vecs, gauss_grid, pes_onebody, dipole_onebody, method="rhf", do_dipole=False):
    r"""Computes the two-mode potential energy surface on a grid in real space, along the normal coordinate directions (or any directions set by the displ_vecs).
    Simultaneously, can compute the dipole one-body elements."""

    freqs = freqs_au * AU_TO_CM
    init_geom = scf_result.mol.atom

    all_mode_combos = []
    for aa in range(len(displ_vecs)):
        for bb in range(len(displ_vecs)):
            all_mode_combos.append([aa, bb])

    all_jobs = []
    for ii, pt1 in enumerate(gauss_grid):
        for jj, pt2 in enumerate(gauss_grid):
            all_jobs.append([ii, pt1, jj, pt2])

    jobs_on_rank = np.array_split(all_jobs, size)[rank]
    local_pes_twobody = np.zeros((len(all_mode_combos)*len(jobs_on_rank)))

    if do_dipole:
        local_dipole_twobody = np.zeros((len(all_mode_combos)*\
                                        len(jobs_on_rank), 3), dtype=float)
        ref_dipole = get_dipole(scf_result, method)

    ll = 0
    for [aa, bb] in all_mode_combos:
        aa, bb = int(aa), int(bb)

        displ_vec_a = displ_vecs[aa]
        scaling_a = np.sqrt( HBAR / (2 * np.pi * freqs[aa] * 100 * C_LIGHT))

        if bb >= aa:
            ll += 1
            continue

        displ_vec_b = displ_vecs[bb]

        if (freqs[aa].imag) > 1e-6 or (freqs[bb].imag) > 1e-6:
            ll += 1
            continue

        scaling_b = np.sqrt( HBAR / (2 * np.pi * freqs[bb] * 100 * C_LIGHT))
        mm = 0
        for [ii, pt1, jj, pt2] in jobs_on_rank:

            ii, jj = int(ii), int(jj)
            positions = np.array([np.array(init_geom[ll][1]) + \
                                  scaling_a * pt1 * displ_vec_a[ll,:] + \
                                  scaling_b * pt2 * displ_vec_b[ll,:] \
                                  for ll in range(len(molecule.symbols))])
            disp_mol = qml.qchem.Molecule(molecule.symbols, positions, basis_name=molecule.basis_name, charge=molecule.charge, mult=molecule.mult, unit="angstrom", load_data=True)
            disp_hf = single_point(disp_mol, method=method)
            idx = ll*len(jobs_on_rank) + mm
            local_pes_twobody[idx] = disp_hf.e_tot - pes_onebody[aa, ii] - pes_onebody[bb, jj] - scf_result.e_tot

            if do_dipole:
                local_dipole_twobody[idx,:] = get_dipole(disp_hf, method) - dipole_onebody[aa,ii,:] -  dipole_onebody[bb,jj,:] - ref_dipole
            mm += 1
        ll+=1

    if do_dipole:
        return local_pes_twobody, local_dipole_twobody

    return local_pes_twobody, None


def _load_pes_twomode(num_pieces, nmodes, quad_order, do_dipole=False):
    """
    Loader to combine pes_twomode from multiple ranks.
    """

    final_shape = (nmodes, nmodes, quad_order, quad_order)
    nmode_combos = nmodes**2
    pes_twobody = np.zeros(np.prod(final_shape))
    if do_dipole:
        dipole_twobody = np.zeros((np.prod(final_shape), 3))

    r0 = 0
    r1 = 0

    for mode_combo in range(nmode_combos):
        local_pes = np.zeros(quad_order**2)
        local_dipole = np.zeros((quad_order**2, 3))
        l0 = 0
        l1 = 0
        for piece in range(num_pieces):
            f = h5py.File("v2data" + f"_{piece}" + '.hdf5', 'r+')
            local_pes_twobody = f['V2_PES'][()]
            local_dipole_twobody = f['D2_DMS'][()]
            pes_chunk = np.array_split(local_pes_twobody, nmode_combos)[mode_combo]
            dipole_chunk = np.array_split(local_dipole_twobody, nmode_combos, \
                                                        axis=0)[mode_combo]
            l1 += len(pes_chunk)
            local_pes[l0:l1] = pes_chunk
            local_dipole[l0:l1,:] = dipole_chunk
            l0 += len(pes_chunk)

        r1 += len(local_pes)
        pes_twobody[r0:r1] = local_pes
        dipole_twobody[r0:r1,:] = local_dipole
        r0 += len(local_pes)

    pes_twobody = pes_twobody.reshape(final_shape)
    dipole_twobody = dipole_twobody.reshape(final_shape+(3,))
    if do_dipole:
        return pes_twobody, dipole_twobody
    return pes_twobody, None

def _local_pes_threemode(molecule, scf_result, freqs_au, displ_vecs, gauss_grid, pes_onebody, pes_twobody, dipole_onebody, dipole_twobody, method="rhf", do_dipole=False):
    r"""
    Computes the three-mode potential energy surface on a grid in real space,
    along the normal coordinate directions (or any directions set by the 
    displ_vecs).
    """
    freqs = freqs_au * AU_TO_CM
    init_geom = scf_result.mol.atom

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
    for [aa, bb, cc] in all_mode_combos:

        aa, bb, cc = int(aa), int(bb), int(cc)
        if bb >= aa or cc >= bb:
            ll += 1
            continue

        if (freqs[aa].imag) > 1e-6 or (freqs[bb].imag) > 1e-6 or (freqs[cc].imag) > 1e-6:
            ll += 1
            continue

        displ_vec_a = displ_vecs[aa]
        scaling_a = np.sqrt( HBAR / (2 * np.pi * freqs[aa] * 100 * C_LIGHT))

        displ_vec_b = displ_vecs[bb]
        scaling_b = np.sqrt( HBAR / (2 * np.pi * freqs[bb] * 100 * C_LIGHT))

        displ_vec_c = displ_vecs[cc]
        scaling_c = np.sqrt( HBAR / (2 * np.pi * freqs[cc] * 100 * C_LIGHT))

        mm = 0
        for [ii, pt1, jj, pt2, kk, pt3] in boscombos_on_rank:

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

    return local_pes_threebody, None

def _load_pes_threemode(num_pieces, nmodes, ngridpoints):
    """
    Loader to combine pes_threemode from multiple ranks.
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
        for piece in range(num_pieces):
            f = h5py.File("v3data" + f"_{piece}" + '.hdf5', 'r+')
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
    Loader to combine dipole_threemode from multiple ranks.
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
        for piece in range(num_pieces):
            f = h5py.File("v3data" + f"_{piece}" + '.hdf5', 'r+')
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
    r"""Computes the three-mode potential energy surface on a grid in real space, along the normal coordinate directions (or any directions set by the displ_vecs).
    Simultaneously, can compute the dipole one-body elements."""

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

        subprocess.run(['rm', 'v3data*'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True, check=True)

    comm.Barrier()
    pes_threebody = comm.bcast(pes_threebody, root=0)
    if do_dipole:
        dipole_threebody = comm.bcast(dipole_threebody, root=0)
        return pes_threebody, dipole_threebody

    return pes_threebody, None

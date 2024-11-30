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

import itertools
import subprocess

import numpy as np

import pennylane as qml
from pennylane.data.base._lazy_modules import h5py

from .vibrational_class import _single_point, get_dipole

# pylint: disable=too-many-arguments, too-many-function-args, c-extension-no-member
# pylint: disable= import-outside-toplevel, too-many-positional-arguments

# constants
HBAR = 6.022 * 1.055e12  # (amu)*(angstrom^2/s)
C_LIGHT = 3 * 10**8  # m/s
AU_TO_CM = 219475
BOHR_TO_ANG = 0.5291772106


def _import_mpi4py():
    """Import mpi4py."""
    try:
        import mpi4py
    except ImportError as Error:
        raise ImportError(
            "This feature requires mpi4py. It can be installed with: pip install mpi4py."
        ) from Error

    return mpi4py


def pes_onemode(
    molecule, scf_result, freqs_au, displ_vecs, gauss_grid, method="rhf", do_dipole=False
):
    r"""Computes the one-mode potential energy surface on a grid along directions defined by displacement vectors.

    Args:
       molecule: Molecule object
       scf_result: pyscf object from electronic structure calculations
       freqs: list of vibrational frequencies in atomic units
       vectors: list of displacement vectors for each normal mode
       gauss_grid: sample points for Gauss-Hermite quadrature
       method: Electronic structure method to define the level of theory
            for harmonic analysis. Default is restricted Hartree-Fock 'rhf'.
       do_dipole: Whether to calculate the dipole elements. Default is ``False``.

    Returns:
       A tuple of one-mode potential energy surface and one-mode dipole along
       the normal-mode coordinates.

    """

    _import_mpi4py()
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    local_pes_onebody, local_dipole_onebody = _local_pes_onemode(
        comm,
        molecule,
        scf_result,
        freqs_au,
        displ_vecs,
        gauss_grid,
        method=method,
        do_dipole=do_dipole,
    )

    filename = f"v1data_{rank}.hdf5"
    with h5py.File(filename, "w") as f:
        f.create_dataset("V1_PES", data=local_pes_onebody)
        if do_dipole:
            f.create_dataset("D1_DMS", data=local_dipole_onebody)
        f.close()

    comm.Barrier()
    pes_onebody = None
    dipole_onebody = None
    if rank == 0:
        pes_onebody, dipole_onebody = _load_pes_onemode(
            comm.Get_size(), len(freqs_au), len(gauss_grid), do_dipole=do_dipole
        )
        subprocess.run(
            ["rm", "v1data*"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True,
            check=False,
        )

    comm.Barrier()
    pes_onebody = comm.bcast(pes_onebody, root=0)

    if do_dipole:
        dipole_onebody = comm.bcast(dipole_onebody, root=0)
        return pes_onebody, dipole_onebody

    return pes_onebody, None


def _local_pes_onemode(
    comm, molecule, scf_result, freqs_au, displ_vecs, gauss_grid, method="rhf", do_dipole=False
):
    r"""Computes the one-mode potential energy surface on a grid in real space, along thedirections set by the displ_vecs for each processor.
    Simultaneously, can compute the dipole one-body elements.

    Args:
       molecule: Molecule object
       scf_result: pyscf object from electronic structure calculations
       freqs_au: list of normal mode frequencies
       displ_vecs: list of displacement vectors for each normal mode
       gauss_grid: sample points for Gauss-Hermite quadrature
       method: Electronic structure method to define the level of theory
            for harmonic analysis. Default is restricted Hartree-Fock 'rhf'.
       do_dipole: Whether to calculate the dipole elements. Default is ``False``.

    Returns:
       A tuple of one-mode potential energy surface and one-mode dipole along
       the normal-mode coordinates.

    """

    size = comm.Get_size()
    rank = comm.Get_rank()
    freqs = freqs_au * AU_TO_CM
    quad_order = len(gauss_grid)
    nmodes = len(freqs)
    init_geom = molecule.coordinates * BOHR_TO_ANG

    jobs_on_rank = np.array_split(range(quad_order), size)[rank]
    local_pes_onebody = np.zeros((nmodes, len(jobs_on_rank)), dtype=float)
    local_harmonic_pes = np.zeros_like(local_pes_onebody)
    if do_dipole:
        local_dipole_onebody = np.zeros((nmodes, len(jobs_on_rank), 3), dtype=float)
        ref_dipole = get_dipole(scf_result, method)
    for mode in range(nmodes):
        displ_vec = displ_vecs[mode]
        if (freqs[mode].imag) > 1e-6:
            continue  # pragma: no cover

        job_idx = 0
        for job in jobs_on_rank:
            pt = gauss_grid[job]
            # numerical scaling out front to shrink region
            scaling = np.sqrt(HBAR / (2 * np.pi * freqs[mode] * 100 * C_LIGHT))
            positions = np.array(init_geom + scaling * pt * displ_vec)

            displ_mol = qml.qchem.Molecule(
                molecule.symbols,
                positions,
                basis_name=molecule.basis_name,
                charge=molecule.charge,
                mult=molecule.mult,
                unit="angstrom",
                load_data=True,
            )

            displ_scf = _single_point(displ_mol, method=method)

            omega = freqs_au[mode]
            ho_const = omega / 2

            local_harmonic_pes[mode][job_idx] = ho_const * (pt**2)

            local_pes_onebody[mode][job_idx] = displ_scf.e_tot - scf_result.e_tot
            if do_dipole:
                local_dipole_onebody[mode, job_idx, :] = get_dipole(displ_scf, method) - ref_dipole
            job_idx += 1

    if do_dipole:
        return local_pes_onebody, local_dipole_onebody
    return local_pes_onebody, None


def _load_pes_onemode(num_proc, nmodes, quad_order, do_dipole=False):
    """
    Loader to combine pes_onebody and dipole_onebody from multiple processors.

    Args:
       num_proc: number of processors
       nmodes: number of normal modes
       quad_order: order for Gauss-Hermite quadratures
       do_dipole: Whether to calculate the dipole elements. Default is ``False``.

    Returns:
       A tuple of one-mode potential energy surface and one-mode dipole along
       the normal-mode coordinates.

    """
    pes_onebody = np.zeros((nmodes, quad_order), dtype=float)

    if do_dipole:
        dipole_onebody = np.zeros((nmodes, quad_order, 3), dtype=float)

    for mode in range(nmodes):
        init_chunk = 0
        for proc in range(num_proc):
            f = h5py.File("v1data" + f"_{proc}" + ".hdf5", "r+")
            local_pes_onebody = f["V1_PES"][()]
            if do_dipole:
                local_dipole_onebody = f["D1_DMS"][()]

            end_chunk = np.array(local_pes_onebody).shape[1]
            pes_onebody[mode][init_chunk : init_chunk + end_chunk] = local_pes_onebody[mode]
            if do_dipole:
                dipole_onebody[mode][init_chunk : init_chunk + end_chunk] = local_dipole_onebody[
                    mode
                ]
            init_chunk += end_chunk

    if do_dipole:
        return pes_onebody, dipole_onebody
    return pes_onebody, None


def pes_twomode(
    molecule,
    scf_result,
    freqs_au,
    displ_vecs,
    gauss_grid,
    pes_onebody,
    dipole_onebody,
    method="rhf",
    do_dipole=False,
):
    r"""Computes the two-mode potential energy surface on a grid in real space, along the directions set by the displ_vecs.
    Simultaneously, can compute the dipole two-mode elements.

    Args:
       molecule: Molecule object.
       scf_result: pyscf object from electronic structure calculations
       freqs_au: list of normal mode frequencies
       displ_vecs: list of displacement vectors for each normal mode
       gauss_grid: sample points for Gauss-Hermite quadrature
       pes_onebody: one-mode PES
       dipole_onebody: one-mode dipole
       method: Electronic structure method to define the level of theory
            for harmonic analysis. Default is restricted Hartree-Fock 'rhf'.
       do_dipole: Whether to calculate the dipole elements. Default is ``False``.

    Returns:
       A tuple of two-mode potential energy surface and two-mode dipole along
       the normal-mode coordinates.

    """

    _import_mpi4py()
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    local_pes_twobody, local_dipole_twobody = _local_pes_twomode(
        comm,
        molecule,
        scf_result,
        freqs_au,
        displ_vecs,
        gauss_grid,
        pes_onebody,
        dipole_onebody,
        method=method,
        do_dipole=do_dipole,
    )

    filename = f"v2data_{rank}.hdf5"
    with h5py.File(filename, "w") as f:
        f.create_dataset("V2_PES", data=local_pes_twobody)
        if do_dipole:
            f.create_dataset("D2_DMS", data=local_dipole_twobody)

    comm.Barrier()
    pes_twobody = None
    dipole_twobody = None
    if rank == 0:
        pes_twobody, dipole_twobody = _load_pes_twomode(
            comm.Get_size(), len(freqs_au), len(gauss_grid), do_dipole=do_dipole
        )
        subprocess.run(
            ["rm", "v2data*"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True,
            check=False,
        )

    comm.Barrier()
    pes_twobody = comm.bcast(pes_twobody, root=0)

    if do_dipole:
        dipole_twobody = comm.bcast(dipole_twobody, root=0)
        return pes_twobody, dipole_twobody
    return pes_twobody, None  # pragma: no cover


def _local_pes_twomode(
    comm,
    molecule,
    scf_result,
    freqs_au,
    displ_vecs,
    gauss_grid,
    pes_onebody,
    dipole_onebody,
    method="rhf",
    do_dipole=False,
):
    r"""Computes the two-mode potential energy surface on a grid in real space, along the directions set by the displ_vecs) for each processor.
    Simultaneously, can compute the dipole two-mode elements."""

    size = comm.Get_size()
    rank = comm.Get_rank()
    freqs = freqs_au * AU_TO_CM
    init_geom = molecule.coordinates * BOHR_TO_ANG
    nmodes = len(freqs)

    all_mode_combos = [(mode_a, mode_b) for mode_a in range(nmodes) for mode_b in range(mode_a)]

    all_jobs = [
        [i, pt1, j, pt2]
        for (i, pt1), (j, pt2) in itertools.product(enumerate(gauss_grid), repeat=2)
    ]
    jobs_on_rank = np.array_split(all_jobs, size)[rank]
    local_pes_twobody = np.zeros((len(all_mode_combos) * len(jobs_on_rank)))

    if do_dipole:
        local_dipole_twobody = np.zeros((len(all_mode_combos) * len(jobs_on_rank), 3), dtype=float)
        ref_dipole = get_dipole(scf_result, method)

    for mode_idx, [mode_a, mode_b] in enumerate(all_mode_combos):

        if (freqs[mode_a].imag) > 1e-6 or (freqs[mode_b].imag) > 1e-6:
            continue  # pragma: no cover

        displ_vec_a = displ_vecs[mode_a]
        scaling_a = np.sqrt(HBAR / (2 * np.pi * freqs[mode_a] * 100 * C_LIGHT))

        displ_vec_b = displ_vecs[mode_b]

        scaling_b = np.sqrt(HBAR / (2 * np.pi * freqs[mode_b] * 100 * C_LIGHT))
        for job_idx, [i, pt1, j, pt2] in enumerate(jobs_on_rank):

            i, j = int(i), int(j)
            positions = np.array(
                init_geom + scaling_a * pt1 * displ_vec_a + scaling_b * pt2 * displ_vec_b
            )
            displ_mol = qml.qchem.Molecule(
                molecule.symbols,
                positions,
                basis_name=molecule.basis_name,
                charge=molecule.charge,
                mult=molecule.mult,
                unit="angstrom",
                load_data=True,
            )
            displ_scf = _single_point(displ_mol, method=method)
            idx = mode_idx * len(jobs_on_rank) + job_idx

            local_pes_twobody[idx] = (
                displ_scf.e_tot - pes_onebody[mode_a, i] - pes_onebody[mode_b, j] - scf_result.e_tot
            )

            if do_dipole:
                local_dipole_twobody[idx, :] = (
                    get_dipole(displ_scf, method)
                    - dipole_onebody[mode_a, i, :]
                    - dipole_onebody[mode_b, j, :]
                    - ref_dipole
                )

    if do_dipole:
        return local_pes_twobody, local_dipole_twobody

    return local_pes_twobody, None  # pragma: no cover


def _load_pes_twomode(num_proc, nmodes, quad_order, do_dipole=False):
    """
    Loader to combine pes_twobody and dipole_twobody from multiple processors.

    Args:
       num_proc: number of processors
       nmodes: number of normal modes
       quad_order: order for Gauss-Hermite quadratures
       do_dipole: Whether to calculate the dipole elements. Default is ``False``.

    Returns:
       A tuple of one-mode potential energy surface and one-mode dipole along
       the normal-mode coordinates.

    """

    final_shape = (nmodes, nmodes, quad_order, quad_order)
    nmode_combos = int(nmodes * (nmodes - 1) / 2)
    pes_twobody = np.zeros((final_shape))
    if do_dipole:
        dipole_twobody = np.zeros((final_shape + (3,)))

    mode_combo = 0
    for mode_a in range(nmodes):
        for mode_b in range(mode_a):
            local_pes = np.zeros(quad_order**2)
            if do_dipole:
                local_dipole = np.zeros((quad_order**2, 3))

            init_idx = 0
            end_idx = 0
            for proc in range(num_proc):
                f = h5py.File("v2data" + f"_{proc}" + ".hdf5", "r+")
                local_pes_twobody = f["V2_PES"][()]
                pes_chunk = np.array_split(local_pes_twobody, nmode_combos)[mode_combo]

                end_idx += len(pes_chunk)
                local_pes[init_idx:end_idx] = pes_chunk
                if do_dipole:
                    local_dipole_twobody = f["D2_DMS"][()]
                    dipole_chunk = np.array_split(local_dipole_twobody, nmode_combos, axis=0)[
                        mode_combo
                    ]
                    local_dipole[init_idx:end_idx, :] = dipole_chunk
                init_idx += len(pes_chunk)

            pes_twobody[mode_a, mode_b, :, :] = local_pes.reshape(quad_order, quad_order)
            if do_dipole:
                dipole_twobody[mode_a, mode_b, :, :, :] = local_dipole.reshape(
                    quad_order, quad_order, 3
                )
            mode_combo += 1

    if do_dipole:
        return pes_twobody, dipole_twobody
    return pes_twobody, None  # pragma: no cover


def _local_pes_threemode(
    comm,
    molecule,
    scf_result,
    freqs_au,
    displ_vecs,
    gauss_grid,
    pes_onebody,
    pes_twobody,
    dipole_onebody,
    dipole_twobody,
    method="rhf",
    do_dipole=False,
):
    r"""
    Computes the three-mode potential energy surface on a grid in real space, along the directions set by the displ_vecs for each processor.
    """
    size = comm.Get_size()
    rank = comm.Get_rank()

    freqs = freqs_au * AU_TO_CM
    init_geom = molecule.coordinates * BOHR_TO_ANG
    nmodes = len(freqs)

    all_mode_combos = [
        (mode_a, mode_b, mode_c)
        for mode_a in range(nmodes)
        for mode_b in range(mode_a)
        for mode_c in range(mode_b)
    ]

    all_jobs = [
        [i, pt1, j, pt2, k, pt3]
        for (i, pt1), (j, pt2), (k, pt3) in itertools.product(enumerate(gauss_grid), repeat=3)
    ]

    jobs_on_rank = np.array_split(all_jobs, size)[rank]
    local_pes_threebody = np.zeros(len(all_mode_combos) * len(jobs_on_rank))

    if do_dipole:
        local_dipole_threebody = np.zeros(
            (len(all_mode_combos) * len(jobs_on_rank), 3), dtype=float
        )
        ref_dipole = get_dipole(scf_result, method)

    mode_combo = 0
    for mode_combo, [mode_a, mode_b, mode_c] in enumerate(all_mode_combos):

        if (
            (freqs[mode_a].imag) > 1e-6
            or (freqs[mode_b].imag) > 1e-6
            or (freqs[mode_c].imag) > 1e-6
        ):
            continue  # pragma: no cover

        displ_vec_a = displ_vecs[mode_a]
        scaling_a = np.sqrt(HBAR / (2 * np.pi * freqs[mode_a] * 100 * C_LIGHT))

        displ_vec_b = displ_vecs[mode_b]
        scaling_b = np.sqrt(HBAR / (2 * np.pi * freqs[mode_b] * 100 * C_LIGHT))

        displ_vec_c = displ_vecs[mode_c]
        scaling_c = np.sqrt(HBAR / (2 * np.pi * freqs[mode_c] * 100 * C_LIGHT))

        for job_idx, [i, pt1, j, pt2, k, pt3] in enumerate(jobs_on_rank):

            i, j, k = int(i), int(j), int(k)
            positions = np.array(
                init_geom
                + scaling_a * pt1 * displ_vec_a
                + scaling_b * pt2 * displ_vec_b
                + scaling_c * pt3 * displ_vec_c
            )

            displ_mol = qml.qchem.Molecule(
                molecule.symbols,
                positions,
                basis_name=molecule.basis_name,
                charge=molecule.charge,
                mult=molecule.mult,
                unit="angstrom",
                load_data=True,
            )
            displ_scf = _single_point(displ_mol, method=method)

            idx = mode_combo * len(jobs_on_rank) + job_idx
            local_pes_threebody[idx] = (
                displ_scf.e_tot
                - pes_twobody[mode_a, mode_b, i, j]
                - pes_twobody[mode_a, mode_c, i, k]
                - pes_twobody[mode_b, mode_c, j, k]
                - pes_onebody[mode_a, i]
                - pes_onebody[mode_b, j]
                - pes_onebody[mode_c, k]
                - scf_result.e_tot
            )
            if do_dipole:
                local_dipole_threebody[idx, :] = (
                    get_dipole(displ_scf, method)
                    - dipole_twobody[mode_a, mode_b, i, j, :]
                    - dipole_twobody[mode_a, mode_c, i, k, :]
                    - dipole_twobody[mode_b, mode_c, j, k, :]
                    - dipole_onebody[mode_a, i, :]
                    - dipole_onebody[mode_b, j, :]
                    - dipole_onebody[mode_c, k, :]
                    - ref_dipole
                )

    comm.Barrier()
    if do_dipole:
        return local_pes_threebody, local_dipole_threebody

    return local_pes_threebody, None  # pragma: no cover


def _load_pes_threemode(num_proc, nmodes, quad_order, do_dipole):
    """
    Loader to combine pes_threebody and dipole_threebody from multiple processors.

    Args:
       num_proc: number of processors
       nmodes: number of normal modes
       quad_order: order for Gauss-Hermite quadratures
       do_dipole: Whether to calculate the dipole elements. Default is ``False``.

    Returns:
       A tuple of one-mode potential energy surface and one-mode dipole along
       the normal-mode coordinates.

    """
    final_shape = (nmodes, nmodes, nmodes, quad_order, quad_order, quad_order)
    nmode_combos = int(nmodes * (nmodes - 1) * (nmodes - 2) / 6)
    pes_threebody = np.zeros(final_shape)
    dipole_threebody = np.zeros((final_shape + (3,))) if do_dipole else None

    mode_combo = 0
    for mode_a in range(nmodes):
        for mode_b in range(mode_a):
            for mode_c in range(mode_b):
                local_pes = np.zeros(quad_order**3)
                local_dipole = np.zeros((quad_order**3, 3)) if do_dipole else None

                init_idx = 0
                end_idx = 0
                for proc in range(num_proc):
                    f = h5py.File("v3data" + f"_{proc}" + ".hdf5", "r+")
                    local_pes_threebody = f["V3_PES"][()]
                    pes_chunk = np.array_split(local_pes_threebody, nmode_combos)[mode_combo]

                    end_idx += len(pes_chunk)
                    local_pes[init_idx:end_idx] = pes_chunk
                    if local_dipole is not None:
                        local_dipole_threebody = f["D3_DMS"][()]
                        dipole_chunk = np.array_split(local_dipole_threebody, nmode_combos, axis=0)[
                            mode_combo
                        ]
                        local_dipole[init_idx:end_idx, :] = dipole_chunk
                    init_idx += len(pes_chunk)

                pes_threebody[mode_a, mode_b, mode_c, :, :] = local_pes.reshape(
                    quad_order, quad_order, quad_order
                )
                if dipole_threebody is not None:
                    dipole_threebody[mode_a, mode_b, mode_c, :, :, :, :] = local_dipole.reshape(
                        quad_order, quad_order, quad_order, 3
                    )
                mode_combo += 1

    if do_dipole:
        return pes_threebody, dipole_threebody
    return pes_threebody, None  # pragma: no cover


def pes_threemode(
    molecule,
    scf_result,
    freqs_au,
    displ_vecs,
    gauss_grid,
    pes_onebody,
    pes_twobody,
    dipole_onebody,
    dipole_twobody,
    method="rhf",
    do_dipole=False,
):
    r"""Computes the three-mode potential energy surface on a grid in real space, along the directions set by the displ_vecs.
    Simultaneously, can compute the dipole three-mode elements.

    Args:
       molecule: Molecule object.
       scf_result: pyscf object from electronic structure calculations
       freqs_au: list of normal mode frequencies
       displ_vecs: list of displacement vectors for each normal mode
       gauss_grid: sample points for Gauss-Hermite quadrature
       pes_onebody: one-mode PES
       pes_twobody: two-mode PES
       dipole_onebody: one-mode dipole
       dipole_twobody: two-mode dipole
       method: Electronic structure method to define the level of theory
            for harmonic analysis. Default is restricted Hartree-Fock 'rhf'.
       do_dipole: Whether to calculate the dipole elements. Default is ``False``.

    Returns:
       A tuple of three-mode potential energy surface and three-mode dipole along
       the normal-mode coordinates.
    """

    _import_mpi4py()
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    local_pes_threebody, local_dipole_threebody = _local_pes_threemode(
        comm,
        molecule,
        scf_result,
        freqs_au,
        displ_vecs,
        gauss_grid,
        pes_onebody,
        pes_twobody,
        dipole_onebody,
        dipole_twobody,
        method=method,
        do_dipole=do_dipole,
    )
    comm.Barrier()

    f = h5py.File("v3data" + f"_{rank}" + ".hdf5", "w")
    f.create_dataset("V3_PES", data=local_pes_threebody)
    if do_dipole:
        dipole_threebody = None
        f.create_dataset("D3_DMS", data=local_dipole_threebody)
    f.close()
    comm.Barrier()

    pes_threebody = None
    if rank == 0:
        pes_threebody, dipole_threebody = _load_pes_threemode(
            comm.Get_size(), len(freqs_au), len(gauss_grid), do_dipole
        )
        subprocess.run(
            ["rm", "v3data*"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True,
            check=False,
        )

    comm.Barrier()
    pes_threebody = comm.bcast(pes_threebody, root=0)
    if do_dipole:
        dipole_threebody = comm.bcast(dipole_threebody, root=0)
        return pes_threebody, dipole_threebody

    return pes_threebody, None  # pragma: no cover

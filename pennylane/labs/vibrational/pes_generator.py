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
from pathlib import Path

import numpy as np
import scipy as sp

import pennylane as qml
from pennylane.data.base._lazy_modules import h5py
from pennylane.qchem import VibrationalPES, localize_normal_modes, optimize_geometry
from pennylane.qchem.vibrational.vibrational_class import (
    _get_dipole,
    _harmonic_analysis,
    _single_point,
)

# pylint: disable=too-many-arguments, too-many-function-args, too-many-branches
# pylint: disable= import-outside-toplevel, too-many-positional-arguments

# constants
# TODO: Make this code work in atomic units only.
HBAR = (
    sp.constants.hbar * (1000 * sp.constants.Avogadro) * (10**20)
)  # kg*(m^2/s) to (amu)*(angstrom^2/s)
BOHR_TO_ANG = (
    sp.constants.physical_constants["Bohr radius"][0] * 1e10
)  # factor to convert bohr to angstrom
CM_TO_AU = 100 / sp.constants.physical_constants["hartree-inverse meter relationship"][0]  # m to cm


def _import_mpi4py():
    """Import mpi4py."""
    try:
        import mpi4py
    except ImportError as Error:
        raise ImportError(
            "This feature requires mpi4py. It can be installed with: pip install mpi4py."
        ) from Error

    return mpi4py


def _pes_onemode(molecule, scf_result, freqs, vectors, grid, method="rhf", dipole=False):
    r"""Computes the one-mode potential energy surface on a grid along directions defined by displacement vectors.

    Args:
        molecule (:func:`~pennylane.qchem.molecule.Molecule`): Molecule object
        scf_result (pyscf.scf object): pyscf object from electronic structure calculations
        freqs (list[float]): list of vibrational frequencies in ``cm^-1``
        vectors (TensorLike[float]): list of displacement vectors for each normal mode
        grid (list[float]): the sample points on the Gauss-Hermite quadrature grid
        method (str): Electronic structure method that can be either restricted and unrestricted
            Hartree-Fock,  ``'rhf'`` and ``'uhf'``, respectively. Default is ``'rhf'``.
        dipole (bool): Flag to calculate the dipole elements. Default is ``False``.

    Returns:
        tuple: A tuple containing the following:
         - TensorLike[float]: one-mode potential energy surface
         - TensorLike[float] or None: one-mode dipole or ``None``
           if dipole is set to ``False``

    """
    _import_mpi4py()
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    local_pes_onebody, local_dipole_onebody = _local_pes_onemode(
        comm,
        molecule,
        scf_result,
        freqs,
        vectors,
        grid,
        method=method,
        dipole=dipole,
    )

    filename = f"v1data_{rank}.hdf5"
    with h5py.File(filename, "w") as f:
        f.create_dataset("V1_PES", data=local_pes_onebody)
        if dipole:
            f.create_dataset("D1_DMS", data=local_dipole_onebody)
        f.close()

    comm.Barrier()
    pes_onebody = None
    dipole_onebody = None
    if rank == 0:
        pes_onebody, dipole_onebody = _load_pes_onemode(
            comm.Get_size(), len(freqs), len(grid), dipole=dipole
        )
        current_directory = Path.cwd()
        for file_path in current_directory.glob("v1data*"):
            file_path.unlink(missing_ok=False)

    comm.Barrier()
    pes_onebody = comm.bcast(pes_onebody, root=0)

    if dipole:
        dipole_onebody = comm.bcast(dipole_onebody, root=0)
        return pes_onebody, dipole_onebody

    return pes_onebody, None


def _local_pes_onemode(
    comm, molecule, scf_result, freqs, vectors, grid, method="rhf", dipole=False
):
    r"""Computes the one-mode potential energy surface on a grid along directions defined by
    displacement vectors for each thread.

    Args:
        comm (mpi4py.MPI.Comm): the MPI communicator to be used for communication between processes
        molecule (:func:`~pennylane.qchem.molecule.Molecule`): Molecule object
        scf_result (pyscf.scf object): pyscf object from electronic structure calculations
        freqs (list[float]): list of normal mode frequencies in ``cm^-1``
        vectors (TensorLike[float]): list of displacement vectors for each normal mode
        grid (list[float]): the sample points on the Gauss-Hermite quadrature grid
        method (str): Electronic structure method that can be either restricted and unrestricted
            Hartree-Fock,  ``'rhf'`` and ``'uhf'``, respectively. Default is ``'rhf'``.
        dipole (bool): Flag to calculate the dipole elements. Default is ``False``.

    Returns:
        tuple: A tuple containing the following:
         - TensorLike[float]: one-mode potential energy surface
         - TensorLike[float] or None: one-mode dipole, returns ``None``
           if dipole is set to ``False``

    """

    size = comm.Get_size()
    rank = comm.Get_rank()
    quad_order = len(grid)
    nmodes = len(freqs)
    init_geom = molecule.coordinates * BOHR_TO_ANG

    jobs_on_rank = np.array_split(range(quad_order), size)[rank]
    local_pes_onebody = np.zeros((nmodes, len(jobs_on_rank)), dtype=float)

    if dipole:
        local_dipole_onebody = np.zeros((nmodes, len(jobs_on_rank), 3), dtype=float)
        ref_dipole = _get_dipole(scf_result, method)
    for mode in range(nmodes):
        vec = vectors[mode]
        if (freqs[mode].imag) > 1e-6:
            continue  # pragma: no cover

        for job_idx, job in enumerate(jobs_on_rank):
            gridpoint = grid[job]
            scaling = np.sqrt(HBAR / (2 * np.pi * freqs[mode] * 100 * sp.constants.c))
            positions = np.array(init_geom + scaling * gridpoint * vec)

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

            local_pes_onebody[mode][job_idx] = displ_scf.e_tot - scf_result.e_tot
            if dipole:
                local_dipole_onebody[mode, job_idx, :] = _get_dipole(displ_scf, method) - ref_dipole

    if dipole:
        return local_pes_onebody, local_dipole_onebody
    return local_pes_onebody, None


def _load_pes_onemode(num_proc, nmodes, quad_order, dipole=False):
    """
    Loader to combine pes_onebody and dipole_onebody from multiple processors.

    Args:
        num_proc (int): number of processors
        nmodes (int): number of normal modes
        quad_order (int): order for Gauss-Hermite quadratures
        dipole (bool): Flag to calculate the dipole elements. Default is ``False``.

    Returns:
        tuple: A tuple containing the following:
         - TensorLike[float]: one-mode potential energy surface
         - TensorLike[float] or None: one-mode dipole, returns ``None``
           if dipole is set to ``False``

    """
    pes_onebody = np.zeros((nmodes, quad_order), dtype=float)

    if dipole:
        dipole_onebody = np.zeros((nmodes, quad_order, 3), dtype=float)

    for mode in range(nmodes):
        init_chunk = 0
        for proc in range(num_proc):
            f = h5py.File("v1data" + f"_{proc}" + ".hdf5", "r+")
            local_pes_onebody = f["V1_PES"][()]
            if dipole:
                local_dipole_onebody = f["D1_DMS"][()]

            end_chunk = np.array(local_pes_onebody).shape[1]
            pes_onebody[mode][init_chunk : init_chunk + end_chunk] = local_pes_onebody[mode]
            if dipole:
                dipole_onebody[mode][init_chunk : init_chunk + end_chunk] = local_dipole_onebody[
                    mode
                ]
            init_chunk += end_chunk

    if dipole:
        return pes_onebody, dipole_onebody
    return pes_onebody, None


def _pes_twomode(
    molecule,
    scf_result,
    freqs,
    vectors,
    grid,
    pes_onebody,
    dipole_onebody,
    method="rhf",
    dipole=False,
):
    r"""Computes the two-mode potential energy surface on a grid along directions defined by
    displacement vectors.

    Args:
        molecule (:func:`~pennylane.qchem.molecule.Molecule`): Molecule object
        scf_result (pyscf.scf object): pyscf object from electronic structure calculations
        freqs (list[float]): list of vibrational frequencies in ``cm^-1``
        vectors (TensorLike[float]): list of displacement vectors for each normal mode
        grid (list[float]): the sample points on the Gauss-Hermite quadrature grid
        pes_onebody (TensorLike[float]): one-mode PES
        dipole_onebody (TensorLike[float]): one-mode dipole
        method (str): Electronic structure method that can be either restricted and unrestricted
            Hartree-Fock,  ``'rhf'`` and ``'uhf'``, respectively. Default is ``'rhf'``.
        dipole (bool): Flag to calculate the dipole elements. Default is ``False``.

    Returns:
        tuple: A tuple containing the following:
         - TensorLike[float]: two-mode potential energy surface
         - TensorLike[float] or None: two-mode dipole, returns ``None``
           if dipole is set to ``False``

    """
    _import_mpi4py()
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    local_pes_twobody, local_dipole_twobody = _local_pes_twomode(
        comm,
        molecule,
        scf_result,
        freqs,
        vectors,
        grid,
        pes_onebody,
        dipole_onebody,
        method=method,
        dipole=dipole,
    )

    filename = f"v2data_{rank}.hdf5"
    with h5py.File(filename, "w") as f:
        f.create_dataset("V2_PES", data=local_pes_twobody)
        if dipole:
            f.create_dataset("D2_DMS", data=local_dipole_twobody)

    comm.Barrier()
    pes_twobody = None
    dipole_twobody = None
    if rank == 0:
        pes_twobody, dipole_twobody = _load_pes_twomode(
            comm.Get_size(), len(freqs), len(grid), dipole=dipole
        )
        current_directory = Path.cwd()
        for file_path in current_directory.glob("v2data*"):
            file_path.unlink(missing_ok=False)

    comm.Barrier()
    pes_twobody = comm.bcast(pes_twobody, root=0)

    if dipole:
        dipole_twobody = comm.bcast(dipole_twobody, root=0)
        return pes_twobody, dipole_twobody
    return pes_twobody, None  # pragma: no cover


def _local_pes_twomode(
    comm,
    molecule,
    scf_result,
    freqs,
    vectors,
    grid,
    pes_onebody,
    dipole_onebody,
    method="rhf",
    dipole=False,
):
    r"""Computes the two-mode potential energy surface on a grid along directions defined by
    displacement vectors for each thread.

    Args:
        comm (mpi4py.MPI.Comm): the MPI communicator to be used for communication between processes
        molecule (:func:`~pennylane.qchem.molecule.Molecule`): Molecule object
        scf_result (pyscf.scf object): pyscf object from electronic structure calculations
        freqs (list[float]): list of vibrational frequencies in ``cm^-1``
        vectors (TensorLike[float]): list of displacement vectors for each normal mode
        grid (list[float]): the sample points on the Gauss-Hermite quadrature grid
        pes_onebody (TensorLike[float]): one-mode PES
        dipole_onebody (TensorLike[float]): one-mode dipole
        method (str): Electronic structure method that can be either restricted and unrestricted
            Hartree-Fock,  ``'rhf'`` and ``'uhf'``, respectively. Default is ``'rhf'``.
        dipole (bool): Flag to calculate the dipole elements. Default is ``False``.

    Returns:
        tuple: A tuple containing the following:
         - TensorLike[float]: two-mode potential energy surface
         - TensorLike[float] or None: two-mode dipole, returns ``None``
           if dipole is set to ``False``

    """

    size = comm.Get_size()
    rank = comm.Get_rank()
    init_geom = molecule.coordinates * BOHR_TO_ANG
    nmodes = len(freqs)

    all_mode_combos = [(mode_a, mode_b) for mode_a in range(nmodes) for mode_b in range(mode_a)]

    all_jobs = [
        [i, gridpoint_1, j, gridpoint_2]
        for (i, gridpoint_1), (j, gridpoint_2) in itertools.product(enumerate(grid), repeat=2)
    ]
    jobs_on_rank = np.array_split(all_jobs, size)[rank]
    local_pes_twobody = np.zeros((len(all_mode_combos) * len(jobs_on_rank)))

    if dipole:
        local_dipole_twobody = np.zeros((len(all_mode_combos) * len(jobs_on_rank), 3), dtype=float)
        ref_dipole = _get_dipole(scf_result, method)

    for mode_idx, [mode_a, mode_b] in enumerate(all_mode_combos):

        if (freqs[mode_a].imag) > 1e-6 or (freqs[mode_b].imag) > 1e-6:
            continue  # pragma: no cover

        vec_a = vectors[mode_a]
        scaling_a = np.sqrt(HBAR / (2 * np.pi * freqs[mode_a] * 100 * sp.constants.c))

        vec_b = vectors[mode_b]

        scaling_b = np.sqrt(HBAR / (2 * np.pi * freqs[mode_b] * 100 * sp.constants.c))
        for job_idx, [i, gridpoint_1, j, gridpoint_2] in enumerate(jobs_on_rank):

            i, j = int(i), int(j)
            positions = np.array(
                init_geom + scaling_a * gridpoint_1 * vec_a + scaling_b * gridpoint_2 * vec_b
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

            if dipole:
                local_dipole_twobody[idx, :] = (
                    _get_dipole(displ_scf, method)
                    - dipole_onebody[mode_a, i, :]
                    - dipole_onebody[mode_b, j, :]
                    - ref_dipole
                )

    if dipole:
        return local_pes_twobody, local_dipole_twobody

    return local_pes_twobody, None  # pragma: no cover


def _load_pes_twomode(num_proc, nmodes, quad_order, dipole=False):
    """
    Loader to combine pes_twobody and dipole_twobody from multiple processors.

    Args:
        num_proc (int): number of processors
        nmodes (int): number of normal modes
        quad_order (int): order for Gauss-Hermite quadratures
        dipole (bool): Flag to calculate the dipole elements. Default is ``False``.

    Returns:
        tuple: A tuple containing the following:
         - TensorLike[float]: two-mode potential energy surface
         - TensorLike[float] or None: two-mode dipole, returns ``None``
           if dipole is set to ``False``

    """

    final_shape = (nmodes, nmodes, quad_order, quad_order)
    nmode_combos = int(nmodes * (nmodes - 1) / 2)
    pes_twobody = np.zeros((final_shape))
    if dipole:
        dipole_twobody = np.zeros((final_shape + (3,)))

    mode_combo = 0
    for mode_a in range(nmodes):
        for mode_b in range(mode_a):
            local_pes = np.zeros(quad_order**2)
            if dipole:
                local_dipole = np.zeros((quad_order**2, 3))

            init_idx = 0
            end_idx = 0
            for proc in range(num_proc):
                f = h5py.File("v2data" + f"_{proc}" + ".hdf5", "r+")
                local_pes_twobody = f["V2_PES"][()]
                pes_chunk = np.array_split(local_pes_twobody, nmode_combos)[mode_combo]

                end_idx += len(pes_chunk)
                local_pes[init_idx:end_idx] = pes_chunk
                if dipole:
                    local_dipole_twobody = f["D2_DMS"][()]
                    dipole_chunk = np.array_split(local_dipole_twobody, nmode_combos, axis=0)[
                        mode_combo
                    ]
                    local_dipole[init_idx:end_idx, :] = dipole_chunk
                init_idx += len(pes_chunk)

            pes_twobody[mode_a, mode_b, :, :] = local_pes.reshape(quad_order, quad_order)
            if dipole:
                dipole_twobody[mode_a, mode_b, :, :, :] = local_dipole.reshape(
                    quad_order, quad_order, 3
                )
            mode_combo += 1

    if dipole:
        return pes_twobody, dipole_twobody
    return pes_twobody, None  # pragma: no cover


def _local_pes_threemode(
    comm,
    molecule,
    scf_result,
    freqs,
    vectors,
    grid,
    pes_onebody,
    pes_twobody,
    dipole_onebody,
    dipole_twobody,
    method="rhf",
    dipole=False,
):
    r"""Computes the three-mode potential energy surface on a grid along directions defined by
    displacement vectors for each thread.

    Args:
        comm (mpi4py.MPI.Comm): the MPI communicator to be used for communication between processes
        molecule (:func:`~pennylane.qchem.molecule.Molecule`): Molecule object
        scf_result (pyscf.scf object): pyscf object from electronic structure calculations
        freqs (list[float]): list of vibrational frequencies in ``cm^-1``
        vectors (TensorLike[float]): list of displacement vectors for each normal mode
        grid (list[float]): the sample points on the Gauss-Hermite quadrature grid
        pes_onebody (TensorLike[float]): one-mode PES
        pes_twobody (TensorLike[float]): two-mode PES
        dipole_onebody (TensorLike[float]): one-mode dipole
        dipole_twobody (TensorLike[float]): one-mode dipole
        method (str): Electronic structure method that can be either restricted and unrestricted
            Hartree-Fock,  ``'rhf'`` and ``'uhf'``, respectively. Default is ``'rhf'``.
        dipole (bool): Flag to calculate the dipole elements. Default is ``False``.

    Returns:
        tuple: A tuple containing the following:
         - TensorLike[float]: three-mode potential energy surface
         - TensorLike[float] or None: three-mode dipole, returns ``None``
           if dipole is set to ``False``

    """

    size = comm.Get_size()
    rank = comm.Get_rank()

    init_geom = molecule.coordinates * BOHR_TO_ANG
    nmodes = len(freqs)

    all_mode_combos = [
        (mode_a, mode_b, mode_c)
        for mode_a in range(nmodes)
        for mode_b in range(mode_a)
        for mode_c in range(mode_b)
    ]

    all_jobs = [
        [i, gridpoint_1, j, gridpoint_2, k, gridpoint_3]
        for (i, gridpoint_1), (j, gridpoint_2), (k, gridpoint_3) in itertools.product(
            enumerate(grid), repeat=3
        )
    ]

    jobs_on_rank = np.array_split(all_jobs, size)[rank]
    local_pes_threebody = np.zeros(len(all_mode_combos) * len(jobs_on_rank))

    if dipole:
        local_dipole_threebody = np.zeros(
            (len(all_mode_combos) * len(jobs_on_rank), 3), dtype=float
        )
        ref_dipole = _get_dipole(scf_result, method)

    mode_combo = 0
    for mode_combo, [mode_a, mode_b, mode_c] in enumerate(all_mode_combos):

        if (
            (freqs[mode_a].imag) > 1e-6
            or (freqs[mode_b].imag) > 1e-6
            or (freqs[mode_c].imag) > 1e-6
        ):
            continue  # pragma: no cover

        vec_a = vectors[mode_a]
        scaling_a = np.sqrt(HBAR / (2 * np.pi * freqs[mode_a] * 100 * sp.constants.c))

        vec_b = vectors[mode_b]
        scaling_b = np.sqrt(HBAR / (2 * np.pi * freqs[mode_b] * 100 * sp.constants.c))

        vec_c = vectors[mode_c]
        scaling_c = np.sqrt(HBAR / (2 * np.pi * freqs[mode_c] * 100 * sp.constants.c))

        for job_idx, [i, gridpoint_1, j, gridpoint_2, k, gridpoint_3] in enumerate(jobs_on_rank):

            i, j, k = int(i), int(j), int(k)
            positions = np.array(
                init_geom
                + scaling_a * gridpoint_1 * vec_a
                + scaling_b * gridpoint_2 * vec_b
                + scaling_c * gridpoint_3 * vec_c
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
            if dipole:
                local_dipole_threebody[idx, :] = (
                    _get_dipole(displ_scf, method)
                    - dipole_twobody[mode_a, mode_b, i, j, :]
                    - dipole_twobody[mode_a, mode_c, i, k, :]
                    - dipole_twobody[mode_b, mode_c, j, k, :]
                    - dipole_onebody[mode_a, i, :]
                    - dipole_onebody[mode_b, j, :]
                    - dipole_onebody[mode_c, k, :]
                    - ref_dipole
                )

    comm.Barrier()
    if dipole:
        return local_pes_threebody, local_dipole_threebody

    return local_pes_threebody, None  # pragma: no cover


def _load_pes_threemode(num_proc, nmodes, quad_order, dipole):
    """
    Loader to combine pes_threebody and dipole_threebody from multiple processors.

    Args:
        num_proc: number of processors
        nmodes: number of normal modes
        quad_order: order for Gauss-Hermite quadratures
        dipole: Flag to calculate the dipole elements. Default is ``False``.

    Returns:
        tuple: A tuple containing the following:
         - TensorLike[float]: three-mode potential energy surface
         - TensorLike[float] or None: three-mode dipole, returns ``None``
           if dipole is set to ``False``

    """
    final_shape = (nmodes, nmodes, nmodes, quad_order, quad_order, quad_order)
    nmode_combos = int(nmodes * (nmodes - 1) * (nmodes - 2) / 6)
    pes_threebody = np.zeros(final_shape)
    dipole_threebody = np.zeros((final_shape + (3,))) if dipole else None

    mode_combo = 0
    for mode_a in range(nmodes):
        for mode_b in range(mode_a):
            for mode_c in range(mode_b):
                local_pes = np.zeros(quad_order**3)
                local_dipole = np.zeros((quad_order**3, 3)) if dipole else None

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

    if dipole:
        return pes_threebody, dipole_threebody
    return pes_threebody, None  # pragma: no cover


def _pes_threemode(
    molecule,
    scf_result,
    freqs,
    vectors,
    grid,
    pes_onebody,
    pes_twobody,
    dipole_onebody,
    dipole_twobody,
    method="rhf",
    dipole=False,
):
    r"""Computes the three-mode potential energy surface on a grid along directions defined by
    displacement vectors.

    Args:
        molecule (:func:`~pennylane.qchem.molecule.Molecule`): Molecule object
        scf_result (pyscf.scf object): pyscf object from electronic structure calculations
        freqs (list[float]): list of vibrational frequencies in ``cm^-1``
        vectors (TensorLike[float]): list of displacement vectors for each normal mode
        grid (list[float]): the sample points on the Gauss-Hermite quadrature grid
        pes_onebody (TensorLike[float]): one-mode PES
        pes_twobody (TensorLike[float]): two-mode PES
        dipole_onebody (TensorLike[float]): one-mode dipole
        dipole_twobody (TensorLike[float]): one-mode dipole
        method (str): Electronic structure method that can be either restricted and unrestricted
            Hartree-Fock,  ``'rhf'`` and ``'uhf'``, respectively. Default is ``'rhf'``.
        dipole (bool): Flag to calculate the dipole elements. Default is ``False``.

    Returns:
        tuple: A tuple containing the following:
         - TensorLike[float]: three-mode potential energy surface
         - TensorLike[float] or None: three-mode dipole, returns ``None``
           if dipole is set to ``False``

    """
    _import_mpi4py()
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    local_pes_threebody, local_dipole_threebody = _local_pes_threemode(
        comm,
        molecule,
        scf_result,
        freqs,
        vectors,
        grid,
        pes_onebody,
        pes_twobody,
        dipole_onebody,
        dipole_twobody,
        method=method,
        dipole=dipole,
    )
    comm.Barrier()

    f = h5py.File("v3data" + f"_{rank}" + ".hdf5", "w")
    f.create_dataset("V3_PES", data=local_pes_threebody)
    if dipole:
        dipole_threebody = None
        f.create_dataset("D3_DMS", data=local_dipole_threebody)
    f.close()
    comm.Barrier()

    pes_threebody = None
    if rank == 0:
        pes_threebody, dipole_threebody = _load_pes_threemode(
            comm.Get_size(), len(freqs), len(grid), dipole
        )
        current_directory = Path.cwd()
        for file_path in current_directory.glob("v3data*"):
            file_path.unlink(missing_ok=False)

    comm.Barrier()
    pes_threebody = comm.bcast(pes_threebody, root=0)
    if dipole:
        dipole_threebody = comm.bcast(dipole_threebody, root=0)
        return pes_threebody, dipole_threebody

    return pes_threebody, None  # pragma: no cover


def vibrational_pes(
    molecule,
    quad_order=9,
    method="rhf",
    localize=True,
    bins=None,
    cubic=False,
    dipole_level=1,
):
    r"""Computes potential energy surfaces along vibrational normal modes.

    Args:
       molecule (~qchem.molecule.Molecule): Molecule object
       quad_order (int): Order for Gauss-Hermite quadratures. Default value is ``9``.
       method (str): Electronic structure method that can be either restricted and unrestricted
           Hartree-Fock,  ``'rhf'`` and ``'uhf'``, respectively. Default is ``'rhf'``.
       localize (bool): Flag to perform normal mode localization. Default is ``False``.
       bins (list[float]): List of upper bound frequencies in ``cm^-1`` for creating separation bins .
           Default is ``[2600]`` which means having one bin for all frequencies between ``0`` and  ``2600 cm^-1``.
       cubic (bool)): Flag to include three-mode couplings. Default is ``False``.
       dipole_level (int): The level up to which dipole matrix elements are to be calculated. Input values can be
           ``1``, ``2``, or ``3`` for up to one-mode dipole, two-mode dipole and three-mode dipole, respectively. Default
           value is ``1``.

    Returns:
       VibrationalPES object.

    **Example**

    >>> symbols  = ['H', 'F']
    >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    >>> mol = qml.qchem.Molecule(symbols, geometry)
    >>> pes = vibrational_pes(mol)
    >>> pes.pes_onemode
    array([[ 6.26177771e-02,  3.62085556e-02,  1.72120120e-02,
         4.71674655e-03, -2.84217094e-14,  6.06717218e-03,
         2.87234966e-02,  8.03213574e-02,  1.95651039e-01]])

    """
    if bins is None:
        bins = [2600]

    if dipole_level > 3 or dipole_level < 1:
        raise ValueError(
            "Currently, one-mode, two-mode and three-mode dipole calculations are supported. Please provide a value"
            "between 1 and 3."
        )
    if quad_order < 1:
        raise ValueError("Number of sample points cannot be less than 1.")

    _import_mpi4py()
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    geom_eq = optimize_geometry(molecule, method)

    mol_eq = qml.qchem.Molecule(
        molecule.symbols,
        geom_eq,
        unit=molecule.unit,
        basis_name=molecule.basis_name,
        charge=molecule.charge,
        mult=molecule.mult,
        load_data=molecule.load_data,
    )

    scf_result = _single_point(mol_eq, method)

    freqs = None
    uloc = None
    vectors = None
    if rank == 0:
        freqs, vectors = _harmonic_analysis(scf_result, method)
        if localize:
            freqs, vectors, uloc = localize_normal_modes(freqs, vectors, bins=bins)

    # Broadcast data to all threads
    freqs = comm.bcast(freqs, root=0)
    vectors = np.array(comm.bcast(vectors, root=0))
    uloc = np.array(comm.bcast(uloc, root=0))

    comm.Barrier()

    grid, gauss_weights = np.polynomial.hermite.hermgauss(quad_order)

    dipole = True
    pes_onebody, dipole_onebody = _pes_onemode(
        mol_eq, scf_result, freqs, vectors, grid, method=method, dipole=dipole
    )
    comm.Barrier()

    # build PES -- two-body
    if dipole_level < 2:
        dipole = False

    pes_twobody, dipole_twobody = _pes_twomode(
        mol_eq,
        scf_result,
        freqs,
        vectors,
        grid,
        pes_onebody,
        dipole_onebody,
        method=method,
        dipole=dipole,
    )
    comm.Barrier()

    pes_data = [pes_onebody, pes_twobody]
    dipole_data = [dipole_onebody, dipole_twobody]

    if cubic:
        if dipole_level < 3:
            dipole = False

        pes_threebody, dipole_threebody = _pes_threemode(
            mol_eq,
            scf_result,
            freqs,
            vectors,
            grid,
            pes_onebody,
            pes_twobody,
            dipole_onebody,
            dipole_twobody,
            method=method,
            dipole=dipole,
        )
        comm.Barrier()
        pes_data = [pes_onebody, pes_twobody, pes_threebody]
        dipole_data = [dipole_onebody, dipole_twobody, dipole_threebody]

    freqs = freqs * CM_TO_AU
    return VibrationalPES(
        freqs, grid, gauss_weights, uloc, pes_data, dipole_data, localize, dipole_level
    )

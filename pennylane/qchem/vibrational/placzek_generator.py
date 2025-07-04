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
from tempfile import TemporaryDirectory

import numpy as np
import scipy as sp

from pennylane import concurrency, qchem
from pennylane.qchem.vibrational.christiansen_utils import _read_data, _write_data
from pennylane.qchem.vibrational.localize_modes import localize_normal_modes
from pennylane.qchem.vibrational.vibrational_class import (
    VibrationalPES,
    _get_dipole,
    _harmonic_analysis,
    _single_point,
    optimize_geometry,
)
from pennylane.qchem.vibrational.polarizable_class import (
    PolarizablePES,
    _get_placzek
)

# pylint: disable=too-many-arguments,too-many-function-args
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


def _pes_onemode(
    molecule,
    scf_result,
    freqs,
    vectors,
    grid,
    method="rhf",
    placzek=False,
    num_workers=1,
    backend="serial",
    path=None,
    excited_method='tddft',
    omega_I=None,
    excited_states=10
):
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
        num_workers (int): the number of concurrent units used for the computation. Default value is set to 1.
        backend (string): the executor backend from the list of supported backend.
            Available options : "mp_pool", "cf_procpool", "cf_threadpool", "serial", "mpi4py_pool", "mpi4py_comm". Default value is set to "serial".
        path (string): the path to the directory where results will be saved. Default value is set to None.
    Returns:
        tuple: A tuple containing the following:
         - TensorLike[float]: one-mode potential energy surface
         - TensorLike[float] or None: one-mode dipole or ``None``
           if dipole is set to ``False``

    """

    quad_order = len(grid)
    all_jobs = range(quad_order)
    jobs_on_rank = np.array_split(all_jobs, num_workers)

    arguments = [
        (j, i, molecule, scf_result, freqs, vectors, grid, path, method, placzek, excited_method, omega_I, excited_states)
        for j, i in enumerate(jobs_on_rank)
    ]

    executor_class = concurrency.backends.get_executor(backend)
    with executor_class(max_workers=num_workers) as executor:
        executor.starmap(_local_pes_onemode, arguments)

    pes_onebody = None
    dipole_onebody = None
    pes_onebody, placzek_onebody = _load_pes_onemode(
        num_workers, len(freqs), len(grid), path, placzek=placzek
    )

    if placzek:
        return pes_onebody, placzek_onebody

    return pes_onebody, None


def _local_pes_onemode(
    rank, jobs_on_rank, molecule, scf_result, freqs, vectors, grid, path, method="rhf", placzek=False, excited_method='tddft', omega_I=None, excited_states=10
):
    r"""Computes the one-mode potential energy surface on a grid along directions defined by
    displacement vectors for each thread.

    Args:
        rank (int) : rank of the process
        jobs_on_rank [int] : list of gridpoint processes by this worker
        molecule (:func:`~pennylane.qchem.molecule.Molecule`): Molecule object
        scf_result (pyscf.scf object): pyscf object from electronic structure calculations
        freqs (list[float]): list of normal mode frequencies in ``cm^-1``
        vectors (TensorLike[float]): list of displacement vectors for each normal mode
        grid (list[float]): the sample points on the Gauss-Hermite quadrature grid
        path (string): the path to the directory where results will be saved.
        method (str): Electronic structure method that can be either restricted and unrestricted
            Hartree-Fock,  ``'rhf'`` and ``'uhf'``, respectively. Default is ``'rhf'``.
        dipole (bool): Flag to calculate the dipole elements. Default is ``False``.

    Returns:
        tuple: A tuple containing the following:
         - TensorLike[float]: one-mode potential energy surface
         - TensorLike[float] or None: one-mode dipole, returns ``None``
           if dipole is set to ``False``

    """
    nmodes = len(freqs)
    init_geom = molecule.coordinates * BOHR_TO_ANG

    local_pes_onebody = np.zeros((nmodes, len(jobs_on_rank)), dtype=float)

    if placzek:
        local_placzek_onebody = np.zeros((nmodes, len(jobs_on_rank), 6), dtype=float)
        ref_placzek = _get_placzek(scf_result, method, excited_method, omega_I, num_states)
    for mode in range(nmodes):
        vec = vectors[mode]
        if (freqs[mode].imag) > 1e-6:
            continue  # pragma: no cover

        for job_idx, job in enumerate(jobs_on_rank):
            gridpoint = grid[job]
            scaling = np.sqrt(HBAR / (2 * np.pi * freqs[mode] * 100 * sp.constants.c))
            positions = np.array(init_geom + scaling * gridpoint * vec)

            displ_mol = qchem.Molecule(
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
            if placzek:
                local_placzek_onebody[mode, job_idx, :] = _get_placzek(displ_scf, method, excited_method, omega_I, num_states) - ref_placzek

    _write_data(path, rank, "v1data", "V1_PES", local_pes_onebody)
    if placzek:
        _write_data(path, rank, "v1data", "P1_PPS", local_placzek_onebody)


def _load_pes_onemode(num_proc, nmodes, quad_order, path, placzek=False):
    """
    Loader to combine pes_onebody and dipole_onebody from multiple processors.

    Args:
        num_proc (int): number of processors
        nmodes (int): number of normal modes
        quad_order (int): order for Gauss-Hermite quadratures
        path (string): the path to the directory where results are saved.
        dipole (bool): Flag to calculate the dipole elements. Default is ``False``.

    Returns:
        tuple: A tuple containing the following:
         - TensorLike[float]: one-mode potential energy surface
         - TensorLike[float] or None: one-mode dipole, returns ``None``
           if dipole is set to ``False``

    """
    pes_onebody = np.zeros((nmodes, quad_order), dtype=float)

    if placzek:
        placzek_onebody = np.zeros((nmodes, quad_order, 6), dtype=float)

    for mode in range(nmodes):
        init_chunk = 0
        for proc in range(num_proc):
            local_pes_onebody = _read_data(path, proc, "v1data", "V1_PES")
            if placzek:
                local_placzek_onebody = _read_data(path, proc, "v1data", "P1_PPS")

            end_chunk = np.array(local_pes_onebody).shape[1]
            pes_onebody[mode][init_chunk : init_chunk + end_chunk] = local_pes_onebody[mode]
            if placzek:
                placzek_onebody[mode][init_chunk : init_chunk + end_chunk] = local_placzek_onebody[
                    mode
                ]
            init_chunk += end_chunk

    if placzek:
        return pes_onebody, placzek_onebody
    return pes_onebody, None


def _pes_twomode(
    molecule,
    scf_result,
    freqs,
    vectors,
    grid,
    pes_onebody,
    placzek_onebody,
    method="rhf",
    placzek=False,
    num_workers=1,
    backend="serial",
    path=None,
    excited_method='tddft',
    omega_I=None,
    excited_states=10
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
        num_workers (int): the number of concurrent units used for the computation. Default value is set to 1.
        backend (string): the executor backend from the list of supported backends.
            Available options : "mp_pool", "cf_procpool", "cf_threadpool", "serial", "mpi4py_pool", "mpi4py_comm". Default value is set to "serial".
        path (string): the path to the directory where results will be saved. Default value is set to None.

    Returns:
        tuple: A tuple containing the following:
         - TensorLike[float]: two-mode potential energy surface
         - TensorLike[float] or None: two-mode dipole, returns ``None``
           if dipole is set to ``False``

    """

    all_jobs = [
        [i, gridpoint_1, j, gridpoint_2]
        for (i, gridpoint_1), (j, gridpoint_2) in itertools.product(enumerate(grid), repeat=2)
    ]

    jobs_on_rank = np.array_split(all_jobs, num_workers)
    arguments = [
        (
            j,
            i,
            molecule,
            scf_result,
            freqs,
            vectors,
            pes_onebody,
            placzek_onebody,
            path,
            method,
            placzek,
            excited_method, 
            omega_I, 
            excited_states
        )
        for j, i in enumerate(jobs_on_rank)
    ]

    executor_class = concurrency.backends.get_executor(backend)
    with executor_class(max_workers=num_workers) as executor:
        executor.starmap(_local_pes_twomode, arguments)

    pes_twobody = None
    placzek_twobody = None

    pes_twobody, placzek_twobody = _load_pes_twomode(
        num_workers, len(freqs), len(grid), path, placzek=placzek
    )

    if placzek:

        return pes_twobody, placzek_twobody
    return pes_twobody, None  # pragma: no cover


def _local_pes_twomode(
    rank,
    jobs_on_rank,
    molecule,
    scf_result,
    freqs,
    vectors,
    pes_onebody,
    placzek_onebody,
    path,
    method="rhf",
    placzek=False,
    excited_method='tddft',
    omega_I=None,
    excited_states=10
):
    r"""Computes the two-mode potential energy surface on a grid along directions defined by
    displacement vectors for each thread.

    Args:
        rank (int) : rank of the process
        jobs_on_rank [int] : list of gridpoint processes by this worker
        molecule (:func:`~pennylane.qchem.molecule.Molecule`): Molecule object
        scf_result (pyscf.scf object): pyscf object from electronic structure calculations
        freqs (list[float]): list of vibrational frequencies in ``cm^-1``
        vectors (TensorLike[float]): list of displacement vectors for each normal mode
        grid (list[float]): the sample points on the Gauss-Hermite quadrature grid
        pes_onebody (TensorLike[float]): one-mode PES
        dipole_onebody (TensorLike[float]): one-mode dipole
        path (string): the path to the directory where results will be saved.
        method (str): Electronic structure method that can be either restricted and unrestricted
            Hartree-Fock,  ``'rhf'`` and ``'uhf'``, respectively. Default is ``'rhf'``.
        dipole (bool): Flag to calculate the dipole elements. Default is ``False``.

    Returns:
        tuple: A tuple containing the following:
         - TensorLike[float]: two-mode potential energy surface
         - TensorLike[float] or None: two-mode dipole, returns ``None``
           if dipole is set to ``False``

    """

    init_geom = molecule.coordinates * BOHR_TO_ANG
    nmodes = len(freqs)

    all_mode_combos = [(mode_a, mode_b) for mode_a in range(nmodes) for mode_b in range(mode_a)]
    local_pes_twobody = np.zeros((len(all_mode_combos) * len(jobs_on_rank)))

    if placzek:
        local_placzek_twobody = np.zeros((len(all_mode_combos) * len(jobs_on_rank), 6), dtype=float)
        ref_placzek = _get_placzek(scf_result, method, excited_method, omega_I, excited_states)

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
            displ_mol = qchem.Molecule(
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

            if placzek:
                local_placzek_twobody[idx, :] = (
                    _get_placzek(displ_scf, method, excited_method, omega_I, excited_states)
                    - placzek_onebody[mode_a, i, :]
                    - placzek_onebody[mode_b, j, :]
                    - ref_placzek
                )

    _write_data(path, rank, "v2data", "V2_PES", local_pes_twobody)
    if placzek:
        _write_data(path, rank, "v2data", "P2_PPS", local_placzek_twobody)


def _load_pes_twomode(num_proc, nmodes, quad_order, path, placzek=False):
    """
    Loader to combine pes_twobody and dipole_twobody from multiple processors.

    Args:
        num_proc (int): number of processors
        nmodes (int): number of normal modes
        quad_order (int): order for Gauss-Hermite quadratures
        path (string): the path to the directory where results are saved.
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
    if placzek:
        placzek_twobody = np.zeros((final_shape + (6,)))

    mode_combo = 0
    for mode_a in range(nmodes):
        for mode_b in range(mode_a):
            local_pes = np.zeros(quad_order**2)
            if placzek:
                local_placzek = np.zeros((quad_order**2, 6))

            init_idx = 0
            end_idx = 0
            for proc in range(num_proc):
                local_pes_twobody = _read_data(path, proc, "v2data", "V2_PES")
                if placzek:
                    local_placzek_twobody = _read_data(path, proc, "v2data", "P2_PPS")

                pes_chunk = np.array_split(local_pes_twobody, nmode_combos)[mode_combo]
                end_idx += len(pes_chunk)
                local_pes[init_idx:end_idx] = pes_chunk
                if placzek:
                    placzek_chunk = np.array_split(local_placzek_twobody, nmode_combos, axis=0)[
                        mode_combo
                    ]
                    local_placzek[init_idx:end_idx, :] = placzek_chunk
                init_idx += len(pes_chunk)

            pes_twobody[mode_a, mode_b, :, :] = local_pes.reshape(quad_order, quad_order)
            if placzek:
                placzek_twobody[mode_a, mode_b, :, :, :] = local_placzek.reshape(
                    quad_order, quad_order, 6
                )
            mode_combo += 1

    if placzek:
        return pes_twobody, placzek_twobody
    return pes_twobody, None  # pragma: no cover


def _local_pes_threemode(
    rank,
    jobs_on_rank,
    molecule,
    scf_result,
    freqs,
    vectors,
    pes_onebody,
    pes_twobody,
    placzek_onebody,
    placzek_twobody,
    path,
    method="rhf",
    placzek=False,
    excited_method='tddft',
    omega_I=None,
    excited_states=10
):
    r"""Computes the three-mode potential energy surface on a grid along directions defined by
    displacement vectors for each thread.

    Args:
        rank (int) : rank of the process
        jobs_on_rank [int] : list of gridpoint processes by this worker
        molecule (:func:`~pennylane.qchem.molecule.Molecule`): Molecule object
        scf_result (pyscf.scf object): pyscf object from electronic structure calculations
        freqs (list[float]): list of vibrational frequencies in ``cm^-1``
        vectors (TensorLike[float]): list of displacement vectors for each normal mode
        grid (list[float]): the sample points on the Gauss-Hermite quadrature grid
        pes_onebody (TensorLike[float]): one-mode PES
        pes_twobody (TensorLike[float]): two-mode PES
        dipole_onebody (TensorLike[float]): one-mode dipole
        dipole_twobody (TensorLike[float]): one-mode dipole
        path (string): the path to the directory where results will be saved.
        method (str): Electronic structure method that can be either restricted and unrestricted
            Hartree-Fock,  ``'rhf'`` and ``'uhf'``, respectively. Default is ``'rhf'``.
        dipole (bool): Flag to calculate the dipole elements. Default is ``False``.

    Returns:
        tuple: A tuple containing the following:
         - TensorLike[float]: three-mode potential energy surface
         - TensorLike[float] or None: three-mode dipole, returns ``None``
           if dipole is set to ``False``

    """

    init_geom = molecule.coordinates * BOHR_TO_ANG
    nmodes = len(freqs)

    all_mode_combos = [
        (mode_a, mode_b, mode_c)
        for mode_a in range(nmodes)
        for mode_b in range(mode_a)
        for mode_c in range(mode_b)
    ]

    local_pes_threebody = np.zeros(len(all_mode_combos) * len(jobs_on_rank))

    if placzek:
        local_placzek_threebody = np.zeros(
            (len(all_mode_combos) * len(jobs_on_rank), 6), dtype=float
        )
        ref_dipole = _get_placzek(scf_result, method, excited_method, omega_I, num_states)

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

            displ_mol = qchem.Molecule(
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
            if placzek:
                local_placzek_threebody[idx, :] = (
                    _get_placzek(displ_scf, method, excited_method, omega_I, num_states)
                    - placzek_twobody[mode_a, mode_b, i, j, :]
                    - placzek_twobody[mode_a, mode_c, i, k, :]
                    - placzek_twobody[mode_b, mode_c, j, k, :]
                    - placzek_onebody[mode_a, i, :]
                    - placzek_onebody[mode_b, j, :]
                    - placzek_onebody[mode_c, k, :]
                    - ref_placzek
                )

    _write_data(path, rank, "v3data", "V3_PES", local_pes_threebody)
    if placzek:
        _write_data(path, rank, "v3data", "P3_PPS", local_placzek_threebody)


def _pes_threemode(
    molecule,
    scf_result,
    freqs,
    vectors,
    grid,
    pes_onebody,
    pes_twobody,
    placzek_onebody,
    placzek_twobody,
    method="rhf",
    placzek=False,
    num_workers=1,
    backend="serial",
    path=None,
    excited_method='tddft',
    omega_I=None,
    excited_states=10
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
        num_workers (int): the number of concurrent units used for the computation. Default value is set to 1.
        backend (string): the executor backend from the list of supported backends.
            Available options : "mp_pool", "cf_procpool", "cf_threadpool", "serial", "mpi4py_pool", "mpi4py_comm". Default value is set to "serial".
        path (string): the path to the directory where results will be saved. Default value is set to None.


    Returns:
        tuple: A tuple containing the following:
         - TensorLike[float]: three-mode potential energy surface
         - TensorLike[float] or None: three-mode dipole, returns ``None``
           if dipole is set to ``False``

    """

    all_jobs = [
        [i, gridpoint_1, j, gridpoint_2, k, gridpoint_3]
        for (i, gridpoint_1), (j, gridpoint_2), (k, gridpoint_3) in itertools.product(
            enumerate(grid), repeat=3
        )
    ]
    jobs_on_rank = np.array_split(all_jobs, num_workers)
    arguments = [
        (
            j,
            i,
            molecule,
            scf_result,
            freqs,
            vectors,
            pes_onebody,
            pes_twobody,
            placzek_onebody,
            placzek_twobody,
            path,
            method,
            placzek,
            excited_method,
            omega_I,
            excited_states
        )
        for j, i in enumerate(jobs_on_rank)
    ]

    executor_class = concurrency.backends.get_executor(backend)
    with executor_class(max_workers=num_workers) as executor:
        executor.starmap(_local_pes_threemode, arguments)

    pes_threebody = None

    pes_threebody, placzek_threebody = _load_pes_threemode(
        num_workers, len(freqs), len(grid), path, placzek
    )

    if placzek:
        return pes_threebody, placzek_threebody

    return pes_threebody, None  # pragma: no cover


def _load_pes_threemode(num_proc, nmodes, quad_order, path, placzek):
    """
    Loader to combine pes_threebody and dipole_threebody from multiple processors.

    Args:
        num_proc: number of processors
        nmodes: number of normal modes
        quad_order: order for Gauss-Hermite quadratures
        dipole: Flag to calculate the dipole elements. Default is ``False``.
        path (string): the path to the directory where results are saved.

    Returns:
        tuple: A tuple containing the following:
         - TensorLike[float]: three-mode potential energy surface
         - TensorLike[float] or None: three-mode dipole, returns ``None``
           if dipole is set to ``False``

    """
    final_shape = (nmodes, nmodes, nmodes, quad_order, quad_order, quad_order)
    nmode_combos = int(nmodes * (nmodes - 1) * (nmodes - 2) / 6)
    pes_threebody = np.zeros(final_shape)
    placzek_threebody = np.zeros((final_shape + (6,))) if placzek else None

    mode_combo = 0
    for mode_a in range(nmodes):
        for mode_b in range(mode_a):
            for mode_c in range(mode_b):
                local_pes = np.zeros(quad_order**3)
                local_placzek = np.zeros((quad_order**3, 6)) if placzek else None

                init_idx = 0
                end_idx = 0
                for proc in range(num_proc):
                    local_pes_threebody = _read_data(path, proc, "v3data", "V3_PES")
                    if placzek:
                        local_placzek_threebody = _read_data(path, proc, "v3data", "P3_PPS")

                    pes_chunk = np.array_split(local_pes_threebody, nmode_combos)[mode_combo]
                    end_idx += len(pes_chunk)
                    local_pes[init_idx:end_idx] = pes_chunk
                    if placzek:
                        placzek_chunk = np.array_split(local_placzek_threebody, nmode_combos, axis=0)[
                            mode_combo
                        ]
                        local_placzek[init_idx:end_idx, :] = placzek_chunk
                    init_idx += len(pes_chunk)

                pes_threebody[mode_a, mode_b, mode_c, :, :] = local_pes.reshape(
                    quad_order, quad_order, quad_order
                )
                if placzek_threebody is not None:
                    placzek_threebody[mode_a, mode_b, mode_c, :, :, :, :] = local_placzek.reshape(
                        quad_order, quad_order, quad_order, 6
                    )
                mode_combo += 1

    if placzek:
        return pes_threebody, placzek_threebody
    return pes_threebody, None  # pragma: no cover


def polarizable_pes(
    molecule,
    n_points=9,
    method="rhf",
    optimize=True,
    localize=True,
    bins=None,
    cubic=False,
    placzek_level=2,
    num_workers=1,
    backend="serial",
    excited_method='tddft',
    omega_I=None,
    excited_states=10
):
    r"""Computes potential energy surfaces along vibrational normal modes.

    Args:
        molecule (~qchem.molecule.Molecule): the molecule object
        n_points (int): number of points for computing the potential energy surface. Default value is ``9``.
        method (str): Electronic structure method used to perform geometry optimization.
            Available options are ``"rhf"`` and ``"uhf"`` for restricted and unrestricted
            Hartree-Fock, respectively. Default is ``"rhf"``.
        optimize (bool): if ``True`` perform geometry optimization. Default is ``True``.
        localize (bool): if ``True`` perform normal mode localization. Default is ``False``.
        bins (List[float]): grid of frequencies for grouping normal modes.
            Default is ``None`` which means all frequencies will be grouped in one bin. For
            instance, ``bins = [1300, 2600]`` allows to separately group and localize modes in three
            groups that have frequencies below :math:`1300`, between :math:`1300-2600` and
            above :math:`2600`.
        cubic (bool)): if ``True`` include three-mode couplings. Default is ``False``.
        dipole_level (int): The level up to which dipole moment data are to be calculated. Input
            values can be ``1``, ``2``, or ``3`` for up to one-mode dipole, two-mode dipole and
            three-mode dipole, respectively. Default value is ``1``.
        num_workers (int): the number of concurrent units used for the computation. Default value is
            set to 1.
        backend (string): the executor backend from the list of supported backends. Available
            options are ``mp_pool``, ``cf_procpool``, ``cf_threadpool``, ``serial``,
            ``mpi4py_pool``, ``mpi4py_comm``. Default value is set to ``serial``. See Usage Details
            for more information.

    Returns:
       VibrationalPES: the VibrationalPES object

    **Example**

    >>> symbols  = ['H', 'F']
    >>> geometry = np.array([[0.0, 0.0, -0.40277116], [0.0, 0.0, 1.40277116]])
    >>> mol = qml.qchem.Molecule(symbols, geometry)
    >>> pes = qml.qchem.vibrational_pes(mol, optimize=False)
    >>> print(pes.freqs)
    [0.02038828]

    .. details::
        :title: Usage Details

        The ``backend`` options allow to run calculations using multiple threads or multiple
        processes.

        - ``serial``: This executor wraps Python standard library calls without support for
            multithreaded or multiprocess execution. Any calls to external libraries that utilize
            threads, such as BLAS through numpy, can still use multithreaded calls at that layer.

        - ``mp_pool``: This executor wraps Python standard library `multiprocessing.Pool <https://docs.python.org/3/library/multiprocessing.html#module-multiprocessing.pool>`_
            interface, and provides support for execution using multiple processes.

        - ``cf_procpool``: This executor wraps Python standard library `concurrent.futures.ProcessPoolExecutor <https://docs.python.org/3/library/concurrent.futures.html#processpoolexecutor>`_
            interface, and provides support for execution using multiple processes.

        - ``cf_threadpool``: This executor wraps Python standard library `concurrent.futures.ThreadPoolExecutor <https://docs.python.org/3/library/concurrent.futures.html#threadpoolexecutor>`_
            interface, and provides support for execution using multiple threads. The threading
            executor may not provide execution speed-ups for tasks when using a GIL-enabled Python.

        - ``mpi4py_pool``: This executor wraps the `mpi4py.futures.MPIPoolExecutor <https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html#mpipoolexecutor>`_
            class, and provides support for execution using multiple processes launched using MPI.

        - ``mpi4py_comm``: This executor wraps the `mpi4py.futures.MPICommExecutor <https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html#mpicommexecutor>`_
            class, and provides support for execution using multiple processes launched using MPI.
    """
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        if bins is None:
            bins = [2600]

        if placzek_level > 3 or placzek_level < 1:
            raise ValueError(
                "Currently, one-mode, two-mode and three-mode placzek calculations are supported. Please provide a value"
                "between 1 and 3."
            )
        if n_points < 1:
            raise ValueError("Number of sample points cannot be less than 1.")

        if optimize:
            geom_eq = optimize_geometry(molecule, method)
            mol_eq = qchem.Molecule(
                molecule.symbols,
                geom_eq,
                unit=molecule.unit,
                basis_name=molecule.basis_name,
                charge=molecule.charge,
                mult=molecule.mult,
                load_data=molecule.load_data,
            )
        else:
            mol_eq = molecule

        scf_result = _single_point(mol_eq, method)

        uloc = None

        freqs, vectors = _harmonic_analysis(scf_result, method)
        if localize:
            freqs, vectors, uloc = localize_normal_modes(freqs, vectors, bins=bins)

        grid, gauss_weights = np.polynomial.hermite.hermgauss(n_points)

        placzek = True

        pes_onebody, placzek_onebody = _pes_onemode(
            mol_eq,
            scf_result,
            freqs,
            vectors,
            grid,
            method=method,
            placzek=placzek,
            num_workers=num_workers,
            backend=backend,
            path=path,
            excited_method=excited_method,
            omega_I=omega_I,
            excited_states=excited_states
        )

        # build PES -- two-body
        if placzek_level < 2:
            placzek = False

        pes_twobody, placzek_twobody = _pes_twomode(
            mol_eq,
            scf_result,
            freqs,
            vectors,
            grid,
            pes_onebody,
            placzek_onebody,
            method=method,
            placzek=placzek,
            num_workers=num_workers,
            backend=backend,
            path=path,
            excited_method=excited_method,
            omega_I=omega_I,
            excited_states=excited_states
        )

        pes_data = [pes_onebody, pes_twobody]
        placzek_data = [placzek_onebody, placzek_twobody]

        if cubic:
            if placzek_level < 3:
                placzek = False

            pes_threebody, placzek_threebody = _pes_threemode(
                mol_eq,
                scf_result,
                freqs,
                vectors,
                grid,
                pes_onebody,
                pes_twobody,
                placzek_onebody,
                placzek_twobody,
                method=method,
                placzek=placzek,
                num_workers=num_workers,
                backend=backend,
                path=path,
                excited_method=excited_method,
                omega_I=omega_I,
                excited_states=excited_states
            )

            pes_data = [pes_onebody, pes_twobody, pes_threebody]
            placzek_data = [placzek_onebody, placzek_twobody, placzek_threebody]

        freqs = freqs * CM_TO_AU
        return PolarizablePES(
            freqs, grid, gauss_weights, uloc, pes_data, placzek_data, localize, placzek_level
        )

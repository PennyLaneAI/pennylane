from pennylane import labs
import itertools
from pathlib import Path

from enum import Enum
from typing import Union, Optional

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
HBAR = (
    sp.constants.hbar * (1000 * sp.constants.Avogadro) * (10**20)
)  # kg*(m^2/s) to (amu)*(angstrom^2/s)
BOHR_TO_ANG = (
    sp.constants.physical_constants["Bohr radius"][0] * 1e10
)  # factor to convert bohr to angstrom
CM_TO_AU = 100 / sp.constants.physical_constants["hartree-inverse meter relationship"][0]  # m to cm

import concurrent.futures
from pennylane import concurrency


def _worker_pes_onemode(rank, start, end, molecule, scf_result, freqs, vectors, grid, method="rhf", dipole=False):
    
    quad_order = len(grid)
    nmodes = len(freqs)
    init_geom = molecule.coordinates * BOHR_TO_ANG

    jobs_on_thread = range(start,end+1)
    local_pes_onebody = np.zeros((nmodes, len(jobs_on_thread)), dtype=float)

    if dipole:
        local_dipole_onebody = np.zeros((nmodes, len(jobs_on_thread), 3), dtype=float)
        ref_dipole = _get_dipole(scf_result, method)
    for mode in range(nmodes):
        vec = vectors[mode]
        if (freqs[mode].imag) > 1e-6:
            continue  # pragma: no cover

        for job_idx, job in enumerate(jobs_on_thread):
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
        
        filename = f"v1data_{rank}.hdf5"
        with h5py.File(filename, "w") as f:
            f.create_dataset("V1_PES", data=local_pes_onebody)
            if dipole:
                f.create_dataset("D1_DMS", data=local_dipole_onebody)
            f.close()


   

    if dipole:
        return local_pes_onebody, local_dipole_onebody,
    return local_pes_onebody, None

def _pes_onemode(molecule, scf_result, freqs, vectors, grid, method="rhf", dipole=False, hardware: Union[concurrency.backends.ExecBackends, str] = "mp_pool" ):
    
    print(hardware)
    num_threads = 2
    n = len(grid)
    chunk_size = (n + num_threads - 1) // num_threads

    # Use a list to store results in order
    pes_values = np.zeros(n)
    if dipole :
        dipole_values = np.zeros(n)

    

    chunk_range = range(0, n, chunk_size)
    arguments = [(j, i, min(i + chunk_size, n) - 1, molecule, scf_result, freqs, vectors, grid, method, dipole)
                 for j, i in enumerate(chunk_range)]
    
    executor_class = concurrency.backends.get_executor(hardware)
    executor = executor_class()

    
    results = list(executor.starmap(_worker_pes_onemode, arguments))

        
    pes_onebody = None
    dipole_onebody = None
    pes_onebody, dipole_onebody = new_load_pes_onemode(
        len(chunk_range), len(freqs), len(grid), dipole=dipole)
    current_directory = Path.cwd()
    for file_path in current_directory.glob("v1data*"):
        file_path.unlink(missing_ok=False)

    if dipole:
        return pes_onebody, dipole_onebody

    return pes_onebody, None

def new_load_pes_onemode(num_proc, nmodes, quad_order, dipole=False):
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
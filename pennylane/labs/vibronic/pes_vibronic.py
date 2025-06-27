# Copyright 2025 Xanadu Quantum Technologies Inc.

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

import numpy as np
import scipy as sp

from pathlib import Path
from tempfile import TemporaryDirectory

import pyscf
from pyscf.hessian import thermo

from scipy.spatial.transform import Rotation

from itertools import combinations

from pyscf import gto, scf, mcscf, tdscf, cc
from pyscf.fci import direct_spin0
from pyscf.symm.geom import SymmSys

from pennylane import concurrency, qchem

from pennylane.qchem.vibrational.vibrational_class import (
    VibrationalPES,
    optimize_geometry,
)

from .pes_vibronic_utils import _harmonic_analysis, _grid_points, _generate_1d_grid
from .pes_solver import _run_casscf, _run_tddft, _run_tddft_gpu


# constants
# TODO: Make this code work in atomic units only.
HBAR = (
    sp.constants.hbar * (1000 * sp.constants.Avogadro) * (10**20)
)  # kg*(m^2/s) to (amu)*(angstrom^2/s)
BOHR_TO_ANG = (
    sp.constants.physical_constants["Bohr radius"][0] * 1e10
)  # factor to convert bohr to angstrom
CM_TO_AU = 100 / sp.constants.physical_constants["hartree-inverse meter relationship"][0]  # m to cm


def vibronic_pes(
    molecule,
    n_points=5,
    grid_range=2.0,
    grid_type="uniform",
    method="rhf",
    method_excited="tddft",
    optimize=True,
    rotate=True,
    pes_level=1,
    num_workers=1,
    backend="serial",
):
    r"""Computes potential energy surfaces along vibrational normal modes.

    Args:
        molecule (~qchem.molecule.Molecule): the molecule object
        n_points (int): number of points for computing the potential energy surface. Default value is ``5``.
        grid_range (float): maximum range of points for computing the potential energy surface. Default value is ``2``.
        grid_type (str): method to generate the grid points. Options are ``'uniform'``, ``'gaussian'`` and ``'chebyshev'``.
            Default is ``'uniform'``.
        method (str): Electronic structure method used to perform geometry optimization.
            Available options are ``"rhf"`` and ``"uhf"`` for restricted and unrestricted
            Hartree-Fock, respectively. Default is ``"rhf"``.
        optimize (bool): if ``True`` perform geometry optimization. Default is ``True``.
        rotate(bool): if ``True`` rotate the molecule to align normal modes with symmetry operators
        pes_level (int): the number of coupled modes for generating the potential energy surface data.
            Available options are ``1``, ``2``, ``3`` for one-mode, two-mode and three-mode levels,
            respctively.
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
    >>> pes = qml.qchem.vibronic_pes(mol, optimize=False)
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

        geom = [
            [symbol, tuple(np.array(mol_eq.coordinates)[i])]
            for i, symbol in enumerate(mol_eq.symbols)
        ]
        spin = int((mol_eq.mult - 1) / 2)
        mol = pyscf.gto.Mole(atom=geom, symmetry="C1", spin=spin, charge=mol_eq.charge, unit="Bohr")
        mol.basis = mol_eq.basis_name
        mol.build()

        freqs, vectors = _harmonic_analysis(mol, rotate)

        grid = _grid_points(grid_type, grid_range, n_points)

        eq_geometry = mol_eq.coordinates * BOHR_TO_ANG

        geometry_1d = _generate_1d_grid(freqs, vectors, eq_geometry, grid)

        # geometry_2d = _generate_2d_grid(freqs, vectors, eq_geometry, grid)
        #
        # geometry_3d = _generate_3d_grid(freqs, vectors, eq_geometry, grid)

        

        if method_excited == "casscf":
            energy_1 = []


            arguments_1d = [
            (mol_eq.symbols, i["coordinates"], 2, 2)
            for i in geometry_1d
            ]
            executor_class = concurrency.backends.get_executor(backend)
            with executor_class(max_workers=num_workers) as executor:
                energy_1 = list(executor.starmap(_run_casscf, arguments_1d))

        if method_excited == "tddft":
            energy_1 = []

            
            for geometry_point in geometry_1d:
                new_coords = geometry_point["coordinates"]
                energy_1.append(_run_tddft(mol_eq.symbols, new_coords))

        # energy_2 = []
        # for geometry_point in geometry_2d:
        #     new_coords = geometry_point["coordinates"]
        #     energy_2.append(_run_casscf(mol_eq.symbols, new_coords, ncas=2, nelecas=2))
        #
        # energy_3 = []
        # for geometry_point in geometry_3d:
        #     new_coords = geometry_point["coordinates"]
        #     energy_3.append(_run_casscf(mol_eq.symbols, new_coords, ncas=2, nelecas=2))

        freqs = freqs * CM_TO_AU

        return VibrationalPES(
            freqs,
            grid,
            pes_data=[energy_1],
        )

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

BOHR_TO_ANG = 0.5291772106  # factor to convert bohr to angstrom


def _harmonic_analysis(scf_result, method="rhf"):
    r"""Performs harmonic analysis by evaluating the Hessian using PySCF routines.

    Args:
        scf_result (pyscf.scf object): pyscf object from electronic structure calculations
        method (str): Electronic structure method that can be either restricted and unrestricted
            Hartree-Fock,  ``'rhf'`` and ``'uhf'``, respectively. Default is ``'rhf'``.

    Returns:
        tuple: A tuple containing the following:
         - list[float]: normal mode frequencies in ``cm^-1``
         - TensorLike[float]: corresponding displacement vectors for each normal mode

    """
    pyscf = _import_pyscf()
    from pyscf.hessian import thermo

    method = method.strip().lower()
    if method not in ["rhf", "uhf"]:
        raise ValueError(f"Specified electronic structure method, {method} is not available.")

    hess = getattr(pyscf.hessian, method).Hessian(scf_result).kernel()
    harmonic_res = thermo.harmonic_analysis(scf_result.mol, hess)

    return harmonic_res["freq_wavenumber"], harmonic_res["norm_mode"]


def _single_point(molecule, method="rhf"):
    r"""Runs electronic structure calculation.

    Args:
        molecule (:func:`~pennylane.qchem.molecule.Molecule`): Molecule object.
        method (str): Electronic structure method that can be either restricted and unrestricted
            Hartree-Fock,  ``'rhf'`` and ``'uhf'``, respectively. Default is ``'rhf'``.

    Returns:
        pyscf.scf object from electronic structure calculation

    """
    pyscf = _import_pyscf()

    method = method.strip().lower()
    if method not in ["rhf", "uhf"]:
        raise ValueError(f"Specified electronic structure method, {method}, is not available.")

    geom = [
        [symbol, tuple(np.array(molecule.coordinates)[i])]
        for i, symbol in enumerate(molecule.symbols)
    ]
    spin = int((molecule.mult - 1) / 2)
    mol = pyscf.gto.Mole(atom=geom, symmetry="C1", spin=spin, charge=molecule.charge, unit="Bohr")
    mol.basis = molecule.basis_name
    mol.build()
    if method == "rhf":
        scf_obj = pyscf.scf.RHF(mol).run(verbose=0)
    else:
        scf_obj = pyscf.scf.UHF(mol).run(verbose=0)
    return scf_obj


def _import_geometric():
    """Import geometric."""
    try:
        import geometric
    except ImportError as Error:
        raise ImportError(
            "This feature requires geometric. It can be installed with: pip install geometric."
        ) from Error

    return geometric


def optimize_geometry(molecule, method="rhf"):
    r"""Computes the equilibrium geometry of a molecule.

    Args:
        molecule (:func:`~pennylane.qchem.molecule.Molecule`): Molecule object
        method (str): Electronic structure method that can be either restricted and unrestricted
            Hartree-Fock,  ``'rhf'`` and ``'uhf'``, respectively. Default is ``'rhf'``.

    Returns:
        tuple: A tuple containing the following:
         - :func:`~pennylane.qchem.molecule.Molecule` object with optimized geometry
         - pyscf.scf object

    """
    pyscf = _import_pyscf()
    geometric = _import_geometric()
    from pyscf.geomopt.geometric_solver import optimize

    scf_res = _single_point(molecule, method)
    geom_eq = optimize(scf_res, maxsteps=100)

    mol_eq = qml.qchem.Molecule(
        molecule.symbols,
        geom_eq.atom_coords(unit="B"),
        unit="Bohr",
        basis_name=molecule.basis_name,
        charge=molecule.charge,
        mult=molecule.mult,
        load_data=molecule.load_data,
    )

    scf_result = _single_point(mol_eq, method)
    return mol_eq, scf_result

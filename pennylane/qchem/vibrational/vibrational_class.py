"""This module contains functions and classes to generate a pes object.
This object stores all the necessary information to construct
vibrational Hamiltonian for a given molecule."""

import numpy as np
import pyscf
from pyscf.geomopt.geometric_solver import optimize
from pyscf.hessian import thermo

import pennylane as qml

BOHR_TO_ANG = 0.529177


def harmonic_analysis(scf_result, method="rhf"):
    r"""Performs harmonic analysis by evaluating the Hessian using PySCF routines.
    Args:
       scf_result: pyscf object from electronic structure calculations
       method: Electronic structure method to define the level of theory
            for harmonic analysis. Default is restricted Hartree-Fock 'rhf'.

    Returns:
       pyscf object containing information about the harmonic analysis
    """

    method = method.strip().lower()
    if method not in ["rhf", "uhf"]:
        raise ValueError(f"Specified electronic structure method, {method} is not available.")

    hess = getattr(pyscf.hessian, method).Hessian(scf_result).kernel()

    harmonic_res = thermo.harmonic_analysis(scf_result.mol, hess)
    return harmonic_res


def single_point(molecule, method="rhf"):
    r"""Runs electronic structure calculation.
    Args:
      molecule: Molecule object.
      method: Electronic structure method to define the level of theory.
              Default is restricted Hartree-Fock 'rhf'.

    Returns:
      pyscf object from electronic structure calculation
    """
    method = method.strip().lower()
    if method not in ["rhf", "uhf"]:
        raise ValueError(f"Specified electronic structure method, {method}, is not available.")

    geom = [
        [symbol, tuple(np.array(molecule.coordinates)[i] * BOHR_TO_ANG)]
        for i, symbol in enumerate(molecule.symbols)
    ]

    spin = int((molecule.mult - 1) / 2)
    mol = pyscf.gto.Mole(
        atom=geom, symmetry="C1", spin=spin, charge=molecule.charge, unit="Angstrom"
    )
    mol.basis = molecule.basis_name
    mol.build()
    if method == "rhf":
        scf_obj = pyscf.scf.RHF(mol).run(verbose=0)
    else:
        scf_obj = pyscf.scf.UHF(mol).run(verbose=0)
    return scf_obj


def optimize_geometry(molecule, method="rhf"):
    r"""Obtains equilibrium geometry for the molecule.

    Args:
      molecule: Molecule object.
      method: Electronic structure method to define the level of theory.
              Default is restricted Hartree-Fock 'rhf'.

    Returns:
      molecule object with optimized geometry

    """

    scf_res = single_point(molecule, method)
    geom_eq = optimize(scf_res, maxsteps=100)

    mol_eq = qml.qchem.Molecule(
        molecule.symbols,
        geom_eq.atom_coords(unit="B"),
        unit="bohr",
        basis_name=molecule.basis_name,
        charge=molecule.charge,
        mult=molecule.mult,
        load_data=molecule.load_data,
    )

    scf_result = single_point(mol_eq, method)

    return mol_eq, scf_result

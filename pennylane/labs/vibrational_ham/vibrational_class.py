import numpy as np
import pennylane as qml
import pyscf
from pyscf import scf
from pyscf.geomopt.geometric_solver import optimize
from utils import *
import h5py
import sys, os, subprocess
from time import time
from mpi4py import MPI 
from dataclasses import dataclass
au_to_cm = 219475
bohr_to_ang = 0.529177
orig_stdout = sys.stdout


@dataclass
class VibrationalPES():
    r"""Data class to save the PES information to an object"""
    def __init__(self, freqs, gauss_grid, gauss_weights, uloc, pes_arr, dipole_arr, localized=True, get_anh_dipole=2):
        self.freqs = freqs
        self.gauss_grid = gauss_grid
        self.gauss_weights = gauss_weights
        self.uloc = uloc
        self.pes_onebody = pes_arr[0]
        self.pes_twobody = pes_arr[1]
        self.pes_threebody = pes_arr[2] if len(pes_arr)>2 else None
        self.dipole_onebody = dipole_arr[0]
        self.dipole_twobody = dipole_arr[1] if len(dipole_arr)>1 else None
        self.dipole_threebody = dipole_arr[2] if len(dipole_arr)>2 else None
        self.localized = localized
        self.get_anh_dipole = get_anh_dipole


def harmonic_analysis(scf_result, method):
    r"""Performs harmonic analysis by evaluating the Hessian using PySCF routines.
    Args:
       scf_result: pyscf object from electronic structure calculations.
       method: Electronic structure method to define the level of theory
            for harmonic analysis.
        
    Returns:
       pyscf object containing information about the the harmonic analysis.
    """
    from pyscf.hessian import thermo

    method = method.strip().lower()
    if method not in ["rhf", "uhf"]:
        raise ValueError(f"Specified electronic structure method {method} is not defined.")
    
    method = method.strip().lower()
    if method == 'rhf':
        hess = pyscf.hessian.rhf.Hessian(scf_result).kernel()
    elif method == 'uhf':
        hess = pyscf.hessian.uhf.Hessian(scf_result).kernel()
        
    harmonic_res = thermo.harmonic_analysis(scf_result.mol, hess)
        
    return harmonic_res


def single_point(molecule, method="rhf"):
    r"""Runs electronic structure calculation.
    Args:
      molecule: Molecule object.
      method: Electronic structure method. Default is restricted Hartree-Fock 'rhf'.

    Returns:
      pyscf object.
    """
    method = method.strip().lower()
    if method not in ["rhf", "uhf"]:
        raise ValueError(f"Specified electronic structure method {method} is not defined.")
        
    geom = [
        [symbol, tuple(np.array(molecule.coordinates)[i]*bohr_to_ang)]
        for i, symbol in enumerate(molecule.symbols)
    ]
            
    spin = (molecule.mult - 1)/2
    mol = pyscf.gto.Mole(atom=geom, symmetry="C1", spin=spin, charge=molecule.charge, unit="Angstrom")
    mol.basis = molecule.basis_name
    mol.build()
    if method == 'rhf':
        return pyscf.scf.RHF(mol).run(verbose=0)
    elif method == 'uhf':
        return pyscf.scf.UHF(mol).run(verbose=0)


def optimize_geometry(molecule, method):
    r"""Obtains equilibrium geometry for the molecule.

    Args:
      molecule: Molecule object.
      method: Electronic structure method. Default is restricted Hartree-Fock 'rhf'.

    Returns:
      molecule object at Equilibrium geometry.
      pyscf object.
      
    """
    
    scf_res = single_point(molecule, method)
    geom_eq = optimize(scf_res, maxsteps=100)
    print("Geometry after optimization: ", geom_eq.atom_coords(unit='A'))

    mol_eq = qml.qchem.Molecule(molecule.symbols, geom_eq.atom_coords(unit='A'), unit="angstrom", basis_name=molecule.basis_name, charge=molecule.charge, mult=molecule.mult, load_data=molecule.load_data)

    scf_result = single_point(mol_eq, method)
    return mol_eq, scf_result




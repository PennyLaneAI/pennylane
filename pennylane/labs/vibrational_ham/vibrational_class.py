import numpy as np
import pennylane as qml
import pyscf
from pyscf import scf
from pyscf.geomopt.geometric_solver import optimize
from utils import *
import h5py
import sys, os, subprocess
from localize_modes import pm_custom_separate_localization
from time import time
from mpi4py import MPI 
from dataclasses import dataclass
au_to_cm = 219475
bohr_to_ang = 0.529177
orig_stdout = sys.stdout


@dataclass
class PES():
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
    # def save_pes(self, do_cubic=True, get_anh_dipole=2, savename="data_pes"):
    #     if rank == 0:
    #         if get_anh_dipole < 2 or get_anh_dipole is False:
    #             do_dip_2 = False
    #         elif get_anh_dipole > 1 or get_anh_dipole is True:
    #             do_dip_2 = True    
                
    #         if get_anh_dipole < 3 or get_anh_dipole is False:
    #             do_dip_3 = False
    #         elif get_anh_dipole > 2 or get_anh_dipole is True:
    #             do_dip_3 = True    

    #         f = h5py.File(savename + '.hdf5', 'w')
    #         f.create_dataset('V1_PES',data=self.pes_onebody)
    #         f.create_dataset('D1_DMS',data=self.dipole_onebody)
    #         f.create_dataset('GH_quad_order', data=self.quad_order)
    #         f.create_dataset('V2_PES',data=self.pes_twobody)
    #         if do_dip_2:
    #             f.create_dataset('D2_DMS',data=self.dipole_twobody)

    #         if do_cubic:
    #             f.create_dataset('V3_PES',data=self.pes_threebody)
    #             if do_dip_3:
    #                 f.create_dataset('D3_DMS',data=self.dipole_threebody)
    #         f.close()
    
    # def plot_pes_onebody(self, do_dipole=True, save_dir="plots"):
    #     fig, ax = plt.subplots()
    #     ax.set_xlim((-2,2))
    #     ax.plot(self.gauss_grid, self.pes_onebody[ii,:], label='PES')
    #     ax.plot(self.gauss_grid, self.harmonic_pes[ii, :], label='Harmonic')
    #     ax.legend()
    #     ymax = ho_const * 4
    #     ax.set_ylim((0,ymax))
    #     fig.savefig(f"plots/pes_onebody_{ii}.png", format="png")
    #     plt.close()
    #     if do_dipole:
    #         fig, ax = plt.subplots()
    #         ax.plot(self.gauss_grid, self.dipole_onebody[ii,:,0], label='d_x')
    #         ax.plot(self.gauss_grid, self.dipole_onebody[ii,:,1], label='d_y')
    #         ax.plot(self.gauss_grid, self.dipole_onebody[ii,:,2], label='d_z')
    #         ax.legend()
    #         fig.savefig(f"plots/dipole_onebody_{ii}.png", format="png")
    #     plt.close()

    # def plot_pes_twobody(self, do_dipole=True):
    #     fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    #     ax.plot_wireframe(gridx, gridy, pes_twobody[aa,bb,:,:])
    #     fig.savefig(f"plots/pes_twobody_{aa,bb}.png", format="png")
    #     plt.close()
    #     if do_dipole:
    #         fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    #         ax.plot_wireframe(gridx, gridy, self.dipole_twobody[aa,bb,:,:,0])
    #         fig.savefig(f"plots/dipole_x_twobody_{aa,bb}.png", format="png")
    #         plt.close()
    #         fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    #         ax.plot_wireframe(gridx, gridy, self.dipole_twobody[aa,bb,:,:,1])
    #         fig.savefig(f"plots/dipole_y_twobody_{aa,bb}.png", format="png")
    #         plt.close()
    #         fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    #         ax.plot_wireframe(gridx, gridy, self.dipole_twobody[aa,bb,:,:,2])
    #         fig.savefig(f"plots/dipole_z_twobody_{aa,bb}.png", format="png")
    #         plt.close()
        

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


def run_electronic_structure(molecule, method="rhf"):
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


def equilibrium_geom(molecule, method):
    r"""Obtains equilibrium geometry for the molecule.

    Args:
      molecule: Molecule object.
      method: Electronic structure method. Default is restricted Hartree-Fock 'rhf'.

    Returns:
      molecule object at Equilibrium geometry.
      pyscf object.
      
    """
    
    scf_res = run_electronic_structure(molecule, method)
    geom_eq = optimize(scf_res, maxsteps=100)
    print("Geometry after optimization: ", geom_eq.atom_coords(unit='A'))

    mol_eq = qml.qchem.Molecule(molecule.symbols, geom_eq.atom_coords(unit='A'), unit="angstrom", basis_name=molecule.basis_name, charge=molecule.charge, mult=molecule.mult, load_data=molecule.load_data)

    scf_result = run_electronic_structure(mol_eq, method)
    return mol_eq, scf_result




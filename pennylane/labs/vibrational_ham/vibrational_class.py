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

au_to_cm = 219475
bohr_to_ang = 0.529177
orig_stdout = sys.stdout

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
        
class Build_PES:
    r"""Builds potential energy surfaces over the normal modes.

    Args:
       molecule: molecule object.
       quad_order: order for Gauss-Hermite quadratures
       method: electronic structure method. Default is restricted Hartree-Fock 'RHF'.
       localize: perform normal mode localization.
       loc_freqs: array of frequencies (in cm-1) where separation happens for mode localization.
       do_cubic: use True to include three-mode couplings
       get_anh_dipole: True for also obtaining anharmonic matrix elements for molecular dipole, takes considerable time. If integer then gets up to that degree of anharmonic dipole


    - saving: whether HDF5 files are created storing the calculated quantities
        - savename: default name for save files, will generate one from atom list if none is specified

        - do_taylor: whether Taylor expansion of potentials is done
        - taylor_deg: if do_taylor is True, then sets degree for Taylor expansion
        - do_cform_cubic: whether the C-form calculations are done for 3-mode couplings. Requires do_cubic and do_cform to be both True
        - load_pes: set True for skipping PES calculation and loading from previously generated run
    """

    def __init__(self, molecule, quad_order=17, method="RHF", localize=True, loc_freqs=[2600], do_cubic=True, get_anh_dipole = 2):

        self.molecule = molecule
        self.quad_order = quad_order
        self.method = method.strip().lower()
        if self.method not in ["rhf", "uhf"]:
            raise ValueError(f"Specified electronic structure method {method} is not defined.")
        self.localize = localize

        self.scf_result, self.molecule = self.get_equilibrium_geom()

        harmonic_res = None
        loc_res = None
        uloc = None
        displ_vecs = None
        if rank == 0:
            harmonic_res = self.harmonic_analysis()
            displ_vecs = harmonic_res["norm_mode"]
            if self.localize:
                loc_res, uloc, displ_vecs = pm_custom_separate_localization(harmonic_res, freq_separation=loc_freqs)

        # Broadcast data to all threads
        self.harmonic_res = comm.bcast(harmonic_res, root=0)
        self.displ_vecs = np.array(comm.bcast(displ_vecs, root=0))
        self.loc_res = comm.bcast(loc_res, root=0)
        self.uloc = np.array(comm.bcast(uloc, root=0))
        
        comm.Barrier()

        self.freqs = self.loc_res['freq_wavenumber'] / au_to_cm

        self.gauss_grid, self.gauss_weights = np.polynomial.hermite.hermgauss(quad_order)

        self.pes_onebody, self.dipole_onebody = self.get_pes_onebody(do_dipole=True)
        comm.Barrier()

        # build PES -- two-body
        if get_anh_dipole < 2 or get_anh_dipole is False:
            do_dip_2 = False
        elif get_anh_dipole > 1 or get_anh_dipole is True:
            do_dip_2 = True
        self.pes_twobody, self.dipole_twobody = self.get_pes_twobody(do_dipole=do_dip_2)
        comm.Barrier()

        # build PES -- three-body
        if do_cubic:

            if get_anh_dipole < 3 or get_anh_dipole is False:
                do_dip_3 = False
            elif get_anh_dipole > 2 or get_anh_dipole is True:
                do_dip_3 = True

            self.pes_threebody = None            
            local_pes_threebody, local_dipole_threebody = self.get_pes_threebody(do_dipole=do_dip_2)
            comm.Barrier()

            f = h5py.File("v3data" + f"_{rank}" + '.hdf5', 'w')
            f.create_dataset('V3_PES',data=local_pes_threebody)
            if do_dip_3:
                self.dipole_threebody = None
                f.create_dataset('D3_DMS',data=local_dipole_threebody)
            f.close()
            comm.Barrier()
            
            if rank==0:
                self.pes_threebody = load_pes_threebody(comm.Get_size(), len(self.freqs), len(self.gauss_grid))
                if do_dip_3:
                    self.dipole_threebody = load_dipole_threebody(comm.Get_size(), len(self.freqs), len(self.gauss_grid))
                
                process = subprocess.Popen('rm ' + 'v3data*', stdout=subprocess.PIPE, shell=True)
                output, error = process.communicate()
            comm.Barrier()
            self.pes_threebody = comm.bcast(self.pes_threebody, root=0)
            if do_dip_3:
                self.dipole_threebody = comm.bcast(self.dipole_threebody, root=0)            
    def run_electronic_structure(self, molecule):
        r"""Runs electronic structure calculation"""

        geom = [
            [symbol, tuple(np.array(molecule.coordinates)[i])]
            for i, symbol in enumerate(molecule.symbols)
        ]
            
        spin = (molecule.mult - 1)/2
        mol = pyscf.gto.Mole(atom=geom, symmetry="C1", spin=spin, charge=molecule.charge, unit="bohr")
        mol.basis = molecule.basis_name
        mol.build()
        if self.method == 'rhf':
            return pyscf.scf.RHF(mol).run(verbose=0)
        elif self.method == 'uhf':
            return pyscf.scf.UHF(mol).run(verbose=0)
        
    def get_equilibrium_geom(self):
        r"""Obtains equilibrium geometry for the molecule"""

        scf_res = self.run_electronic_structure(self.molecule)
        geom_eq = optimize(scf_res, maxsteps=100)
        print("Geometry after optimization: ", geom_eq.atom_coords(unit='A'))
        mol_eq = qml.qchem.Molecule(self.molecule.symbols, geom_eq.atom_coords(unit='A'), unit="angstrom", 
                                    basis_name=self.molecule.basis_name, charge=self.molecule.charge, mult=self.molecule.mult, load_data=True)

        scf_result = self.run_electronic_structure(mol_eq)
        return scf_result, mol_eq

    def harmonic_analysis(self):
        r"""Performs harmonic analysis by evaluating the Hessian using PySCF routines. 
        Returns the results, as well as displacement vectors.
    """
        from pyscf.hessian import thermo

        if self.method == 'rhf':
            hess = pyscf.hessian.rhf.Hessian(self.scf_result).kernel()
        elif self.method == 'uhf':
            hess = pyscf.hessian.uhf.Hessian(self.scf_result).kernel()

        harmonic_res = thermo.harmonic_analysis(self.scf_result.mol, hess)
#        red_mass = self.scf_result.mol.atom_mass_list(isotope_avg=True)
    
        return harmonic_res

    def get_pes_onebody(self, do_dipole=False):
        r"""Computes the one-mode potential energy surface on a grid in real space, along the normal coordinate directions (or any directions set by the displ_vecs).
        Simultaneously, can compute the dipole one-body elements."""

        freqs = self.freqs * au_to_cm
        quad_order = len(self.gauss_grid)
        nmodes = len(freqs)
        init_geom = self.scf_result.mol.atom

        pes_onebody = np.zeros((nmodes, quad_order), dtype=float)
        harmonic_pes = np.zeros_like(pes_onebody)

        if do_dipole:
            dipole_onebody = np.zeros((nmodes, quad_order, 3), dtype=float)
            ref_dipole = get_dipole(self.scf_result, self.method)

        local_pes_onebody = np.zeros((quad_order), dtype=float)
        local_harmonic_pes = np.zeros_like(local_pes_onebody)
        if do_dipole:
            local_dipole_onebody = np.zeros((quad_order, 3), dtype=float)
            ref_dipole = get_dipole(self.scf_result, self.method)
    
        for ii in tqdm(range(len(self.displ_vecs)), desc='Loop one-body pes'):
            displ_vec = self.displ_vecs[ii]
            # imaginary frequency check
            if (freqs[ii].imag) > 1e-6:
                continue

            jobs_on_rank = np.array_split(range(quad_order), size)[rank]

            for jj in jobs_on_rank:
                pt = self.gauss_grid[jj]
                # numerical scaling out front to shrink region
                scaling = np.sqrt( hbar / (2 * np.pi * freqs[ii] * 100 * c_light))
                positions = np.array([np.array(init_geom[ll][1])*bohr_to_ang + \
                              scaling * pt * displ_vec[ll,:] \
                              for ll in range(self.scf_result.mol.natm)])

                disp_mol = qml.qchem.Molecule(self.molecule.symbols,
                                              positions,
                                              basis_name=self.molecule.basis_name,
                                              charge=self.molecule.charge,
                                              mult=self.molecule.mult,
                                              unit="angstrom",
                                              load_data=True)

                disp_hf = self.run_electronic_structure(disp_mol)

                omega = self.freqs[ii]
                ho_const = omega / 2
                local_harmonic_pes[jj] = ho_const * (pt**2)

                local_pes_onebody[jj] = disp_hf.e_tot - self.scf_result.e_tot  
                if do_dipole:
                    local_dipole_onebody[jj,:] = get_dipole(disp_hf, self.method) - ref_dipole

            # gather the results on head process only
            pes_onebody[ii,:] = np.sum(np.array(comm.gather(local_pes_onebody)), axis=0)
            harmonic_pes[ii,:] = np.sum(np.array(comm.gather(local_harmonic_pes)), axis=0)
            if do_dipole:
                dipole_onebody[ii,:] = np.sum(np.array(comm.gather(local_dipole_onebody)), axis=0)

        # broadcast the result to everybody
        final_pes_onebody = np.array(comm.bcast(pes_onebody, root=0))
        if do_dipole:
            final_dipole_onebody = np.array(comm.bcast(dipole_onebody, root=0))

        if do_dipole:
            return final_pes_onebody, final_dipole_onebody
        else:
            return final_pes_onebody, None

    def get_pes_twobody(self, do_dipole=False):
        """
        Computes the two-mode potential energy surface on a grid in real space,
        along the normal coordinate directions (or any directions set by the 
        displ_vecs).
        """
        freqs = self.freqs * au_to_cm
        quad_order = len(self.gauss_grid)
        nmodes = len(freqs)
        init_geom = self.scf_result.mol.atom

        pes_twobody = np.zeros((nmodes, nmodes, quad_order, quad_order), dtype=float)
        local_pes_twobody = np.zeros((quad_order, quad_order), dtype=float)
        gridx, gridy = np.meshgrid(self.gauss_grid, self.gauss_grid)

        if do_dipole:
            dipole_twobody = np.zeros((nmodes, nmodes, quad_order, quad_order, 3), dtype=float)
            local_dipole_twobody = np.zeros((quad_order, quad_order, 3), dtype=float)
            ref_dipole = get_dipole(self.scf_result, self.method)

        for aa in tqdm(range(len(self.displ_vecs)), desc='Outer Loop two-body pes'):
            displ_vec_a = self.displ_vecs[aa]
            scaling_a = np.sqrt( hbar / (2 * np.pi * freqs[aa] * 100 * c_light))

            for bb in tqdm(range(len(self.displ_vecs)), desc='Inner Loop two-body pes'):
                # skip the pieces that are not part of the Hamiltonian
                if bb >= aa:
                    continue

                displ_vec_b = self.displ_vecs[bb]
                # imaginary frequency check
                if (freqs[aa].imag) > 1e-6 or (freqs[bb].imag) > 1e-6:
                    continue

                scaling_b = np.sqrt( hbar / (2 * np.pi * freqs[bb] * 100 * c_light))

                all_jobs = []
                for ii, pt1 in enumerate(self.gauss_grid):
                    for jj, pt2 in enumerate(self.gauss_grid):
                        all_jobs.append([ii, pt1, jj, pt2])

                jobs_on_rank = np.array_split(all_jobs, size)[rank]

                for [ii, pt1, jj, pt2] in jobs_on_rank:

                    ii, jj = int(ii), int(jj)
                    positions = np.array([np.array(init_geom[ll][1])*bohr_to_ang + \
                                  scaling_a * pt1 * displ_vec_a[ll,:] + \
                                  scaling_b * pt2 * displ_vec_b[ll,:] \
                                  for ll in range(self.scf_result.mol.natm)])
                    disp_mol = qml.qchem.Molecule(self.molecule.symbols, positions, basis_name=self.molecule.basis_name, charge=self.molecule.charge, mult=self.molecule.mult, unit="angstrom", load_data=True)
                    disp_hf = self.run_electronic_structure(disp_mol)
                    local_pes_twobody[ii, jj] = disp_hf.e_tot - self.pes_onebody[aa, ii] - self.pes_onebody[bb, jj] - self.scf_result.e_tot
                    
                    if do_dipole:
                        local_dipole_twobody[ii,jj,:] = get_dipole(disp_hf, self.method) - self.dipole_onebody[aa,ii,:] -  self.dipole_onebody[bb,jj,:] - ref_dipole

                # gather the results on head process only
                pes_twobody[aa,bb,:,:] = np.sum(np.array(comm.gather(local_pes_twobody, root=0)), axis=0)
                if do_dipole:
                    dipole_twobody[aa,bb,:,:] = np.sum(np.array(comm.gather(local_dipole_twobody, root=0)), axis=0)

        # broadcast the result to everybody
        final_pes_twobody = np.array(comm.bcast(pes_twobody, root=0))
        if do_dipole:
            final_dipole_twobody = np.array(comm.bcast(dipole_twobody, root=0))
            
        if do_dipole:
            return final_pes_twobody, final_dipole_twobody
        else:
            return final_pes_twobody, None


    def get_pes_threebody(self, do_dipole=False):
        """
        Computes the three-mode potential energy surface on a grid in real space,
        along the normal coordinate directions (or any directions set by the 
        displ_vecs).
        """
        freqs = self.freqs * au_to_cm
        quad_order = len(self.gauss_grid)
        init_geom = self.scf_result.mol.atom
        nmodes = len(freqs)

        all_mode_combos = []
        for aa in range(len(self.displ_vecs)):
            for bb in range(len(self.displ_vecs)):
                for cc in range(len(self.displ_vecs)):
                    all_mode_combos.append([aa, bb, cc])

        all_bos_combos = []
        for ii, pt1 in enumerate(self.gauss_grid):
            for jj, pt2 in enumerate(self.gauss_grid):
                for kk, pt3 in enumerate(self.gauss_grid):
                    all_bos_combos.append([ii, pt1, jj, pt2, kk, pt3])

        boscombos_on_rank = np.array_split(all_bos_combos, size)[rank]
        local_pes_threebody = np.zeros(len(all_mode_combos)*len(boscombos_on_rank))
        if do_dipole:
            local_dipole_threebody = np.zeros((len(all_mode_combos)*\
                                        len(boscombos_on_rank), 3), dtype=float)
            ref_dipole = get_dipole(self.scf_result, self.method)

        ll = 0
        for [aa, bb, cc] in tqdm(all_mode_combos, desc = "Outer loop three-body pes"):

            aa, bb, cc = int(aa), int(bb), int(cc)
            # skip the ones that are not needed
            if bb >= aa or cc >= bb:
                ll += 1
                continue

            # imaginary frequency check
            if (freqs[aa].imag) > 1e-6 or (freqs[bb].imag) > 1e-6 or (freqs[cc].imag) > 1e-6:
                ll += 1
                continue

            displ_vec_a = self.displ_vecs[aa]
            scaling_a = np.sqrt( hbar / (2 * np.pi * freqs[aa] * 100 * c_light))
            
            displ_vec_b = self.displ_vecs[bb]
            scaling_b = np.sqrt( hbar / (2 * np.pi * freqs[bb] * 100 * c_light))

            displ_vec_c = self.displ_vecs[cc]
            scaling_c = np.sqrt( hbar / (2 * np.pi * freqs[cc] * 100 * c_light))

            mm = 0
            for [ii, pt1, jj, pt2, kk, pt3] in tqdm(boscombos_on_rank, desc="Inner loop three-body pes"):

                ii, jj, kk = int(ii), int(jj), int(kk)

                positions = np.array([ np.array(init_geom[ll][1])*bohr_to_ang + \
                              scaling_a * pt1 * displ_vec_a[ll,:] + \
                              scaling_b * pt2 * displ_vec_b[ll,:] + \
                              scaling_c * pt3 * displ_vec_c[ll,:]
                                          for ll in range(self.scf_result.mol.natm)])
                disp_mol = qml.qchem.Molecule(self.molecule.symbols, positions, basis_name=self.molecule.basis_name, charge=self.molecule.charge, mult=self.molecule.mult, unit="angstrom", load_data=True)
                disp_hf = self.run_electronic_structure(disp_mol)

                ind = ll*len(boscombos_on_rank) + mm
                local_pes_threebody[ind] = disp_hf.e_tot - self.pes_twobody[aa,bb,ii,jj] - self.pes_twobody[aa,cc,ii,kk] -\
                    self.pes_twobody[bb,cc,jj,kk] - self.pes_onebody[aa, ii] - self.pes_onebody[bb, jj] - self.pes_onebody[cc,kk] - self.scf_result.e_tot
                if do_dipole:
                    local_dipole_threebody[ind,:] = get_dipole(disp_hf, self.method) - self.dipole_twobody[aa,bb,ii,jj,:] - self.dipole_twobody[aa,cc,ii,kk,:] - \
                        self.dipole_twobody[bb,cc,jj,kk,:] - self.dipole_onebody[aa,ii,:] -  self.dipole_onebody[bb,jj,:] - self.dipole_onebody[cc,kk,:] - ref_dipole
                mm += 1
            
            ll += 1

        if do_dipole:
            return local_pes_threebody, local_dipole_threebody
        else:
            return local_pes_threebody, None

    def save_pes(self, do_cubic=True, get_anh_dipole=2, savename="data_pes"):
        if rank == 0:
            if get_anh_dipole < 2 or get_anh_dipole is False:
                do_dip_2 = False
            elif get_anh_dipole > 1 or get_anh_dipole is True:
                do_dip_2 = True    
                
            if get_anh_dipole < 3 or get_anh_dipole is False:
                do_dip_3 = False
            elif get_anh_dipole > 2 or get_anh_dipole is True:
                do_dip_3 = True    

            f = h5py.File(savename + '.hdf5', 'w')
            f.create_dataset('V1_PES',data=self.pes_onebody)
            f.create_dataset('D1_DMS',data=self.dipole_onebody)
            f.create_dataset('GH_quad_order', data=self.quad_order)
            f.create_dataset('V2_PES',data=self.pes_twobody)
            if do_dip_2:
                f.create_dataset('D2_DMS',data=self.dipole_twobody)

            if do_cubic:
                f.create_dataset('V3_PES',data=self.pes_threebody)
                if do_dip_3:
                    f.create_dataset('D3_DMS',data=self.dipole_threebody)
            f.close()
    
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
        


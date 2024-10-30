import pennylane as qml
from pennylane.labs import vibrational_ham
import numpy as np
from mpi4py import MPI 
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()



sym = ["H", "F"]
geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

#sym = ["C", "O"]
#geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

sym = ["H", "H", "S"]
geometry = np.array([[0.0,-1.0,-1.0], [0.0,1.0,-1.0], [0.0,0.0,0.0]])
load_data = True
localize = True


## Generate PES object
molecule = qml.qchem.Molecule(sym, geometry, basis_name="6-31g", unit="Angstrom", load_data=load_data)
pes = vibrational_ham.vibrational_pes(molecule, quad_order=9, localize=localize, do_cubic=True, get_anh_dipole=3)

if rank == 0:
    print("onebody: ", pes.pes_onebody, pes.dipole_onebody)
    print("twobody: ", pes.pes_twobody, pes.dipole_twobody)
    print("threebody: ", pes.pes_threebody, pes.dipole_threebody)


## Generate Real-space/Taylor Hamiltonian and dipole
# if rank == 0:
#     if localize:
#         min_deg = 2
#     else:
#         min_deg = 3
#     t_ham = vibrational_ham.taylor_integrals(pes, min_deg=min_deg)
#     t_dipole = vibrational_ham.taylor_integrals_dipole(pes, min_deg=min_deg)


#exit()
## Generate Christiansen Hamiltonian and dipole
c_ham = vibrational_ham.christiansen_integrals(pes, nbos=4, do_cubic=True)
c_dipole = vibrational_ham.christiansen_integrals_dipole(pes, nbos=9)
## Generate vibrational Hamiltonian
ham = vibrational_ham.christiansen_bosonic(one=c_ham[0], two=c_ham[1])

## Generate vibrational Dipole
dipole_x = vibrational_ham.christiansen_bosonic(one=c_dipole[0][0,:,:,:], two=c_dipole[1][0,:,:,:,:,:,:])
dipole_y = vibrational_ham.christiansen_bosonic(one=c_dipole[0][1,:,:,:], two=c_dipole[1][1,:,:,:,:,:,:])
dipole_z = vibrational_ham.christiansen_bosonic(one=c_dipole[0][2,:,:,:], two=c_dipole[1][2,:,:,:,:,:,:])

## Jordan-Wigner transformed Hamiltonian
ham_jw = vibrational_ham.christiansen_mapping(ham)

# Bosonic Hamiltonian
#ham = bosonic_form(pes)
## Standard binary mapped Hamiltonian
#ham_sb = binary_mapping(ham, d=4)
#print(ham_sb)

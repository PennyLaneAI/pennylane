import pennylane as qml
import pes_generator
import numpy as np
from christiansenForm import christiansen_ham, christiansen_dipole
from real_space_ham import realspace_ham_coeff, realspace_dipole_coeff
from vib_observables import *
from bosonic_mapping import *
from mpi4py import MPI 
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()



sym = ["H", "F"]
geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

#sym = ["C", "O"]
#geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

#sym = ["H", "H", "S"]
#geometry = np.array([[0.0,-1.0,-1.0], [0.0,1.0,-1.0], [0.0,0.0,0.0]])
load_data = True
localize = True


## Generate PES object
molecule = qml.qchem.Molecule(sym, geometry, basis_name="6-31g", unit="Angstrom", load_data=load_data)
pes = pes_generator.vibrational(molecule, quad_order=9, localize=localize, do_cubic=True, get_anh_dipole=2)

if rank == 0:
    print("onebody: ", pes.pes_onebody, pes.dipole_onebody)
    print("twobody: ", pes.pes_twobody, pes.dipole_twobody)
    print("threebody: ", pes.pes_threebody, pes.dipole_threebody)


## Generate Real-space/Taylor Hamiltonian and dipole
if rank == 0:
    if localize:
        min_deg = 2
    else:
        min_deg = 3
    t_ham = realspace_ham_coeff(pes, min_deg=min_deg)
    t_dipole = realspace_dipole_coeff(pes, min_deg=min_deg)


exit()
## Generate Christiansen Hamiltonian and dipole
c_ham = christiansen_ham(pes, nbos=4, do_cubic=False)
c_dipole = christiansen_dipole(pes, nbos=4)

## Generate vibrational Hamiltonian
_, ham = vib_obs(one=c_ham[0], two=c_ham[1])

## Jordan-Wigner transformed Hamiltonian
ham_jw = jordan_wigner(ham)
print(ham_jw)


# Bosonic Hamiltonian
#ham = bosonic_form(pes)
## Standard binary mapped Hamiltonian
#ham_sb = binary_mapping(ham, d=4)
#print(ham_sb)

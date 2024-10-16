import pennylane as qml
import pes_generator
import numpy as np
from christiansenForm import christiansen_ham, christiansen_dipole
from vib_observables import *
from bosonic_mapping import *

sym = ["H", "F"]
geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

#sym = ["C", "O"]
#geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

#sym = ["H", "H", "S"]
#geometry = np.array([[0.0,-1.0,-1.0], [0.0,1.0,-1.0], [0.0,0.0,0.0]])
load_data = True

## Generate PES object
molecule = qml.qchem.Molecule(sym, geometry, basis_name="6-31g", unit="Angstrom", load_data=load_data)
pes = pes_generator.vibrational(molecule, quad_order=17, do_cubic=True, get_anh_dipole=2)

## Generate Christiansen Hamiltonian and dipole
c_ham = christiansen_ham(pes, nbos=4, do_cubic=False)
c_dipole = christiansen_dipole(pes, nbos=4)

## Generate vibrational Hamiltonian
_, ham = vib_obs(one=c_ham[0], two=c_ham[1])

## Jordan-Wigner transformed Hamiltonian
ham_jw = jordan_wigner(ham)
print(ham_jw)

## Standard binary mapped Hamiltonian
ham_sb = binary_mapping(ham)
print(ham_sb)

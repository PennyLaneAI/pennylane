import pennylane as qml
from vibrational_class import Build_PES
import numpy as np
from christiansenForm import get_christiansen_form

sym = ["H", "F"]
geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

sym = ["C", "O"]
geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

sym = ["H", "H", "S"]
geometry = np.array([[0.0,-1.0,-1.0], [0.0,1.0,-1.0], [0.0,0.0,0.0]])
load_data = True

molecule = qml.qchem.Molecule(sym, geometry, basis_name="6-31g", unit="Angstrom", load_data=load_data)
pes = Build_PES(molecule, quad_order=17, do_cubic=False)
pes.save_pes(do_cubic=False, get_anh_dipole=2, savename="data_pes")
get_christiansen_form(pes, do_cubic=False, nbos=8)


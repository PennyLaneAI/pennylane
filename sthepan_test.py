import numpy as np
import time

from pennylane.qchem import taylor_hamiltonian, vibrational_pes, \
                                Molecule, taylor_coeffs
import numpy as np
import pennylane as qml

# @profile
def main():
    # CH2O Formaldehyde, 6 modes
    symbols = ['C', 'H', 'H', 'O']
    geometry = np.array([[0.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [-1.0, -1.0, 0.0], [1.0, 0.0, 0.0]])
    
    # CH4 methane, 9 modes
    # symbols = ['C', 'H', 'H', 'H', 'H']
    # geometry = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-0.7, -1.0, 0.0], [-0.7, 0.7, 0.7], [-0.7, 0.7, -0.7]])


    # create the molecule object in Pennylane
    mol = Molecule(symbols, geometry, basis_name="cc-pvdz", unit="Angstrom")

    # time the geometry optimization, Hessian, vibrational PES
    start = time.time()
    pes = vibrational_pes(mol, method='rhf', num_workers=3, backend="mpi4py_comm", n_points=9)
    end = time.time() - start
    print(f"Total PES generation runtime {end:.2f} seconds")

    # time the vibrational Hamiltonian construction
    start_hamiltonian = time.time()
    one, two = taylor_coeffs(pes)
    h_qubit2 = taylor_hamiltonian(pes, n_states=4)
    time_hamiltonian = time.time() - start_hamiltonian
    print(f"Vibrational Hamiltonian construction took {time_hamiltonian:.2f} seconds")
    
    
if __name__ == "__main__":
    main()
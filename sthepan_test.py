import numpy as np
import time

from pennylane.qchem import taylor_hamiltonian, vibrational_pes, \
                                Molecule, taylor_coeffs
import numpy as np
import pennylane as qml

# @profile
def main():
    # CH2O Formaldehyde, 6 modes
    # symbols = ['C', 'H', 'H', 'O']
    # geometry = np.array([[0.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [-1.0, -1.0, 0.0], [1.0, 0.0, 0.0]])
    
    # CH4 methane, 9 modes
    symbols = ['C', 'H', 'H', 'H', 'H']
    geometry = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-0.7, -1.0, 0.0], [-0.7, 0.7, 0.7], [-0.7, 0.7, -0.7]])


    # create the molecule object in Pennylane
    mol = Molecule(symbols, geometry, basis_name="cc-pvdz", unit="Angstrom")
    # mol["omp_threads"] = 4  # set the number of threads for pyscf

    # time the geometry optimization, Hessian, vibrational PES
    start = time.time()
    pes = vibrational_pes(mol, method='rhf', num_workers=1, backend="mpi4py_pool", n_points=16//4)
    end = time.time() - start
    print(f"Total PES generation runtime {end:.2f} seconds")

    # time the vibrational Hamiltonian construction
    start_hamiltonian = time.perf_counter()
    one, two = taylor_coeffs(pes)
    h_qubit2 = taylor_hamiltonian(pes, n_states=4)
    time_hamiltonian = time.perf_counter() - start_hamiltonian
    print(f"Vibrational Hamiltonian construction took {time_hamiltonian:.2f} seconds")
    
    
if __name__ == "__main__":
    main()
    
"""
Notes:
The correct way to run it is 
    mpiexec python -m mpi4py.futures sthepan_test.py

Using 
    mpiexec -n X python -m mpi4py.futures sthepan_test.py
    limit the workers spawn to X-1 because the main process doesn't count as a worker.`
    
Using 
    python stephan_test.py
    will not work because the mpi4py_comm backend requires the MPI environment to be set up
    and don't spawn any new worker.
"""
import numpy as np
from pennylane import qchem
from numpy.linalg import eigvalsh
from scipy.optimize import minimize

from pennylane.labs.vibronic.pes_vibronic_utils import harmonic_analysis, generate_grid
from pennylane.labs.vibronic.pes_vibronic import pes_mode

import time

if __name__ == "__main__":

    symbols = ["H", "O", "H"]
    geometry = np.array([[-0.0399, -0.0038, 0.0000],
                         [ 1.5780,  0.8540, 0.0000],
                         [ 2.7909, -0.5159, 0.0000]])
    mol = qchem.Molecule(symbols, geometry, basis_name="sto-3g", load_data=True)
    
    geom_eq = qchem.optimize_geometry(mol, method = "rhf")
    
    mol.coordinates = geom_eq
    
    freqs, vectors = harmonic_analysis(mol)
    
    grid = generate_grid(mol, freqs, vectors, n_points=5)
    
    energy_1 = pes_mode(mol, freqs, vectors, grid, restrict_spin='mixed')
    
    dis = np.array([-1.0, -0.5, 0.5, 1.0])
    
    e_zero = np.array([-75.286633, -75.27155696, -75.27413167])
    
    coeffs_matrix_total = potential_matrix(freqs, dis, e_zero)
    
    E_target_vectors = [item for item in energy_1]
    coeffs_matrix_4d = [item for item in coeffs_matrix_total]
    
    N = coeffs_matrix_4d[0].shape[0]
    M = coeffs_matrix_4d[0].shape[-1]
    Q = len(coeffs_matrix_4d)
    
    a_initial_guess_3d = np.random.rand(N, N, M)
    a_initial_guess_flat = a_initial_guess_3d.flatten()
    
    
    result = minimize(
        cost_function,
        a_initial_guess_flat,
        args=(N, M, Q, coeffs_matrix_4d, E_target_vectors),
        method='Nelder-Mead',
        options={'maxfev': 50000} 
    )
    
    if result.success:
        optimized_a_flat = result.x
        optimized_a_3d = optimized_a_flat.reshape((N, N, M))
        
        print("Optimization successful!")
        
        for q in range(Q):
            A_final = create_matrix(optimized_a_3d, coeffs_matrix_4d[q])
            eigenvalues_final = eigvalsh(A_final)
            print(f"\nQ {q+1}:")
            print(f"Final Eigenvalues: {eigenvalues_final}")
            print(f"Target Eigenvalues: {np.sort(E_target_vectors[q])}")
    else:
        print("Optimization failed.")
        print(result.message)
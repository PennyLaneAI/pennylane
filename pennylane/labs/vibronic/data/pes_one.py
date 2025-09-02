import numpy as np
from pennylane import qchem
from pennylane.labs.vibronic.pes_vibronic_utils import harmonic_analysis, generate_grid
from pennylane.labs.vibronic.pes_vibronic import pes_mode

import pickle
import time

if __name__ == "__main__":

    symbols = ["B", "N", "N", "F", "F", "C", "H", "C", "H", "C", "H", "C", "C", "H", "C", "C", "H", "C", "H", "C", "H"]

    geometry = np.array([[ 3.30845987,  2.67026749,  6.90736431],
                       [ 2.2547038 ,  3.42227066,  6.01485241],
                       [ 2.93015328,  3.03235064,  8.39003679],
                       [ 4.56951862,  3.11625558,  6.62198749],
                       [ 3.21096881,  1.32003214,  6.71341259],
                       [ 2.18744016,  3.39713669,  4.70089506],
                       [ 2.89139969,  2.82141242,  4.11964743],
                       [ 1.12101991,  4.19111342,  4.24132489],
                       [ 0.84356922,  4.34407689,  3.21134175],
                       [ 0.52166606,  4.71848385,  5.3665893 ],
                       [-0.3318649 ,  5.37754591,  5.40923531],
                       [ 1.23613299,  4.23356034,  6.47955254],
                       [ 1.05995137,  4.4448983 ,  7.83808035],
                       [ 0.25412679,  5.08426055,  8.17187101],
                       [ 1.89004066,  3.85602863,  8.77908993],
                       [ 1.89227735,  3.92737821, 10.18592255],
                       [ 1.19681937,  4.49520009, 10.78476492],
                       [ 2.93748449,  3.1427768 , 10.62811119],
                       [ 3.24236396,  2.9596993 , 11.64534706],
                       [ 3.54536667,  2.61343195,  9.47522558],
                       [ 4.39648985,  1.95284014,  9.41132754]]) / 0.529177210544

    mol = qchem.Molecule(symbols, geometry, basis_name="def2-svp", load_data=True)


    with open('geom_eq.pkl', 'rb') as file:
        geom_eq = pickle.load(file)

    with open('freqs.pkl', 'rb') as file:
        freqs = pickle.load(file)

    with open('vectors.pkl', 'rb') as file:
        vectors = pickle.load(file)


    mol.coordinates = geom_eq


    grid = generate_grid(mol, freqs, vectors, n_points=5)

    t1 = time.time()

    energy = pes_mode(mol, freqs, vectors, grid, restrict_spin='mixed', num_workers=16, backend='mpi4py_comm')

    print(time.time() - t1)

    with open('grid_1.pkl', 'wb') as file:
        pickle.dump(grid, file)

    with open('energy_1.pkl', 'wb') as file:
        pickle.dump(energy, file)

    print(energy)

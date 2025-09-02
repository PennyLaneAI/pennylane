import numpy as np
from pennylane import qchem
from pennylane.labs.vibronic.pes_vibronic_utils import harmonic_analysis

import pickle

if __name__ == "__main__":

    with open('geom_eq.pkl', 'rb') as file:
        geom_eq = pickle.load(file)

    with open('freqs.pkl', 'rb') as file:
        freqs = pickle.load(file)

    with open('vectors.pkl', 'rb') as file:
        vectors = pickle.load(file)

    with open('grid_1.pkl', 'rb') as file:
        grid_1 = pickle.load(file)

    with open('energy_1.pkl', 'rb') as file:
        energy_1 = pickle.load(file)


    print(geom_eq)
    print()
    print(freqs)
    print()
    print(vectors)
    print(grid_1)
    print()
    print(energy_1)

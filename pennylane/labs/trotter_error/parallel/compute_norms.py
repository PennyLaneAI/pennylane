import csv
import pickle
import numpy as np
from pennylane.labs.trotter_error import ProductFormula, RealspaceMatrix, effective_hamiltonian, vibronic_fragments
from mode_selector import get_reduced_model
from vibronic_norm import vibronic_norm

# FILE, GRIDPOINTS, MODES
jobs = [
    ("maleimide_5s_24m.pkl", 4, 5),
]

def build_error_term(freqs, taylor_coeffs, modes):
    freqs, taylor_coeffs = get_reduced_model(freqs, taylor_coeffs, modes, strategy="PT")
    states = taylor_coeffs[0].shape[0]

    frags = vibronic_fragments(states, modes, np.array(freqs), taylor_coeffs.values())
    frags = dict(enumerate(frags))
    ham = sum(frags.values(), RealspaceMatrix.zero(states, modes))

    frag_labels = list(frags.keys()) + list(frags.keys())[::-1]
    second_order = ProductFormula(frag_labels, [1/2]*len(frag_labels))
    eff = effective_hamiltonian(second_order, frags, order=3, timestep=1)

    return (eff - 1j*ham)*(1/1j)



if __name__ == '__main__':
    for file, gridpoints, modes in jobs:
        with open(file, 'rb') as f:
            freqs, taylor_coeffs = pickle.load(f)

        err = build_error_term(freqs, taylor_coeffs, modes)
        norm = vibronic_norm(err, gridpoints)

        with open("output.csv", 'a+') as output:
            csv_writer = csv.writer(output)
            csv_writer.writerow((file, gridpoints, modes, norm))

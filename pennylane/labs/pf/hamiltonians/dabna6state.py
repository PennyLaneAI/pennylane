import pickle

import numpy as np

from pennylane.labs.pf import VibronicHamiltonian


def six_mode_ham():
    """Return the hamiltonian"""

    filehandler = open("./dabna_6s10m.pkl", "rb")
    omegas, couplings = pickle.load(filehandler)
    omegas = np.array(omegas)

    # couplings is a dictionary {0: lambdas, 1: alphas, 2: betas}

    lambdas = couplings[0]
    alphas = couplings[1]
    betas = couplings[2]

    n = np.shape(lambdas)[0]
    m = len(omegas)

    h_operator = VibronicHamiltonian(n, m, omegas, [lambdas, alphas, betas], sparse=True)

    return h_operator

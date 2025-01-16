"""Benchmark the norm"""

import time

from pickled_ham import get_coeffs as coeffs2
from spin_vibronic_ham import get_coeffs as coeffs1

from pennylane.labs.vibronic import VibronicHamiltonian, coeffs

N_STATES_1 = 8
N_MODES_1 = 10

vham1 = VibronicHamiltonian(N_STATES_1, N_MODES_1, *coeffs1(), sparse=True)

N_STATES_2 = 8
N_MODES_2 = 21

vham2 = VibronicHamiltonian(N_STATES_2, N_MODES_2, *coeffs2())

dense_vham1 = VibronicHamiltonian(4, 5, *coeffs(4, 5))


def time_norm(vham: VibronicHamiltonian):
    start = time.time()
    res = vham.block_operator().norm(4)
    end = time.time()

    return end - start, res


def time_norm_ep(vham: VibronicHamiltonian):
    ep = vham.epsilon(1)

    start = time.time()
    res = ep.norm(4)
    end = time.time()

    return end - start, res


def time_coefficients(vham: VibronicHamiltonian):
    print("computing epsilon...")
    ep = vham.epsilon(1)

    print("iterating over coefficients...")
    start = time.time()
    ep.coefficients()
    end = time.time()

    return end - start


if __name__ == "__main__":
    print(time_norm_ep(dense_vham1))

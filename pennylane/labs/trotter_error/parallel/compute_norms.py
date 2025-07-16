import csv
import os
import pickle
import time
from itertools import product

import numpy as np
from mpi4py import MPI
from vibronic_norm import _compute_norm, build_error_term, vibronic_norm

# FILE, GRIDPOINTS, MODES
jobs = [
    # ("VCHLIB/maleimide_5s_24m.pkl", 4, 6),
    ("no4a_sf.pkl", 4, 19),
]


def chunkify(lst, n):
    """Split list of indices into n roughly equal chunks."""
    return [lst[i::n] for i in range(n)]


if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    for file, gridpoints, modes in jobs:
        path = f"hamiltonians/{file}"
        with open(path, "rb") as f:
            freqs, taylor_coeffs = pickle.load(f)

        err = build_error_term(freqs, taylor_coeffs, modes)

    if rank == 0:

        tt1 = time.time()

        numbers = list(i for i in product(range(err.states), repeat=2))

        batches = chunkify(numbers, size)

    else:
        batches = None

    # Scatter the batches
    batch = comm.scatter(batches, root=0)

    # Each process computes a batch of matrix elements
    local_result = vibronic_norm(err, gridpoints, batch)

    # Collect all matrices
    all_results = comm.gather(local_result, root=0)

    if rank == 0:

        tt3 = time.time()
        # Sum the resulting matrices to obtain the final matrix
        final_result = sum(all_results)

        # Compute the norm of the matrix
        norm = _compute_norm(final_result)

        tt2 = time.time()
        print(tt2 - tt1, tt2 - tt3)

        with open("output.csv", "a+") as output:
            csv_writer = csv.writer(output)
            csv_writer.writerow((file, gridpoints, modes, norm))

        path = f"norms/{file}_g{gridpoints}_m{modes}.pkl"
        os.makedirs("norms", exist_ok=True)
        with open(path, "wb+") as f:
            pickle.dump(final_result[: err.states, : err.states], f)

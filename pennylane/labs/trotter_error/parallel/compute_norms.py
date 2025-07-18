import csv
import os
import pickle
import time
from itertools import product

import numpy as np
np.set_printoptions(linewidth=100, suppress=True, precision=6)
from mpi4py import MPI
from vibronic_norm import _compute_norm, build_error_term, finalize_timing#, vibronic_norm
from pennylane.labs.trotter_error import (
    ProductFormula,
    RealspaceMatrix,
    RealspaceSum,
    effective_hamiltonian,
    vibronic_fragments,
)

import itertools

from vibronic_norm import _next_pow_2, _block_norm, chunkify, _get_eigenvalue_batch

# FILE, GRIDPOINTS, MODES
jobs = [
    # ("VCHLIB/maleimide_5s_24m.pkl", 4, 6),
    ("no4a_dimer.pkl", 4, 10),
]

if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    start = MPI.Wtime()

    #if rank == 0:

    for file, gridpoints, modes in jobs:
        path = f"hamiltonians/{file}"
        with open(path, "rb") as f:
            freqs, taylor_coeffs = pickle.load(f)

        err = build_error_term(freqs, taylor_coeffs, modes)
        numbers = list(i for i in product(range(err.states), repeat=2))

    #else:
    #    err, numbers, gridpoints, modes = None, None, None, None


# --------------------------------------------------------------------------

        padded = RealspaceMatrix(_next_pow_2(err.states), err.modes, err._blocks)
        norms = np.zeros(shape=(padded.states, padded.states))

        if rank == 0:

            all_matrices = [_block_norm(padded.block(i, j), gridpoints) for i, j in numbers]
            list_lens = [len(m) for m in all_matrices]
            flat_matrices = [vals for m in all_matrices for vals in m.values()]
            batches = chunkify(flat_matrices, size)

        else:
            batches = None

        batch = comm.scatter(batches, root=0)

        print(rank, size, len(batch))

        local_result = _get_eigenvalue_batch(batch, gridpoints)

        all_results = comm.gather(local_result, root=0)

        comm.Barrier()
        print(f"Rank {rank} finished computing. Results: {all_results}")
        comm.Barrier()

        if rank == 0:

            all_eigvals = np.concatenate(all_results)

            grouped_eigvals = zip([0] + list(itertools.accumulate(list_lens))[: -1], list_lens)

            grouped_sums = [sum(all_eigvals[i: i + n]) for i, n in grouped_eigvals]

            norms[tuple(zip(*numbers))] = grouped_sums

            norm = _compute_norm(norms)

            end = MPI.Wtime()

            print(f"Norm: {norm} | Time: {end - start:.4f}")

            with open("output.csv", "a+") as output:
                csv_writer = csv.writer(output)
                csv_writer.writerow((file, gridpoints, modes, norm))

            path = f"norms/{file.replace('.pkl', '')}_g{gridpoints}_m{modes}.pkl"
            os.makedirs("norms", exist_ok=True)
            with open(path, "wb+") as f:
                pickle.dump(norms[: err.states, : err.states], f)

    print("Execution complete")
    finalize_timing()

import math
from itertools import product

import numpy as np
import scipy as sp
from mode_selector import get_reduced_model

from pennylane.labs.trotter_error import (
    ProductFormula,
    RealspaceMatrix,
    RealspaceSum,
    effective_hamiltonian,
    vibronic_fragments,
)
from pennylane.labs.trotter_error.realspace.matrix import _momentum_operator, _position_operator


from mpi4py import MPI
import time

TIMES_TABLE = {(MPI.COMM_WORLD.Get_rank()): {}}

def timeit(func):
    def wrapper(*args, **kwargs):
        rank = MPI.COMM_WORLD.Get_rank()
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        # print(f"Rank {rank} finished {func.__name__} in {end - start} seconds")
        
        if func.__name__ not in TIMES_TABLE[rank]:
            TIMES_TABLE[rank][func.__name__] = {"hit_count": 0, "runs": []}
        TIMES_TABLE[rank][func.__name__]["hit_count"] += 1
        TIMES_TABLE[rank][func.__name__]["runs"].append(end - start)
        
        return result
    return wrapper

def get_times_table():
    """Return the times table for all ranks merged in rank 0."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # First, compute total and average times for local data
    for rank_id, data in TIMES_TABLE.items():
        for func_name, stats in data.items():
            if "runs" in stats and stats["runs"]:
                total_time = sum(stats["runs"])
                avg_time = total_time / stats["hit_count"]
                stats["total_time"] = total_time
                stats["avg_time"] = avg_time
    
    # Gather all timing data to rank 0
    all_times_tables = comm.gather(TIMES_TABLE, root=0)
    
    if rank == 0:
        # Merge all ranks' data into a single global table
        global_times = {}
        
        for rank_data in all_times_tables:
            for rank_id, timing_data in rank_data.items():
                global_times[rank_id] = timing_data
        
        # Print the merged times table
        print("Global Times Table (merged from all ranks):")
        for rank_id in sorted(global_times.keys()):
            data = global_times[rank_id]
            print(f"Rank {rank_id}:")
            # for func_name, stats in data.items():
            for func_name in sorted(data.keys()):
                stats = data[func_name]
                if "total_time" in stats and "avg_time" in stats:
                    print(f"  {func_name}: hit_count={stats['hit_count']}, total_time={stats['total_time']:.4f}s, avg_time={stats['avg_time']:.4f}s")
                    # precision print for runs > 0.1s
                    if stats['avg_time'] > 0.01:
                        print(f"     Runs: ", end="")
                        for run in stats['runs']:
                            print(f"{run:7.4f}s,", end=" ")
                        print()

        return global_times
    else:
        return None        
    

@timeit
def vibronic_norm(rs_mat: RealspaceMatrix, gridpoints: int, batch: list):
    if not _is_pow_2(gridpoints) or gridpoints <= 0:
        raise ValueError(f"Number of gridpoints must be a positive power of 2, got {gridpoints}.")

    padded = RealspaceMatrix(_next_pow_2(rs_mat.states), rs_mat.modes, rs_mat._blocks)

    norms = np.zeros(shape=(padded.states, padded.states))

    for i, j in batch:
        norms[i, j] = _block_norm(padded.block(i, j), gridpoints)

    return norms

@timeit
def build_error_term(freqs, taylor_coeffs, modes):
    freqs, taylor_coeffs = get_reduced_model(freqs, taylor_coeffs, modes, strategy="PT")
    states = taylor_coeffs[0].shape[0]

    frags = vibronic_fragments(states, modes, np.array(freqs), taylor_coeffs.values())
    frags = dict(enumerate(frags))
    ham = sum(frags.values(), RealspaceMatrix.zero(states, modes))

    frag_labels = list(frags.keys()) + list(frags.keys())[::-1]
    second_order = ProductFormula(frag_labels, [1 / 2] * len(frag_labels))
    eff = effective_hamiltonian(second_order, frags, order=3, timestep=1)

    return (eff - 1j * ham) * (1 / 1j)


# @timeit
def _compute_norm(norm_mat: np.ndarray):
    if norm_mat.shape == (1, 1):
        return norm_mat[0, 0]

    half = norm_mat.shape[0] // 2

    top_left = norm_mat[0:half, 0:half]
    top_right = norm_mat[0:half, half:]
    bottom_left = norm_mat[half:, 0:half]
    bottom_right = norm_mat[half:, half:]

    norm1 = max(_compute_norm(top_left), _compute_norm(bottom_right))
    norm2 = math.sqrt(_compute_norm(top_right) * _compute_norm(bottom_left))

    return norm1 + norm2

@timeit
@profile
def _block_norm(rs_sum: RealspaceSum, gridpoints: int):
    mode_groups = {}

    for op in rs_sum.ops:
        for index, coeff in op.coeffs.nonzero().items():
            group = frozenset(index)

            mode_mats = {i: sp.sparse.eye(gridpoints, dtype=np.complex128) for i in group}
            for i, mat in zip(index, op.ops):
                if mat == "P":
                    mode_mats[i] @= _momentum_operator(gridpoints, basis="harmonic", sparse=True)
                elif mat == "Q":
                    mode_mats[i] @= _position_operator(gridpoints, basis="harmonic", sparse=True)

            sorted_group = sorted(group)
            if len(sorted_group) == 0:
                mat = sp.sparse.eye(gridpoints)
            else:
                mat = mode_mats[sorted_group[0]]
                for i in sorted_group[1:]:
                    mat = sp.sparse.kron(mat, mode_mats[i], format="csr")

            mat = coeff * mat

            try:
                mode_groups[group] += mat
            except KeyError:
                mode_groups[group] = mat

    return sum_get_eigenvalue(mode_groups)

@timeit
def sum_get_eigenvalue(mode_groups):
    """Sum the eigenvalues of the matrices in mode_groups."""
    return sum(_get_eigenvalue(mat) for mat in mode_groups.values())

def _get_eigenvalue(mat):
    
    _, _, values = sp.sparse.find(mat)
    if np.allclose(values, 0):
        return 0

    # print(f"Computing eigenvalue for matrix of shape {mat.shape}, rank {rank}")
    if mat.shape[0] <= 2:
        eigvals, _ = sp.linalg.eig(mat.toarray())
    else:
        eigvals, _ = sp.sparse.linalg.eigs(mat, k=1)
    return np.abs(eigvals[0])


def _is_pow_2(k: int) -> bool:
    """Test if k is a power of two"""
    return k & (k - 1) == 0


def _next_pow_2(k: int) -> int:
    """Return the smallest power of 2 greater than or equal to k"""
    return 2 ** (k - 1).bit_length()

def print_global_timing_stats():
    """Convenience function to print timing statistics across all ranks."""
    global_times = get_times_table()
    return global_times

def finalize_timing():
    """Call this at the end of your program to collect and print all timing data."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Synchronize all ranks before collecting timing data
    comm.Barrier()
    
    if rank == 0:
        print("\n" + "="*50)
        print("FINAL TIMING STATISTICS")
        print("="*50)
    
    global_times = get_times_table()
    
    if rank == 0:
        print("="*50 + "\n")
        return global_times
    
    return None

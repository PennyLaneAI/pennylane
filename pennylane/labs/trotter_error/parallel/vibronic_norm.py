import math
from functools import cache

import numpy as np
import scipy as sp
from mode_selector import get_reduced_model
import itertools

from mpi4py import MPI

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
            print("-" * 150)
            print(f"Rank {rank_id}:")
            # for func_name, stats in data.items():
            for func_name in sorted(data.keys()):
                stats = data[func_name]
                if "total_time" in stats and "avg_time" in stats:
                    print(f"  {func_name}: hit_count={stats['hit_count']}, total_time={stats['total_time']:.4f}s, avg_time={stats['avg_time']:.4f}s, min_run={min(stats['runs']):.4f}s, max_run={max(stats['runs']):.4f}s")
                    
                    if False and stats['avg_time'] > 0.01:
                        print(f"     Runs: ", end="")
                        for run in stats['runs']:
                            print(f"{run:7.4f}s,", end=" ")
                        print()

        return global_times
    else:
        return None        

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
    


def chunkify(lst, n):
    """Split list into n chunks and preserve order."""
    size = len(lst) // n
    remainder = len(lst) % n
    chunks = []
    start = 0
    for i in range(n):
        end = start + size + (1 if i < remainder else 0)
        chunks.append(lst[start:end])
        start = end
    return chunks
@timeit
def _get_eigenvalue_batch(batch_matrices, gridpoints):
    """Compute eigvals for a batch of matrices."""
    eigenvalue_batch = np.array([_get_eigenvalue(group_ops, gridpoints) for group_ops in batch_matrices])
    return eigenvalue_batch

# def vibronic_norm(rs_mat: RealspaceMatrix, gridpoints: int, numbers: list):
#     if not _is_pow_2(gridpoints) or gridpoints <= 0:
#         raise ValueError(f"Number of gridpoints must be a positive power of 2, got {gridpoints}.")

#     comm = MPI.COMM_WORLD
#     rank = comm.Get_rank()
#     size = comm.Get_size()

#     padded = RealspaceMatrix(_next_pow_2(rs_mat.states), rs_mat.modes, rs_mat._blocks)
#     norms = np.zeros(shape=(padded.states, padded.states))
   
#     if rank == 0:

#         all_matrices = [_block_norm(padded.block(i, j), gridpoints) for i, j in numbers]
#         list_lens = [len(m) for m in all_matrices]
#         flat_matrices = [vals for m in all_matrices for vals in m.values()]
#         batches = chunkify(flat_matrices, size)
        
#     else:
#         batches = None

#     batch = comm.scatter(batches, root=0)

#     print(rank, size, len(batch))
 
#     local_result = _get_eigenvalue_batch(batch, gridpoints)

#     all_results = comm.gather(local_result, root=0)

#     if rank == 0:

#         all_eigvals = np.concatenate(all_results)

#         grouped_eigvals = zip([0] + list(itertools.accumulate(list_lens))[: -1], list_lens)
        
#         grouped_sums = [sum(all_eigvals[i: i + n]) for i, n in grouped_eigvals]
    
#         norms[tuple(zip(*numbers))] = grouped_sums
    
#     return norms


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
def _block_norm(rs_sum: RealspaceSum, gridpoints: int):
    mode_groups = {}

    for op in rs_sum.ops:
        for index, coeff in op.coeffs.nonzero().items():
            group = frozenset(index)

            mode_strs = {i: "" for i in group}
            for i, mat in zip(index, op.ops):
                if mat == "P":
                    mode_strs[i] += "P"
                elif mat == "Q":
                    mode_strs[i] += "Q"

            sorted_ops = tuple(mode_strs[i] for i in sorted(group))

            if group not in mode_groups:
                mode_groups[group] = {"ops": [sorted_ops], "coeffs": [coeff]}
            else:
                mode_groups[group]["ops"].append(sorted_ops)
                mode_groups[group]["coeffs"].append(coeff)


    return mode_groups

@cache
def build_mat(gridpoints, ops):
    if len(ops) == 0:
        return sp.sparse.eye(gridpoints)

    mats = [sp.sparse.eye(gridpoints, dtype=np.complex128)] * len(ops)
    for i, op in enumerate(ops):
        for ch in op:
            if ch == "P":
                mats[i] @= _momentum_operator(gridpoints, basis="harmonic", sparse=True)
            elif ch == "Q":
                mats[i] @= _position_operator(gridpoints, basis="harmonic", sparse=True)

    ret = mats[0]

    for mat in mats[1:]:
        ret = sp.sparse.kron(ret, mat, format="csr")

    return ret


@timeit
def _get_eigenvalue(group_ops, gridpoints):
    ops, coeffs = group_ops["ops"], group_ops["coeffs"]
    mat = coeffs[0] * build_mat(gridpoints, ops[0])

    for op, coeff in zip(ops[1:], coeffs[1:]):
        mat += coeff * build_mat(gridpoints, op)

    _, _, values = sp.sparse.find(mat)
    if np.allclose(values, 0):
        return 0

    try:
        eigvals, _ = sp.sparse.linalg.eigs(mat, k=1)
        return np.abs(eigvals[0])
    except Exception:
        pass

    try:
        return sp.sparse.linalg.norm(mat, ord=2)
    except Exception:
        pass

    return 0


def _is_pow_2(k: int) -> bool:
    """Test if k is a power of two"""
    return k & (k - 1) == 0


def _next_pow_2(k: int) -> int:
    """Return the smallest power of 2 greater than or equal to k"""
    return 2 ** (k - 1).bit_length()

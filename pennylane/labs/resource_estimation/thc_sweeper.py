"""
THC (Tensor Hyper-Contraction) rank factorization for quantum chemistry.

This module uses THC utility in thc_decomp.py to sweep over different ranks and obtain required THC rank for associated accuracy
Uses criterion of maximum difference for first nroots excited states
"""
import numpy as np
from jax import numpy as jnp
from tqdm import tqdm
import pennylane.labs.resource_estimation.thc_decomp as thc
from pennylane.labs.resource_estimation.auxiliary import QuantumSolver

#Remove printing from unnecessary routines
import sys
import io
from time import time

def _calculate_energies_with_fci(obt, eri, ncas, nelecas, nroots):
    """Common utility to calculate energies using FCI solver."""
    from pyscf import fci
    
    # Convert to numpy if needed
    obt_np = np.array(obt)
    eri_np = np.array(eri)
    
    # Calculate energies
    energies, _ = fci.direct_spin1.kernel(obt_np, eri, ncas, nelecas, nroots=nroots)
    
    # Ensure list format
    if nroots == 1:
        energies = [energies]
        
    return energies


def _calculate_energies(obt, eri, ncas, nelecas, nroots, solver, **kwargs):
    QS = QuantumSolver(solver_type=solver, **kwargs)
    if solver == "fci":
        return QS._solve_fci(obt, eri, ncas, nelecas, nroots)
    elif solver == "dmrg":
        eri_writeable = np.array(eri)
        return QS._solve_dmrg(obt, eri_writeable, ncas, nelecas, nroots)[0]
    else:
        raise ValueError(f"Trying to use electronic structure solver {solver}, not implemented!")


def _silent_thc(eri, Nthc, obt=None, ob_sym_list=[], **kwargs):
    """
    Wrapper for get_thc that suppresses all text output.
    
    Args:
        eri: Two-electron repulsion integrals
        Nthc: Rank for THC factorization
        obt: one-body tensor
        ob_sym_list: List of tuples, each containing symmetry eigenvalue and associated one-body operator
        
    Returns:
        Same as get_thc but without any printed output
    """
    # Capture both stdout and stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    try:
        # Redirect both stdout and stderr to nowhere
        # sys.stdout = io.StringIO()
        # sys.stderr = io.StringIO()
        
        # Call the original function
        return thc.get_thc(eri, Nthc=Nthc, obt=obt, ob_sym_list=ob_sym_list, **kwargs)
        
    finally:
        # Always restore stdout and stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def _thc_cost(energy_deltas, criterion="avg"):
    if criterion == "max":
        my_cost = np.max(energy_deltas)
    elif criterion == "avg":
        my_cost = np.mean(energy_deltas)
    else:
        raise ValueError(f"""Trying to use criterion {criterion} for energy cost, only implemented ones are "max" and "avg".""")

    return my_cost


def _bliss_constant_shift(avec, bvec, ob_sym_vals, num_ob_syms):
    const_shift = 0.0

    idx = 0
    for pp in range(num_ob_syms):
        const_shift += avec[pp] * ob_sym_vals[pp]
        for qq in range(pp+1):
            const_shift -= bvec[idx] * ob_sym_vals[pp] * ob_sym_vals[qq]
            idx += 1

    return const_shift


def thc_rank_finder(obt, eri, num_elecs, epsilon=1.6e-3, criterion="avg", verbose=True, regularize=1e-5, maxiter=int(5e5), learning_rate=5e-4,\
                    Nthc_min=None, Nthc_max=None, num_thc_calcs=10, ob_sym_list=[], nroots=10, solver="dmrg", bond_dim=None, sub_verbose=False, **kwargs):
    print(f"Starting calculation of energies for exact tensors...")
    ncas = obt.shape[0]
    TIMES_ARR = [time()]
    exact_energies = _calculate_energies(obt, eri, ncas, num_elecs, nroots, solver, bond_dim=bond_dim)
    TIMES_ARR.append(time())
    print(f"Finished initial energy calculation for {nroots} first electronic energies after {TIMES_ARR[1]-TIMES_ARR[0]:.2f} seconds")

    if Nthc_min is None:
        Nthc_min = 2*ncas

    if Nthc_max is None:
        Nthc_max = 5*ncas

    try:
        assert Nthc_min < Nthc_max
    except:
        raise ValueError(f"Minimum THC rank {Nthc_min} is larger than maximum rank {Nthc_max}")

    thc_ranks = np.round(np.linspace(Nthc_min, stop=Nthc_max, num=num_thc_calcs)).astype(int)
    if verbose:
        print(f"Will do THC calculations for ranks {thc_ranks}")

    num_ob_syms = len(ob_sym_list)
    if num_ob_syms > 0:
        include_bliss = True
        ob_sym_mats = np.array([ob_sym_list[kk][0] for kk in range(num_ob_syms)])
        ob_sym_vals = np.array([ob_sym_list[kk][1] for kk in range(num_ob_syms)])
    else:
        include_bliss = False

    thc_energies = []
    diffs_arr = []
    thc_lambdas = []
    thc_cost = []
    thc_params = []
    undetermined_rank = True

    curr_idx = 0
    while undetermined_rank and curr_idx < num_thc_calcs:
        t_rank = thc_ranks[curr_idx]
        if verbose:
            print(f"Starting calculation for THC rank {t_rank}")
        my_params, my_lam = _silent_thc(eri, t_rank, obt, ob_sym_list, regularize=regularize, maxiter=maxiter, learning_rate=learning_rate, verbose=sub_verbose, **kwargs)
        thc_params.append(my_params)
        thc_lambdas.append(my_lam)
        my_eri = thc._unfold_thc(my_params["MPQ"], my_params["etaPp"])

        if include_bliss:
            obt_killer, eri_killer = thc._BLISS_corrections(my_params["avec"], my_params["bvec"], my_params["beta_mats_params"], my_params["dvec"], ncas, ob_sym_mats, ob_sym_vals, num_ob_syms)
            eri_killer = np.array(eri_killer)
        else:
            eri_killer = np.zeros_like(eri)

        diffs_arr.append(np.sqrt(np.sum(np.abs(eri - my_eri - eri_killer)**2)))

        thc_energies.append(_calculate_energies(obt, my_eri + eri_killer, ncas, num_elecs, nroots, solver))
        energy_deltas = np.abs(exact_energies - thc_energies[-1])
        thc_cost.append(_thc_cost(energy_deltas, criterion))

        if verbose:
            print(f"Finished calculation for THC rank {t_rank} with associated cost 1-norm {my_lam:.2f}, 2-norm difference {diffs_arr[-1]:.2e}, and {criterion} cost {thc_cost[-1]:.2e}")
            TIMES_ARR.append(time())
            print(f"Time for calculation was {TIMES_ARR[-1]-TIMES_ARR[-2]:.2f} seconds\n")

        if thc_cost[-1] < epsilon:
            print(f"Found converged THC for rank = {t_rank} with associated {criterion} eigenvalue deviation of {thc_cost[-1]}")
            undetermined_rank = False

        curr_idx += 1

    if undetermined_rank:
        raise ValueError(f"Finished calculations with maximum THC rank of {thc_ranks[-1]} with minimum cost function of {np.min(thc_cost)}, not converged! Modify hyperparameters or increase maximum THC rank")

    return t_rank, thc_lambdas[-1], thc_params[-1]

######## DEPRECATED FUNCTIONS

def _thc_sweep(obt, eri, num_elecs, Nthc_min=None, Nthc_max=None, num_thc_calcs=10, ob_sym_list=[], nroots=10, solver="dmrg", bond_dim=None, **kwargs):
    ncas = obt.shape[0]
    print(f"Starting calculation of energies for exact tensors...")
    exact_energies = _calculate_energies(obt, eri, ncas, num_elecs, nroots, solver, bond_dim=bond_dim)

    if Nthc_min is None:
        Nthc_min = 2*ncas

    if Nthc_max is None:
        Nthc_max = 5*ncas

    try:
        assert Nthc_min < Nthc_max
    except:
        raise ValueError(f"Minimum THC rank {Nthc_min} is larger than maximum rank {Nthc_max}")

    thc_ranks = np.round(np.linspace(Nthc_min, stop=Nthc_max, num=num_thc_calcs)).astype(int)
    print(f"Will do THC calculations for ranks {thc_ranks}")

    thc_energies = np.zeros((num_thc_calcs, nroots))
    diffs_arr = np.zeros(num_thc_calcs)
    thc_lambdas = np.zeros(num_thc_calcs)

    num_ob_syms = len(ob_sym_list)
    if num_ob_syms > 0:
        include_bliss = True
        ob_sym_mats = np.array([ob_sym_list[kk][0] for kk in range(num_ob_syms)])
        ob_sym_vals = np.array([ob_sym_list[kk][1] for kk in range(num_ob_syms)])
    else:
        include_bliss = False

    for ii,t_rank in tqdm(enumerate(thc_ranks),desc="THC ranks"):
        my_params, my_lam = _silent_thc(eri, t_rank, obt, ob_sym_list, **kwargs)
        thc_lambdas[ii] = my_lam
        my_eri = thc._unfold_thc(my_params["MPQ"], my_params["etaPp"])

        if include_bliss:
            obt_killer, eri_killer = _BLISS_corrections(my_params["avec"], my_params["bvec"], my_params["beta_mats_params"], my_params["dvec"], ncas, ob_sym_mats, ob_sym_vals, num_ob_syms)
        else:
            obt_killer = np.zeros_like(obt)
            eri_killer = np.zeros_like(eri)

        diffs_arr[ii] = np.sqrt(np.sum(np.abs(eri - my_eri + eri_killer)**2))
        thc_energies[ii,:] = _calculate_energies(obt - obt_killer, my_eri - eri_killer, ncas, num_elecs, nroots, solver)


    return thc_ranks, diffs_arr, thc_energies, thc_lambdas, exact_energies

def thc_for_eigenaccuracy(obt, eri, num_elecs, epsilon=1.6e-3, criterion="avg", verbose=True, regularize=1e-5, maxiter=int(5e5), **kwargs):
    thc_ranks, diffs_arr, thc_energies, thc_lambdas, exact_energies = _thc_sweep(obt, eri, num_elecs, regularize=regularize, maxiter=maxiter, **kwargs)
    num_thc_calcs = len(thc_ranks)
    nroots = len(thc_energies)

    if verbose:
        print("THC ranks and associated one-norms are:")
        for i_thc in range(num_thc_calcs):
            print(f"Nthc = {thc_ranks[i_thc]}, lambda = {thc_lambdas[i_thc]:.2f}")

    nroots = len(exact_energies)
    print(f"Finished running THC and electronic structure calculations, starting convergence analysis")
    energy_deltas = np.zeros((num_thc_calcs, nroots))
    for ii_root in range(nroots):
        energy_deltas[:,ii_root] = np.abs(exact_energies[ii_root] - thc_energies[:,ii_root])

    thc_cost = np.zeros(num_thc_calcs)
    if criterion == "max":
        print(f"Using max criterion: maximum absolute energy deviation over first {nroots} eigenvalues")
        for i_thc in range(num_thc_calcs):
            thc_cost[i_thc] = np.max(energy_deltas[i_thc,:])
    elif criterion == "avg":
        print(f"Using average criterion: average absolute energy deviation over first {nroots} eigenvalues")
        for i_thc in range(num_thc_calcs):
            thc_cost[i_thc] = np.mean(energy_deltas[i_thc,:])
    else:
        raise ValueError(f"""Trying to use criterion {criterion} for energy cost, only implemented ones are "max" and "avg".""")

    is_bad = thc_cost > epsilon

    if verbose:
        print(f"THC costs are {thc_cost}")
        print(f"THC 2-norm of differences are {np.round(diffs_arr, decimals=4)}")
        print(f"Total energies are {exact_energies}")
        for ii in range(num_thc_calcs):
            print(f"THC rank {thc_ranks[ii]} energy errors are {np.round(energy_deltas[ii,:], decimals=3)}")
    
    if is_bad[-1]:
        raise ValueError(f"Final THC calculation for THC rank Nthc={thc_ranks[-1]} did not achieve criterion! Associated cost is {thc_cost[-1]}. Rank needs to be increased with argument Nthc_max")


    i_min = num_thc_calcs-1
    while not(is_bad[i_min]):
        i_min -= 1

    i_min += 1

    print(f"Found THC rank of {thc_ranks[i_min]} with target accuracy {thc_cost[i_min]:.2e}")
    print("Higher and lower ranks are:")
    if i_min < num_thc_calcs-1:
        print(f"High THC rank of {thc_ranks[i_min+1]} with target accuracy {thc_cost[i_min+1]:.2e}")
    else:
        print(f"Found THC rank was highest available!")
    
    if i_min > 0:
        print(f"Low THC rank of {thc_ranks[i_min-1]} with target accuracy {thc_cost[i_min-1]:.2e}")
    else:
        print(f"Found THC rank was lowest available!")

    return thc_ranks[i_min], thc_lambdas[i_min]

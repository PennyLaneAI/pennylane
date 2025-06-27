import pennylane as qml
from pennylane.labs.resource_estimation.resource_tracking import estimate_resources, resource_config
import numpy as np
import math

from time import time

import bliss
import orbital_optimization as oo
from hyperoptimization import resource_optimizer, resource_cost, cost_heuristic

from templates.compact_hamiltonian import CompactHamiltonian
# from templates.walk import ResourceWalk

from templates.LCU_decomps import optax_lbfgs_opt_thc_l2reg_enhanced, thc_one_norm, sparse_matrix, double_factorization, compressed_double_factorization, l4
from templates.thc import ResourceSelectTHC, ResourcePrepTHC, ResourcePrepCDF, ResourceSelectCDF, ResourcePrepSparsePauli, ResourceSelectSparsePauli

from cow_print import cow_print

CustomGateSet = {
    "X",
    "Y",
    "Z",
    "Hadamard",
    "CNOT",
    "S",
    "T"
}

decomposition_methods = ["THC", "BLISS-THC", "CDF", "BLISS-CDF", "Sparse"]
def create_compact_ham(obt, tbt, method, **kwargs):
	if method not in decomposition_methods:
		raise ValueError(f"Trying to do decomposition for method {method}, not defined! Only {decomposition_methods} have been defined")

	Norbs = obt.shape[0]

	if method == "Sparse":
		if "tol_factor" in kwargs:
			tol_factor = kwargs["tol_factor"]
		else:
			tol_factor = 1e-5

		one_norm, num_unitaries = sparse_matrix(obt, tbt, tol_factor=1e-5)
		return CompactHamiltonian.sparsepauli(Norbs, num_unitaries), one_norm

	if method == "CDF" or method == "BLISS-CDF":
		if "nfrags" in kwargs:
			ncdf = kwargs["nfrags"]
		else:
			ncdf = 2*Norbs

		if "tol_factor" in kwargs:
			tol_factor = kwargs["tol_factor"]
		else:
			tol_factor = 1e-5

		num_unitaries, cores, leaves = compressed_double_factorization(obt, tbt, tol_factor=tol_factor)
		if method == "CDF":
			do_bliss = False
			return CompactHamiltonian.cdf(Norbs, num_unitaries), bliss.cdf_one_norm(obt, cores, leaves)
		else:
			do_bliss = True
			if "eta" not in kwargs:
				raise ValueError(f"Trying to do BLISS-CDF decomposition but number of electrons eta not passed as kwarg!")
			eta = kwargs["eta"]
			new_obt, new_cores = bliss.bliss_cdf(obt, cores, leaves, eta)
			return CompactHamiltonian.cdf(Norbs, num_unitaries), bliss.cdf_one_norm(new_obt, new_cores, leaves)

	if method == "THC" or method == "BLISS-THC":
		if "nfrags" in kwargs:
			nthc = kwargs["nfrags"]
		else:
			nthc = 3*Norbs

		if method == "THC":
			do_bliss = False
		else:
			do_bliss = True
			if "eta" not in kwargs:
				raise ValueError(f"Trying to do BLISS-THC decomposition but number of electrons eta not passed as kwarg!")
			eta = kwargs["eta"]

		if "maxiter" in kwargs:
			maxiter = kwargs["maxiter"]
		else:
			maxiter = 500

		thc_params = optax_lbfgs_opt_thc_l2reg_enhanced(
			eri=tbt,
			nthc=nthc,
			maxiter=maxiter,
			random_seed=42,
			verbose=False,
			include_bias_terms=do_bliss,
			chkfile_name="thc_results.h5")

		zeta_tsr = thc_params["MPQ"]
		obt_corrected = obt + bliss.ob_correction(tbt)

		if do_bliss:
			beta = thc_params["beta"]
			obt_corrected += 0.5 * eta * beta
			obt_corrected = obt_corrected - bliss.bliss_one_body(obt_corrected)*np.eye(Norbs) 

		one_norm = thc_one_norm(obt_corrected, zeta_tsr)

		return CompactHamiltonian.thc(Norbs,nthc), one_norm

	raise ValueError(f"Tried to create CompactHamiltonian for method {method}, not defined!")

def create_resource_function(compact_ham, method, max_selwap = 8, **kwargs):
	if method not in decomposition_methods:
		raise ValueError(f"Trying to do decomposition for method {method}, not defined! Only {decomposition_methods} have been defined")

	selswap_range = 2**np.arange(max_selwap+1)
	if method == "Sparse":
		PREP = ResourcePrepSparsePauli(compact_ham, **kwargs)
		prep_kwargs = ["select_swap_depth"]
		prep_ranges = [selswap_range]

		SEL = ResourceSelectSparsePauli(compact_ham, **kwargs)
		sel_kwargs = []
		sel_ranges = []


	if method in ["CDF", "BLISS-CDF"]:
		PREP = ResourcePrepCDF(compact_ham, **kwargs)
		prep_kwargs = ["select_swap_depth"]
		prep_ranges = [selswap_range]

		par_range = np.arange(1,compact_ham.params["num_orbitals"])
		SEL = ResourceSelectCDF(compact_ham, **kwargs)
		sel_kwargs = ["select_swap_depth", "parallel_rotations"]
		sel_ranges = [selswap_range, par_range]


	if method in ["THC", "BLISS-THC"]:
		PREP = ResourcePrepTHC(compact_ham, **kwargs)
		prep_kwargs = ["select_swap_depth"]
		prep_ranges = [selswap_range]

		par_range = np.arange(1,compact_ham.params["num_orbitals"])
		SEL = ResourceSelectTHC(compact_ham, **kwargs)
		sel_kwargs = ["select_swap_depth", "parallel_rotations"]
		sel_ranges = [selswap_range, par_range]

	def prep_cost(*args):
		prep_config = {}
		prep_config.update(resource_config)
		for i_k,kw in enumerate(prep_kwargs):
			prep_config[kw] = args[i_k]

		return estimate_resources(PREP, gate_set=CustomGateSet, config=prep_config)

	def sel_cost(*args):
		sel_config = {}
		sel_config.update(resource_config)
		for i_k,kw in enumerate(sel_kwargs):
			sel_config[kw] = args[i_k]

		return estimate_resources(SEL, gate_set=CustomGateSet, config=sel_config)
	
	def tot_cost(*args):
		num_prep = len(prep_kwargs)

		my_prep_cost = prep_cost(*args[:num_prep])
		my_sel_cost = sel_cost(*args[num_prep:])

		return my_sel_cost + 2*my_prep_cost

	return tot_cost, (*prep_ranges, *sel_ranges)


def optimize_method(obt, tbt, method, eta, compact_ham_kwargs={}, alpha=0.95, heuristic="full_Q", verbose=True, **kwargs):
	'''
	Loop for obtaining best decomposition for a given method
	alpha is optimization heuristic parameter
	'''
	if verbose:
		print(f"Starting optimization for method {method}!")

	compact_ham, one_norm = create_compact_ham(obt, tbt, method, eta=eta, **compact_ham_kwargs)
	cost_func, opt_ranges = create_resource_function(compact_ham, method, **kwargs)

	opt_resources, opt_params = resource_optimizer(cost_func, *opt_ranges, heuristic=heuristic, verbose=verbose, alpha=alpha)

	return opt_resources, one_norm, opt_params

preopt_list = ["Sparse", "DF", "AC"]
def find_optimum(obt, tbt, eta, method_list, mixing_arr = np.linspace(0,1,num=11), compact_ham_kwargs={}, alpha=0.95, heuristic="full_Q", verbose=True, **kwargs):
	TIMES_ARR = [time()]

	num_methods = len(method_list)
	do_preopt = np.zeros(num_methods,dtype=bool)
	for i_method, method in enumerate(method_list):
		if method in preopt_list:
			do_preopt[i_method] = True

	if np.sum(do_preopt) > 0:
		if verbose:
			print(f"Found method which requires pre-optimization, applying orbital optimization and BLISS...")
		oo_obt, oo_tbt = oo.full_optimization(obt, tbt, mixing_arr)
		oo_obt = np.array(oo_obt)
		oo_tbt = np.array(oo_tbt)
		TIMES_ARR.append(time())
		if verbose:
			print(f"Optimized orbital frame, optimization time was {TIMES_ARR[-1] - TIMES_ARR[-2]:.2f} seconds")
		
		bliss_obt, bliss_tbt = bliss.bliss_linprog(obt, tbt, eta)
		bliss_obt = np.array(bliss_obt)
		bliss_tbt = np.array(bliss_tbt)
		
		oo_bliss_obt, oo_bliss_tbt = bliss.bliss_linprog(oo_obt, oo_tbt, eta)
		oo_bliss_obt = np.array(oo_bliss_obt)
		oo_bliss_tbt = np.array(oo_bliss_tbt)

		TIMES_ARR.append(time())
		if verbose:
			print(f"Ran BLISS, optimization time was {TIMES_ARR[-1] - TIMES_ARR[-2]:.2f} seconds")

	one_norms_list = []
	resources_list = []
	costs_list = []
	params_list = []
	method_final_list = []

	if verbose:
		print("\n\nStarting optimization over methods!")

	for i_method, method in enumerate(method_list):
		if verbose:
			print(f"\nOptimizing {method}...")
		my_res, my_one_norm, my_params = optimize_method(obt, tbt, method, eta, compact_ham_kwargs, alpha, heuristic, verbose=False, **kwargs)
		resources_list.append(my_res)
		params_list.append(my_params)
		one_norms_list.append(my_one_norm)
		costs_list.append(resource_cost(my_res, heuristic, alpha=alpha))
		method_final_list.append(method)
		
		if do_preopt[i_method]:
			bliss_res, bliss_one_norm, bliss_params = optimize_method(bliss_obt, bliss_tbt, method, eta, compact_ham_kwargs, alpha, heuristic, verbose=False, **kwargs)	
			resources_list.append(bliss_res)
			params_list.append(bliss_params)
			one_norms_list.append(bliss_one_norm)
			costs_list.append(resource_cost(bliss_res, heuristic, alpha=alpha))
			method_final_list.append(method + "-(BLISS)")

			oo_res, oo_one_norm, oo_params = optimize_method(oo_obt, oo_tbt, method, eta, compact_ham_kwargs, alpha, heuristic, verbose=False, **kwargs)	
			resources_list.append(oo_res)
			params_list.append(oo_params)
			one_norms_list.append(oo_one_norm)
			costs_list.append(resource_cost(oo_res, heuristic, alpha=alpha))
			method_final_list.append(method + "-(OO)")

			oo_bliss_res, oo_bliss_one_norm, oo_bliss_params = optimize_method(oo_bliss_obt, oo_bliss_tbt, method, eta, compact_ham_kwargs, alpha, heuristic, verbose=False, **kwargs)	
			resources_list.append(oo_bliss_res)
			params_list.append(oo_bliss_params)
			one_norms_list.append(oo_bliss_one_norm)
			costs_list.append(resource_cost(oo_bliss_res, heuristic, alpha=alpha))
			method_final_list.append(method + "-(OO-BLISS)")

		TIMES_ARR.append(time())
		if verbose:
			# print(f"Finished optimizing {method} method, found 1-norm of {my_one_norm:.2e} and resources \n{my_res}")
			print(f"{method} optimization took {TIMES_ARR[-1] - TIMES_ARR[-2]:.2f} seconds")

	tot_methods = len(method_final_list)
	hardness_list = [resources_list[ii].clean_gate_counts["T"] * one_norms_list[ii] for ii in range(tot_methods)]
	qubits_list = [resources_list[ii].qubit_manager.total_qubits for ii in range(tot_methods)]
	hardness_heuristic_list = [cost_heuristic(hardness_list[ii], qubits_list[ii], heuristic, alpha=alpha) for ii in range(tot_methods)]

	TIMES_ARR.append(time())
	if verbose:
		print(f"Finished optimizing all methods after {TIMES_ARR[-1] - TIMES_ARR[-2]:.2f} seconds!\n\n\n")
		for ii in range(tot_methods):
			print(f"Method {method_final_list[ii]} uses {qubits_list[ii]} qubits and {resources_list[ii].clean_gate_counts["T"]:.2e} T-gates with {one_norms_list[ii]:.2e} one-norm and {hardness_heuristic_list[ii]:.2e} cost heuristic")

	min_cost = min(hardness_heuristic_list)
	min_index = hardness_heuristic_list.index(min_cost)

	best_method = method_final_list[min_index]
	best_resources = resources_list[min_index]
	best_one_norm = one_norms_list[min_index]
	best_params = [int(pp) for pp in params_list[min_index]]
	cow_print(best_method, best_one_norm, best_resources.clean_gate_counts["T"], best_resources.qubit_manager.total_qubits, best_params)
	
	return best_method, best_resources, best_one_norm
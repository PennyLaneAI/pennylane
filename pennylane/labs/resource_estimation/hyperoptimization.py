import pennylane as qml
import numpy as np
import scipy as sp
import itertools

def extract_cost(resource_object):
	T_num = resource_object.clean_gate_counts["T"]
	num_qubits = resource_object.qubit_manager.total_qubits

	return T_num, num_qubits

heuristic_list = ["T", "Q", "full_T", "full_Q", "linear_mix", "Q3", "Q500"]
linear_heuristics = ["T", "Q", "full_T", "full_Q", "linear_mix"]
'''
Cost heuristics and what they do:
		legend:	- Q = # qubits
				- T = # T-gates
	- full_T: minimize T
	- full_Q: minimize Q
	- linear_mix: minimize alpha*T + (1-alpha)*Q
		- T: linear_mix for alpha = 0.95
		- Q: linear_mix for alpha = 0.05
	- Q3: minimize (Q**3) * T
'''
def cost_heuristic(T, Q, heuristic, **kwargs):
	if heuristic == "Q3":
		return T * (Q**3)
	if heuristic == "Q500":
		if Q < 500:
			return T
		else:
			return Q*1e20 + T
	if heuristic in linear_heuristics:
		if heuristic == "T":
			alpha = 0.95
		elif heuristic == "Q":
			alpha = 0.05
		elif heuristic == "full_T":
			alpha = 1 - 1e-8
		elif heuristic == "full_Q":
			alpha = 0 + 1e-8
		else:
			if "alpha" in kwargs:
				alpha = kwargs["alpha"]
			else:
				raise ValueError(f"Trying to use linear_mix heuristic but alpha not defined in kwargs!")

		return alpha*T + (1-alpha)*Q

def resource_cost(resource_object, heuristic, **kwargs):
	if heuristic not in heuristic_list:
		raise ValueError(f"Trying to use optimization heuristic {heuristic}, not implemented! Need to choose from list {heuristic_list}")

	T,Q = extract_cost(resource_object)

	return cost_heuristic(T, Q, heuristic, **kwargs)

def resource_optimizer(resource_func, *ranges, heuristic="Q3", verbose=True, **kwargs):
	'''
	For a function resource_func which for a set of N parameters returns a resource object, finds the best combination
	of parameters for each in the specified possible ranges according to the optimization heuristic
	'''
	num_params = len(ranges)
	if verbose:
		print(f"Starting resource optimizer function with heuristic {heuristic}")

	params_dims = [len(rr) for rr in ranges]
	params_iterables = [range(pp) for pp in params_dims]
	
	if verbose:
		print(f"Detected {num_params} variables to optimize:")
		print(f"Total number of algorithm variants will be {np.prod(params_dims)}")

	cost_matrix = np.zeros(tuple([*params_dims]))

	min_val = None
	min_combo = None
	for params_idxs in itertools.product(*params_iterables):
		current_combo = tuple(ranges[jj][params_idxs[jj]] for jj in range(num_params))
		if verbose:
			print(F"Trying current combo of {[int(combo) for combo in current_combo]}")
		my_resources = resource_func(*current_combo)
		my_T, my_Q = extract_cost(my_resources)
		if verbose:
			print(f"Number of qubits is {my_Q} and number of T-gates is {my_T:.2e}")
		cost_matrix[params_idxs] = resource_cost(my_resources, heuristic, **kwargs)

		if min_val is None:
			min_val = cost_matrix[params_idxs]
			min_combo = current_combo
		else:
			if cost_matrix[params_idxs] < min_val:
				min_val = cost_matrix[params_idxs]
				min_combo = current_combo

	if verbose:
		print(f"Found minimum cost value for combo {min_combo} with cost {min_val}")

	return resource_func(*min_combo), min_combo
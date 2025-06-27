import pennylane.labs.resource_estimation as plre
from pennylane.labs.resource_estimation.resource_tracking import resource_config
print(resource_config)

config = {'error_rx': 1e-09, 'error_ry': 1e-09, 'error_rz': 1e-09, 'precision_multiplexer': 1e-09, 'precision_qrom_state_prep': 1e-09, 'select_swap_depth': 1, 'parallel_rotations': 1}

qft = plre.ResourceQFT(5)

pa = plre.ResourcePhaseAdder(2, 10)

gate_set = {
    "X",
    "Y",
    "Z",
    "Hadamard",
    "CNOT",
    "S",
    "T"}
mult_rot = plre.ResourceMultiplexedRotation(10)

unif_state_prep = plre.ResourceUniformStatePrep(20)

ch = plre.CompactHamiltonian.thc(20, 40)
thc_select = plre.ResourceSelectTHC(ch)
res = plre.estimate_resources(thc_select, gate_set, config)


print(res)
print(res.clean_gate_counts["T"])
print(res.qubit_manager.total_qubits)

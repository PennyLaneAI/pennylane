import pennylane.labs.resource_estimation as plre


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

ch = plre.CompactHamiltonian.sparsepauli(3)
thc_select = plre.ResourceSelectSparsePauli(ch)
res = plre.estimate_resources(thc_select, gate_set)


print(res)
print(res.clean_gate_counts["T"])
print(res.qubit_manager.total_qubits)

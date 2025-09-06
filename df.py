import pennylane.labs.resource_estimation as plre

ham = plre.CompactHamiltonian.df(10,19,10,10)
res = plre.estimate_resources(plre.ResourceQubitizeDF(ham, select_swap_depths=2))

print(res)

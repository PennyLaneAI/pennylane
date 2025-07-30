import time
import numpy as np
from pennylane.labs.trotter_error import ProductFormula, perturbation_error, generic_fragments

if __name__ == "__main__":
    n_states = 1
    
    run_serial = False
    
    run_parallel = True

    for n_frags in [2]:
        for dim in [10_000]:

            print("n_frags: ", n_frags)
            frag_labels = list(range(n_frags)) + list(reversed(range(n_frags)))
            frag_coeffs = [1/2]*len(frag_labels)
            second_order = ProductFormula(frag_labels, frag_coeffs)
            u = 1 / (4 - 4**(1/3))
            v = 1 - 4*u
            fourth_order = second_order(u)**2 @ second_order(v) @ second_order(u)**2

            print("dim: ", dim)
            fragments = dict(enumerate(generic_fragments([np.random.random(size=(dim, dim)) for x in range(n_frags)])))
            states = [np.random.random(size=(dim,)) for x in range(1)]

            if run_serial:
                backend ="serial"
                num_workers = 1
                parallel_mode = "commutator"

                start = time.time()
                actual = perturbation_error(
                    fourth_order,
                    fragments,
                    states,
                    order=5,
                    backend=backend,
                    num_workers=num_workers,
                    parallel_mode=parallel_mode,
                )
                end = time.time()
                print("serial: ", end - start)
                print("result: ", actual)

            if run_parallel:
                backend ="mpi4py_pool"
                num_workers = 4
                parallel_mode = "state"
                n_states = 1

                start = time.time()
                actual = perturbation_error(
                    fourth_order,
                    fragments,
                    states,
                    order=6,
                    backend=backend,
                    num_workers=num_workers,
                    parallel_mode=parallel_mode,
                )
                end = time.time()
                print(f"Mode:{parallel_mode}[thrd={num_workers}]: ", end - start)
                print("result: ", actual)
            
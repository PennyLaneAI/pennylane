import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
from tqdm import tqdm
from subroutines import basic_simulation, load_hamiltonian, split_hamiltonian

import matplotlib.pyplot as plt
import scienceplots

plt.style.use('science')

time_steps = 1/(2**np.array(range(1, 8)))
n_steps = 1
random_weights = True
device_name = 'default.qubit'

H, n_wires = load_hamiltonian(layout="1x4")
H0, H1 = split_hamiltonian(H)
hamiltonian = (H0, H1)

device = qml.device(device_name, wires=n_wires)

method_errors={}

range_s = [1, 2, 2] #, 3, 3]
range_m = [1, 2, 3] #, 5, 6]
for s, m in tqdm(zip(range_s, range_m), desc='CFQM'):
    method_errors[(s, m)] = []
    for time in time_steps:
        error = basic_simulation(hamiltonian, time, n_steps, s, m, device, n_wires,
                                n_samples = 4)
        method_errors[(s, m)].append(error)

# Plot the results
ax, fig = plt.subplots()
for s, m in method_errors.keys():
    plt.plot(time_steps, method_errors[(s, m)], label = f"$(s, m) = ({s}, {m})$")
    # Compute slope b*x**a
    a, b = np.polyfit(np.log(time_steps), np.log(method_errors[(s, m)]), 1)
    print(f"({s}, {m}): {a}")

plt.yscale('log')
plt.xscale('log')

plt.xlabel('Time step')
plt.ylabel('Error')

plt.legend(fontsize='small')

plt.savefig('labs/2termHamiltonian/results/method_errors.pdf', bbox_inches='tight', format='pdf')


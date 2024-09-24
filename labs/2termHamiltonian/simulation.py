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

orders = [[4, 2], [6, 2], [8, 2],
        [8, 4], [10, 4],
        [8, 6], [10, 6]]
stages = [2, 3, 4,
        5, 7,
        7, 8]
method_errors['NearIntegrable'] = {}
for order, stage in tqdm(zip(orders, stages), desc='NI'):
    method_errors['NearIntegrable'][tuple(order)] = []
    for time in time_steps:
        error = basic_simulation(hamiltonian, time, n_steps, 'NearIntegrable', order, device, n_wires,
                                n_samples = 4, **{'processing': False, 'stages': stage})
        method_errors['NearIntegrable'][tuple(order)].append(error)

range_s, range_m = [1, 2], [1, 2]#, 2] #, 3, 3]
method_errors['InteractionPicture'] = {}
for s, m in tqdm(zip(range_s, range_m), desc='CFQM'):
    method_errors['InteractionPicture'][(s, m)] = []
    for time in time_steps:
        error = basic_simulation(hamiltonian, time, n_steps, 'InteractionPicture', 2*s, device, n_wires,
                                n_samples = 4, **{'m': m})
        method_errors['InteractionPicture'][(s, m)].append(error)

method_errors['ProductFormula'] = {}
for order in tqdm([1, 2, 4], desc='PF'):
    method_errors['ProductFormula'][order] = []
    for time in time_steps:
        error = basic_simulation(hamiltonian, time, n_steps, 'ProductFormula', order, device, n_wires,
                                n_samples = 4, **{'m': m})
        method_errors['ProductFormula'][order].append(error)

# Plot the results
ax, fig = plt.subplots()

for o in method_errors['NearIntegrable'].keys():
    plt.plot(time_steps, method_errors['NearIntegrable'][o], label = f"NI, order = {o}", linestyle=':')

for s, m in method_errors['InteractionPicture'].keys():
    plt.plot(time_steps, method_errors['InteractionPicture'][(s, m)], label = f"IP, order = {2*s}")

for order in [1, 2, 4]:
    plt.plot(time_steps, method_errors['ProductFormula'][order], label = f"PF, order = {order}", linestyle='--')

plt.yscale('log')
plt.xscale('log')

plt.xlabel('Time step')
plt.ylabel('Error')

plt.legend(fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.savefig('labs/2termHamiltonian/results/method_errors.pdf', bbox_inches='tight', format='pdf')


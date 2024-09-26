import os
import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
from tqdm import tqdm
from subroutines import basic_simulation, load_hamiltonian, split_hamiltonian

import matplotlib.pyplot as plt
import scienceplots

plt.style.use('science')

time_steps = 1/(2**np.array(range(0, 7)))
n_steps = 1
random_weights = True
device_name = 'lightning.qubit'
name = 'Ising'
layout = '1x4'
periodicity = 'open'

H, n_wires = load_hamiltonian(name = name, layout=layout, periodicity=periodicity)
H0, H1 = split_hamiltonian(H)
hamiltonian = (H0, H1)

device = qml.device(device_name, wires=n_wires)

method_errors={}
method_costs={}

method_errors['ProductFormula'] = {}
method_costs['ProductFormula'] = {}
orders = [2, 4, 6, 8, 10]
stages = [1, 5, 13, 21, 35]
identifiers = ['Strang', 'SUZ90', 'SS05', 'SS05', 'SS05']
for order, stage, identifier in tqdm(zip(orders, stages, identifiers), desc='PF'):
    method_errors['ProductFormula'][order] = []
    for time in time_steps:
        error, resources = basic_simulation(hamiltonian, time, n_steps, 'ProductFormula', order, device, n_wires,
                                n_samples = 4, **{'processing': False, 'stages': stage, 'identifier': identifier})
        method_errors['ProductFormula'][order].append(error)
        method_costs['ProductFormula'][order] = resources.gate_types['RX'] + resources.gate_types['RZ'] + resources.gate_types['RY']


range_s, range_m = [1, 2, 3], [1, 2, 5]
stages = [1, 5, 13]
identifiers = ['Strang', 'SUZ90', 'SS05']
method_errors['InteractionPicture'] = {}
method_costs['InteractionPicture'] = {}
for s, m, stage, identifier in tqdm(zip(range_s, range_m, stages, identifiers), desc='CFQM'):
    method_errors['InteractionPicture'][(s, m)] = []
    for time in time_steps:
        error, resources = basic_simulation(hamiltonian, time, n_steps, 'InteractionPicture', 2*s, device, n_wires,
                                n_samples = 4, **{'m': m, 'stages': stage, 'identifier': identifier})
        method_errors['InteractionPicture'][(s, m)].append(error)
        method_costs['InteractionPicture'][(s, m)] = resources.gate_types['RX'] + resources.gate_types['RZ'] + resources.gate_types['RY']


orders = [[4, 2], [6, 2], [8, 2],
        [8, 4], [10, 4],
        [8, 6], [10, 6]]
stages = [2, 3, 4,
        5, 7,
        7, 8]
method_errors['NearIntegrable'] = {}
method_costs['NearIntegrable'] = {}
for order, stage in tqdm(zip(orders, stages), desc='NI'):
    method_errors['NearIntegrable'][tuple(order)] = []
    for time in time_steps:
        error, resources = basic_simulation(hamiltonian, time, n_steps, 'NearIntegrable', order, device, n_wires,
                                n_samples = 4, **{'processing': False, 'stages': stage})
        method_errors['NearIntegrable'][tuple(order)].append(error)
        method_costs['NearIntegrable'][tuple(order)] = resources.gate_types['RX'] + resources.gate_types['RZ'] + resources.gate_types['RY']


# Definir la ruta del directorio
directory = f'labs/2termHamiltonian/results/{name}_{layout}_{periodicity}'

# Verificar si el directorio existe, si no, crearlo
if not os.path.exists(directory):
    os.makedirs(directory)
    print(f'Directorio creado: {directory}')
else:
    print(f'El directorio ya existe: {directory}')

# Plot the results
ax, fig = plt.subplots()

for o in method_errors['NearIntegrable'].keys():
    plt.plot(time_steps, method_errors['NearIntegrable'][o], label = f"NI, order = {o}", linestyle=':')

colors = ['b', 'g', 'r']#, 'c']
for (s, m), c in zip(method_errors['InteractionPicture'].keys(), colors):
    plt.plot(time_steps, method_errors['InteractionPicture'][(s, m)], label = f"IP, order = {2*s}", color=c)

colors = ['b', 'g', 'r', 'c', 'm']
for order, c in zip(method_errors['ProductFormula'].keys(),colors):
    plt.plot(time_steps, method_errors['ProductFormula'][order], label = f"PF, order = {order}", linestyle='--', color = c)

plt.yscale('log')
plt.xscale('log')

plt.xlabel('Time step')
plt.ylabel('Error')

plt.legend(fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')

path = os.path.join(directory, 'method_errors.pdf')
plt.savefig(path, bbox_inches='tight', format='pdf')

plt.close()

fig, ax = plt.subplots()
for o in method_errors['NearIntegrable'].keys():
    plt.plot(method_costs['NearIntegrable'][o]/time_steps,
            method_errors['NearIntegrable'][o]/time_steps,
            label = f"NI, order = {o}", linestyle=':')

colors = ['b', 'g', 'r', 'c']
for (s, m), c in zip(method_errors['InteractionPicture'].keys(), colors):
    plt.plot(method_costs['InteractionPicture'][(s, m)]/time_steps,
            method_errors['InteractionPicture'][(s, m)]/time_steps,
            label = f"IP, order = {2*s}", color=c)

colors = ['b', 'g', 'r', 'c', 'm']
for order, c in zip(method_errors['ProductFormula'].keys(), colors):
    plt.plot(method_costs['ProductFormula'][order]/time_steps,
            method_errors['ProductFormula'][order]/time_steps,
            label = f"PF, order = {order}", linestyle='--', color = c)

plt.yscale('log')
plt.xscale('log')

plt.xlabel('Single qubit rotation cost ($T = 1$)')
plt.ylabel('Error')

plt.legend(fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')

# Create folder f'labs/2termHamiltonian/results/{name}_{layout}_{periodicity}' if it does not exist

path = os.path.join(directory, 'method_errors_cost.pdf')
plt.savefig(path, bbox_inches='tight', format='pdf')



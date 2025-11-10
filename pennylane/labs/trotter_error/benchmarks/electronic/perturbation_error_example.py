import time
import numpy as np

from pennylane_block2 import MPOperator, MPState, from_cdf, from_molecule

from pennylane.labs.trotter_error import ProductFormula
from pennylane.labs.trotter_error.product_formulas.error import perturbation_error


molecules = {
    'LiMnO_N6': (6, 8),
    'LiMnO_N9': (9, 14),
    'LiMnO_N10': (10, 10),
    'LiMnO_N11': (11, 14),
    'LiMnO_N14': (14, 20),
    'LiMnO_N18': (18, 24),
    'LiMnO_N28': (28, 28),
}

# Define system size
k = 6

ncas, nelecas = molecules[f"LiMnO_N{k}"]
nroots = 1
dmrg_nroots = max(2, nroots)

# Define data path
cdf_folder = f'/home/soran/Soran/Data/LiMnO/LiMnO_N{k}'

U0 = np.load(cdf_folder + '/cdf_results/onebody_U0.npy')
Z0 = np.load(cdf_folder + '/cdf_results/onebody_Z0.npy')
Z0 = np.diag(Z0)

U = np.load(cdf_folder + '/cdf_results/two_body_leaves.npy')
Z = np.load(cdf_folder + '/cdf_results/two_body_cores.npy')

core_const = np.load(cdf_folder + '/core_const.npy').item()

driver, frags, mp_states, h1e, eri = from_cdf(U, Z, U0, Z0, core_const, ncas, nelecas, dmrg_nroots, pyblock2=True)
state = mp_states[0]

# Trotter constants
u = 1 / (4 - 4 ** (1 / 3))
v = 1 - 4 * u

timestep, order = 1, 5

frags = dict(enumerate(frags))
frag_labels = list(frags.keys()) + list(frags.keys())[::-1]

second_order = ProductFormula(list(zip(frag_labels, [1 / 2] * len(frag_labels))))
fourth_order = ProductFormula.prod([second_order(u)**2, second_order(v), second_order(u)**2])

start = time.time()
error = perturbation_error(second_order, frags, [state], max_order=3)
end = time.time()
print("time: ", end - start, error, type(frags[0]))

import os
import numpy as np

from pennylane.labs.trotter_error.subroutines import H, PTerrorTensor, _load_cdf
from pyscf import gto, scf
from scipy.linalg import expm
import matplotlib.pyplot as plt
# Load the cdf data and use the subroutines to compute Y3 error.

# where the cdf integrals are stored
sizes = ([5, 4], [6,8], [8, 10], [10, 12])
fig, ax = plt.subplots(1, 1, figsize=(7, 7))
Y3s = []
for ncas, nelecas in sizes:
    cdf_folder = os.path.join(os.path.dirname(__file__), f"C4_CDFs/opt_hamiltonian_c4_sq_{ncas}_{nelecas}")

    X, Z, U0, Z0, core_const = _load_cdf(cdf_folder, verbose=True)
    U = expm(X)

    h1e = np.einsum('pq,qr,rs', U0, Z0, U0)
    eri = np.einsum('tpk,tqk,tkl,trl,tsl->pqrs', U, U, Z, U, U)

    hamiltonian = H(ncas, nelecas, 0, None, h1e, eri, core_const)

    pt = PTerrorTensor(
        H = hamiltonian,
        driver = None,
        name = 'C4',
        cdf_loc = cdf_folder,
        bond_dim = 250
    )

    Y3 = pt.compute_error()
    Y3s.append(Y3)

    print(f"Error terms: Y3 = {Y3}")



fig, ax = plt.subplots(1, 1, figsize=(7, 7))


# Fit line to log data
y = np.log(np.array(Y3s))
x = np.array(sizes)[:, 0]

slope, intercept = np.polyfit(x, y, 1)
ax.plot(x, np.exp(slope*x + intercept), label=fr"Fit: $\mathrm{{(\exp(N))^{{{round(slope, 3)}}} \cdot \exp({round(intercept, 3)})}}$")
ax.plot(x, np.exp(y), 'o', label=r"$C_{4}$")

ax.set_xlabel("N", fontsize=12)
ax.set_ylabel(r"$\|Y_3\| $", fontsize=12)

ax.set_yscale('log')

ax.legend(fontsize=12)
plt.savefig("C4_error.pdf")


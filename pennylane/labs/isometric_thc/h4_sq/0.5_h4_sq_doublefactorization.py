import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import scienceplots # needed to execute the line below
plt.style.use('science')
from scipy.linalg import expm

from pyscf import fci

from jax import numpy as jnp
import jax
from jax import config
config.update("jax_enable_x64", True)

from pennylane.labs.isometric_thc.mol_prep import molecule, Hamiltonian, to_density
from pennylane.labs.isometric_thc.compressed_factorize \
            import factorize_onebody, minimize_k1, one_body_correction, optimization_schedule

# directories to save data
dirpath = os.path.dirname(os.path.realpath(__file__))
save_h_dir = "opt_hamiltonian_h4_sq"
save_h_dir = os.path.join(dirpath, save_h_dir)
if not os.path.isdir(save_h_dir): os.mkdir(save_h_dir)
save_fig_dir = "figures_h4_sq"
save_fig_dir = os.path.join(dirpath, save_fig_dir)
if not os.path.isdir(save_fig_dir): os.mkdir(save_fig_dir)

bliss = True
commutator = False

### ==============    Set up the problem    ==================
# Create a mol object
r = 0.5
geom = f"""
H 0.0 0.0 0.0
H 0.0 0.0 {r}
H 0.0 {1.5*r} {r}
H 0.0 {2.5*r} 0.0
"""
basis = 'sto3g'
mol = molecule(geom, basis, sym=None)

# Get the MO coefficients and the one and two body integrals
nelec, norb = mol.nelectron, mol.nao
# get integrals
core_const, one, two = Hamiltonian(mol)
h1e, eri = to_density(one, two)

# fix how many roots we will check the energy error over
nroots = 10

### ==============    Optimization of two-electron part    ==================
pts = 30 # Number of parallel optimization instances (???)
opt_X, opt_Z = jnp.zeros((0, norb, norb)), jnp.zeros((0, norb, norb))


k1 = jnp.float64(0.0)
k2 = jnp.float64(0.0)
F = jnp.zeros((norb, norb))

max_ncdf = 2*norb
energies = []
N1 = jnp.eye(norb)
N2 = jnp.einsum('ij,kl->ijkl', N1, N1)
for n_cdf in tqdm(range(1, max_ncdf+2), desc = 'Number of terms in the CDF'):
    # expand pts copies of k2 and F along their first dimension
    k2 = jnp.repeat(jnp.expand_dims(k2, 0), pts, axis = 0)
    F = jnp.repeat(jnp.expand_dims(F, 0), pts, axis = 0)

    cdf_eri, opt_X, opt_Z, k2, F = optimization_schedule(pts, n_cdf, eri, \
                                n_steps=500, prior_X = opt_X, prior_Z = opt_Z, k2 = k2, F = F,
                                additional_cdf_terms = not (bliss and n_cdf == max_ncdf+1),
                                bliss = (bliss and n_cdf == max_ncdf+1), cdf = not (bliss and n_cdf == max_ncdf+1), M = 1, verbose=False)

    # compute the energies for reference
    print(f"Calculating lowest {nroots} states with factorized"
        f" two-electron integrals to check relative energy error")
    
    T = jnp.einsum('pq,rs->pqrs', F, N1)/2. + jnp.einsum('pq,rs->pqrs', N1, F)/2.
    approx_eri = cdf_eri + (k2*N2 + T).reshape(norb**2, norb**2)

    e_CDF, civec = fci.direct_spin1.kernel(h1e, approx_eri, norb, \
                        mol.nelectron, tol=1e-12, lindep=1e-14, \
                            max_cycle=1000, nroots = nroots, verbose=0)

    energies.append(e_CDF)
    print(f"Reference energies for n_cdf = {n_cdf} obtained\n")

    # combine the variables for saving
    opt_XZ_params = {'X': opt_X, 'Z': opt_Z, 'k2': k2, 'F': F}

### ==============    Optimization of one-electron part    ==================
if bliss: k1 = minimize_k1(k1, h1e, nelec, F,
                            cdf_eri.reshape(norb, norb, norb, norb), 
                            commutator = commutator) 

h1ebliss = h1e + nelec*F/2. - k1*N1
core_const_bliss = core_const + k1*nelec + k2*nelec**2/2.

XZ = jnp.stack((opt_X, opt_Z), axis = 0)

### ==============    Saving    ==================
optU = expm(np.array(opt_X))
obc = one_body_correction(optU, opt_Z)
UZ0 = factorize_onebody(h1ebliss+obc)
if not os.path.isdir(save_h_dir): os.mkdir(save_h_dir)
np.save(f'{save_h_dir}/CDF_twobody.npy', XZ)
np.save(f'{save_h_dir}/CDF_onebody.npy', UZ0)
np.save(f'{save_h_dir}/CDF_core.npy', core_const_bliss)
np.save(f'{save_h_dir}/BLISS_k1.npy', k1)
np.save(f'{save_h_dir}/BLISS_k2.npy', k2)
np.save(f'{save_h_dir}/BLISS_F.npy', F)


# generate some classical low-energy subspace for comparison
print(f"Calculating lowest {nroots} states with exact two-electron integrals"
      f" to check relative energy error")
e_exact, civecs = fci.direct_spin1.kernel(h1e, eri, norb, mol.nelectron, \
                    tol=1e-12, lindep=1e-14, max_cycle=1000, \
                        nroots = nroots, verbose=0)
print(f"Exact reference energies obtained\n"
      f"{[np.round(e, 5) for e in e_exact]}")
spins_exact = []
for ii, civec in enumerate(civecs):
    s2, m = fci.spin_op.spin_square(civec, norb, mol.nelectron)
    spins_exact.append(s2)
print(f"S^2 value of the states\n{[np.round(s2, 5) for s2 in spins_exact]}")


### =========    Plotting the results ===============

fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 1, max_ncdf+1))
for ii in range(max_ncdf+1):
    ax.semilogy(range(1,len(e_exact)+1) , np.abs(energies[ii]-e_exact) / \
                np.abs(e_exact), label=f"CDF-{ii+1}", color = colors[ii])

ax.set_ylabel("relative energy error" , fontsize=15)
ax.set_xlabel("state" , fontsize=15)

ax.tick_params(labelsize=15)

ax.legend(fontsize=15, bbox_to_anchor=(1.03, 1))

if not os.path.isdir(save_fig_dir):
    os.mkdir(save_fig_dir)
plt.savefig(f"{save_fig_dir}/relative_error.png", format="png")
plt.close()

# Plot the norm of the l+1 terms in the CDF
fig, ax = plt.subplots(figsize=(10, 6))

ys = [jnp.max(jnp.abs(UZ0[1]))]
for ii in range(max_ncdf):
    ys.append(jnp.max(jnp.abs(XZ[1,ii])))

ax.plot(range(0, max_ncdf+1), ys, marker='o', markersize=10)

# y logscale
ax.set_yscale('log')

# x labels are only integer
ax.set_xticks(range(0, max_ncdf+1))

ax.set_ylabel(r"$\max |Z_{ij}|$" , fontsize=15)
ax.set_xlabel("CDF terms" , fontsize=15)

ax.tick_params(labelsize=15)

if not os.path.isdir(save_fig_dir):
    os.mkdir(save_fig_dir)
plt.savefig(f"{save_fig_dir}/norm_CDF_terms.png", format="png")
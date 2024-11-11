import numpy as np

from pennylane.labs.trotter_error.subroutines import H, PTerrorTensor
from pyscf import gto, scf
from scipy.linalg import expm
# Load the cdf data and use the subroutines to compute Y3 error.

name = "h4_sq"
r = 0.5
geom = f"""
H 0.0 0.0 0.0
H 0.0 0.0 {r}
H 0.0 {1.5*r} {r}
H 0.0 {2.5*r} 0.0
"""
# simple molecule, trivial basis, no symmetry
mol = gto.Mole(atom=geom, basis='sto-3g', symmetry='C1')
mol.build()
print(f"full space: nao = {mol.nao}, nelectrons = {mol.nelectron}")

ncas = mol.nao
nelecas = mol.nelectron

cdf_path = "/Users/pablo.casares/Developer/xas_simulation/opt_hamiltonian_h4_sq"

def _load_cdf(folder, verbose = False):
    """
    Helper function to load the pre-computed cdf one 
    and two electron integrals.
    """
    # load the one-electron integrals
    UZ0 = np.load(f'{folder}/CDF_onebody.npy', allow_pickle=True)
    U0, Z0 = UZ0[0], UZ0[1]

    # load the two-electron integrals
    opt_XZ = np.load(f'{folder}/CDF_twobody.npy', allow_pickle=True)
    X, Z = opt_XZ[0], opt_XZ[1]

    # load the core constant
    core_const = np.load(f'{folder}/CDF_core.npy', allow_pickle=True)

    #check whether X is antisymmetric and Z is symmetric
    if verbose:
        print(f'Is X antisymmetric', np.isclose(X, -np.transpose(X, (0,2,1))).all())
        print(f'Is Z symmetric', np.isclose(Z, np.transpose(Z, (0,2,1))).all())

    return X, Z, U0, Z0, core_const

# broadening
eta = 0.05
bond_dim = 250

# perform Hartree-Fock to get orbitals
hf = scf.RHF(mol)
# this is essential -- prevents the PySCF 
# flip-flopping from multithreading
def round_eig(f):
    return lambda h, s: f(h.round(12), s)
hf.eig = round_eig(hf.eig)
hf.run(verbose=0)

X, Z, U0, Z0, core_const = _load_cdf(cdf_path, verbose=True)
U = expm(X)

h1e = np.einsum('pq,qr,rs', U0, Z0, U0)
eri = np.einsum('tpk,tqk,tkl,trl,tsl->pqrs', U, U, Z, U, U)

hamiltonian = H(ncas, nelecas, 0, None, h1e, eri, core_const)

pt = PTerrorTensor(
    H = hamiltonian,
    driver = None,
    name = 'H4',
    cdf_loc = cdf_path,
    bond_dim = bond_dim
)

Y3 = pt.compute_error()

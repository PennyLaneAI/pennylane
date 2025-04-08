from pyscf import ao2mo, scf, gto
import numpy as np
from pennylane.qchem import active_space

def molecule(geom, basis="sto3g", sym="C1"):
    mol = gto.Mole(atom=geom, basis=basis, symmetry=sym)
    mol.build()
    return mol

def Hamiltonian(mol, ncas=None, nelecas=None):

    # get MOs
    hf = scf.RHF(mol)
    # this is essential -- prevents the PySCF 
    # flip-flopping from multithreading
    def round_eig(f):
        return lambda h, s: f(h.round(12), s)
    hf.eig = round_eig(hf.eig)
    hf.run(verbose=0)

    # create h1 -- one-body terms
    h_core = hf.get_hcore(mol)
    orbs = hf.mo_coeff
    core_constant = mol.energy_nuc()
    one = np.einsum("qr,rs,st->qt", orbs.T, h_core, orbs)
    # create h2 -- two-body terms
    two = ao2mo.full(hf._eri, orbs, compact=False).reshape([mol.nao]*4)
    two = np.swapaxes(two, 1, 3)

    # take active space into account (from pennylane)
    core, active = active_space(mol.nelectron, mol.nao, \
                                        2*mol.spin+1, nelecas, ncas)

    if core and active:
        for i in core:
            core_constant = core_constant + 2 * one[i][i]
            for j in core:
                core_constant = core_constant + \
                    2 * two[i][j][j][i] - two[i][j][i][j]

        for p in active:
            for q in active:
                for i in core:
                    one[p, q] = one[p, q] + \
                        (2 * two[i][p][q][i] - two[i][p][i][q])

        one = one[qml.math.ix_(active, active)]
        two = two[qml.math.ix_(active, active, active, active)]

    return core_constant, one, two

def to_density(one, two):
    """
    Converts the integrals from the physicist to the chemist convention.

    The 1 body correction has to be divided by 2.
    However, the 2 body term already incorporates the factor of 2 in the
    circuit.
    """
    eri =  np.einsum('prsq->pqrs', two)
    h1e = one - np.einsum('pqrr->pq', two)/2.

    return h1e, eri
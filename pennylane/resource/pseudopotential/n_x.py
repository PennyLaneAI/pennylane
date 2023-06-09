# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This module contains the functions needed for estimating the number of logical qubits and
non-Clifford gates for quantum algorithms in first quantization using a plane-wave basis
and pseudopotentials.
"""

import pennylane as qml
from pennylane import numpy as np

# constants, configs, utility functions

angs2bohr = 1.8897259886

eta = 100
basis_vectors = np.array([[5.02, 0, 0], [0, 5.40, 0], [0, 0, 6.26]])  # Li2FeSiO4
PP_NL_params = {
    'Li': {'Z_ion': 1, 'rs': [0.666375, 1.079306, 0.], 'Bi_inv': [1.858811, -0.005895, 0.]},
    'O': {'Z_ion': 6, 'rs': [0.221786, 0.256829, 0.], 'Bi_inv': [18.266917, 0., 0.]},
    'Si': {'Z_ion': 4, 'rs': [0.422738, 0.484278, 0.], 'Bi_inv': [5.906928, 2.727013, 0.]},
    'Fe': {'Z_ion': 8, 'rs': [0.454482, 0.638903, 0.308732],
           'Bi_inv': [3.016640, 1.499642, -9.145354]}}

# PP parameters, Bi_inv are the inverse values of the B_i's in the paper
PP_loc_params = {'Li': {'Z_ion': 1, 'r_loc': 0.787553, 'Cs': [-1.892612, 0.286060]},
                 'O': {'Z_ion': 6, 'r_loc': 0.247621, 'Cs': [-16.580318, 2.395701]},
                 'Si': {'Z_ion': 4, 'r_loc': 0.440000, 'Cs': [-7.336103, 0.]},
                 'Fe': {'Z_ion': 8, 'r_loc': 0.430000, 'Cs': [-6.654220, 0.]}}

atoms_and_rep_uc = {'Li': 4, 'O': 8, 'Si': 2, 'Fe': 2}

list_atoms, atoms_rep_uc = list(atoms_and_rep_uc.keys()), np.array(list(atoms_and_rep_uc.values()))


def abs_sum(recip_bv):
    abs_sum = 0
    for b_omega in recip_bv:
        for b_omega_prime in recip_bv:
            abs_sum += np.abs(np.sum(b_omega * b_omega_prime))
    return abs_sum


def compute_Omega(vecs):
    # print('Omega in Angs^3 ', np.abs(np.sum((np.cross(vecs[0],vecs[1])*vecs[2]))))
    return np.abs(np.sum((np.cross(vecs[0], vecs[1]) * vecs[2]))) * angs2bohr ** 3


def compute_recip_bv(omega, basis_vectors):
    return 2 * np.pi / omega * np.array([np.cross(basis_vectors[i], basis_vectors[j]) for i, j in
                                         [(1, 2), (2, 0), (0, 1)]]) * angs2bohr ** 2


# from error_X2n_X
def G_p(p, recip_bv):
    return np.sum(np.array(p) * recip_bv, axis=0)


def G_lattice(n_p):
    return list(itertools.product(range(-2 ** (n_p - 1) + 1, 2 ** (n_p - 1)), repeat=3))


def Ps(n, br):
    theta = 2 * np.pi / (2 ** br) * np.round(
        (2 ** br) / (2 * np.pi) * np.arcsin(np.sqrt(2 ** (np.ceil(np.log2(n))) / (4 * n))))
    braket = (1 + (2 - (4 * n) / (2 ** np.ceil(np.log2(n)))) * (np.sin(theta)) ** 2) ** 2 + (
        np.sin(2 * theta)) ** 2
    return n / (2 ** np.ceil(np.log2(n))) * braket


def compute_lambda_NL_prime(eta, omega, n_p, list_atoms, recip_bv, PP_NL_params, atoms_rep):
    NL0 = np.array([0.] * len(list_atoms))
    NL1omega = np.array([[0.] * 3] * len(list_atoms))
    NL20 = np.array([0.] * len(list_atoms))
    NL2omegaomega = np.array([[0.] * 6] * len(list_atoms))

    mcG = G_lattice(n_p)
    G_ps = np.array([G_p(p, recip_bv) for p in mcG])
    G_pnorms = np.array([np.linalg.norm(G_p) for G_p in G_ps])
    G_pnormsexps = np.array([np.exp(-nGp ** 2) for nGp in G_pnorms])

    G_psomega = [np.array([Gp[omega] for Gp in G_ps]) for omega in range(3)]
    G_psomegasquared = [x ** 2 for x in G_psomega]
    G_psomegaomega = [G_psomega[i] * G_psomega[j] for i, j in
                      [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (2, 1)]]  # why this combination
    G_psomegaomegasquared = [x ** 2 for x in G_psomegaomega]
    G_pnorms = np.array([np.linalg.norm(G_p) for G_p in G_ps])
    G_pnormsquad = G_pnorms ** 4

    multipliers = [{}, {}, {}, {}]

    for idx, atom in enumerate(list_atoms):
        I = PP_NL_params[atom]
        rs, Bi_inv = I['rs'], I['Bi_inv']

        multipliers[0][atom] = (G_pnormsexps ** (rs[0] ** 2)).sum()
        NL0[idx] = atoms_rep[idx] * rs[0] ** 3 * abs(Bi_inv[0]) * multipliers[0][atom]
        multipliers[1][atom] = []
        for i in range(3):
            multipliers[1][atom].append((G_psomegasquared[i] * G_pnormsexps ** (rs[1] ** 2)).sum())
            NL1omega[idx][i] = atoms_rep[idx] * rs[1] ** 5 * abs(Bi_inv[1]) * multipliers[1][atom][
                -1]
        multipliers[2][atom] = (G_pnormsquad * G_pnormsexps ** (rs[2] ** 2)).sum()
        NL20[idx] = atoms_rep[idx] * rs[2] ** 7 * abs(Bi_inv[2]) * multipliers[2][atom]

        multipliers[3][atom] = []
        for i in range(6):
            multipliers[3][atom].append(
                (G_psomegaomegasquared[i] * G_pnormsexps ** (rs[2] ** 2)).sum())
            NL2omegaomega[idx][i] = (int(i > 2) + 1) * atoms_rep[idx] * rs[2] ** 7 * abs(
                Bi_inv[2]) * multipliers[3][atom][-1]

    multipliers = multipliers

    NL0 = NL0 * 8 * np.pi * eta / omega
    NL1omega = NL1omega * 32 * np.pi * eta / (3 * omega)
    NL20 = NL20 * 64 * np.pi * eta / (45 * omega)
    NL2omegaomega = NL2omegaomega * 64 * np.pi * eta / (15 * omega)

    NL0sum = NL0.sum()
    NL1omegasum = list(NL1omega.sum(axis=0))
    NL20sum = NL20.sum()
    NL2omegaomegasum = list(NL2omegaomega.sum(axis=0))

    lambda_NL_prime = [NL0sum] + NL1omegasum + [NL20sum] + NL2omegaomegasum

    return lambda_NL_prime, multipliers

def compute_nx(lambda_total, error_x):
    r"""Eq. (J6)"""

    scalar = 4 * np.pi * lambda_total

    integral = 1

    return np.ceil(np.log2(scalar * integral / error_x))

def compute_nb(recip_bv, error_b, lattice = None):
    r"""Eq. (J7)"""

    scalar = 2 * np.pi * eta * 2**(2 * n_p - 2) * abs_sum(recip_bv)

    if lattice = 'ortho':
        scalar *= 2

    integral = 1

    return np.ceil(np.log2(scalar * integral / error_b))

def compute_nnl(list_atoms, lambda_NL_prime, error_nl):
    r"""Eq. (J8)"""

    n_Ltype = np.ceil(np.log2(len(list_atoms)))

    scalar = 2 * (n_Ltype + 4) * np.pi * eta * sum(lambda_NL_prime) # eta is missing

    integral = 1

    return np.ceil(np.log2(scalar * integral / error_nl))

def compute_nmv(eta, omega, b_min, n_p, error_mv):
    r"""Eq. (J10)"""

    scalar = 8 * np.pi * eta * (eta - 1) / (omega * b_min**2) # (self.Omega**self.bmin**2) in code

    integral = 7 * 2**(n_p + 1) - 9 * n_p - 11 - 3 / 2**n_p # +11

    return np.ceil(np.log2(scalar * integral / error_mv))

def compute_nmloc(error_mloc):
    r"""Eq. (J10)"""

    n_Ltype = np.ceil(np.log2(len(list_atoms)))

    maxnaPa = np.max([atoms_rep[idx] / Ps(atoms_rep[idx], b_r) for idx,_ in enumerate(list_atoms)])

    scalar = 8 * np.pi**2 * eta / omega

    integral = integral_loc_error * maxnaPa * (3 * n_p + n_Ltype)

    return np.ceil(np.log2(scalar * integral / error_mloc))


def compute_integral_loc_error():

    integral_loc = 0
    integral_loc_error = 0

    for idx, atom in enumerate(list_atoms):
        I = PP_loc_params[atom]

        # Gnu_norm?
        X_I = np.sum([atoms_rep[idx] * abs(k_loc(Gnu_norm, I)) / (Ps(atoms_rep[idx], b_r) * Gnu_norm**2 )\
              if abs(Gnu_norm) > 1e-7 else 0 \
              for Gnu_norm in G_pnorms])

        integral_loc += X_I
        integral_loc_error += (X_I * Ps(atoms_rep[idx], b_r) / atoms_rep[idx])

    return integral_loc_error


def k_loc(G_norm, I): #I contains Z_ion, r_loc, Cs, this is function gamma in the paper

    Z_ion, r_loc, Cs = I['Z_ion'], I['r_loc'], I['Cs']
    D1 = (Cs[0] + 3 * Cs[1]) * np.sqrt(np.pi) * r_loc**3 / 2
    D2 = Cs[1] * np.sqrt(np.pi) * r_loc**5 / 2
    return np.exp(-(G_norm * r_loc)**2 / 2) * (-Z_ion + (D1 - D2 * G_norm**2) * G_norm**2)

def compute_nr(basis_vectors, error_r):

    integral = sum_kloc_over_Gnu(n_p) + sum_kNL_over_Gnu(n_p)

    return np.ceil(np.log2(scalar * integral / error_r))


def sum_kloc_over_Gnu(n_p, atoms_rep, recip_bv):

    mcG = G_lattice(n_p)
    # mcG = mcG if n_p == self.n_p else G_lattice(n_p)    # ?

    integral = 0
    integral2 = 0
    for nu in mcG:
        if np.all(nu) != 0:
            for idx, atom in enumerate(list_atoms):
                I = PP_loc_params[atom]
                nxt = atoms_rep[idx] * np.abs(k_loc(np.linalg.norm(G_p(nu, recip_bv)), I)) / np.linalg.norm(G_p(nu, recip_bv))
                integral += nxt
                integral2 += nxt / np.linalg.norm(G_p(nu, recip_bv))
    # self.sum_kloc_over_Gnu_squared_result = integral2      # ?
    return integral


def sum_kNL_over_Gnu(n_p, PP_NL_params, recip_bv, atoms_rep, fast = True):

    mcG = G_lattice(n_p) # ?
    # mcG = self.mcG if n_p == self.n_p else self.G_lattice(n_p) # ?

    integral = 0
    if fast:
        for idx, atom in enumerate(list_atoms):
            I = PP_NL_params[atom]
            integral += atoms_rep[idx] * np.abs(k_NL_fast(I))
    else:
        Gsq = list(itertools.product(mcG, repeat = 2))
        for p, q in Gsq:
            if p != q:
                for idx, atom in enumerate(list_atoms):
                    I = PP_NL_params[atom]
                    integral += atoms_rep[idx] * np.abs(k_NL(G_p(p, recip_bv), G_p(q, recip_bv), I)) / norm(G_p(p, recip_bv) - G_p(q, recip_bv))
    return integral


def k_NL_fast(I): #based on |G_p-G_q| * fpq <= (|G_p| + |G_q|)fpq = ...

    # soran
    mcG = G_lattice(n_p)
    G_ps = np.array([G_p(p, recip_bv) for p in mcG])
    G_pnorms = np.array([np.linalg.norm(G_p) for G_p in G_ps])
    G_pnormsexps = np.array([np.exp(-nGp**2) for nGp in G_pnorms])



    rs, Bi_inv = I['rs'], I['Bi_inv']

    a0 = abs(4*rs[0]**3 * Bi_inv[0])
    exps0 = G_pnormsexps**(rs[0]**2 / 2)
    nGpexp0 = np.sum(G_pnorms * exps0)
    exp0 = np.sum(exps0)
    nondiagest0 = nGpexp0 * exp0 - np.sum(G_pnorms * exps0**2)
    Int0 = 2 * a0 * nondiagest0

    a1 = abs(16 / 3 * rs[1]**5 * Bi_inv[1])
    exps1 = G_pnormsexps**(rs[1]**2 / 2)
    nGpexp1 = np.sum(G_pnorms * exps1)
    nGp2exp1 = np.sum(G_pnorms**2 * exps1)
    nondiagest1 = nGpexp1*nGp2exp1 - np.sum(G_pnorms**3 * exps1**2)
    Int1 = 2 * a1 * nondiagest1

    a2 = abs(128 / 45 * rs[2]**7 * Bi_inv[2])
    exps2 = G_pnormsexps**(rs[2]**2 / 2)
    nGp3exp2 = np.sum(G_pnorms**3 * exps2)
    nGp2exp2 = np.sum(G_pnorms**2 * exps2)
    nondiagest2 = nGp3exp2 * nGp2exp2 - np.sum(G_pnorms**5 * exps2**2)
    Int2 = 2 * a2 * nondiagest2

    return Int0 + Int1 + Int2

def compute_nb(recip_bv, error_b):
    r"""Eq. (J7)"""

    abs_sum = 0
    for b_omega in recip_bv:
        for b_omega_prime in recip_bv:
            abs_sum += np.abs(np.sum(b_omega * b_omega_prime))

    scalar = 2 * np.pi * eta * 2**(2 * n_p - 2) * abs_sum

    if not self.material_ortho_lattice:
        scalar *= 2

    # why needed?
    # if 'sum_1_over_Gnu_squared_result' not in self.__dict__.keys():
    #     sum_1_over_Gnu_squared(n_p)

    integral = 1

    return np.ceil(np.log2(scalar * integral / error_b))


# def sum_1_over_Gnu_squared(self, n_p):
#     return np.sum([1 / normGp**2 if abs(normGp) > 1e-7 else 0 for normGp in G_pnorms])

def compute_npsi(list_atoms, omega, PP_NL_params, atoms_rep, error_psi):
    r"""Eq. (J29)"""

    atom_n_NL_error = 0

    n_Ltype = np.ceil(np.log2(len(list_atoms)))

    _, multipliers = compute_lambda_NL_prime(eta, omega, n_p, list_atoms, recip_bv, PP_NL_params, atoms_rep)

    for idx, atom in enumerate(list_atoms):
        I = PP_NL_params[atom]
        rs, Bi_inv = I['rs'], I['Bi_inv']
        atom_n_NL_error += rs[0]**3 * abs(Bi_inv[0]) * multipliers[0][atom]
        atom_n_NL_error += 32/3 * rs[1]**5 * abs(Bi_inv[1]) * sum(multipliers[1][atom])
        atom_n_NL_error += 64/45 * rs[2]**7 * abs(Bi_inv[2]) * multipliers[2][atom]
        atom_n_NL_error += 64/15 * rs[2]**7 * abs(Bi_inv[2]) * sum(multipliers[3][atom])
        atom_n_NL_error *= atoms_rep[idx]

    sum_c = atom_n_NL_error / omega

    scalar = 18 * (n_p + 4 + n_Ltype) * np.pi * eta * sum_c # np.pi**2

    # if not self.material_ortho_lattice:
    #     scalar = scalar * (2 * n_p + 4 + n_Ltype) / (n_p + 4 + n_Ltype) # Eq (?)

    integral = 1

    return np.ceil(np.log2(scalar * integral / error_psi))


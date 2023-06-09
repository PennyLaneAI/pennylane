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
basis_vectors = np.array([[5.02,0,0],[0,5.40,0],[0,0,6.26]]) # Li2FeSiO4
PP_NL_params = {'Li': {'Z_ion':1 , 'rs': [0.666375, 1.079306, 0.], 'Bi_inv': [1.858811, -0.005895, 0.]},
                'O': {'Z_ion': 6, 'rs': [0.221786, 0.256829, 0.], 'Bi_inv': [18.266917, 0., 0.]},
                'Si': {'Z_ion': 4, 'rs': [0.422738, 0.484278, 0.], 'Bi_inv': [5.906928, 2.727013, 0.]},
                'Fe': {'Z_ion': 8, 'rs': [0.454482, 0.638903, 0.308732], 'Bi_inv': [3.016640, 1.499642, -9.145354]}}

# PP parameters, Bi_inv are the inverse values of the B_i's in the paper
PP_loc_params = {'Li': {'Z_ion':1, 'r_loc': 0.787553, 'Cs': [-1.892612, 0.286060]},
                 'O': {'Z_ion':6, 'r_loc': 0.247621, 'Cs': [-16.580318, 2.395701]},
                 'Si': {'Z_ion':4, 'r_loc': 0.440000, 'Cs': [-7.336103, 0.]},
                 'Fe': {'Z_ion':8,'r_loc': 0.430000, 'Cs': [-6.654220, 0.]}}

atoms_and_rep_uc = {'Li': 4, 'O': 8, 'Si': 2, 'Fe': 2}

list_atoms, atoms_rep_uc = list(atoms_and_rep_uc.keys()), np.array(list(atoms_and_rep_uc.values()))

def abs_sum(recip_bv):
    abs_sum = 0
    for b_omega in recip_bv:
        for b_omega_prime in recip_bv:
            abs_sum += np.abs(np.sum(b_omega * b_omega_prime))
    return abs_sum

def Ps(n, br):
        theta = 2 * np.pi / (2**br) * np.round((2**br) / (2*np.pi) * np.arcsin(np.sqrt(2**(np.ceil(np.log2(n)))/(4*n))))
        braket = (1 + (2 - (4*n) / (2**np.ceil(np.log2(n)))) * (np.sin(theta))**2)**2 + (np.sin(2*theta))**2
        return n / (2**np.ceil(np.log2(n)))*braket

def lambda_t(eta, recip_bv, n_p, br):

    lamb = eta * 2**(2 * n_p - 3) * Ps(eta, br)**(-2) * abs_sum(recip_bv) # * Ps ? , n_p - 2, / 2 ?

    if self.material_ortho_lattice:
        lamb /= 2 # Eq (?)

    return lamb

def lambda_nl(): #HGH specific implementation

    lambda_NL_prime = compute_lambda_NL_prime()

    lambda_NL = [x for x in lambda_NL_prime]

    lamb = lambda_NL

    return lamb


def compute_lambda_NL_prime():
    NL0 = np.array([0.] * len(list_atoms))
    NL1omega = np.array([[0.] * 3] * len(list_atoms))
    NL20 = np.array([0.] * len(list_atoms))
    NL2omegaomega = np.array([[0.] * 6] * len(list_atoms))

    # from error_X2n_X
    def G_p(p):
        return np.sum(np.array(p) * recip_bv, axis=0)
    def G_lattice(n_p):
        return list(itertools.product(range(-2**(n_p - 1) + 1, 2**(n_p - 1)), repeat = 3))
    mcG = G_lattice(n_p)
    G_ps = np.array([G_p(p) for p in mcG])

    G_psomega = [np.array([Gp[omega] for Gp in G_ps]) for omega in range(3)]
    G_psomegasquared = [x**2 for x in G_psomega]
    G_psomegaomega = [G_psomega[i] * G_psomega[j] for i, j in [(0,0),(1,1),(2,2),(0,1),(1,2),(2,1)]] # why this combination
    G_psomegaomegasquared = [x**2 for x in G_psomegaomega]
    G_pnorms = np.array([np.linalg.(G_p) for G_p in G_ps])
    G_pnormsquad = G_pnorms**4

    multipliers = [{},{},{},{}]

    for idx, atom in enumerate(list_atoms):
        I = PP_NL_params[atom]
        rs, Bi_inv = I['rs'], I['Bi_inv']

        multipliers[0][atom] = (G_pnormsexps**(rs[0]**2)).sum()
        NL0[idx] = atoms_rep[idx] * rs[0]**3 * abs(Bi_inv[0]) * multipliers[0][atom]
        multipliers[1][atom] = []
        for i in range(3):
            multipliers[1][atom].append((G_psomegasquared[i] * G_pnormsexps**(rs[1]**2)).sum())
            NL1omega[idx][i] = atoms_rep[idx] * rs[1]**5 * abs(Bi_inv[1]) * multipliers[1][atom][-1]
        multipliers[2][atom] = (G_pnormsquad * G_pnormsexps**(rs[2]**2)).sum()
        NL20[idx] =  atoms_rep[idx] * rs[2]**7 * abs(Bi_inv[2]) * multipliers[2][atom]

        multipliers[3][atom] = []
        for i in range(6):
            multipliers[3][atom].append((G_psomegaomegasquared[i] * G_pnormsexps**(rs[2]**2)).sum())
            NL2omegaomega[idx][i] = (int(i > 2) + 1) * atoms_rep[idx] * rs[2]**7 * abs(Bi_inv[2]) * multipliers[3][atom][-1]

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

    return lambda_NL_prime # multipliers

def lambda_v():
    return 2 * np.pi * eta * (eta - 1) * lambda_nu / omega * Ps(eta, br)**(-2) # P_nu^amp, Ps

def lambda_nu():

    n_M_V = compute_nmv(error_mv)

    # why needed?
    # if 'B_mus' not in self.__dict__.keys():
    #     compute_B_mus()

    M_V = 2**n_M_V

    lambda_nu_one = 0
    p_nu_one = 0

    for mu in range(2, (n_p + 2)):
        for nu in B_mus[mu]:
            Gnu_norm = norm(G_p(nu))
            p_nu_one += np.ceil( M_V * (bmin * 2**(mu-2) / Gnu_norm)**2) / (M_V * (2**(mu - 2))**2)
            lambda_nu_one += np.ceil(M_V * (bmin * 2**(mu - 2) / Gnu_norm)**2) / (M_V * (bmin * 2**(mu - 2))**2)

    lambda_nu_one = lambda_nu_one # lambda_nu_one?

    return p_nu_one # lambda_nu_one, Gnu_norm




# def compute_p_nu_V():
#     compute_lambda_nu_one()
#     p_nu_V = p_nu_one / 2**(n_p + 6)
# def compute_p_nu_amp_V(self):
#     compute_p_nu_V()
#     p_nu_amp_V, AA_steps_V = compute_AA_steps(p_nu_V)

# def compute_p_nu_loc(self):
#     compute_lambda_nu_loc_one()
#     p_nu_loc = p_nu_loc_one / 2**(n_p + 6)
# def compute_p_nu_amp_loc(self):
#     compute_p_nu_loc()
#     p_nu_amp_loc, AA_steps_loc = compute_AA_steps(p_nu_loc)

# def compute_B_mus():
#     B_mus = {}
#     for j in range(2, n_p + 3):
#         B_mus[j] = []
#     for nu in itertools.product(range(-2**(n_p), 2**(n_p) + 1), repeat = 3):
#         nu = np.array(nu)
#         if list(nu) != [0, 0, 0]:
#             mu = int(np.floor(np.log2(np.max(abs(nu))))) + 2
#             B_mus[mu].append(nu)
#     return B_mus

def lambda_loc():

    integral_loc = 0
    integral_loc_error = 0

    # maxnaPa = np.max([atoms_rep[idx] / Ps(atoms_rep[idx], b_r) for idx,_ in enumerate(list_atoms)]) # why here?

    for idx, atom in enumerate(list_atoms):

        I = PP_loc_params[atom]

        # Ps**-2 ?
        X_I = np.sum([atoms_rep[idx]*abs(k_loc(Gnu_norm, I)) / (Ps(atoms_rep[idx], b_r) * Gnu_norm**2 )\
            if abs(Gnu_norm)>1e-7 else 0 \
              for Gnu_norm in G_pnorms])

        integral_loc += X_I

        # integral_loc_error += (X_I * Ps(atoms_rep[idx], b_r) / atoms_rep[idx]) # why here?

    scalar = 4 * np.pi * eta / omega

    # n_Mloc = error2n(scalar * integral_loc_error * 2 * np.pi * maxnaPa * (3 * n_p + n_Ltype) / error_Mloc) # why here?

    lambda_loc = scalar * integral_loc

    lamb = lambda_loc

    return lamb


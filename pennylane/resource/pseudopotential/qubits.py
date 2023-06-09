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

def qubit_cost(self): #qubit costings
    clean_cost = compute_clean_cost(self)
    dirty_cost = compute_dirty_cost(self)
    qubit_cost = max([clean_cost, dirty_cost])
    return qubit_cost

def compute_beta_loc():

    x = 2**(n_qrom_bits + 1) - 2**ortho_additional

    if material_ortho_lattice:
        y = n_NL * n_p
    else:
        y = n_NL * 2 * n_p

    if material_ortho_lattice:
        beta_NL_dirty = np.floor(n_dirty / (3 * n_NL))
        beta_NL_gate = np.floor(2 * x / (3 * (y / kappa)) * np.log(2))
        beta_NL_parallel = np.floor(n_parallel/(3 * kappa))
        beta_NL = np.min([beta_NL_dirty, beta_NL_gate, beta_NL_parallel])
    else:
        beta_NL_dirty = np.floor(n_dirty / (2 * n_NL))
        beta_NL_gate = np.floor(2 * x / (3 * (y / kappa)) * np.log(2))
        beta_NL_parallel = np.floor(n_parallel / (2 * kappa))
        beta_NL = np.min([beta_NL_dirty, beta_NL_gate, beta_NL_parallel])

    if n_parallel == 1:
        beta_NL_gate = np.floor(np.sqrt(2 * x / (3 * y)))
        beta_NL = np.min([beta_NL_dirty, beta_NL_gate])

    return beta_NL # ...


def compute_beta_NL():

    n_NL_prime = 50 #default high value set by us

    if material_ortho_lattice:
        beta_NL_prime_dirty = np.floor(n_dirty / (3 * n_NL))
        beta_NL_prime_gate = np.floor(2 * x / (3 * (y / kappa)) * np.log(2))
        beta_NL_prime_parallel = np.floor(n_parallel / (3 * kappa))
        beta_NL_prime = np.min([beta_NL_prime_dirty, beta_NL_prime_gate, beta_NL_prime_parallel])
    else:
        beta_NL_prime_dirty = np.floor(n_dirty / (2 * n_NL))
        beta_NL_prime_gate = np.floor(2 * x / (3 * (y / kappa)) * np.log(2))
        beta_NL_prime_parallel = np.floor(n_parallel / (2 * kappa))
        beta_NL_prime = np.min([beta_NL_prime_dirty, beta_NL_prime_gate, beta_NL_prime_parallel])

    if n_parallel == 1:
        beta_NL_prime_gate = np.floor(np.sqrt(2 * x / (3 * y)))
        beta_NL_prime = np.min([beta_NL_prime_dirty, beta_NL_prime_gate])

    return beta_NL_prime # ...

def compute_logn_a_max():

        list_atoms, atoms_rep_uc = list(atoms_and_rep_uc.keys()), np.array(list(atoms_and_rep_uc.values()))
        natoms_type = len(list_atoms)
        n_Ltype = np.ceil(np.log2(natoms_type))

        atoms_rep = atoms_rep_uc

        self.n_a_max = max([self.atoms_rep[idx] for idx, atom in enumerate(self.list_atoms)])
logn_a_max
        self.logn_a_max = np.ceil(np.log2(self.n_a_max))


def compute_clean_cost():

    beta_loc = compute_beta_loc()
    beta_V =  compute_beta_V()
    beta_k = compute_beta_k()
    beta_NL = compute_beta_NL()
    beta_NL_prime = compute_beta_NL_prime()

    n_a_max = max([atoms_rep[idx] for idx, atom in enumerate(list_atoms)])
    logn_a_max = np.ceil(np.log2(n_a_max))

    n_Ltype = np.ceil(np.log2(natoms_type))

    n_NL_prime = 50 #default high value set by us

    # the 35 in max([g.n_R+1, g.n_op, 35, g.n_b, g.n_k, g.n_NL]) belongs to the n_AA estimate for exact AA
    clean_cost = (3 * eta * n_p, np.ceil(np.log2(np.ceil(np.pi * lambda_val / (2 * error_qpe)))),\
        max([n_R + 1, n_op, 35,  n_b,  n_k,  n_Mloc,  n_NL_prime,  n_NL]), 1 , 2, 4, 2 * eta + 5, 8, \
            2 * eta,  n_Ltype + logn_a_max, \
             n_Ltype+ logn_a_max+5 , 4, 4, 2, 3*( n_p + 1) ,  n_p,  n_M_V,\
            3* n_p+2 , 2* n_p+1 ,  n_M_V, 1, 2, 3*( n_p + 1),  n_Ltype +  logn_a_max, 1,  \
            3+3+3+ n_Ltype+9+4, 9, 2, 2*1) #2*1 for using n_AA one time in total

    clean_prep_temp_cost = max([5, 2*( n_Ltype +  logn_a_max+1),  n_k + \
        ( n_Ltype+4), 4,  n_Ltype+2,  n_M_V+3* n_p, ( n_Mloc+1)+(3* n_p+ n_Ltype)+ n_Mloc])

    clean_temp_H_cost = max([5* n_p+1, 5* n_R-4]) + max([clean_prep_temp_cost, 3* n_NL + \
        3*( n_p +  n_Ltype+4), 3* n_p-1, 3* n_NL + 3*( n_p+ n_Ltype+2) + 3])

    clean_temp_cost = max([clean_temp_H_cost, 2* n_eta + 9* n_p +  n_Mloc +  n_M_V + 35 + \
        2*( n_Ltype +  logn_a_max)])

    return sum(clean_cost) + clean_temp_cost


def compute_dirty_cost():
    if material_ortho_lattice:
        return max([ beta_k *  n_k ,  beta_V * ( n_M_V),  beta_loc * ( n_Mloc + 1),\
            3 * beta_NL * n_NL, 3 * beta_NL_prime * n_NL])
    else:
        return max([ beta_k *  n_k ,  beta_V * ( n_M_V),  beta_loc * ( n_Mloc + 1),\
            2 * beta_NL* n_NL, 2 * beta_NL_prime * n_NL])

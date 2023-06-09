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

# n_op = error_X_PP.compute_n_X(4*lambda_val, error_op)

def cost(self):

    prep_nondiagonal_V = 14 * n_eta + 8 * b_r - 36
    binary_decomp_indices = binary_decomp_indices(...)

    v2n_a_max =  np.ceil(np.log2((n_a_max & (~(n_a_max - 1)))))
    prep_uniform_nuclei_a_locV = 2 * 2 * (3 * np.ceil(np.log2(n_a_max)) - \
                                 3 * v2n_a_max + 2 * b_r - 9 + 2 * 2**n_Ltype)

    prep_R_nuclei_coords = 2 * 2 * (2**(n_Ltype + np.ceil(np.log2(n_a_max)) + 1))
    binary_decomp_register_fgh = 2*(2*(2**(4+1)-1) + (n_b-3)*4 + 2**4 + (n_p-2))
    prep_op_registers = 6 + 3 + 3 + n_Ltype + 9 + 4 # compact

    momentum_state_V = momentum_state_V(...)
    momentum_state_loc_qrom = momentum_state_loc_qrom(...)

    _, AA_steps_V = compute_AA_steps()
    AA_V = 2 * AA_steps_V + 1
    AA_loc = 2 # we don't use AA for U_loc, and 2 here is just to take into account the double of the PREP cost


    cost =  prep_nondiagonal_V
    cost += binary_decomp_indices
    cost += prep_uniform_nuclei_a_locV
    cost += prep_R_nuclei_coords
    cost += binary_decomp_register_fgh
    cost += prep_op_registers
    cost += AA_V * momentum_state_V
    cost += AA_loc * momentum_state_loc_qrom

    return cost


def cost_qpe():
    return np.ceil(np.pi * lambda_val / (2 * error_qpe))

def binary_decomp_indices(beta_k): #preparing the PREP_NL state; subscript k is NL in the paper, e.g. n_k is n_NL etc.

    if self.n_parallel == 1:
        qrom_cost =  2*(2*np.ceil(x/self.beta_k) + 3*self.n_k*\
                self.beta_k*n_qrom_bits + 2*n_qrom_bits)+\
                (self.n_k-3)*n_qrom_bits
    else:
        qrom_cost = 2*(2*np.ceil(x/self.beta_k) + 3*np.ceil(self.n_k/self.kappa)*\
                np.ceil(np.log2(self.beta_k))*n_qrom_bits + 2*n_qrom_bits)+\
                (self.n_k-3)*n_qrom_bits
    self.binary_QROM_decomp_indices_cost = 2*qrom_cost + 2*2**(self.n_Ltype+2) + 12

    return self.binary_QROM_decomp_indices_cost

def momentum_state_V(self): #momentum state superposition for V
    nqrom_bits = 3 * n_p
    x = 2**nqrom_bits
    y = n_M_V

    beta_V_dirty = np.floor(n_dirty / y)
    beta_V_parallel = np.floor(n_parallel / kappa)

    beta_V_gate = ...
    beta_V = ...

    if n_parallel == 1:
        momentum_state_V_cost_qrom = 2 * np.ceil(x / beta_V) + 3 * y * beta_V
    else:
        momentum_state_V_cost_qrom = 2 * np.ceil(x / beta_V) + 3 * np.ceil(y / kappa) * np.ceil(np.log2(beta_V))

    #below takes into account the PREP and PREP dagger cost, but not the AA steps multiplier
    momentum_state_V_cost = 2 * momentum_state_V_cost_qrom + y + 8 * (n_p - 1) + 6 * n_p + 2

    return momentum_state_V_cost

def momentum_state_loc_qrom(self): #momentum state superposition for U_loc
    n_qrom_bits = 3*self.n_p+self.n_Ltype
    x = 2**(n_qrom_bits + 1)-1
    y = self.n_Mloc * (n_qrom_bits-1) + (self.n_Mloc+1)  #the very last one has to output sign k_i(G_nu)

    beta_loc_dirty = ...
    beta_loc_gate = ...
    beta_loc_parallel = ...
    beta_loc = ...


    if n_parallel == 1:
        momentum_state_qrom_loc_cost = 2 * np.ceil(x / beta_loc) + \
        3 * beta_loc * (n_Mloc) * (3 * n_p-1) + \
        3 * beta_loc * (n_Mloc + 1) + 6 * n_p
    else:
        momentum_state_qrom_loc_cost = 2 * np.ceil(x / beta_loc) + \
        3 * np.ceil(np.log2(beta_loc)) * np.ceil((n_Mloc) / kappa) * (3 * n_p - 1) + \
        3 * np.ceil(np.log2(beta_loc)) * np.ceil((n_Mloc + 1) / kappa) + 6 * n_p

    momentum_state_loc_cost = 2 * momentum_state_qrom_loc_cost + (n_Mloc - 3) * n_qrom_bits
    return momentum_state_loc_cost #this will get multiplied by two since self.AA_loc = 2

def compute_AA_steps(pnu, th = None):

    th = pnuth if th is None else th

    amplitude_amplified = 0
    index = 0
    for i in range(29,-1,-1):
        amplitude = (np.sin((2 * i + 1) * np.arcsin(np.sqrt(pnu))))**2
        if amplitude > th:
            index = i
            amplitude_amplified = amplitude
    pnu_amps = [(np.sin((2 * i + 1)*np.arcsin(np.sqrt(pnu))))**2 for i in range(6)]

    return amplitude_amplified, index
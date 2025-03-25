# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Compact input classes for resource estimation."""
from collections import defaultdict

from pennylane.labs import resource_estimation as re


class CompactState:
    r"""A compact representation for the state of a quantum system."""

    def __init__(
        self,
        num_qubits=None,
        num_coeffs=None,
        precision=None,
        num_work_wires=None,
        num_bit_flips=None,
    ):
        self.num_qubits = num_qubits  # total dimension
        self.num_coeffs = num_coeffs  # num basis states in the linear combination
        self.precision = precision  # accuracy requirement for approx-prep
        self.num_work_wires = num_work_wires  # num extra work wires
        self.num_bit_flips = num_bit_flips

    @classmethod
    def from_mps(cls, num_mps_matrices, max_bond_dim, precision=1e-3):
        """Instantiate a CompactState for a state coming from an MPS"""
        return cls(num_qubits=num_mps_matrices, num_work_wires=max_bond_dim, precision=precision)

    @classmethod
    def from_bitstring(cls, num_qubits, num_bit_flips, precision=1e-3):
        """Instantiate a CompactState for a state coming from a bitstring"""
        cost_per_prep = defaultdict(int)
        cost_per_prep[re.ResourceX.resource_rep()] = num_bit_flips
        return cls(
            num_qubits=num_qubits, num_coeffs=1, cost_per_prep=cost_per_prep, precision=precision
        )

    @classmethod
    def from_state_vector(cls, num_qubits, num_coeffs, precision=1e-3, num_work_wires=0):
        """Instantiate a CompactState for a state coming from a statevector (dense or sparse)"""
        return cls(
            num_qubits=num_qubits,
            num_coeffs=num_coeffs,
            precision=precision,
            num_work_wires=num_work_wires,
        )

class CompactHamiltonian: 
    r"""A compact representation for the state of a hamiltonian."""

    def __init__(
        self,
        num_qubits,
        num_terms,
        cost_per_term,
        cost_per_exp_term, 
        cost_per_
    ):
        return
    
    @classmethod
    def from_pauli_lcu(cls, num_qubits, num_terms, k_local=None, pauli_term_dist=None):

        return cls()
    
    @classmethod 
    def from_factorized_lcu(cls, num_qubits, num_terms):
        return cls()

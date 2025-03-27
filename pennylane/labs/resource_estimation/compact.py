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
        positive_and_real=None,
    ):
        self.num_qubits = num_qubits  # total dimension
        self.num_coeffs = num_coeffs  # num basis states in the linear combination
        self.precision = precision  # accuracy requirement for approx-prep
        self.num_work_wires = num_work_wires  # num extra work wires
        self.num_bit_flips = num_bit_flips
        self.positive_and_real = positive_and_real

    @classmethod
    def from_mps(cls, num_mps_matrices, max_bond_dim):
        """Instantiate a CompactState for a state coming from an MPS"""
        return cls(num_qubits=num_mps_matrices, num_work_wires=max_bond_dim)

    @classmethod
    def from_bitstring(cls, num_qubits, num_bit_flips, precision=1e-3):
        """Instantiate a CompactState for a state coming from a bitstring"""
        cost_per_prep = defaultdict(int)
        cost_per_prep[re.ResourceX.resource_rep()] = num_bit_flips
        return cls(
            num_qubits=num_qubits, num_coeffs=1, cost_per_prep=cost_per_prep, precision=precision
        )

    @classmethod
    def from_state_vector(
        cls, num_qubits, num_coeffs, precision=1e-3, num_work_wires=0, positive_and_real=False
    ):
        """Instantiate a CompactState for a state coming from a statevector (dense or sparse)"""
        return cls(
            num_qubits=num_qubits,
            num_coeffs=num_coeffs,
            precision=precision,
            num_work_wires=num_work_wires,
            positive_and_real=positive_and_real,
        )

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
import math
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
        self.num_bit_flips = num_bit_flips  # num |1> state qubits in basis state
        self.positive_and_real = (
            positive_and_real  # True if coefficients are real and positive valued
        )

    def __eq__(self, other: object) -> bool:
        return all(
            (
                (self.num_qubits == other.num_qubits),
                (self.num_coeffs == other.num_coeffs),
                (self.precision == other.precision),
                (self.num_work_wires == other.num_work_wires),
                (self.num_bit_flips == other.num_bit_flips),
                (self.positive_and_real == other.positive_and_real),
            )
        )

    @classmethod
    def from_mps(cls, num_mps_matrices, max_bond_dim):
        """Instantiate a CompactState for a state coming from an MPS"""
        num_work_wires = math.ceil(math.log2(max_bond_dim))
        return cls(num_qubits=num_mps_matrices, num_work_wires=num_work_wires)

    @classmethod
    def from_bitstring(cls, num_qubits, num_bit_flips):
        """Instantiate a CompactState for a state coming from a bitstring"""
        return cls(
            num_qubits=num_qubits,
            num_coeffs=1,
            num_bit_flips=num_bit_flips,
        )

    @classmethod
    def from_state_vector(
        cls,
        num_qubits,
        num_coeffs,
        precision=1e-3,
        num_work_wires=0,
        positive_and_real=False,
    ):
        """Instantiate a CompactState for a state coming from a statevector (dense or sparse)"""
        return cls(
            num_qubits=num_qubits,
            num_coeffs=num_coeffs,
            precision=precision,
            num_work_wires=num_work_wires,
            positive_and_real=positive_and_real,
        )

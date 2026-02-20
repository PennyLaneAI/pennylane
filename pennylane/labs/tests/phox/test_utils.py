# Copyright 2026 Xanadu Quantum Technologies Inc.

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
Tests for the Phox simulator utility functions.
"""
import numpy as np
from pennylane.labs.phox.utils import (
    create_local_gates,
    create_lattice_gates,
    create_random_gates,
    generate_pauli_observables,
)


class TestGateGeneration:
    """Tests for gate generation helper functions."""

    def test_create_local_gates(self):
        """Test generating all local gates up to max weight."""
        n_qubits = 3
        max_weight = 2

        gates = create_local_gates(n_qubits, max_weight)

        # Expected:
        # Weight 1: [0], [1], [2] (3 gates)
        # Weight 2: [0,1], [0,2], [1,2] (3 gates)
        # Total = 6 gates
        assert len(gates) == 6

        gate_list = [g[0] for g in gates.values()]
        assert [0] in gate_list
        assert [0, 1] in gate_list
        # Ensure correct formatting
        assert all(isinstance(v, list) and isinstance(v[0], list) for v in gates.values())

    def test_create_lattice_gates_2x2(self):
        """Test lattice gate generation for a small grid."""
        rows, cols = 2, 2
        # Grid:
        # 0 - 1
        # |   |
        # 2 - 3
        # Edges: (0,1), (0,2), (1,3), (2,3) -> 4 edges
        # Nodes: 0, 1, 2, 3 -> 4 nodes
        # If max_weight=2, we get 1-body and 2-body terms.
        # Total = 4 (1-body) + 4 (2-body) = 8 gates.

        gates = create_lattice_gates(rows, cols, distance=1, max_weight=2, periodic=False)

        # Check uniqueness
        gate_set = {tuple(sorted(v[0])) for v in gates.values()}
        assert len(gate_set) == len(gates)

        # Check specific known interactions
        assert (0,) in gate_set
        assert (0, 1) in gate_set
        assert (0, 3) not in gate_set  # Diagonal not in nearest neighbor

        # Expect 8 gates total
        assert len(gates) == 8

    def test_create_lattice_gates_periodic(self):
        """Test lattice gate generation with periodic boundaries."""
        # 2x2 periodic adds edges (0,2) vertical, (0,1) horizontal
        # (0,1) horiz, (1,0) wrapped? No, (0,1) is existing.
        # (0,1), (2,3) -> horizontal
        # (0,2), (1,3) -> vertical
        # Wrapped horizontal: (0,1) and (2,3) covered. (1,0) is same.
        # Wrapped vertical: (0,2) and (1,3) covered.

        # Let's try 3x1 chain periodic. 0-1-2. Edge 0-2 should exist.
        gates = create_lattice_gates(1, 3, periodic=True, max_weight=2)
        gate_set = {tuple(sorted(v[0])) for v in gates.values()}

        assert (0, 2) in gate_set

    def test_create_random_gates(self):
        """Test random gate generation."""
        n_qubits = 5
        n_gates = 10
        seed = 42

        gates = create_random_gates(n_qubits, n_gates, min_weight=2, max_weight=3, seed=seed)

        assert len(gates) == n_gates
        for v in gates.values():
            gate = v[0]
            assert 2 <= len(gate) <= 3
            assert max(gate) < n_qubits
            assert len(set(gate)) == len(gate)  # No duplicates in a single gate

    def test_create_random_gates_determinism(self):
        """Test that seeding produces deterministic results."""
        g1 = create_random_gates(4, 5, seed=123)
        g2 = create_random_gates(4, 5, seed=123)
        g3 = create_random_gates(4, 5, seed=999)

        # Dictionaries may order differently if not careful, but param keys 0..N are consistent
        assert str(g1) == str(g2)
        assert str(g1) != str(g3)


class TestObservableGeneration:
    """Tests for observable generation helper."""

    def test_generate_pauli_observables_single_z(self):
        """Test generating simple 1-body Z observables."""
        n_qubits = 2
        obs = generate_pauli_observables(n_qubits, orders=[1], bases=["Z"])

        # Expect ['Z', 'I'] and ['I', 'Z']
        assert len(obs) == 2
        assert ["Z", "I"] in obs
        assert ["I", "Z"] in obs

    def test_generate_pauli_observables_multi_basis(self):
        """Test mixed bases and orders."""
        n_qubits = 2
        obs = generate_pauli_observables(n_qubits, orders=[1, 2], bases=["X", "Z"])

        # Order 1 (1-body):
        # Base X: XI, IX
        # Base Z: ZI, IZ
        # -> 4 terms

        # Order 2 (2-body):
        # Base X: XX
        # Base Z: ZZ
        # -> 2 terms

        # Total 6
        assert len(obs) == 6
        assert ["X", "X"] in obs
        assert ["Z", "Z"] in obs
        assert [
            "X",
            "Z",
        ] not in obs  # Function usually doesn't mix bases in one term unless bases=["XZ"]?
        # Looking at implementation: `for base in bases`. It applies ONE base to the chosen positions.

    def test_generate_pauli_observables_higher_order(self):
        """Test generating higher order terms."""
        n_qubits = 3
        obs = generate_pauli_observables(n_qubits, orders=[3], bases=["Y"])

        assert len(obs) == 1
        assert obs[0] == ["Y", "Y", "Y"]

    def test_generate_pauli_observables_skip_invalid_order(self):
        """Test that orders > n_qubits are ignored."""
        obs = generate_pauli_observables(2, orders=[5])
        assert len(obs) == 0

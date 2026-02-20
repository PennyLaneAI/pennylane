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
from pennylane.labs.phox.utils import (
    create_local_gates,
    create_lattice_gates,
    create_random_gates,
    generate_pauli_observables,
)


class TestGateGeneration:
    """Tests for gate generation helper functions."""

    def test_create_local_gates(self):
        n_qubits = 3
        max_weight = 2
        gates = create_local_gates(n_qubits, max_weight)
        assert len(gates) == 6
        gate_list = [g[0] for g in gates.values()]
        assert [0] in gate_list
        assert [0, 1] in gate_list
        assert all(isinstance(v, list) and isinstance(v[0], list) for v in gates.values())

    def test_create_lattice_gates_2x2(self):
        rows, cols = 2, 2
        gates = create_lattice_gates(rows, cols, distance=1, max_weight=2, periodic=False)
        gate_set = {tuple(sorted(v[0])) for v in gates.values()}
        assert len(gate_set) == len(gates)
        assert (0,) in gate_set
        assert (0, 1) in gate_set
        assert (0, 3) not in gate_set
        assert len(gates) == 8

    def test_create_lattice_gates_periodic(self):
        gates = create_lattice_gates(1, 3, periodic=True, max_weight=2)
        gate_set = {tuple(sorted(v[0])) for v in gates.values()}
        assert (0, 2) in gate_set

    def test_create_random_gates(self):
        n_qubits = 5
        n_gates = 10
        seed = 42
        gates = create_random_gates(n_qubits, n_gates, min_weight=2, max_weight=3, seed=seed)
        assert len(gates) == n_gates
        for v in gates.values():
            gate = v[0]
            assert 2 <= len(gate) <= 3
            assert max(gate) < n_qubits
            assert len(set(gate)) == len(gate)

    def test_create_random_gates_determinism(self):
        g1 = create_random_gates(4, 5, seed=123)
        g2 = create_random_gates(4, 5, seed=123)
        g3 = create_random_gates(4, 5, seed=999)
        assert str(g1) == str(g2)
        assert str(g1) != str(g3)


class TestObservableGeneration:
    """Tests for observable generation helper mapped to integers."""

    def test_generate_pauli_observables_single_z(self):
        n_qubits = 2
        obs = generate_pauli_observables(n_qubits, orders=[1], bases=["Z"])

        assert len(obs) == 2
        assert [3, 0] in obs
        assert [0, 3] in obs

    def test_generate_pauli_observables_multi_basis(self):
        n_qubits = 2
        obs = generate_pauli_observables(n_qubits, orders=[1, 2], bases=["X", "Z"])

        assert len(obs) == 6
        assert [1, 1] in obs
        assert [3, 3] in obs
        assert [1, 3] not in obs

    def test_generate_pauli_observables_higher_order(self):
        n_qubits = 3
        obs = generate_pauli_observables(n_qubits, orders=[3], bases=["Y"])

        assert len(obs) == 1
        assert obs[0] == [2, 2, 2]

    def test_generate_pauli_observables_skip_invalid_order(self):
        obs = generate_pauli_observables(2, orders=[5])
        assert len(obs) == 0

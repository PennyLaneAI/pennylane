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
Utility functions for generating gates, observables, and analyzing circuits for the Phox simulator.
"""
import itertools
from itertools import combinations
import networkx as nx
import numpy as np


def create_local_gates(n_qubits: int, max_weight: int = 2) -> dict[int, list[list[int]]]:
    """
    Generates a gate dictionary for the Phox simulator containing all gates whose
    generators have Pauli weight less or equal to max_weight.

    Each gate is assigned a unique parameter index.

    Args:
        n_qubits (int): The number of qubits.
        max_weight (int): Maximum Pauli weight of gate generators.

    Returns:
        dict[int, list[list[int]]]: Gate structure mapping parameter indices to qubit lists.
    """
    gates = {}
    param_idx = 0
    for weight in range(1, max_weight + 1):
        for gate in combinations(range(n_qubits), weight):
            gates[param_idx] = [list(gate)]
            param_idx += 1
    return gates


def create_lattice_gates(
    rows: int, cols: int, distance: int = 1, max_weight: int = 2, periodic: bool = False
) -> dict[int, list[list[int]]]:
    """
    Generates gates based on nearest-neighbor interactions on a 2D lattice.

    Args:
        rows (int): Lattice height.
        cols (int): Lattice width.
        distance (int): Maximum graph distance to consider for interactions.
        max_weight (int): Maximum weight of the generators.
        periodic (bool): Whether to use periodic boundary conditions.

    Returns:
        dict[int, list[list[int]]]: Gate structure.
    """
    G = nx.grid_2d_graph(rows, cols, periodic=periodic)
    mapping = {(i, j): i * cols + j for i in range(rows) for j in range(cols)}
    G = nx.relabel_nodes(G, mapping)

    gates_list = []

    for source in list(G.nodes):
        lengths = nx.single_source_shortest_path_length(G, source, cutoff=distance)
        neighbors = [n for n in lengths if n != source]

        for weight in range(max_weight):
            for combo in combinations(neighbors, weight):
                new_gate = sorted(list(combo) + [source])
                if new_gate not in gates_list:
                    gates_list.append(new_gate)

    gates_dict = {}
    for i, g in enumerate(gates_list):
        gates_dict[i] = [g]

    return gates_dict


def create_random_gates(
    n_qubits: int, n_gates: int, min_weight: int = 1, max_weight: int = 2, seed: int = None
) -> dict[int, list[list[int]]]:
    """
    Generates a dictionary of random gates.

    Args:
        n_qubits (int): Total number of qubits.
        n_gates (int): Number of gates to generate.
        min_weight (int): Minimum weight of a gate.
        max_weight (int): Maximum weight of a gate.
        seed (int): Random seed.

    Returns:
        dict[int, list[list[int]]]: Gate structure.
    """
    rng = np.random.default_rng(seed)
    gates_dict = {}

    for i in range(n_gates):
        weight = rng.integers(min_weight, max_weight + 1)
        gate = sorted(list(rng.choice(range(n_qubits), size=weight, replace=False)))
        gates_dict[i] = [[int(q) for q in gate]]

    return gates_dict


def generate_pauli_observables(
    n_qubits: int, orders: list[int] = (1,), bases: list[str] = ("Z",)
) -> list[list[int]]:
    """
    Generates a batch of Pauli observables represented as integers (I=0, X=1, Y=2, Z=3).

    Args:
        n_qubits (int): Number of qubits.
        orders (list[int]): Orders of interactions to generate (e.g., [1, 2] for one-body and two-body).
        bases (list[str]): Pauli bases to use ('X', 'Y', 'Z').

    Returns:
        list[list[int]]: A list of observables mapped to ints.
                         Example for 2 qubits, order 1, base Z: [[3, 0], [0, 3]]
    """
    observables = []
    base_map = {"I": 0, "X": 1, "Y": 2, "Z": 3}

    for order in orders:
        if order > n_qubits:
            continue
        for base in bases:
            base_int = base_map[base.upper()]
            for positions in combinations(range(n_qubits), order):
                obs_row = [0] * n_qubits
                for pos in positions:
                    obs_row[pos] = base_int
                observables.append(obs_row)

    return observables

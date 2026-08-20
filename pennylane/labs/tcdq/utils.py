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
"""Helpers for building TCDQ gate dictionaries and observable lists.

These functions generate the ``gates`` and ``observables`` inputs expected by
:class:`~pennylane.labs.tcdq.CircuitConfig`.
"""

from itertools import combinations

import networkx as nx
import numpy as np


def create_local_gates(n_qubits: int, max_weight: int = 2) -> dict[int, list[list[int]]]:
    """Generate a gate dictionary containing all qubit subsets up to a given weight.

    Creates one gate (and one trainable parameter) for every combination of
    qubits whose size is between 1 and ``max_weight``. For example, with
    ``max_weight=2`` on 3 qubits, this produces single-qubit Z gates
    ``[0], [1], [2]`` and two-qubit ZZ gates ``[0,1], [0,2], [1,2]``.

    Args:
        n_qubits (int): Total number of qubits.
        max_weight (int): Maximum number of qubits per gate (Pauli weight).
            Defaults to ``2``.

    Returns:
        dict[int, list[list[int]]]: A gate dictionary suitable for
        :class:`~pennylane.labs.tcdq.CircuitConfig`. Keys are parameter
        indices (starting from 0) and values are single-element lists
        containing the list of qubit indices for that gate.

    **Example**

    >>> from pennylane.labs.tcdq import create_local_gates
    >>> gates = create_local_gates(n_qubits=3, max_weight=2)
    >>> gates
    {0: [[0]], 1: [[1]], 2: [[2]], 3: [[0, 1]], 4: [[0, 2]], 5: [[1, 2]]}
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
    """Generate gates based on spatial locality on a 2-D qubit lattice.

    Qubits are arranged on a ``rows × cols`` grid and numbered in row-major
    order (qubit index = row * cols + col). For each qubit, gates are created
    involving that qubit and up to ``max_weight - 1`` neighbours within graph
    distance ``distance``.

    Args:
        rows (int): Number of rows in the lattice.
        cols (int): Number of columns in the lattice.
        distance (int): Maximum graph distance (Manhattan steps on the grid)
            for two qubits to be considered neighbours. Defaults to ``1``
            (nearest neighbours only).
        max_weight (int): Maximum number of qubits per gate. Defaults to ``2``.
        periodic (bool): If ``True``, use periodic (toroidal) boundary
            conditions. Defaults to ``False``.

    Returns:
        dict[int, list[list[int]]]: A gate dictionary suitable for
        :class:`~pennylane.labs.tcdq.CircuitConfig`.

    **Example**

    >>> from pennylane.labs.tcdq import create_lattice_gates
    >>> gates = create_lattice_gates(rows=2, cols=2, distance=1, max_weight=2)
    >>> len(gates)  # 4 single-qubit + 4 nearest-neighbour pairs
    8
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
    """Generate a gate dictionary with randomly chosen qubit subsets.

    Each gate acts on a uniformly random subset of qubits whose size is
    drawn uniformly from ``[min_weight, max_weight]``.

    Args:
        n_qubits (int): Total number of qubits.
        n_gates (int): Number of gates (and trainable parameters) to generate.
        min_weight (int): Minimum number of qubits per gate. Defaults to ``1``.
        max_weight (int): Maximum number of qubits per gate. Defaults to ``2``.
        seed (int): Random seed for reproducibility. Defaults to ``None``.

    Returns:
        dict[int, list[list[int]]]: A gate dictionary suitable for
        :class:`~pennylane.labs.tcdq.CircuitConfig`.

    **Example**

    >>> from pennylane.labs.tcdq import create_random_gates
    >>> gates = create_random_gates(n_qubits=6, n_gates=10, max_weight=3, seed=0)
    >>> len(gates)
    10
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
    """Generate Pauli observables as integer-encoded arrays.

    Produces all tensor-product observables of the requested Pauli bases at
    the given interaction orders. Each observable is a list of length
    ``n_qubits`` with integer entries: I=0, X=1, Y=2, Z=3.

    For example, with ``orders=[2]`` and ``bases=["Z"]`` on 3 qubits, this
    generates all two-body ZZ operators: ``[3,3,0]``, ``[3,0,3]``,
    ``[0,3,3]``.

    Args:
        n_qubits (int): Number of qubits.
        orders (list[int]): List of interaction orders (number of non-identity
            sites). For instance, ``[1, 2]`` generates all single-qubit and
            two-qubit observables.
        bases (list[str]): Pauli bases to use. Any subset of
            ``["X", "Y", "Z"]``.

    Returns:
        list[list[int]]: A list of observables, each of length ``n_qubits``,
        suitable for use as the ``observables`` argument of
        :class:`~pennylane.labs.tcdq.CircuitConfig`.

    **Example**

    >>> from pennylane.labs.tcdq import generate_pauli_observables
    >>> obs = generate_pauli_observables(n_qubits=3, orders=[1], bases=["Z"])
    >>> obs
    [[3, 0, 0], [0, 3, 0], [0, 0, 3]]
    >>> obs_2body = generate_pauli_observables(3, orders=[2], bases=["Z"])
    >>> obs_2body
    [[3, 3, 0], [3, 0, 3], [0, 3, 3]]
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

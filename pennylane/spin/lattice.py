# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
This module contains functions and classes to create a
:class:`~pennylane.spin.lattice` object. This object stores all
the necessary information about a lattice.
"""

import numpy as np
from scipy.spatial import cKDTree
from .utils import map_vertices, get_custom_edges

# pylint: disable=too-many-arguments, too-many-instance-attributes


class Lattice:
    r"""Constructs a Lattice object.

    Args:
       L: Number of unit cells in a direction, it is a list depending on the dimensions of the lattice.
       unit_cell: Primitive vectors for the lattice.
       basis: Initial positions of spins.
       boundary_condition: defines boundary conditions, boolean or series of bools with dimensions same as L.
       neighbour_order: Range of neighbours a spin interacts with.
       custom_edges: List of edges in a unit cell along with the operations associated with them.
       distance_tol: Distance below which spatial points are considered equal for the purpose of identifying nearest neighbours.

    Returns:
       Lattice object
    """

    def __init__(
        self,
        L,
        unit_cell,
        basis=None,
        boundary_condition=False,
        neighbour_order=1,
        custom_edges=None,
        distance_tol=1e-5,
    ):

        self.L = np.asarray(L)
        self.n_dim = len(L)
        self.unit_cell = np.asarray(unit_cell)
        if basis is None:
            basis = np.zeros(self.unit_cell.shape[0])[None, :]
        self.basis = np.asarray(basis)
        self.n_sl = len(self.basis)
        self.n_sites = np.prod(L) * self.n_sl

        if isinstance(boundary_condition, bool):
            boundary_condition = [boundary_condition for _ in range(self.n_dim)]

        self.boundary_condition = boundary_condition
        self.test_input_accuracy()

        if True in self.boundary_condition:
            extra_shells = np.where(self.boundary_condition, neighbour_order, 0)
        else:
            extra_shells = None

        self.coords, self.sl_coords, self.lattice_points = self.generate_grid(extra_shells)

        if custom_edges is None:
            cutoff = neighbour_order * np.linalg.norm(self.unit_cell, axis=1).max() + distance_tol
            self.identify_neighbours(cutoff, neighbour_order)
            self.generate_true_edges()
        else:
            if neighbour_order > 1:
                raise ValueError(
                    "custom_edges and neighbour_order cannot be specified at the same time"
                )
            self.edges = get_custom_edges(
                self.unit_cell,
                self.L,
                self.basis,
                self.boundary_condition,
                self.lattice_points,
                self.n_sites,
                custom_edges,
            )

    def test_input_accuracy(self):
        r"""Tests the accuracy of the input provided"""

        for l in self.L:
            if (not isinstance(l, np.int64)) or l <= 0:
                raise TypeError("Argument `L` must be a list of positive integers")

        if self.unit_cell.ndim != 2:
            raise ValueError("'unit_cell' must have ndim==2, as array of primitive vectors.")

        if self.basis.ndim != 2:
            raise ValueError("'basis' must have ndim==2, as array of initial coordinates.")

        if self.unit_cell.shape[0] != self.unit_cell.shape[1]:
            raise ValueError("The number of primitive vectors must match their length")

        if not all(isinstance(b, bool) for b in self.boundary_condition):
            raise ValueError(
                "Argument 'boundary_condition' must be a bool or a list of bools of same dimensions as the unit_cell"
            )

        if len(self.boundary_condition) != self.n_dim:
            raise ValueError(
                "Argument 'boundary_condition' must be a bool or a list of bools of same dimensions as the unit_cell"
            )

    def identify_neighbours(self, cutoff, neighbour_order):
        r"""Identifies the connections between lattice points and returns the unique connections
        based on the neighbour_order"""

        tree = cKDTree(self.lattice_points)
        indices = tree.query_ball_tree(tree, cutoff)
        unique_pairs = set()
        row = []
        col = []
        distance = []
        for i, neighbours in enumerate(indices):
            for neighbour in neighbours:
                if neighbour != i:
                    pair = (min(i, neighbour), max(i, neighbour))
                    if pair not in unique_pairs:
                        unique_pairs.add(pair)
                        dist = np.linalg.norm(
                            self.lattice_points[i] - self.lattice_points[neighbour]
                        )
                        row.append(i)
                        col.append(neighbour)
                        distance.append(dist)

        row = np.array(row)
        col = np.array(col)

        # Sort distance into bins for comparison
        bin_density = 21621600  # multiple of expected denominators
        distance = np.asarray(np.rint(np.asarray(distance) * bin_density), dtype=int)

        _, ii = np.unique(distance, return_inverse=True)

        self.edges = [sorted(list(zip(row[ii == k], col[ii == k]))) for k in range(neighbour_order)]

    def generate_true_edges(self):
        r"""Modifies the edges to remove hidden nodes and create connections based on boundary_conditions"""

        map = map_vertices(self.coords, self.sl_coords, self.L, self.basis)
        colored_edges = []
        for k, edge in enumerate(self.edges):
            true_edges = set()
            for node1, node2 in edge:
                node1 = map[node1]
                node2 = map[node2]
                if node1 == node2:
                    raise RuntimeError(f"Lattice contains self-referential edge {(node1, node2)}.")
                true_edges.add((min(node1, node2), max(node1, node2)))
            for edge in true_edges:
                colored_edges.append((*edge, k))
        self.edges = colored_edges

    def add_edge(self, edge_index):
        r"""Adds a specific edge based on the site index without translating it"""

        if len(edge_index) == 2:
            edge_index = (*edge_index, 0)

        self.edges.append(edge_index)

    def generate_grid(self, extra_shells):
        """Generates the coordinates of all lattice sites.

        Args:
           extra_shells (np.ndarray): Optional. The number of unit cells added along each lattice direction.
           This is used for near-neighbour searching in periodic boundary conditions (PBC).
           It must be a vector of the same length as L.

        Returns:
           basis_coords: The coordinates of the basis sites in each unit cell.
           sl_coords: The coordinates of sublattice sites in each lattice site.
           lattice_points: The coordinates of all lattice sites.
        """

        # Initialize extra_shells if not provided
        if extra_shells is None:
            extra_shells = np.zeros(self.L.size, dtype=int)

        shell_min = -extra_shells
        shell_max = self.L + extra_shells

        range_dim = []
        for i in range(self.n_dim):
            range_dim.append(np.arange(shell_min[i], shell_max[i]))

        range_dim.append(np.arange(0, self.n_sl))

        coords = np.meshgrid(*range_dim, indexing="ij")

        sl_coords = coords[-1].ravel()
        basis_coords = np.column_stack([c.ravel() for c in coords[:-1]])

        lattice_points = (np.dot(basis_coords, self.unit_cell)).astype(float)

        for i in range(0, len(lattice_points), self.n_sl):
            lattice_points[i : i + self.n_sl] = lattice_points[i : i + self.n_sl] + self.basis

        return basis_coords, sl_coords, lattice_points

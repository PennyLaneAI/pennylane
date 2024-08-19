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
import itertools

import numpy as np
from scipy.spatial import cKDTree

# pylint: disable=too-many-arguments, too-many-instance-attributes
# pylint: disable=use-a-generator, too-few-public-methods


class Lattice:
    r"""Constructs a Lattice object.

    Args:
       L: Number of unit cells in each direction.
       unit_cell: Primitive vectors for the lattice.
       basis: Initial positions of spins.
       boundary_condition: defines boundary conditions, boolean or series of bools with dimensions same as L.
       neighbour_order: Range of neighbours a spin interacts with.
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
        self._test_input_accuracy()

        self.lattice_points, lattice_map = self._generate_grid(neighbour_order)

        cutoff = neighbour_order * np.linalg.norm(self.unit_cell, axis=1).max() + distance_tol
        edges = self._identify_neighbours(cutoff)
        self._generate_true_edges(edges, lattice_map, neighbour_order)

    def _test_input_accuracy(self):
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

    def _identify_neighbours(self, cutoff):
        r"""Identifies the connections between lattice points and returns the unique connections
        based on the neighbour_order"""

        tree = cKDTree(self.lattice_points)
        indices = tree.query_ball_tree(tree, cutoff)
        unique_pairs = set()
        edges = {}
        for i, neighbours in enumerate(indices):
            for neighbour in neighbours:
                if neighbour != i:
                    pair = (min(i, neighbour), max(i, neighbour))
                    if pair not in unique_pairs:
                        unique_pairs.add(pair)
                        dist = np.linalg.norm(
                            self.lattice_points[i] - self.lattice_points[neighbour]
                        )
                        # Scale the distance
                        bin_density = 21621600  # multiple of expected denominators
                        scaled_dist = np.rint(dist * bin_density)

                        if scaled_dist not in edges:
                            edges[scaled_dist] = []
                        edges[scaled_dist].append((i, neighbour))
        return edges

    def _generate_true_edges(self, edges, map, neighbour_order):
        r"""Modifies the edges to remove hidden nodes and create connections based on boundary_conditions"""

        self.edges = []
        for i, (_, edge) in enumerate(sorted(edges.items())):
            if i >= neighbour_order:
                break
            for e1, e2 in edge:
                true_edge = (min(map[e1], map[e2]), max(map[e1], map[e2]), i)
                if true_edge not in self.edges:
                    self.edges.append(true_edge)

    def _generate_grid(self, neighbour_order):
        """Generates the coordinates of all lattice sites and their indices.

        Args:
           neighbour_order: The number of nearest neighbour interactions.
        Returns:
           lattice_points: The coordinates of all lattice sites.
           lattice_map: A list to represent the node number for each lattice_point
        """

        n_sl = len(self.basis)
        if self.boundary_condition:
            wrap_grid = np.where(self.boundary_condition, neighbour_order, 0)
        else:
            wrap_grid = np.zeros(self.L.size, dtype=int)

        ranges_dim = [range(-wrap_grid[i], Lx + wrap_grid[i]) for i, Lx in enumerate(self.L)]
        ranges_dim.append(range(n_sl))
        lattice_points = []
        lattice_map = []

        for Lx in itertools.product(*ranges_dim):
            point = np.dot(Lx[:-1], self.unit_cell) + self.basis[Lx[-1]]
            node_index = 0
            for i in range(self.n_dim):
                node_index += (
                    (Lx[i] % self.L[i]) * np.prod(self.L[self.n_dim - 1 - i : 0 : -1]) * n_sl
                )
            node_index += Lx[-1]
            lattice_points.append(point)
            lattice_map.append(node_index)

        return np.array(lattice_points), np.array(lattice_map)

    def add_edge(self, edge_indices):
        r"""Adds a specific edge based on the site index without translating it.
        Args:
          edge_indices: List of edges to be added.
        Returns:
          Updates the edges attribute to include provided edges.
        """

        edges_nocolor = [(v1, v2) for (v1, v2, color) in self.edges]
        for edge_index in edge_indices:
            edge_index = tuple(edge_index)
            if len(edge_index) > 3 or len(edge_index) < 2:
                raise ValueError("Edge length can only be 2 or 3.")

            if len(edge_index) == 2:
                if edge_index in edges_nocolor:
                    raise ValueError("Edge is already present")
                new_edge = (*edge_index, 0)
            else:
                if edge_index in self.edges:
                    raise ValueError("Edge is already present")
                new_edge = edge_index

            self.edges.append(new_edge)

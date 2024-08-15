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
This module contains helper functions to create 
:class:`~pennylane.spin.lattice` objects.
"""
import numpy as np

# pylint: disable=too-many-arguments
# pylint: disable=use-a-generator


def map_vertices(basis_coords, sl, L, basis):
    """Generates lattice site indices for unit cell + sublattice coordinates."""

    basis_coords = basis_coords % L

    site_indices = np.zeros(basis_coords.shape[0], dtype=int)

    num_sl = len(basis)
    num_dim = len(L)

    nsites_axis = np.zeros(num_dim, dtype=int)
    nsites_axis[-1] = num_sl

    for j in range(num_dim - 1, 0, -1):
        nsites_axis[j - 1] = nsites_axis[j] * L[num_dim - j]

    for index in range(basis_coords.shape[0]):
        site_indices[index] = np.dot(basis_coords[index], nsites_axis)
    site_indices += sl

    return site_indices


def get_custom_edges(
    unit_cell, L, basis, boundary_condition, lattice_points, n_sites, custom_edges
):
    """Generates the edges described in `custom_edges` for all unit cells."""

    if not all([len(edge) in (1, 2) for edge in custom_edges]):
        raise TypeError(
            """
            custom_edges must be a list of tuples of length 1 or 2.
            Every tuple must contain two lattice indices to represent the edge
            and can optionally include a list to represent the operation and coefficient for that edge.
            """
        )

    def define_custom_edges(edge):
        if edge[0] >= n_sites or edge[1] >= n_sites:
            raise ValueError(f"The edge {edge} has vertices greater than n_sites, {n_sites}")
        num_sl = len(basis)
        sl1 = edge[0] % num_sl
        sl2 = edge[1] % num_sl
        new_coords = lattice_points[edge[1]] - lattice_points[edge[0]]
        return (sl1, sl2, new_coords)

    def translated_edges(sl1, sl2, distance, operation):
        # get distance in terms of unit cells, each axis represents the difference between two points in terms of unit_cells
        d_cell = (distance + basis[sl1] - basis[sl2]) @ np.linalg.inv(unit_cell)
        d_cell = np.asarray(np.rint(d_cell), dtype=int)

        # Unit cells of starting points
        edge_ranges = []
        for i, Li in enumerate(L):
            if boundary_condition[i]:
                edge_ranges.append(np.arange(0, Li))
            else:
                edge_ranges.append(
                    np.arange(np.maximum(0, -d_cell[i]), Li - np.maximum(0, d_cell[i]))
                )

        start_grid = np.meshgrid(*edge_ranges, indexing="ij")
        start_grid = np.column_stack([g.ravel() for g in start_grid])
        end_grid = (start_grid + d_cell) % L

        # Convert to site indices
        edge_start = map_vertices(start_grid, sl1, L, basis)
        edge_end = map_vertices(end_grid, sl2, L, basis)
        return [(*edge, operation) for edge in zip(edge_start, edge_end)]

    edges = []
    for i, custom_edge in enumerate(custom_edges):
        edge = custom_edge[0]
        edge_operation = custom_edge[1] if len(custom_edge) == 2 else i
        edge_data = define_custom_edges(edge)
        edges += translated_edges(*edge_data, edge_operation)
    return edges

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
This file contains functions and classes to create a
:class:`~pennylane.spin.Lattice` object. This object stores all
the necessary information about a lattice.
"""
import itertools

from scipy.spatial import KDTree

from pennylane import math

# pylint: disable=too-many-arguments, too-many-instance-attributes
# pylint: disable=use-a-generator, too-few-public-methods


class Lattice:
    r"""Constructs a Lattice object.

    Args:
       n_cells (list[int]): Number of cells in each direction of the grid.
       vectors (list[list[float]]): Primitive vectors for the lattice.
       positions (list[list[float]]): Initial positions of spin cites. Default value is
           ``[[0.0]`` :math:`\times` ``number of dimensions]``.

       boundary_condition (bool or list[bool]): Defines boundary conditions in different lattice axes,
           default is ``False`` indicating open boundary condition.
       neighbour_order (int): Specifies the interaction level for neighbors within the lattice.
           Default is 1 (nearest neighbour).
       distance_tol (float): Distance below which spatial points are considered equal for the
           purpose of identifying nearest neighbours. Default value is 1e-5.

    Raises:
       TypeError:
          if ``n_cells`` contains numbers other than positive integers.
       ValueError:
          if ``positions`` doesn't have a dimension of 2.
          if ``vectors`` doesn't have a dimension of 2 or the length of vectors is not equal to the number of vectors.
          if ``boundary_condition`` is not a bool or a list of bools with length equal to the number of vectors

    Returns:
       Lattice object

    **Example**

    >>> n_cells = [2,2]
    >>> vectors = [[0, 1], [1, 0]]
    >>> boundary_condition = [True, False]
    >>> lattice = qml.spin.Lattice(n_cells, vectors,
    >>>           boundary_condition=boundary_condition)
    >>> lattice.edges
    [(2, 3, 0), (0, 2, 0), (1, 3, 0), (0, 1, 0)]

    """

    def __init__(
        self,
        n_cells,
        vectors,
        positions=None,
        boundary_condition=False,
        neighbour_order=1,
        distance_tol=1e-5,
    ):

        if not all(isinstance(l, int) for l in n_cells) or any(l <= 0 for l in n_cells):
            raise TypeError("Argument `n_cells` must be a list of positive integers")

        self.vectors = math.asarray(vectors)

        if self.vectors.ndim != 2:
            raise ValueError(f"The dimensions of vectors array must be 2, got {self.vectors.ndim}.")

        if self.vectors.shape[0] != self.vectors.shape[1]:
            raise ValueError("The number of primitive vectors must match their length")

        if positions is None:
            positions = math.zeros(self.vectors.shape[0])[None, :]
        self.positions = math.asarray(positions)

        if self.positions.ndim != 2:
            raise ValueError(
                f"The dimensions of positions array must be 2, got {self.positions.ndim}."
            )

        if isinstance(boundary_condition, bool):
            boundary_condition = [boundary_condition] * len(n_cells)

        if not all(isinstance(b, bool) for b in boundary_condition) or len(
            boundary_condition
        ) != len(n_cells):
            raise ValueError(
                "Argument 'boundary_condition' must be a bool or a list of bools with length equal to number of vectors"
            )

        self.n_cells = math.asarray(n_cells)
        self.n_dim = len(n_cells)
        self.boundary_condition = boundary_condition

        n_sl = len(self.positions)
        self.n_sites = math.prod(n_cells) * n_sl
        self.lattice_points, lattice_map = self._generate_grid(neighbour_order)

        cutoff = neighbour_order * math.max(math.linalg.norm(self.vectors, axis=1)) + distance_tol
        edges = self._identify_neighbours(cutoff)
        self.edges = Lattice._generate_true_edges(edges, lattice_map, neighbour_order)
        self.edges_indices = [(v1, v2) for (v1, v2, color) in self.edges]

    def _identify_neighbours(self, cutoff):
        r"""Identifies the connections between lattice points and returns the unique connections
        based on the neighbour_order. This function uses KDTree to identify neighbours, which
        follows depth-first search traversal."""

        tree = KDTree(self.lattice_points)
        indices = tree.query_ball_tree(tree, cutoff)
        # Number to scale the distance, needed to sort edges into appropriate bins, it is currently
        # set as a multiple of expected denominators.
        bin_density = 2 ^ 5 * 3 ^ 3 * 5 ^ 2 * 7 * 11 * 13
        unique_pairs = set()
        edges = {}
        for i, neighbours in enumerate(indices):
            for neighbour in neighbours:
                if neighbour != i:
                    pair = (min(i, neighbour), max(i, neighbour))
                    if pair not in unique_pairs:
                        unique_pairs.add(pair)
                        dist = math.linalg.norm(
                            self.lattice_points[i] - self.lattice_points[neighbour]
                        )
                        scaled_dist = math.rint(dist * bin_density)

                        if scaled_dist not in edges:
                            edges[scaled_dist] = []
                        edges[scaled_dist].append((i, neighbour))

        edges = [value for _, value in sorted(edges.items())]
        return edges

    @staticmethod
    def _generate_true_edges(edges, map, neighbour_order):
        r"""Modifies the edges to remove hidden nodes and create connections based on boundary_conditions"""

        true_edges = []
        for i, edge in enumerate(edges):
            if i >= neighbour_order:
                break
            for e1, e2 in edge:
                true_edge = (min(map[e1], map[e2]), max(map[e1], map[e2]), i)
                if true_edge not in true_edges:
                    true_edges.append(true_edge)
        return true_edges

    def _generate_grid(self, neighbour_order):
        """Generates the coordinates of all lattice sites and their indices.

        Args:
           neighbour_order (int): Specifies the interaction level for neighbors within the lattice.

        Returns:
           lattice_points: The coordinates of all lattice sites.
           lattice_map: A list to represent the node number for each lattice_point.
        """

        n_sl = len(self.positions)
        wrap_grid = math.where(self.boundary_condition, neighbour_order, 0)

        ranges_dim = [
            range(-wrap_grid[i], cell + wrap_grid[i]) for i, cell in enumerate(self.n_cells)
        ]
        ranges_dim.append(range(n_sl))
        nsites_axis = math.cumprod([n_sl, *self.n_cells[:0:-1]])[::-1]
        lattice_points = []
        lattice_map = []

        for cell in itertools.product(*ranges_dim):
            point = math.dot(cell[:-1], self.vectors) + self.positions[cell[-1]]
            node_index = math.dot(math.mod(cell[:-1], self.n_cells), nsites_axis) + cell[-1]
            lattice_points.append(point)
            lattice_map.append(node_index)

        return math.array(lattice_points), math.array(lattice_map)

    def add_edge(self, edge_indices):
        r"""Adds a specific edge based on the site index without translating it.

        Args:
          edge_indices: List of edges to be added, an edge is defined as a list of integers
               specifying the corresponding node indices.

        Returns:
          Updates the edges attribute to include provided edges.
        """

        for edge_index in edge_indices:
            edge_index = tuple(edge_index)
            if len(edge_index) > 3 or len(edge_index) < 2:
                raise TypeError("Length of the tuple representing each edge can only be 2 or 3.")

            if len(edge_index) == 2:
                if edge_index in self.edges_indices:
                    raise ValueError("Edge is already present")
                new_edge = (*edge_index, 0)
            else:
                if edge_index in self.edges:
                    raise ValueError("Edge is already present")
                new_edge = edge_index

            self.edges.append(new_edge)


def _chain(n_cells, boundary_condition=False, neighbour_order=1):
    r"""Generates a chain lattice"""
    vectors = [[1]]
    n_cells = n_cells[0:1]
    lattice_chain = Lattice(
        n_cells=n_cells,
        vectors=vectors,
        neighbour_order=neighbour_order,
        boundary_condition=boundary_condition,
    )
    return lattice_chain


def _square(n_cells, boundary_condition=False, neighbour_order=1):
    r"""Generates a square lattice"""
    vectors = [[1, 0], [0, 1]]
    positions = [[0, 0]]
    n_cells = n_cells[0:2]
    lattice_square = Lattice(
        n_cells=n_cells,
        vectors=vectors,
        positions=positions,
        neighbour_order=neighbour_order,
        boundary_condition=boundary_condition,
    )

    return lattice_square


def _rectangle(n_cells, boundary_condition=False, neighbour_order=1):
    r"""Generates a rectangle lattice"""
    vectors = [[1, 0], [0, 1]]
    positions = [[0, 0]]

    n_cells = n_cells[0:2]
    lattice_rec = Lattice(
        n_cells=n_cells,
        vectors=vectors,
        positions=positions,
        neighbour_order=neighbour_order,
        boundary_condition=boundary_condition,
    )

    return lattice_rec


def _honeycomb(n_cells, boundary_condition=False, neighbour_order=1):
    r"""Generates a honeycomb lattice"""
    vectors = [[1, 0], [0.5, math.sqrt(3) / 2]]
    positions = [[0, 0], [0.5, 0.5 / 3**0.5]]

    n_cells = n_cells[0:2]
    lattice_honeycomb = Lattice(
        n_cells=n_cells,
        vectors=vectors,
        positions=positions,
        neighbour_order=neighbour_order,
        boundary_condition=boundary_condition,
    )

    return lattice_honeycomb


def _triangle(n_cells, boundary_condition=False, neighbour_order=1):
    r"""Generates a triangular lattice"""
    vectors = [[1, 0], [0.5, math.sqrt(3) / 2]]
    positions = [[0, 0]]

    n_cells = n_cells[0:2]
    lattice_triangle = Lattice(
        n_cells=n_cells,
        vectors=vectors,
        positions=positions,
        neighbour_order=neighbour_order,
        boundary_condition=boundary_condition,
    )

    return lattice_triangle


def _kagome(n_cells, boundary_condition=False, neighbour_order=1):
    r"""Generates a kagome lattice"""
    vectors = [[1, 0], [0.5, math.sqrt(3) / 2]]
    positions = [[0.0, 0], [-0.25, math.sqrt(3) / 4], [0.25, math.sqrt(3) / 4]]

    n_cells = n_cells[0:2]
    lattice_kagome = Lattice(
        n_cells=n_cells,
        vectors=vectors,
        positions=positions,
        neighbour_order=neighbour_order,
        boundary_condition=boundary_condition,
    )

    return lattice_kagome


# TODO Check the efficiency of this function with a dictionary instead.
def _generate_lattice(lattice, n_cells, boundary_condition=False, neighbour_order=1):
    r"""Generates the lattice object for a given shape and n_cells.

    Args:
        lattice (str): Shape of the lattice. Input Values can be ``'chain'``, ``'square'``, ``'rectangle'``, ``'honeycomb'``, ``'triangle'``, or ``'kagome'``.
        n_cells (list[int]): Number of cells in each direction of the grid.
        boundary_condition (bool or list[bool]): Defines boundary conditions, False for open boundary condition, each element represents the axis for lattice. It defaults to False.
        neighbour_order (int): Specifies the interaction level for neighbors within the lattice. Default is 1 (nearest neighbour).

    Returns:
        lattice object.
    """

    lattice_shape = lattice.strip().lower()

    if lattice_shape not in ["chain", "square", "rectangle", "honeycomb", "triangle", "kagome"]:
        raise ValueError(
            f"Lattice shape, '{lattice}' is not supported."
            f"Please set lattice to: chain, square, rectangle, honeycomb, triangle, or kagome"
        )

    if lattice_shape == "chain":
        lattice = _chain(n_cells, boundary_condition, neighbour_order)
    elif lattice_shape == "square":
        lattice = _square(n_cells, boundary_condition, neighbour_order)
    elif lattice_shape == "rectangle":
        lattice = _rectangle(n_cells, boundary_condition, neighbour_order)
    elif lattice_shape == "honeycomb":
        lattice = _honeycomb(n_cells, boundary_condition, neighbour_order)
    elif lattice_shape == "triangle":
        lattice = _triangle(n_cells, boundary_condition, neighbour_order)
    elif lattice_shape == "kagome":
        lattice = _kagome(n_cells, boundary_condition, neighbour_order)

    return lattice

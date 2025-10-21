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

import scipy as sp

from pennylane import math

# pylint: disable=too-many-arguments, too-many-instance-attributes
# pylint: disable=too-few-public-methods
# pylint: disable=too-many-branches


class Lattice:
    r"""Constructs a Lattice object.

    Args:
       n_cells (list[int]): Number of cells in each direction of the grid.
       vectors (list[list[float]]): Primitive vectors for the lattice.
       positions (list[list[float]]): Initial positions of the lattice nodes. Default value is
           ``[[0.0]`` :math:`\times` ``number of dimensions]``.
       boundary_condition (bool or list[bool]): Specifies whether or not to enforce periodic
            boundary conditions for the different lattice axes.  Default is ``False`` indicating
            open boundary condition.
       neighbour_order (int): Specifies the interaction level for neighbors within the lattice.
           Default is 1, indicating nearest neighbour. Must be 1 if ``custom_edges`` is defined.
       custom_edges (Optional[list(list(tuples))]): Specifies the edges to be added in the lattice.
           Default value is ``None``, which adds the edges based on ``neighbour_order``.
           Each element in the list is for a separate edge, and can contain 1 or 2 tuples.
           First tuple contains the indices of the starting and ending vertices of the edge.
           Second tuple is optional and contains the operator on that edge and coefficient
           of that operator. Default value is the index of edge in custom_edges list.
       custom_nodes (Optional(list(list(int, tuples)))): Specifies the on-site potentials and
           operators for nodes in the lattice. The default value is `None`, which means no on-site
           potentials. Each element in the list is for a separate node. For each element, the first
           value is the index of the node, and the second element is a tuple which contains the
           operator and coefficient.
       distance_tol (float): Distance below which spatial points are considered equal for the
           purpose of identifying nearest neighbours. Default value is 1e-5.

    Raises:
       TypeError:
          if ``n_cells`` contains numbers other than positive integers.
       ValueError:
          if ``positions`` doesn't have a dimension of 2.
       ValueError:
          if ``vectors`` doesn't have a dimension of 2 or the length of vectors is not equal to the number of vectors.
       ValueError:
          if ``boundary_condition`` is not a bool or a list of bools with length equal to the number of vectors.
       ValueError:
          if ``custom_nodes`` contains nodes with negative indices or indices greater than number of sites

    Returns:
       Lattice object

    **Example**

    We can define the positions of nodes in the lattice unit cell along with the lattice vectors
    to create a custom lattice layout.

    .. code-block:: python

        from pennylane.spin import Lattice

        positions = [[0.2, 0.5],
                     [0.5, 0.2],
                     [0.5, 0.8],
                     [0.8, 0.5]]

        vectors = [[1, 0], [0, 1]]

        n_cells = [2, 2]

        # periodic boundary conditions applied along the [1,0] axis only
        boundary_condition = [True, False]

        lattice = Lattice(n_cells, vectors, positions, boundary_condition=boundary_condition)

    >>> lattice.edges
    [(10, 13, 0), (0, 11, 0), (4, 15, 0), (2, 5, 0), (3, 8, 0), (7, 12, 0)]

    .. details::
        :title: Usage Details

        Unless otherwise specified, the edges will be added based on the ``neighbour_order``,
        which defaults to 1. Increasing ``neighbour_order`` will add additional connections
        in the lattice.

        .. code-block:: python

            positions = [[0.2, 0.5],
                         [0.5, 0.2],
                         [0.5, 0.8],
                         [0.8, 0.5]]

            lattice = Lattice(n_cells=[2, 2],
                              vectors=[[1, 0], [0, 1]],
                              positions=positions,
                              neighbour_order=2,
                              boundary_condition=[True, False])

        >>> len(lattice.edges)
        22

        We can also define edges with custom interactions, as well as adding on-site potentials for the
        nodes:

        .. code-block:: python

            # defining on-site potential at each node in the unit cell
            custom_nodes = [[(0), ('X', 0.5)],
                            [(1), ('X', 0.6)],
                            [(2), ('X', 0.7)],
                            [(3), ('X', 0.8)]]

            # defining custom edges (instead of nearest-neigbour connections) and their interactions
            custom_edges = [[(0, 1), ('XX', 0.5)],
                            [(0, 2), ('YY', 0.6)],
                            [(1, 3), ('ZZ', 0.7)],
                            [(2, 3), ('ZZ', 0.7)]]

        >>> lattice = Lattice(n_cells,
        ...                   vectors,
        ...                   positions,
        ...                   custom_edges=custom_edges,
        ...                   custom_nodes=custom_nodes)
        >>> lattice.edges
        [(0, 1, ('XX', 0.5)),
        (4, 5, ('XX', 0.5)),
        (8, 9, ('XX', 0.5)),
        (12, 13, ('XX', 0.5)),
        (0, 2, ('YY', 0.6)),
        (4, 6, ('YY', 0.6)),
        (8, 10, ('YY', 0.6)),
        (12, 14, ('YY', 0.6)),
        (1, 3, ('ZZ', 0.7)),
        (5, 7, ('ZZ', 0.7)),
        (9, 11, ('ZZ', 0.7)),
        (13, 15, ('ZZ', 0.7)),
        (2, 3, ('ZZ', 0.7)),
        (6, 7, ('ZZ', 0.7)),
        (10, 11, ('ZZ', 0.7)),
        (14, 15, ('ZZ', 0.7))]

    """

    def __init__(
        self,
        n_cells,
        vectors,
        positions=None,
        boundary_condition=False,
        neighbour_order=1,
        custom_edges=None,
        custom_nodes=None,
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
        if custom_edges is None:
            cutoff = (
                neighbour_order * math.max(math.linalg.norm(self.vectors, axis=1)) + distance_tol
            )
            edges = self._identify_neighbours(cutoff)
            self.edges = Lattice._generate_true_edges(edges, lattice_map, neighbour_order)
        else:
            if neighbour_order != 1:
                raise ValueError(
                    "custom_edges cannot be specified if neighbour_order argument is set to a value other than 1."
                )

            lattice_map = dict(zip(lattice_map, self.lattice_points))
            self.edges = self._get_custom_edges(custom_edges, lattice_map)

        self.edges_indices = [(v1, v2) for (v1, v2, color) in self.edges]

        if custom_nodes is not None:
            for node in custom_nodes:
                if node[0] > self.n_sites:
                    raise ValueError(
                        "The custom node has an index larger than the number of sites."
                    )
                if node[0] < 0:
                    raise ValueError("The custom node has an index smaller than 0.")

        self.nodes = custom_nodes

    def _identify_neighbours(self, cutoff):
        r"""Identifies the connections between lattice points and returns the unique connections
        based on the neighbour_order. This function uses KDTree to identify neighbours, which
        follows depth-first search traversal."""

        tree = sp.spatial.KDTree(self.lattice_points)
        indices = tree.query_ball_tree(tree, cutoff)
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
                        dist = math.round(dist, 4)
                        if dist not in edges:
                            edges[dist] = []
                        edges[dist].append((i, neighbour))

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
            lattice_map.append(int(node_index))  # convert from numpy int to inbuilt int

        return math.array(lattice_points), lattice_map

    def _get_custom_edges(self, custom_edges, lattice_map):
        """Generates the edges described in `custom_edges` for all unit cells.

        Args:
          custom_edges (Optional[list(list(tuples))]): Specifies the edges to be added in the lattice.
              Default value is None, which adds the edges based on neighbour_order.
              Each element in the list is for a separate edge, and can contain 1 or 2 tuples.
              First tuple contains the index of the starting and ending vertex of the edge.
              Second tuple is optional and contains the operator on that edge and coefficient
              of that operator.
          lattice_map (list[int]): A list to represent the node number for each lattice_point.

        Returns:
          List of edges.

        **Example**

        Generates a square lattice with a single diagonal and assigns a different operation
        to horizontal, vertical, and diagonal edges.

        >>> n_cells = [3,3]
        >>> vectors = [[1, 0], [0,1]]
        >>> custom_edges = [
        ...         [(0, 1), ("XX", 0.1)],
        ...         [(0, 3), ("YY", 0.2)],
        ...         [(0, 4), ("XY", 0.3)],
        ...     ]
        >>> lattice = qml.spin.Lattice(n_cells=n_cells, vectors=vectors, custom_edges=custom_edges)
        >>> from pprint import pprint
        >>> pprint(lattice.edges)
        [(0, 1, ('XX', 0.1)),
         (1, 2, ('XX', 0.1)),
         (3, 4, ('XX', 0.1)),
         (4, 5, ('XX', 0.1)),
         (6, 7, ('XX', 0.1)),
         (7, 8, ('XX', 0.1)),
         (0, 3, ('YY', 0.2)),
         (1, 4, ('YY', 0.2)),
         (2, 5, ('YY', 0.2)),
         (3, 6, ('YY', 0.2)),
         (4, 7, ('YY', 0.2)),
         (5, 8, ('YY', 0.2)),
         (0, 4, ('XY', 0.3)),
         (1, 5, ('XY', 0.3)),
         (3, 7, ('XY', 0.3)),
         (4, 8, ('XY', 0.3))]

        """

        for edge in custom_edges:
            if len(edge) not in (1, 2):
                raise TypeError(
                    """
                    The elements of custom_edges should be lists of length 1 or 2.
                    Inside said lists should be a tuple that contains two lattice
                    indices to represent the edge and, optionally, a tuple that represents
                    the operation and coefficient for that edge.
                    Every tuple must contain two lattice indices to represent the edge
                    and can optionally include a list to represent the operation and coefficient for that edge.
                    """
                )

            if edge[0][0] >= self.n_sites or edge[0][1] >= self.n_sites:
                raise ValueError(
                    f"The edge {edge[0]} has vertices greater than n_sites, {self.n_sites}"
                )

        edges = []
        n_sl = len(self.positions)
        nsites_axis = math.cumprod([n_sl, *self.n_cells[:0:-1]])[::-1]

        for i, custom_edge in enumerate(custom_edges):
            edge = custom_edge[0]

            edge_operation = custom_edge[1] if len(custom_edge) == 2 else i

            # Finds the coordinates of starting and ending vertices of the edge
            # and the vector distance between the coordinates
            vertex1 = lattice_map[edge[0]]
            vertex2 = lattice_map[edge[1]]
            edge_distance = vertex2 - vertex1

            # Calculates the number of unit cells that a given edge spans in each direction
            v1, v2 = math.mod(edge, n_sl)
            translation_vector = (
                edge_distance + self.positions[v1] - self.positions[v2]
            ) @ math.linalg.inv(self.vectors)
            translation_vector = math.asarray(math.rint(translation_vector), dtype=int)

            # Finds the minimum and maximum range for a given edge based on boundary_conditions
            edge_ranges = []
            for idx, cell in enumerate(self.n_cells):
                t_point = 0 if self.boundary_condition[idx] else translation_vector[idx]
                edge_ranges.append(
                    range(math.maximum(0, -t_point), cell - math.maximum(0, t_point))
                )

            # Finds the indices for starting and ending vertices of the edge
            for cell in itertools.product(*edge_ranges):
                node1_idx = math.dot(math.mod(cell, self.n_cells), nsites_axis) + v1
                node2_idx = (
                    math.dot(math.mod(cell + translation_vector, self.n_cells), nsites_axis) + v2
                )
                edges.append((int(node1_idx), int(node2_idx), edge_operation))

        return edges

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


def generate_lattice(lattice, n_cells, boundary_condition=False, neighbour_order=1):
    r"""Generates a :class:`~pennylane.spin.Lattice` object for a given lattice shape and number of
    cells.

    Args:
        lattice (str): Shape of the lattice. Input values can be ``'chain'``, ``'square'``,
            ``'rectangle'``, ``'triangle'``, ``'honeycomb'``,  ``'kagome'``, ``'lieb'``,
            ``'cubic'``, ``'bcc'``, ``'fcc'`` or ``'diamond'``.
        n_cells (list[int]): Number of cells in each direction of the grid.
        boundary_condition (bool or list[bool]): Defines boundary conditions in different lattice axes.
            Default is ``False`` indicating open boundary condition.
        neighbour_order (int): Specifies the interaction level for neighbors within the lattice.
            Default is 1, indicating nearest neighbour.

    Returns:
        ~pennylane.spin.Lattice: lattice object.

    **Example**

    >>> shape = 'square'
    >>> n_cells = [2, 2]
    >>> boundary_condition = [True, False]
    >>> lattice = qml.spin.generate_lattice(shape, n_cells, boundary_condition)
    >>> lattice.edges
    [(2, 3, 0), (0, 2, 0), (1, 3, 0), (0, 1, 0)]

    .. details::
        :title: Lattice details

        The following lattice shapes are currently supported.

        * ``'chain'``: linear arrangement of sites in one dimension

        * ``'square'``: square arrangement of sites in two dimensions

        * ``'rectangle'``: rectangular arrangement of sites in two dimensions

        * ``'triangle'``: triangular arrangement of sites in two dimensions [`Phys. Rev. B 7, 5017 (1973) <https://journals.aps.org/pr/abstract/10.1103/PhysRev.79.357>`_]

        * ``'honeycomb'``: `honeycomb <https://en.wikipedia.org/wiki/Hexagonal_lattice#Honeycomb_point_set>`_ arrangement of sites in two dimensions

        * ``'kagome'``: kagome arrangement of sites in two dimensions [`Prog. Theor. Phys. 6, 306 (1951) <https://academic.oup.com/ptp/article/6/3/306/1852171>`_]

        * ``'lieb'``: Lieb arrangement of sites in two dimensions [`arXiv:1004.5172 <https://arxiv.org/abs/1004.5172>`_]

        * ``'cubic'``: `cubic <https://en.wikipedia.org/wiki/Cubic_crystal_system>`_ arrangement of sites in three dimensions

        * ``'bcc'``: `body-centered cubic <https://en.wikipedia.org/wiki/Cubic_crystal_system>`_ arrangement of sites in three dimensions

        * ``'fcc'``: `face-centered cubic <https://en.wikipedia.org/wiki/Cubic_crystal_system>`_ arrangement of sites in three dimensions

        * ``'diamond'``: `diamond <https://en.wikipedia.org/wiki/Diamond_cubic>`_ arrangement of sites in three dimensions

    """

    lattice_shape = lattice.strip().lower()

    if lattice_shape not in [
        "chain",
        "square",
        "rectangle",
        "honeycomb",
        "triangle",
        "kagome",
        "lieb",
        "cubic",
        "bcc",
        "fcc",
        "diamond",
    ]:
        raise ValueError(
            f"Lattice shape, '{lattice}' is not supported."
            f"Please set lattice to: 'chain', 'square', 'rectangle', 'honeycomb', 'triangle', 'kagome', 'lieb',"
            f"'cubic', 'bcc', 'fcc', or 'diamond'."
        )

    lattice_dict = {
        "chain": {"dim": 1, "vectors": [[1]], "positions": None},
        "square": {"dim": 2, "vectors": [[0, 1], [1, 0]], "positions": None},
        "rectangle": {"dim": 2, "vectors": [[0, 1], [1, 0]], "positions": None},
        "triangle": {"dim": 2, "vectors": [[1, 0], [0.5, math.sqrt(3) / 2]], "positions": None},
        "honeycomb": {
            "dim": 2,
            "vectors": [[1, 0], [0.5, math.sqrt(3) / 2]],
            "positions": [[0, 0], [0.5, 0.5 / 3**0.5]],
        },
        "kagome": {
            "dim": 2,
            "vectors": [[1, 0], [0.5, math.sqrt(3) / 2]],
            "positions": [[0.0, 0], [-0.25, math.sqrt(3) / 4], [0.25, math.sqrt(3) / 4]],
        },
        "lieb": {"dim": 2, "vectors": [[0, 1], [1, 0]], "positions": [[0, 0], [0.5, 0], [0, 0.5]]},
        "cubic": {"dim": 3, "vectors": math.eye(3), "positions": None},
        "bcc": {"dim": 3, "vectors": math.eye(3), "positions": [[0, 0, 0], [0.5, 0.5, 0.5]]},
        "fcc": {
            "dim": 3,
            "vectors": math.eye(3),
            "positions": [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
        },
        "diamond": {
            "dim": 3,
            "vectors": [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],
            "positions": [[0, 0, 0], [0.25, 0.25, 0.25]],
        },
    }

    lattice_dim = lattice_dict[lattice_shape]["dim"]
    if len(n_cells) != lattice_dim:
        raise ValueError(
            f"Argument `n_cells` must be of the correct dimension for the given lattice shape."
            f" {lattice_shape} lattice is of dimension {lattice_dim}, got {len(n_cells)}."
        )

    lattice_obj = Lattice(
        n_cells=n_cells,
        vectors=lattice_dict[lattice_shape]["vectors"],
        positions=lattice_dict[lattice_shape]["positions"],
        neighbour_order=neighbour_order,
        boundary_condition=boundary_condition,
    )
    return lattice_obj

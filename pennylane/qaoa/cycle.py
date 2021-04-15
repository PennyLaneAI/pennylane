# Copyright 2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
Methods for finding max weighted cycle of weighted directed graphs
"""
import itertools
from typing import Dict, Tuple, Iterable, List
import networkx as nx
import pennylane as qml


def edges_to_wires(graph: nx.Graph) -> Dict[Tuple[int], int]:
    r"""Maps the edges of a graph to corresponding wires.

    **Example**

    >>> g = nx.complete_graph(4).to_directed()
    >>> edges_to_wires(g)
    {(0, 1): 0,
     (0, 2): 1,
     (0, 3): 2,
     (1, 0): 3,
     (1, 2): 4,
     (1, 3): 5,
     (2, 0): 6,
     (2, 1): 7,
     (2, 3): 8,
     (3, 0): 9,
     (3, 1): 10,
     (3, 2): 11}

    Args:
        graph (nx.Graph): the graph specifying possible edges

    Returns:
        Dict[Tuple[int], int]: a mapping from graph edges to wires
    """
    return {edge: i for i, edge in enumerate(graph.edges)}


def wires_to_edges(graph: nx.Graph) -> Dict[int, Tuple[int]]:
    r"""Maps the wires of a register of qubits to corresponding edges.

    **Example**

    >>> g = nx.complete_graph(4).to_directed()
    >>> wires_to_edges(g)
    {0: (0, 1),
     1: (0, 2),
     2: (0, 3),
     3: (1, 0),
     4: (1, 2),
     5: (1, 3),
     6: (2, 0),
     7: (2, 1),
     8: (2, 3),
     9: (3, 0),
     10: (3, 1),
     11: (3, 2)}

    Args:
        graph (nx.Graph): the graph specifying possible edges

    Returns:
        Dict[Tuple[int], int]: a mapping from wires to graph edges
    """
    return {i: edge for i, edge in enumerate(graph.edges)}


def _square_hamiltonian_terms(
    coeffs: Iterable[float], ops: Iterable[qml.operation.Observable]
) -> Tuple[List[float], List[qml.operation.Observable]]:
    """Calculates the coefficients and observables that compose the squared Hamiltonian.

    Args:
        coeffs (Iterable[float]): coeffients of the input Hamiltonian
        ops (Iterable[qml.operation.Observable]): observables of the input Hamiltonian

    Returns:
        Tuple[List[float], List[qml.operation.Observable]]: The list of coefficients and list of observables
            of the squared Hamiltonian.
    """
    squared_coeffs, squared_ops = [], []
    pairs = [(coeff, op) for coeff, op in zip(coeffs, ops)]
    products = itertools.product(pairs, repeat=2)

    for (coeff1, op1), (coeff2, op2) in products:
        squared_coeffs.append(coeff1 * coeff2)

        if isinstance(op1, qml.Identity):
            squared_ops.append(op2)
        elif isinstance(op2, qml.Identity):
            squared_ops.append(op1)
        elif op1.wires == op2.wires and type(op1) == type(op2):
            squared_ops.append(qml.Identity(0))
        elif op2.wires[0] < op1.wires[0]:
            squared_ops.append(op2 @ op1)
        else:
            squared_ops.append(op1 @ op2)

    return squared_coeffs, squared_ops


def _collect_duplicates(
    coeffs: Iterable[float], ops: Iterable[qml.operation.Observable]
) -> Tuple[List[float], List[qml.operation.Observable]]:
    """Collects duplicate observables together into one observable.

    Args:
        coeffs (Iterable[float]): coeffients of the input Hamiltonian
        ops (Iterable[qml.operation.Observable]): observables of the input Hamiltonian

    Returns:
        Tuple[List[float], List[qml.operation.Observable]]: The list of coefficients and list of
            observables without any duplicates
    """
    # Create a new list of coefficients and operations to add to without duplicates present
    reduced_coeffs, reduced_ops = [], []

    for coeff, op in zip(coeffs, ops):

        # We now loop through coefficients and operations from the input lists. The following checks
        # if an operation is already present in reduced_ops. If so, it just adds to the
        # corresponding coefficient
        included = False
        for i, op_red in enumerate(reduced_ops):
            if type(op) == type(op_red) and op.wires == op_red.wires:
                reduced_coeffs[i] += coeff
                included = True

        # If the operation is not already present in reduced_ops, we add to the reduces lists
        if not included:
            reduced_coeffs.append(coeff)
            reduced_ops.append(op)

    reduced_coeffs_no_zeros = []
    reduced_ops_no_zeros = []

    for coeff, op in zip(reduced_coeffs, reduced_ops):
        if coeff != 0:
            reduced_coeffs_no_zeros.append(coeff)
            reduced_ops_no_zeros.append(op)

    return reduced_coeffs_no_zeros, reduced_ops_no_zeros


def out_flow_constraint(graph: nx.DiGraph) -> qml.Hamiltonian:
    r"""Calculates the Hamiltonian which imposes the constraint that each node has
    an outflow of at most one.

    The out flow constraint is, for all :math:`i`:

    .. math:: \sum_{j,(i,j)\in E}x_{ij} \leq 1,

    where :math:`E` are the edges of the graph and :math:`x_{ij}` is a binary number that selects
    whether to include the edge :math:`(i, j)`.

    The corresponding qubit Hamiltonian is:

    .. math::

        \frac{1}{4}\sum_{i\in V}\left(d_{i}^{out}(d_{i}^{out} - 2)\mathbb{I}
        - 2(d_{i}^{out}-1)\sum_{j,(i,j)\in E}\hat{Z}_{ij} +
        \left( \sum_{j,(i,j)\in E}\hat{Z}_{ij} \right)^{2}\right)


    where :math:`V` are the graph vertices and :math:`Z_{ij}` is a qubit Pauli-Z matrix acting
    upon the qubit specified by the pair :math:`(i, j)`. Note that this function omits the
    :math:`1/4` constant factor.

    This Hamiltonian is minimized by selecting edges such that each node has an outflow of at most one.

    Args:
        graph (nx.DiGraph): the graph specifying possible edges

    Returns:
        qml.Hamiltonian: the out flow constraint Hamiltonian

    Raises:
        ValueError: if the input graph is not directed
    """
    if not hasattr(graph, "out_edges"):
        raise ValueError("Input graph must be directed")

    hamiltonian = qml.Hamiltonian([], [])

    for node in graph.nodes:
        hamiltonian += _inner_out_flow_constraint_hamiltonian(graph, node)

    return hamiltonian


def _inner_out_flow_constraint_hamiltonian(
    graph: nx.DiGraph, node: int
) -> Tuple[List[float], List[qml.operation.Observable]]:
    r"""Calculates the expanded inner portion of the Hamiltonian in :func:`out_flow_constraint`.

    For a given :math:`i`, this function returns:

    .. math::

        d_{i}^{out}(d_{i}^{out} - 2)\mathbb{I}
        - 2(d_{i}^{out}-1)\sum_{j,(i,j)\in E}\hat{Z}_{ij} +
        ( \sum_{j,(i,j)\in E}\hat{Z}_{ij}) )^{2}

    Args:
        graph (nx.DiGraph): the graph specifying possible edges
        node (int): a fixed node

    Returns:
        Tuple[List[float], List[qml.operation.Observable]]: The list of coefficients and list of
        observables of the inner part of the node-constraint Hamiltonian.
    """
    coeffs = []
    ops = []

    edges_to_qubits = edges_to_wires(graph)
    out_edges = graph.out_edges(node)
    d = len(out_edges)

    for edge in out_edges:
        wire = (edges_to_qubits[edge],)
        coeffs.append(1)
        ops.append(qml.PauliZ(wire))

    coeffs, ops = _square_hamiltonian_terms(coeffs, ops)

    for edge in out_edges:
        wire = (edges_to_qubits[edge],)
        coeffs.append(-2 * (d - 1))
        ops.append(qml.PauliZ(wire))

    coeffs.append(d * (d - 2))
    ops.append(qml.Identity(0))

    coeffs, ops = _collect_duplicates(coeffs, ops)

    hamiltonian = qml.Hamiltonian(coeffs, ops)

    return hamiltonian

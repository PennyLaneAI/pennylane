# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module contains the implementations of the decompose_graph_state and
null_decompose_graph_state transforms, written using xDSL.

.. note::

    The transforms contained in this module make frequent use of the *densely packed adjacency
    matrix* graph representation. For a detailed description of this graph representation, see the
    documentation for the GraphStatePrepOp operation in the MBQC dialect in Catalyst.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generator, TypeAlias

from xdsl import context, passes, pattern_rewriter
from xdsl.dialects import builtin
from xdsl.pattern_rewriter import PatternRewriter, RewritePattern

from pennylane.exceptions import CompileError

from ..dialects import mbqc, quantum
from .api import compiler_transform

DenselyPackedAdjMatrix: TypeAlias = Sequence[int] | Sequence[bool]


@dataclass(frozen=True)
class DecomposeGraphStatePass(passes.ModulePass):
    """The decompose-graph-state pass replaces ``graph_state_prep`` operations with their
    corresponding sequence of quantum operations for execution on state simulators.
    """

    name = "decompose-graph-state"

    # pylint: disable=arguments-renamed,no-self-use
    def apply(self, _ctx: context.Context, module: builtin.ModuleOp) -> None:
        """Apply the decompose-graph-state pass."""

        walker = pattern_rewriter.PatternRewriteWalker(DecomposeGraphStatePattern())
        walker.rewrite_module(module)


decompose_graph_state_pass = compiler_transform(DecomposeGraphStatePass)


# pylint: disable=too-few-public-methods
class DecomposeGraphStatePattern(RewritePattern):
    """Rewrite pattern for the decompose-graph-state transform."""

    # pylint: disable=arguments-renamed
    @pattern_rewriter.op_type_rewrite_pattern
    def match_and_rewrite(self, graph_prep_op: mbqc.GraphStatePrepOp, rewriter: PatternRewriter, /):
        """Match and rewrite pattern for graph_state_prep ops."""
        # These are the names of the gates that realize the desired initial individual qubit state
        # and entangled state, respectively
        init_op_gate_name = graph_prep_op.init_op.data
        entangle_op_gate_name = graph_prep_op.entangle_op.data

        adj_matrix = _parse_adj_matrix(graph_prep_op)
        n_vertices = _n_vertices_from_packed_adj_matrix(adj_matrix)

        # Allocate a register with as many qubits as vertices in the graph
        alloc_op = quantum.AllocOp(n_vertices)
        rewriter.insert_op(alloc_op)

        # This dictionary maps wires indices in the register to qubit SSA values
        graph_qubits_map: dict[int, quantum.QubitSSAValue] = {}

        # In this section, we create the sequences of quantum.extract, quantum.custom (for the init
        # and entangle gates) and quantum.insert ops, and gather them up into list for insertion
        # later on.
        qextract_ops: list[quantum.ExtractOp] = []
        for i in range(n_vertices):
            qextract_op = quantum.ExtractOp(alloc_op.qreg, i)
            qextract_ops.append(qextract_op)
            graph_qubits_map[i] = qextract_op.qubit

        init_ops: list[quantum.CustomOp] = []
        for i in range(n_vertices):
            init_op = quantum.CustomOp(in_qubits=graph_qubits_map[i], gate_name=init_op_gate_name)
            init_ops.append(init_op)
            graph_qubits_map[i] = init_op.out_qubits[0]

        entangle_ops: list[quantum.CustomOp] = []
        for edge in _edge_iter(adj_matrix):
            q0 = graph_qubits_map[edge[0]]
            q1 = graph_qubits_map[edge[1]]
            entangle_op = quantum.CustomOp(in_qubits=(q0, q1), gate_name=entangle_op_gate_name)
            entangle_ops.append(entangle_op)
            graph_qubits_map[edge[0]] = entangle_op.out_qubits[0]
            graph_qubits_map[edge[1]] = entangle_op.out_qubits[1]

        qinsert_ops: list[quantum.InsertOp] = []
        qreg = alloc_op.qreg
        for i in range(n_vertices):
            qinsert_op = quantum.InsertOp(in_qreg=qreg, idx=i, qubit=graph_qubits_map[i])
            qinsert_ops.append(qinsert_op)
            qreg = qinsert_op.out_qreg

        # In this section, we iterate over the ops created above and insert them.
        # Note that we do not need to specify the insertion point here; all ops are inserted before
        # the matched op, automatically putting them in the order we want them in.
        for qextract_op in qextract_ops:
            rewriter.insert_op(qextract_op)

        for init_op in init_ops:
            rewriter.insert_op(init_op)

        for entangle_op in entangle_ops:
            rewriter.insert_op(entangle_op)

        for qinsert_op in qinsert_ops:
            rewriter.insert_op(qinsert_op)

        # The register that is the result of the last quantum.insert op replaces the register that
        # was the result of the graph_state_prep op
        rewriter.replace_all_uses_with(graph_prep_op.results[0], qinsert_ops[-1].results[0])

        # Finally, erase the ops that have now been replaced with quantum ops
        rewriter.erase_matched_op()

        # Erase the constant op that returned the adjacency matrix only if it has no other uses
        if graph_prep_op.adj_matrix.uses.get_length() == 0:
            rewriter.erase_op(graph_prep_op.adj_matrix.owner)


@dataclass(frozen=True)
class NullDecomposeGraphStatePass(passes.ModulePass):
    """The null-decompose-graph-state pass replaces ``graph_state_prep`` operations with a single
    quantum-register allocation operation for execution on null devices.
    """

    name = "null-decompose-graph-state"

    # pylint: disable=arguments-renamed,no-self-use
    def apply(self, _ctx: context.Context, module: builtin.ModuleOp) -> None:
        """Apply the null-decompose-graph-state pass."""

        walker = pattern_rewriter.PatternRewriteWalker(NullDecomposeGraphStatePattern())
        walker.rewrite_module(module)


null_decompose_graph_state_pass = compiler_transform(NullDecomposeGraphStatePass)


# pylint: disable=too-few-public-methods
class NullDecomposeGraphStatePattern(RewritePattern):
    """Rewrite pattern for the null-decompose-graph-state transform."""

    # pylint: disable=arguments-renamed
    @pattern_rewriter.op_type_rewrite_pattern
    def match_and_rewrite(self, graph_prep_op: mbqc.GraphStatePrepOp, rewriter: PatternRewriter, /):
        """Match and rewrite pattern for graph_state_prep ops."""
        adj_matrix = _parse_adj_matrix(graph_prep_op)
        n_vertices = _n_vertices_from_packed_adj_matrix(adj_matrix)

        # Allocate a register with as many qubits as vertices in the graph
        alloc_op = quantum.AllocOp(n_vertices)
        rewriter.insert_op(alloc_op)

        # The newly allocated register replaces the register that was the result of the
        # graph_state_prep op
        rewriter.replace_all_uses_with(graph_prep_op.results[0], alloc_op.results[0])

        # Finally, erase the ops that have now been replaced with quantum ops
        rewriter.erase_matched_op()

        # Erase the constant op that returned the adjacency matrix only if it has no other uses
        if graph_prep_op.adj_matrix.uses.get_length() == 0:
            rewriter.erase_op(graph_prep_op.adj_matrix.owner)


def _parse_adj_matrix(graph_prep_op: mbqc.GraphStatePrepOp) -> list[int]:
    """Parse the adjacency matrix from the result of the ConstantOp given as input to the
    graph_state_prep op.

    We assume that the adjacency matrix is stored as a DenseIntOrFPElementsAttr, whose data is
    accessible as a Python 'bytes' array. Converting this bytes array to a list results in integer
    elements, whose values are typically either 0 for 'false' or 255 for 'true'.

    Returns:
        list[int]: The densely packed adjacency matrix as a list of ints. See the note in the module
        documentation for a description of this format.
    """
    adj_matrix_const_op = graph_prep_op.adj_matrix.owner
    adj_matrix_value = adj_matrix_const_op.properties.get("value")
    assert adj_matrix_value is not None and hasattr(
        adj_matrix_value, "data"
    ), f"Unable to read graph adjacency matrix from op `{adj_matrix_const_op}`"

    adj_matrix_bytes = adj_matrix_value.data
    assert isinstance(adj_matrix_bytes, builtin.BytesAttr), (
        f"Expected graph adjacency matrix data to be of type 'builtin.BytesAttr', but got "
        f"{type(adj_matrix_bytes).__name__}"
    )

    return list(adj_matrix_bytes.data)


def _n_vertices_from_packed_adj_matrix(adj_matrix: DenselyPackedAdjMatrix) -> int:
    """Returns the number of vertices in the graph represented by the given densely packed adjacency
    matrix.

    Args:
        adj_matrix (DenselyPackedAdjMatrix): The densely packed adjacency matrix, given as a
            sequence of bools or ints. See the note in the module documentation for a description of
            this format.

    Raises:
        CompileError: If the number of elements in `adj_matrix` is not compatible with the number of
            elements in the lower-triangular part of a square matrix, excluding the elements along
            the diagonal.

    Returns:
        int: The number of vertices in the graph.

    Example:
        >>> _n_vertices_from_packed_adj_matrix([1, 1, 0, 0, 1, 1])
        4
    """
    assert isinstance(
        adj_matrix, Sequence
    ), f"Expected `adj_matrix` to be a sequence, but got {type(adj_matrix).__name__}"

    m = len(adj_matrix)

    # The formula to compute the number of vertices, N, in the graph from the number elements in the
    # densely packed adjacency matrix, m, is
    #   N = (1 + sqrt(1 + 8m)) / 2
    # To avoid floating-point errors in the sqrt function, we break it down into integer-arithmetic
    # operations and ensure that the solution is one where N is mathematically a true integer.

    discriminant = 1 + 8 * m
    sqrt_discriminant = math.isqrt(discriminant)

    # Check if it's a perfect square
    if sqrt_discriminant * sqrt_discriminant != discriminant:
        raise CompileError(
            f"The number of elements in the densely packed adjacency matrix is {m}, which does not "
            f"correspond to an integer number of graph vertices"
        )

    # The numerator, 1 + sqrt(1 + 8m), must be even for the result to be an integer. The quantity
    # sqrt(1 + 8m) will always be odd if it's a perfect square, so the quantity (1 + sqrt(1 + 8m))
    # will always be even. We can therefore safely divide (using integer division).
    return (1 + sqrt_discriminant) // 2


def _edge_iter(adj_matrix: DenselyPackedAdjMatrix) -> Generator[tuple[int, int], None, None]:
    """Generate an iterator over the edges in a graph represented by the given densely packed
    adjacency matrix.

    Args:
        adj_matrix (DenselyPackedAdjMatrix): The densely packed adjacency matrix, given as a
            sequence of bools or ints. See the note in the module documentation for a description of
            this format.

    Yields:
        tuple[int, int]: The next edge in the graph, represented as the pair of vertices labelled
            according to their indices in the adjacency matrix.

    Example:
        >>> for edge in _edge_iter([1, 1, 0, 0, 1, 1]):
        ...     print(edge)
        (0, 1)
        (0, 2)
        (1, 3)
        (2, 3)
    """
    # Calling `_n_vertices_from_packed_adj_matrix()` asserts that the input `adj_matrix` is in the
    # correct format and is valid.
    _n_vertices_from_packed_adj_matrix(adj_matrix)

    j = 1
    k = 0
    for entry in adj_matrix:
        if entry:
            yield (k, j)
        k += 1
        if k == j:
            k = 0
            j += 1

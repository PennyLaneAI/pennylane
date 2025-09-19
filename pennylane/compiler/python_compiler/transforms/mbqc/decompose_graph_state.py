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

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

from xdsl import context, passes, pattern_rewriter
from xdsl.dialects import builtin
from xdsl.pattern_rewriter import PatternRewriter, RewritePattern

from ...dialects import mbqc, quantum
from ...pass_api import compiler_transform
from .graph_state_utils import edge_iter, n_vertices_from_packed_adj_matrix

DenselyPackedAdjMatrix: TypeAlias = Sequence[int] | Sequence[bool]


@dataclass(frozen=True)
class DecomposeGraphStatePass(passes.ModulePass):
    """The decompose-graph-state pass replaces ``graph_state_prep`` operations with their
    corresponding sequence of quantum operations for execution on state simulators.
    """

    name = "decompose-graph-state"

    # pylint: disable=no-self-use
    def apply(self, _ctx: context.Context, module: builtin.ModuleOp) -> None:
        """Apply the decompose-graph-state pass."""

        walker = pattern_rewriter.PatternRewriteWalker(DecomposeGraphStatePattern())
        walker.rewrite_module(module)


decompose_graph_state_pass = compiler_transform(DecomposeGraphStatePass)


# pylint: disable=too-few-public-methods
class DecomposeGraphStatePattern(RewritePattern):
    """Rewrite pattern for the decompose-graph-state transform."""

    @pattern_rewriter.op_type_rewrite_pattern
    def match_and_rewrite(self, graph_prep_op: mbqc.GraphStatePrepOp, rewriter: PatternRewriter, /):
        """Match and rewrite pattern for graph_state_prep ops."""
        # These are the names of the gates that realize the desired initial individual qubit state
        # and entangled state, respectively
        init_op_gate_name = graph_prep_op.init_op.data
        entangle_op_gate_name = graph_prep_op.entangle_op.data

        adj_matrix = _parse_adj_matrix(graph_prep_op)
        n_vertices = n_vertices_from_packed_adj_matrix(adj_matrix)

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
        for edge in edge_iter(adj_matrix):
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

    # pylint: disable=no-self-use
    def apply(self, _ctx: context.Context, module: builtin.ModuleOp) -> None:
        """Apply the null-decompose-graph-state pass."""

        walker = pattern_rewriter.PatternRewriteWalker(NullDecomposeGraphStatePattern())
        walker.rewrite_module(module)


null_decompose_graph_state_pass = compiler_transform(NullDecomposeGraphStatePass)


# pylint: disable=too-few-public-methods
class NullDecomposeGraphStatePattern(RewritePattern):
    """Rewrite pattern for the null-decompose-graph-state transform."""

    @pattern_rewriter.op_type_rewrite_pattern
    def match_and_rewrite(self, graph_prep_op: mbqc.GraphStatePrepOp, rewriter: PatternRewriter, /):
        """Match and rewrite pattern for graph_state_prep ops."""
        adj_matrix = _parse_adj_matrix(graph_prep_op)
        n_vertices = n_vertices_from_packed_adj_matrix(adj_matrix)

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

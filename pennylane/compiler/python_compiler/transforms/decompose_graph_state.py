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

"""This module contains the implementation of the decompose_graph_state transform,
written using xDSL.
"""

from dataclasses import dataclass
from typing import Generator

import numpy as np
from xdsl import context, ir, passes, pattern_rewriter
from xdsl.dialects import arith, builtin, func, tensor
from xdsl.pattern_rewriter import PatternRewriter, RewritePattern
from xdsl.rewriter import InsertPoint

from pennylane.exceptions import CompileError

from ..dialects import mbqc, quantum
from .api import compiler_transform


@dataclass(frozen=True)
class DecomposeGraphStatePass(passes.ModulePass):
    """Pass that ... [TODO]"""

    name = "decompose-graph-state"

    # pylint: disable=arguments-renamed,no-self-use
    def apply(self, _ctx: context.Context, module: builtin.ModuleOp) -> None:
        """Apply the decompose-graph-state pass."""

        greedy_applier = pattern_rewriter.GreedyRewritePatternApplier(
            [DecomposeGraphStatePattern()]
        )
        walker = pattern_rewriter.PatternRewriteWalker(greedy_applier)
        walker.rewrite_module(module)


decompose_graph_state_pass = compiler_transform(DecomposeGraphStatePass)


# pylint: disable=too-few-public-methods
class DecomposeGraphStatePattern(RewritePattern):
    """Rewrite pattern ... [TODO]"""

    def __init__(self):
        super().__init__()

    # pylint: disable=arguments-renamed
    @pattern_rewriter.op_type_rewrite_pattern
    def match_and_rewrite(self, graph_prep_op: mbqc.GraphStatePrepOp, rewriter: PatternRewriter, /):
        # These are the names of the gates that realize the desired initial individual qubit state
        # and entangled state, respectively
        init_op_gate_name = graph_prep_op.init_op.data
        entangle_op_gate_name = graph_prep_op.entangle_op.data

        # Parse the adjacency matrix from the result of the ConstantOp given as input to the
        # graph_state_prep op. We assume that the adjacency matrix is stored as a
        # DenseIntOrFPElementsAttr, whose data is accessible as a 'bytes' array.
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

        adj_matrix = list(adj_matrix_bytes.data)
        n_vertices = _n_vertices_from_packed_adj_matrix(adj_matrix)

        alloc_op = quantum.AllocOp(n_vertices)
        rewriter.insert_op(alloc_op)

        graph_qubits_map: dict[int, quantum.QubitSSAValue] = {}

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

        for qextract_op in qextract_ops:
            rewriter.insert_op(qextract_op)

        for init_op in init_ops:
            rewriter.insert_op(init_op)

        for entangle_op in entangle_ops:
            rewriter.insert_op(entangle_op)

        for qinsert_op in qinsert_ops:
            rewriter.insert_op(qinsert_op)

        rewriter.erase_matched_op()
        rewriter.erase_op(graph_prep_op.adj_matrix.owner)


def _n_vertices_from_packed_adj_matrix(adj_matrix: list) -> int:
    """TODO"""
    m = len(adj_matrix)

    N = (1 + np.sqrt(1 + 8 * m)) / 2

    if N != int(N):
        raise CompileError(
            f"The number of elements in the densely packed adjacency matrix is {m}, which does not "
            f"correspond to an integer number of graph vertices"
        )

    return int(N)


def _edge_iter(adj_matrix: list) -> Generator[tuple[int, int], None, None]:
    """TODO"""
    j = 1
    k = 0
    for i in range(len(adj_matrix)):
        if adj_matrix[i]:
            yield (k, j)
        k += 1
        if k == j:
            k = 0
            j += 1

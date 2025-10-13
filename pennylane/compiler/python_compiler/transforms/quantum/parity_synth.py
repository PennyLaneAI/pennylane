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

"""This file contains the implementation of the ``ParitySynth`` compiler pass. Given that it
operates on the phase polynomial representation of subcircuits, the implementation splits into
an xDSL-agnostic synthesis functionality and an integration thereof into xDSL."""

from dataclasses import dataclass
from itertools import product

import networkx as nx
import numpy as np
from xdsl import context, passes, pattern_rewriter
from xdsl.dialects import arith, builtin, func
from xdsl.ir import SSAValue
from xdsl.rewriter import InsertPoint

from .....transforms.intermediate_reps.rowcol import _rowcol_parity_matrix, postorder_traverse
from ...dialects.quantum import CustomOp, QubitType
from ...pass_api import compiler_transform

### xDSL-agnostic part


def _loop_body_parity_network_synth(
    P: np.ndarray,
    inv_synth_matrix: np.ndarray,
    circuit: list[int, list[tuple[int]]],
) -> tuple[np.ndarray, list]:
    """Loop body function for ``_parity_network_synth``, the main subroutine of ``parity_synth``.
    The loop body corresponds to synthesizing one parity in the parity table ``P``, and updating
    all relevant data accordingly. It is the ``for``-loop body in Algorithm 1
    in https://arxiv.org/abs/2104.00934.

    Args:
        P (np.ndarray): (Remaining) parity table for which to synthesize the parity network.
        inv_synth_matrix (np.ndarray): Inverse of the parity _matrix_ implemented within
            the parity network that has been synthesized so far.
        circuit (list[int, list[tuple[int]]]): Circuit for the parity network that has been
            synthesized so far. Each entry of the list consists of a _relative_ index into
            the list of parities (or rotation angles) of the phase polynomial, a qubit
            index onto which the rotation should be applied, and the subcircuit that should
            be applied _before_ the rotation to achieve the respective parity.

    Returns:
        tuple[np.ndarray, list]: Same as inputs, with updates applied; ``P`` has a column less
        and has been transformed in addition. ``inv_synth_matrix`` has been transformed
        according to the newly synthesized subcircuit implementing the next parity. The
        ``circuit`` representation is grown by one entry, corresponding to that parity.

    """
    parity_idx = np.argmin(np.sum(P, axis=0))
    parity = P[:, parity_idx]
    graph_nodes = list(map(int, np.where(parity)[0]))
    if len(graph_nodes) == 1:
        # The parity already has Hamming weight 1, so we don't need any modifications
        # Just slice out the parity and append the parity/angle index as well as the qubit
        # on which the parity has support
        P = np.concatenate([P[:, :parity_idx], P[:, parity_idx + 1 :]], axis=1)
        circuit.append((parity_idx, graph_nodes[0], []))
        return P, inv_synth_matrix, circuit
    single_weights = np.sum(P, axis=1)
    parity_graph = nx.DiGraph()
    parity_graph.add_weighted_edges_from(
        [
            (i, j, np.sum(np.mod(P[i] + P[j], 2)) - single_weights[j])
            for i, j in product(graph_nodes, repeat=2)
            if i != j
        ]
    )
    arbor = nx.minimum_spanning_arborescence(parity_graph)
    roots = [node for node, degree in arbor.in_degree() if degree == 0]
    assert len(roots) == 1
    P = np.concatenate([P[:, :parity_idx], P[:, parity_idx + 1 :]], axis=1)
    sub_circuit = []
    for i, j in postorder_traverse(arbor, source=roots[0]):
        sub_circuit.append((i, j))
        P[i] = np.mod(P[i] + P[j], 2)
        inv_synth_matrix[:, i] += inv_synth_matrix[:, j]
    circuit.append((parity_idx, roots[0], sub_circuit))
    return P, inv_synth_matrix, circuit


def _parity_network_synth(P: np.ndarray) -> list[int, list[tuple[int]]]:
    """Main subroutine for the ``ParitySynth`` pass, mostly a ``for``-loop wrapper around
    ``_loop_body_parity_network_synth``. It synthesizes the parity network, as described
    in Algorithm 1 in https://arxiv.org/abs/2104.00934.

    Args:
        P (np.ndarray): Parity table to be synthesized.

    Returns:
        tuple[list[int, list[tuple[int]]], np.ndarray]: Synthesized parity network, as a
        circuit with structure as described in ``_loop_body_parity_network_synth``. Also,
        inverse of the parity matrix implemented by the synthesized circuit.

    """
    if P.shape[-1] == 0:
        return [], None

    circuit = []
    num_wires, num_parities = P.shape
    inv_synth_matrix = np.eye(num_wires, dtype=int)
    for _ in range(num_parities):
        P, inv_synth_matrix, circuit = _loop_body_parity_network_synth(P, inv_synth_matrix, circuit)

    return circuit, inv_synth_matrix % 2


### end of xDSL-agnostic part

valid_phase_polynomial_ops = {"CNOT", "RZ"}


def make_phase_polynomial(
    ops: list[CustomOp],
    init_wire_map: dict[QubitType, int],
) -> tuple[np.ndarray]:
    r"""Compute the phase polynomial representation of a list of ``CustomOp``\ s.
    This implementation is very similar to :func:`~.transforms.intermediate_reps.phase_polynomial`
    but adjusted to work with xDSL objects."""
    wire_map = init_wire_map

    parity_matrix = np.eye(len(wire_map), dtype=int)
    parity_table = []
    angles = []
    arith_ops = []
    for op in ops:
        name = op.gate_name.data
        if name == "CNOT":
            control, target = wire_map.pop(op.in_qubits[0]), wire_map.pop(op.in_qubits[1])
            parity_matrix[target] += parity_matrix[control]
            wire_map[op.out_qubits[0]] = control
            wire_map[op.out_qubits[1]] = target
            continue
        # RZ
        angle = op.operands[0]
        if getattr(op, "adjoint", False):
            neg_op = arith.NegfOp(angle)
            arith_ops.append(neg_op)
            angle = neg_op.result
        angles.append(angle)
        wire = wire_map[op.in_qubits[0]]
        parity_table.append(parity_matrix[wire].copy())  # append _current_ parity (hence the copy)
        wire_map[op.out_qubits[0]] = wire

    return parity_matrix % 2, np.array(parity_table).T % 2, angles, arith_ops


class ParitySynthPattern(pattern_rewriter.RewritePattern):
    """Rewrite pattern that applies ``ParitySynth`` to subcircuits that constitute
    phase polynomials.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reset_vars()

    def _reset_vars(self):
        """Initialize/reset variables that are used in ``match_and_rewrite`` as well as
        ``rewrite_phase_polynomial``."""
        self.phase_polynomial_ops: list[CustomOp] = []
        self.init_wire_map: dict[QubitType, int] = {}
        self.phase_polynomial_qubits: set[QubitType] = set()
        self.num_phase_polynomial_qubits: int = 0

    def _record_phase_poly_op(self, op: CustomOp):
        """Add a ``CustomOp`` to the phase polynomial ops, remove its input qubits
        from ``self.phase_polynomial_qubits`` if present or add them to ``self.init_wire_map``
        if not, and insert its output qubits in ``self.phase_polynomial_qubits``."""
        for i, q in enumerate(op.in_qubits):
            if q in self.phase_polynomial_qubits:
                self.phase_polynomial_qubits.remove(q)
            else:
                self.init_wire_map[q] = self.num_phase_polynomial_qubits
                self.num_phase_polynomial_qubits += 1
            self.phase_polynomial_qubits.add(op.out_qubits[i])
        self.phase_polynomial_ops.append(op)

    @pattern_rewriter.op_type_rewrite_pattern
    def match_and_rewrite(self, funcOp: func.FuncOp, rewriter: pattern_rewriter.PatternRewriter):
        r"""Implementation of rewriting ``FuncOps`` that may contain phase poynomials
        with ``ParitySynth``.

        Args:
            funcOp (func.FuncOp): function containing the operations to rewrite.
            rewriter (pattern_rewriter.PatternRewriter): Rewriter that executed operation erasure
                and insertion.

        The logic of this implementation is centered around :attr:`~.rewrite_phase_polynomial`,
        which is able to rewrite a collection of ``CustomOp``\ s that forms a phase polynomial
        (see ``valid_phase_polynomial_ops`` for the supported types) into a new collection of
        ``CustomOp``\ s that is equivalent. In addition to the operators, which are stored in
        ``self.phase_polynomial_ops``, the ``rewrite_phase_polynomial`` subroutine requires
        the initial mapping from input qubits to integer-valued wire positions, which is computed
        in ``self.init_wire_map`` using temporary variables ``self.phase_polynomial_qubits``
        and ``self.num_phase_polynomial_qubits``.

        Iterating over all operations, the collected phase polynomial ops are rewritten as soon
        as a non-phase-polynomial operation is encountered. Note that this makes the (size of the)
        rewritten phase polynomials dependent on the order in which we walk over the operations.
        """
        for op in funcOp.body.walk():
            if not isinstance(op, CustomOp):
                # Non-quantum operation. Global phases are ignored as well.
                continue

            gate_name = op.gate_name.data
            if gate_name in valid_phase_polynomial_ops:
                # Include op in phase polynomial ops and track its qubits
                self._record_phase_poly_op(op)
                continue

            # not a phase polynomial op, so we activate rewriting of the phase polynomial
            self.rewrite_phase_polynomial(rewriter)

        # end of operations; rewrite terminal phase polynomial
        self.rewrite_phase_polynomial(rewriter)

        # Mock the rewriter to think it reached a steady state already, because re-applying
        # ParitySynth is not useful
        # todo: to this properly by using a different rewriter or so
        rewriter.has_done_action = False

    @staticmethod
    def _cnot(i: int, j: int, inv_wire_map: dict[int, QubitType]):
        """Create a CNOT operator acting on the qubits that map to wires ``i`` and ``j``
        and update the wire map so that ``i`` and ``j`` point to the output qubits afterwards."""
        cnot_op = CustomOp(
            in_qubits=[inv_wire_map[i], inv_wire_map[j]],
            gate_name="CNOT",
            params=tuple(),
        )
        inv_wire_map[i] = cnot_op.out_qubits[0]
        inv_wire_map[j] = cnot_op.out_qubits[1]
        return cnot_op

    @staticmethod
    def _rz(wire: int, angle: SSAValue[builtin.Float64Type], inv_wire_map: dict[int, QubitType]):
        """Create a CNOT operator acting on the qubit that maps to ``wire``
        and update the wire map so that ``wire`` points to the output qubit afterwards."""
        rz_op = CustomOp(in_qubits=[inv_wire_map[wire]], gate_name="RZ", params=(angle,))
        inv_wire_map[wire] = rz_op.out_qubits[0]
        return rz_op

    def rewrite_phase_polynomial(self, rewriter: pattern_rewriter.PatternRewriter):
        """Rewrite a single region of a circuit that represents a phase polynomial."""
        if not self.phase_polynomial_ops:
            # Nothing to do
            return

        if len(self.phase_polynomial_ops) == 1:
            # Phase polynomials of length 1 are left untouched. Reset internal state
            self._reset_vars()
            return

        insertion_point: InsertPoint = InsertPoint.after(self.phase_polynomial_ops[-1])

        # Mapping from integer-valued wire positions to qubits, corresponding to state before
        # phase polynomial
        inv_wire_map: dict[int, QubitType] = {val: key for key, val in self.init_wire_map.items()}

        # Calculate the new circuit by going to phase polynomial IR and back, including synthesis
        # of trailing CNOTs via rowcol
        M, P, angles, arith_ops = make_phase_polynomial(
            self.phase_polynomial_ops, self.init_wire_map
        )

        # Insert arithmetic operations produced within `make_phase_polynomial`
        for op in arith_ops:
            rewriter.insert_op(op, insertion_point)

        subcircuits, inv_network_parity_matrix = _parity_network_synth(P)
        # `inv_network_parity_matrix` might be None if the parity table was empty
        if inv_network_parity_matrix is not None:
            M = (M @ inv_network_parity_matrix) % 2
        rowcol_circuit: list[tuple[int]] = _rowcol_parity_matrix(M)

        # Apply the parity network part of the new circuit
        for idx, phase_wire, subcircuit in subcircuits:
            for i, j in subcircuit:
                rewriter.insert_op(self._cnot(i, j, inv_wire_map), insertion_point)

            rewriter.insert_op(self._rz(phase_wire, angles.pop(idx), inv_wire_map), insertion_point)

        # Apply the remaining parity matrix part of the new circuit
        for i, j in rowcol_circuit:
            rewriter.insert_op(self._cnot(i, j, inv_wire_map), insertion_point)

        # Replace the output qubits of the old phase polynomial operations by the output qubits of
        # the new circuit
        for old_qubit, int_wire in self.init_wire_map.items():
            rewriter.replace_all_uses_with(old_qubit, inv_wire_map[int_wire])

        # Erase the old phase polynomial operations.
        for op in self.phase_polynomial_ops[::-1]:
            rewriter.erase_op(op)

        # Reset internal state
        self._reset_vars()


@dataclass(frozen=True)
class ParitySynthPass(passes.ModulePass):
    """Pass for applying ParitySynth to phase polynomials in a circuit."""

    name = "xdsl-parity-synth"

    # pylint: disable=no-self-use
    def apply(self, _ctx: context.Context, module: builtin.ModuleOp) -> None:
        """Apply the ParitySynth pass."""
        pattern_rewriter.PatternRewriteWalker(
            pattern_rewriter.GreedyRewritePatternApplier([ParitySynthPattern()])
        ).rewrite_module(module)


parity_synth_pass = compiler_transform(ParitySynthPass)

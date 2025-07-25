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
"""Pattern rewriter API for quantum compilation passes."""

from xdsl.ir import Operation
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriterListener, PatternRewriteWalker
from xdsl.rewriter import InsertPoint

from ...dialects import quantum


class PLPatternRewriter(PatternRewriter):
    """A ``PatternRewriter`` with abstractions for quantum compilation passes.

    This is a subclass of ``xdsl.pattern_rewriter.PatternRewriter`` that exposes
    methods to abstract away low-level pattern-rewriting details relevant to
    quantum compilation passes.
    """

    def erase_quantum_gate_op(self, op: Operation) -> None:
        """Erase a quantum gate.

        Safely erase a quantum gate from the module being transformed. This method automatically
        handles and pre-processing required before safely erasing an operation. To erase quantum
        gates, which include ``CustomOp``, ``MultiRZOp``, and ``QubitUnitaryOp``, it is recommended
        to use this method instead of ``erase_op``.

        Args:
            op (xdsl.ir.Operation): The operation to erase
        """
        if not isinstance(op, (quantum.CustomOp, quantum.MultiRZOp, quantum.QubitUnitaryOp)):
            return

        # We can also use the following code to perform the same task:
        # self.replace_op(op, (), op.in_qubits + op.in_ctrl_qubits)

        for iq, oq in zip(op.in_qubits, op.out_qubits, strict=True):
            self.replace_all_uses_with(oq, iq)
        for icq, ocq in zip(op.in_ctrl_qubits, op.out_ctrl_qubits, strict=True):
            self.replace_all_uses_with(ocq, icq)

        self.erase_op(op)


class PLPatternRewriteWalker(PatternRewriteWalker):
    """A ``PatternRewriteWalker`` for traversing and rewriting modules.

    This is a subclass of ``xdsl.pattern_rewriter.PatternRewriteWalker that uses a custom
    rewriter that contains abstractions for quantum compilation passes."""

    def _process_worklist(self, listener: PatternRewriterListener) -> bool:
        """
        Process the worklist until it is empty.
        Returns true if any modification was done.
        """
        rewriter_has_done_action = False

        # Handle empty worklist
        op = self._worklist.pop()
        if op is None:
            return rewriter_has_done_action

        # Create a rewriter on the first operation
        # Here, we use our custom rewriter instead of the default PatternRewriter.
        rewriter = PLPatternRewriter(op)
        rewriter.extend_from_listener(listener)

        # do/while loop
        while True:
            # Reset the rewriter on `op`
            rewriter.has_done_action = False
            rewriter.current_operation = op
            rewriter.insertion_point = InsertPoint.before(op)

            # Apply the pattern on the operation
            try:
                self.pattern.match_and_rewrite(op, rewriter)
            except Exception as err:  # pylint: disable=broad-exception-caught
                op.emit_error(
                    f"Error while applying pattern: {err}",
                    underlying_error=err,
                )
            rewriter_has_done_action |= rewriter.has_done_action

            # If the worklist is empty, we are done
            op = self._worklist.pop()
            if op is None:
                return rewriter_has_done_action

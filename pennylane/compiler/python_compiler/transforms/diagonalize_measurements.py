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
"""This file contains the implementation of the diagonalize_measurements transform,
written using xDSL.


Known Limitations
-----------------
  * Only observables PauliX, PauliY, PauliZ and Identity are currently supported when using this transform.
  * Usage patterns that are not yet supported with program capture are also not supported in the
    compilation pass. For example, operator arithmetic is not currently supported, such as
    qml.expval(qml.Y(0) @ qml.X(1)).
  * The current implementation does not allow for support for observables with parametrized diagonalizing gates
    (for example, qml.H(0).diagonalizing_gates())
  * Unlike the current tape-based implementation of the transform, it doesn't allow for diagonalization of a subset
    of observables.
  * Unlike the current tape-based implementation of the transform, conversion to measurements based on eigvals
  and wires (rather than the PauliZ observable) is not currently supported.
"""

from dataclasses import dataclass

from xdsl import context, passes, pattern_rewriter
from xdsl.dialects import builtin
from xdsl.rewriter import InsertPoint

from pennylane import ops as pl_ops

from ..quantum_dialect import CustomOp, NamedObservable, NamedObservableAttr, NamedObsOp
from .api import compiler_transform


def _diagonalize(obs: NamedObsOp) -> bool:
    """Whether to diagonalize a given observable."""
    if obs.type.data in ("PauliZ", "Identity"):
        return False
    if obs.type.data in ("PauliX", "PauliY"):
        return True
    raise NotImplementedError(f"Observable {obs.type.data} is not supported for diagonalization")


def _diagonalizing_gates(obs: NamedObsOp):
    """Get the names of the observable's diagonalizing gates from the corresponding
    PennyLane Operation, and use them to get an xDSL representation of the diagonalizing
    gates.

    Returns a list of CustomOps representing the chain of diagonalizing gates, and the
    new SSA value for the observable qubit wire following diagonalization."""
    pl_obs = getattr(pl_ops, obs.type.data)(0)
    gates = [o.name for o in pl_obs.diagonalizing_gates()]

    diagonalizing_gates = []
    qubit = obs.qubit

    for name in gates:
        diagonalizing_gates.append(CustomOp(in_qubits=qubit, gate_name=name))
        qubit = diagonalizing_gates[-1].out_qubits[0]

    return diagonalizing_gates, qubit


class DiagonalizeFinalMeasurementsPattern(
    pattern_rewriter.RewritePattern
):  # pylint: disable=too-few-public-methods
    """RewritePattern for diagonalizing final measurements."""

    @pattern_rewriter.op_type_rewrite_pattern
    def match_and_rewrite(
        self, observable: NamedObsOp, rewriter: pattern_rewriter.PatternRewriter
    ):  # pylint: disable=arguments-differ
        """Replace non-diagonalized observables with their diagonalizing gates and PauliZ."""

        if _diagonalize(observable):

            diagonalizing_gates, obs_qubit = _diagonalizing_gates(observable)
            for gate in diagonalizing_gates:
                rewriter.insert_op(gate, InsertPoint.before(observable))

            diag_obs = NamedObsOp(
                qubit=obs_qubit, obs_type=NamedObservableAttr(NamedObservable("PauliZ"))
            )
            rewriter.replace_op(observable, diag_obs)


@dataclass(frozen=True)
class DiagonalizeFinalMeasurementsPass(passes.ModulePass):
    """Pass for merging consecutive composable rotation gates."""

    name = "diagonalize-measurements"

    # pylint: disable=arguments-renamed
    def apply(self, _ctx: context.Context, module: builtin.ModuleOp) -> None:
        """Apply the diagonalize final measurements pass."""
        pattern_rewriter.PatternRewriteWalker(
            pattern_rewriter.GreedyRewritePatternApplier([DiagonalizeFinalMeasurementsPattern()])
        ).rewrite_module(module)


diagonalize_measurements_pass = compiler_transform(DiagonalizeFinalMeasurementsPass)

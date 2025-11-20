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
"""This file contains the implementation of the diagonalize_final_measurements transform,
written using xDSL.


Known Limitations
-----------------
  * Only observables PauliX, PauliY, PauliZ, Hadamard and Identity are currently supported when using this transform (but
    these are also the only observables currently supported in the Quantum dialect as NamedObservable).
  * Unlike the current tape-based implementation of the transform, it doesn't allow for diagonalization of a subset
    of observables.
  * Unlike the current tape-based implementation of the transform, conversion to measurements based on eigvals
  and wires (rather than the PauliZ observable) is not currently supported.
  * Unlike the tape-based implementation, this pass will NOT raise an error if given a circuit that is invalid because
    it contains non-commuting measurements. It should be assumed that this transform results in incorrect outputs unless
    split_non_commuting is applied to break non-commuting measurements into separate tapes.
"""

from dataclasses import dataclass

from xdsl import context, passes, pattern_rewriter
from xdsl.dialects import arith, builtin
from xdsl.rewriter import InsertPoint

from pennylane.ops import Hadamard, PauliX, PauliY

from ...dialects.quantum import (
    CustomOp,
    GlobalPhaseOp,
    MultiRZOp,
    NamedObservable,
    NamedObservableAttr,
    NamedObsOp,
    QubitUnitaryOp,
)
from ...pass_api import compiler_transform


def _generate_mapping():
    _gate_map = {}
    _params_map = {}

    for op in PauliX(0), PauliY(0), Hadamard(0):
        diagonalizing_gates = op.diagonalizing_gates()

        _gate_map[op.name] = [gate.name for gate in diagonalizing_gates]
        _params_map[op.name] = [gate.data for gate in diagonalizing_gates]

    return _gate_map, _params_map


_gate_map, _params_map = _generate_mapping()


def _diagonalize(obs: NamedObsOp) -> bool:
    """Whether to diagonalize a given observable."""
    if obs.type.data in {"PauliZ", "Identity"}:
        return False
    if obs.type.data in _gate_map:
        return True
    raise NotImplementedError(f"Observable {obs.type.data} is not supported for diagonalization")


class DiagonalizeFinalMeasurementsPattern(
    pattern_rewriter.RewritePattern
):  # pylint: disable=too-few-public-methods
    """RewritePattern for diagonalizing final measurements."""

    # pylint: disable=no-self-use
    @pattern_rewriter.op_type_rewrite_pattern
    def match_and_rewrite(self, observable: NamedObsOp, rewriter: pattern_rewriter.PatternRewriter):
        """Replace non-diagonalized observables with their diagonalizing gates and PauliZ."""

        if _diagonalize(observable):

            diagonalizing_gates = _gate_map[observable.type.data]
            params = _params_map[observable.type.data]

            qubit = observable.qubit

            insert_point = InsertPoint.before(observable)

            for name, op_data in zip(diagonalizing_gates, params):
                if op_data:
                    param_ssa_values = []
                    for param in op_data:
                        paramOp = arith.ConstantOp(
                            builtin.FloatAttr(data=param, type=builtin.Float64Type())
                        )
                        rewriter.insert_op(paramOp, insert_point)
                        param_ssa_values.append(paramOp.results[0])

                    gate = CustomOp(in_qubits=qubit, gate_name=name, params=param_ssa_values)
                else:
                    gate = CustomOp(in_qubits=qubit, gate_name=name)

                rewriter.insert_op(gate, insert_point)

                qubit = gate.out_qubits[0]

            # we need to replace the initial qubit use everwhere EXCEPT the use that is now the
            # input to the first diagonalizing gate. Its not enough to only change the NamedObsOp,
            # because the qubit might be inserted/deallocated later
            uses_to_change = [
                use
                for use in observable.qubit.uses
                if not isinstance(
                    use.operation, (CustomOp, GlobalPhaseOp, MultiRZOp, QubitUnitaryOp)
                )
            ]
            num_observables = len(
                [use for use in uses_to_change if isinstance(use.operation, NamedObsOp)]
            )

            if num_observables > 1:
                raise RuntimeError(
                    "Each wire can only have one set of diagonalizing gates applied, but the circuit contains multiple observables with the same wire."
                )

            observable.qubit.replace_by_if(qubit, lambda use: use in uses_to_change)
            for use in uses_to_change:
                rewriter.notify_op_modified(use.operation)

            # then we also update the observable to be in the Z basis. Since this is done with the
            # rewriter, we don't need to call `rewriter.notify_modified(observable)` regarding this
            diag_obs = NamedObsOp(
                qubit=qubit, obs_type=NamedObservableAttr(NamedObservable("PauliZ"))
            )
            rewriter.replace_op(observable, diag_obs)


@dataclass(frozen=True)
class DiagonalizeFinalMeasurementsPass(passes.ModulePass):
    """Pass for diagonalizing final measurements."""

    name = "diagonalize-final-measurements"

    # pylint: disable= no-self-use
    def apply(self, _ctx: context.Context, module: builtin.ModuleOp) -> None:
        """Apply the diagonalize final measurements pass."""
        pattern_rewriter.PatternRewriteWalker(DiagonalizeFinalMeasurementsPattern()).rewrite_module(
            module
        )


diagonalize_final_measurements_pass = compiler_transform(DiagonalizeFinalMeasurementsPass)

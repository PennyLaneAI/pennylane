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
"""This file contains the implementation of the decompose transform,
written using xDSL."""

from dataclasses import dataclass
from typing import Iterable

from xdsl import context, passes, pattern_rewriter
from xdsl.dialects import builtin, func
from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.builtin import Float64Type, FloatAttr, IntegerAttr, IntegerType, StringAttr
from xdsl.rewriter import InsertPoint

import pennylane as qml
from pennylane.compiler.python_compiler.quantum_dialect import (
    CustomOp,
    ExtractOp,
    GlobalPhaseOp,
    QubitType,
    AllocOp,
)
from pennylane.operation import Operator


# This is just a preliminary structure for mapping of PennyLane gates to xDSL operations.
from_str_to_PL_gate = {
    "RX": qml.RX,
    "RY": qml.RY,
    "RZ": qml.RZ,
    "Rot": qml.Rot,
    "CNOT": qml.CNOT,
    "Hadamard": qml.Hadamard,
    "PhaseShift": qml.PhaseShift,
}


def flatten(nested: Iterable):
    """Recursively flatten an arbitrarily nested iterable (except strings)."""
    flat = []
    for item in nested:
        if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
            flat.extend(flatten(item))
        else:
            flat.append(item)
    return flat


def resolve_constant_params(op: Operator):
    """Walk back to the producing op and resolve the constant parameters."""

    while hasattr(op, "owner"):
        op = op.owner

    if isinstance(op, ConstantOp):
        val = op.value

        if isinstance(val, (FloatAttr, IntegerAttr)):
            return val.value.data

    if isinstance(op, ExtractOp):
        return op.idx_attr.parameters[0].data

    if isinstance(op, CustomOp):

        raise NotImplementedError("Cannot resolve params from CustomOp")

    raise NotImplementedError(f"Cannot resolve params from {op}")


def resolve_constant_wire(operand):
    """Resolve the integer wire index that this operand originates from."""

    while hasattr(operand, "owner"):
        result_index = operand.index
        operand = operand.owner

        if isinstance(operand, GlobalPhaseOp):
            return None

        if isinstance(operand, ConstantOp):
            val = operand.value
            if isinstance(val, IntegerAttr):
                return val.value.data

        elif isinstance(operand, ExtractOp):
            return operand.idx_attr.parameters[0].data

        elif isinstance(operand, CustomOp):

            if result_index < len(operand.in_qubits):
                input_operand = operand.in_qubits[result_index]
                return resolve_constant_wire(input_operand)

            raise IndexError(
                f"Result index {result_index} out of bounds for CustomOp with {len(operand.in_qubits)} in_qubits."
            )

        else:
            raise NotImplementedError(f"Cannot resolve wires from operation type: {type(operand)}")

    raise TypeError(f"Operand {operand} is not a result or constant-producing op")


def get_parameters(op: Operator) -> list[float | int]:
    """Get the parameters from the operation."""
    return [resolve_constant_params(p) for p in op.params if p is not None]


def get_wires(op: Operator) -> list[int]:
    """Get the wires from the operation."""
    if not hasattr(op, "in_qubits"):
        return []
    return [resolve_constant_wire(w) for w in op.in_qubits if w is not None]


def get_op_name(op: Operator) -> str:
    """Get the name of the operation from the properties."""
    return op.properties["gate_name"].data


def resolve_gate(name: str):
    """Resolve the gate from the name."""
    try:
        return from_str_to_PL_gate[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported gate: {name}") from exc


def reconstruct_gate(op: CustomOp):
    """Reconstruct the gate from the operation."""
    gate_name = get_op_name(op)
    parameters = get_parameters(op)
    wires = flatten(get_wires(op))
    return resolve_gate(gate_name)(*parameters, wires=wires)


# pylint: disable=too-few-public-methods
class DecompositionTransform(pattern_rewriter.RewritePattern):
    """A pattern that rewrites CustomOps to their decomposition."""

    def __init__(self, module):
        self.module = module
        super().__init__()

    # pylint:disable=arguments-differ, too-many-branches
    @pattern_rewriter.op_type_rewrite_pattern
    def match_and_rewrite(self, funcOp: func.FuncOp, rewriter: pattern_rewriter.PatternRewriter):
        """Rewrite the function by replacing CustomOps with their decomposition."""
        for op in funcOp.body.walk():

            if not isinstance(op, CustomOp):
                continue

            concrete_op = reconstruct_gate(op)

            if not concrete_op.has_decomposition:
                continue

            decomp_ops = concrete_op.decomposition()
            last_custom_op = None

            qreg_val = None
            for candidate_op in op.parent.walk():
                if isinstance(candidate_op, AllocOp):
                    qreg_val = candidate_op.qreg
                    break

            if qreg_val is None:
                raise Exception("AllocOp not found; cannot extract qubits")

            for qml_op in decomp_ops:
                parameters_xdsl = []
                wires_xdsl = []

                for param in qml_op.parameters:
                    val = param.item() if hasattr(param, "item") else param
                    if isinstance(val, int):
                        attr = IntegerAttr(val, IntegerType(64))
                    elif isinstance(val, float):
                        attr = FloatAttr(val, Float64Type())
                    else:
                        raise TypeError(f"Unsupported parameter type: {type(val)}")

                    const_op = ConstantOp(value=attr)
                    rewriter.insert_op(const_op, InsertPoint.before(op))
                    parameters_xdsl.append(const_op.result)

                # TODO: I will change this since it extracts wires unnecessarily
                for wire in qml_op.wires:
                    wire_attr = IntegerAttr(wire, IntegerType(64))

                    extract_op = ExtractOp(
                        operands=(qreg_val, None),
                        properties={"idx_attr": wire_attr},
                        attributes={},
                        successors=(),
                        regions=(),
                        result_types=(QubitType(),),
                    )

                    if last_custom_op is None:
                        rewriter.insert_op(extract_op, InsertPoint.before(op))

                    wires_xdsl.append(extract_op.qubit)

                wires_xdsl = wires_xdsl if last_custom_op is None else last_custom_op.out_qubits

                # At this stage, this is the only 'special' operator we handle (GlobalPhase does not have wires)
                if qml_op.name == "GlobalPhase":

                    custom_op = GlobalPhaseOp(
                        operands=(*parameters_xdsl, None, None),
                        properties={
                            "gate_name": StringAttr(qml_op.name),
                        },
                        attributes={},
                        successors=op.successors,
                        regions=(),
                        result_types=(QubitType(),),
                    )

                else:

                    custom_op = CustomOp(
                        operands=(*parameters_xdsl, *wires_xdsl, None, None),
                        properties={
                            "gate_name": StringAttr(qml_op.name),
                        },
                        attributes={},
                        successors=op.successors,
                        regions=(),
                        result_types=(QubitType(), []),
                    )

                rewriter.insert_op(custom_op, InsertPoint.before(op))

                last_custom_op = custom_op

            if op.results:
                op.results[0].replace_by(last_custom_op.results[0])

            rewriter.erase_op(op)


@dataclass(frozen=True)
class DecompositionTransformPass(passes.ModulePass):
    """A pass that applies the Transform pattern to a module."""

    name = "decomposition-transform"

    # pylint: disable=arguments-renamed,no-self-use, arguments-differ
    def apply(self, _ctx: context.MLContext, module: builtin.ModuleOp) -> None:
        pattern = DecompositionTransform(module)
        pattern_rewriter.PatternRewriteWalker(
            pattern_rewriter.GreedyRewritePatternApplier([pattern])
        ).rewrite_module(module)

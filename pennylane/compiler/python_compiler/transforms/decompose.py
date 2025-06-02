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
from typing import Iterable, Union, Dict

from xdsl import context, passes, pattern_rewriter
from xdsl.dialects import builtin, func
from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.builtin import Float64Type, FloatAttr, IntegerAttr, IntegerType
from xdsl.ir import SSAValue
from xdsl.rewriter import InsertPoint

import pennylane as qml
from pennylane.compiler.python_compiler.quantum_dialect import (
    AllocOp,
    CustomOp,
    ExtractOp,
    GlobalPhaseOp,
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


def from_qml_to_xdsl_params(op: Operator) -> list[float | int]:
    """Get the parameters from the operation."""
    return [resolve_constant_params(p) for p in op.params if p is not None]


def from_qml_to_xdsl_wires(op: Operator) -> list[int]:
    """Get the wires from the operation."""
    if not hasattr(op, "in_qubits"):
        return []
    return [resolve_constant_wire(w) for w in op.in_qubits if w is not None]


def resolve_gate(name: str) -> Operator:
    """Resolve the gate from the name."""
    try:
        return from_str_to_PL_gate[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported gate: {name}") from exc


def from_xdsl_to_qml_op(op: CustomOp):
    """Convert an xDSL CustomOp to a PennyLane operation."""
    gate_name = op.properties["gate_name"].data
    parameters = from_qml_to_xdsl_params(op)
    wires = flatten(from_qml_to_xdsl_wires(op))
    return resolve_gate(gate_name)(*parameters, wires=wires)


def from_qml_to_xdsl_param(param: Union[int, float]):
    """Convert a PennyLane parameter to an xDSL SSAValue."""
    val = param.item() if hasattr(param, "item") else param
    if isinstance(val, int):
        attr = IntegerAttr(val, IntegerType(64))
    elif isinstance(val, float):
        attr = FloatAttr(val, Float64Type())
    else:
        raise TypeError(f"Unsupported parameter type: {type(val)}")
    return attr


# pylint: disable=too-few-public-methods
class DecompositionTransform(pattern_rewriter.RewritePattern):
    """A pattern that rewrites CustomOps to their decomposition."""

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.extracted_qubits: Dict[int, SSAValue] = {}
        self.extracted_params: Dict[Union[FloatAttr, IntegerAttr], SSAValue] = {}
        self.quantum_register: Union[SSAValue, None] = None

    # pylint:disable=arguments-differ, too-many-branches
    @pattern_rewriter.op_type_rewrite_pattern
    def match_and_rewrite(self, funcOp: func.FuncOp, rewriter: pattern_rewriter.PatternRewriter):
        """Rewrite the function by replacing CustomOps with their decomposition."""

        for xdsl_op in funcOp.body.walk():

            # This assumes that the AllocOp is placed at the top of the program
            # TODO: I feel this should be moved out of the loop, but I am not sure how to do it yet.
            if isinstance(xdsl_op, AllocOp):
                self.quantum_register = xdsl_op.qreg
                continue

            # TODO: This doesn't decompose PL operators that are not CustomOp (e.g., QubitUnaryOp)
            if not isinstance(xdsl_op, CustomOp):
                continue

            if self.quantum_register is None:
                raise ValueError("Quantum register (AllocOp) not found in the function.")

            qml_op = from_xdsl_to_qml_op(xdsl_op)

            # TODO: Decide what to do with the parameters and wires of the operation.
            # It could be worth registering the parameters and wires of the operation
            # in the `self.extracted_params` and `self.extracted_qubits` dictionaries
            # at this point, or erasing them if they are not used later.

            for idx, wire in enumerate(qml_op.wires):
                self.extracted_qubits[wire] = xdsl_op.in_qubits[idx]

            if not qml_op.has_decomposition:
                continue

            qml_decomp_ops = qml_op.decomposition()
            last_custom_op = None

            for qml_decomp_op in qml_decomp_ops:

                params_xdsl: list[SSAValue] = []
                for param in qml_decomp_op.parameters:

                    xdsl_param = from_qml_to_xdsl_param(param)

                    if xdsl_param in self.extracted_params:
                        const_ssa = self.extracted_params[xdsl_param]
                    else:
                        const_op = ConstantOp(value=xdsl_param)
                        rewriter.insert_op(const_op, InsertPoint.before(xdsl_op))
                        const_ssa = const_op.result
                        self.extracted_params[xdsl_param] = const_ssa

                    params_xdsl.append(const_ssa)

                wires_xdsl: list[SSAValue] = []
                for wire in qml_decomp_op.wires:

                    if wire in self.extracted_qubits:
                        qubit_ssa = self.extracted_qubits[wire]
                    else:
                        wire_xdsl = IntegerAttr(wire, IntegerType(64))
                        extract_op = ExtractOp(qreg=self.quantum_register, idx=wire_xdsl)
                        rewriter.insert_op(extract_op, InsertPoint.before(xdsl_op))
                        qubit_ssa = extract_op.qubit
                        self.extracted_qubits[wire] = qubit_ssa

                    wires_xdsl.append(qubit_ssa)

                if last_custom_op is not None:
                    wires_xdsl = last_custom_op.out_qubits

                # At this stage, this is the only 'special' operator we handle (GlobalPhase does not have wires)
                if qml_decomp_op.name == "GlobalPhase":
                    # pylint: disable=unexpected-keyword-arg
                    custom_op = GlobalPhaseOp(params=params_xdsl)
                else:
                    custom_op = CustomOp(
                        params=params_xdsl,
                        in_qubits=wires_xdsl,
                        gate_name=qml_decomp_op.name,
                    )

                rewriter.insert_op(custom_op, InsertPoint.before(xdsl_op))
                last_custom_op = custom_op

            if xdsl_op.results:
                old_results = list(xdsl_op.results)
                new_results = list(last_custom_op.results)
                if len(old_results) != len(new_results):
                    raise NotImplementedError
                for old_res, new_res in zip(old_results, new_results, strict=True):
                    old_res.replace_by(new_res)

            rewriter.erase_op(xdsl_op)


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

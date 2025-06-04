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

import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, Type, Union

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
)
from pennylane.operation import Operator
from pennylane.transforms.decompose import _operator_decomposition_gen

# This is just a preliminary structure for mapping of PennyLane gates to xDSL operations.
from_str_to_PL_gate = {
    "RX": qml.RX,
    "RY": qml.RY,
    "RZ": qml.RZ,
    "Rot": qml.Rot,
    "CNOT": qml.CNOT,
    "Hadamard": qml.Hadamard,
    "PhaseShift": qml.PhaseShift,
    "PauliX": qml.PauliX,
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


def resolve_gate(name: str) -> Operator:
    """Resolve the gate from the name."""
    try:
        return from_str_to_PL_gate[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported gate: {name}") from exc


def qml_to_xdsl_param(param: Union[int, float]) -> Union[IntegerAttr, FloatAttr]:
    """Convert a PennyLane parameter to an xDSL SSAValue."""
    val = param.item() if hasattr(param, "item") else param
    if isinstance(val, int):
        attr = IntegerAttr(val, IntegerType(64))
    elif isinstance(val, float):
        attr = FloatAttr(val, Float64Type())
    else:
        raise TypeError(f"Unsupported parameter type: {type(val)}")
    return attr


# pylint: disable=too-many-instance-attributes
class DecompositionTransform(pattern_rewriter.RewritePattern):
    """A pattern that rewrites CustomOps to their decomposition."""

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.wire_to_ssa_qubits: Dict[int, SSAValue] = {}
        self.ssa_qubits_to_wires: Dict[SSAValue, int] = {}
        self.params_to_ssa_params: Dict[Union[FloatAttr, IntegerAttr], SSAValue] = {}
        self.quantum_register: Union[SSAValue, None] = None

        # TODO: These should be configurable (e.g., parameters of the pass)
        self.max_expansion = None
        self._current_depth = 0
        self.gate_set: set[Type[Operator] | str] = {"CNOT", "RX", "RY", "RZ"}

    def gate_set_contains(self, op: Operator) -> bool:
        """Check if the operator is in the gate set."""
        return op.name in self.gate_set

    def stopping_condition(self, op: Operator) -> bool:
        """Function to determine whether an operator needs to be decomposed or not."""

        if not op.has_decomposition:
            if not self.gate_set_contains(op):
                warnings.warn(
                    f"Operator {op.name} does not define a decomposition and was not "
                    f"found in the target gate set. To remove this warning, add the operator "
                    f"name ({op.name}) or type ({type(op)}) to the gate set.",
                    UserWarning,
                )
            return True

        return self.gate_set_contains(op)

    def decompose_operation(self, op: Operator):
        """Decompose the operation if it is not in the gate set."""

        if self.gate_set_contains(op):
            return [op]

        max_expansion = (
            self.max_expansion - self._current_depth if self.max_expansion is not None else None
        )

        decomposition = list(
            _operator_decomposition_gen(
                op,
                self.stopping_condition,
                max_expansion=max_expansion,
                current_depth=self._current_depth,
                decomp_graph=None,
            )
        )

        return decomposition

    def resolve_constant_params(self, param: SSAValue) -> Union[float, int]:
        """Resolve the constant parameter from the SSA value."""
        op = param.owner
        if isinstance(op, ConstantOp):
            val = op.value
            if isinstance(val, (FloatAttr, IntegerAttr)):
                if val not in self.params_to_ssa_params:
                    self.params_to_ssa_params[val] = op.result
                return val.value.data
        raise NotImplementedError(f"Cannot resolve parameters for operation {op}")

    def ssa_to_qml_params(self, op: CustomOp) -> list[float | int]:
        """Get the parameters from the operation."""
        return [self.resolve_constant_params(p) for p in op.params if p is not None]

    def ssa_to_qml_wires(self, op: CustomOp) -> list[int]:
        """Get the wires from the operation."""
        if not hasattr(op, "in_qubits"):
            return []
        return [self.resolve_constant_wire(q) for q in op.in_qubits if q is not None]

    def xdsl_to_qml_op(self, op: CustomOp) -> Operator:
        """Convert an xDSL CustomOp to a PennyLane operation."""
        gate_name = op.properties["gate_name"].data
        parameters = self.ssa_to_qml_params(op)
        wires = flatten(self.ssa_to_qml_wires(op))
        return resolve_gate(gate_name)(*parameters, wires=wires)

    def resolve_constant_wire(self, in_qubit: SSAValue) -> int:
        """Resolve the wire for the given SSA qubit."""
        op = in_qubit.owner
        if isinstance(op, ExtractOp):
            return op.idx_attr.parameters[0].data
        if isinstance(op, ConstantOp):
            val = op.value
            return val.value.data
        if isinstance(op, CustomOp):
            # The CustomOp should be the last one that updated
            # the mapping for this SSA qubit.
            wire_position = in_qubit.index
            ssa_qubit = op.out_qubits[wire_position]
            return self.ssa_qubits_to_wires[ssa_qubit]

        # TODO: Add support for other operations (e.g., QubitUnaryOp, GlobalPhaseOp, etc.)
        raise NotImplementedError(f"Cannot resolve wire for operation {op}")

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

            # Pre-populate the extracted qubits
            # TODO: I feel this should be moved out of the loop, but I am not sure how to do it yet.
            if isinstance(xdsl_op, ExtractOp):
                wire = xdsl_op.idx_attr.parameters[0].data
                if wire in self.wire_to_ssa_qubits:
                    continue
                ssa_qubit = xdsl_op.qubit
                self.wire_to_ssa_qubits[wire] = ssa_qubit
                continue

            # TODO: Add decomposition for other operations (e.g., QubitUnaryOp, GlobalPhaseOp, etc.)
            if not isinstance(xdsl_op, CustomOp):
                continue

            if self.quantum_register is None:
                raise ValueError("Quantum register (AllocOp) not found in the function.")

            qml_op = self.xdsl_to_qml_op(xdsl_op)
            qml_decomp_ops = self.decompose_operation(qml_op)

            for qml_decomp_op in qml_decomp_ops:

                params_xdsl: list[SSAValue] = []
                for param in qml_decomp_op.parameters:
                    xdsl_param = qml_to_xdsl_param(param)
                    if xdsl_param in self.params_to_ssa_params:
                        const_ssa = self.params_to_ssa_params[xdsl_param]
                    else:
                        const_op = ConstantOp(value=xdsl_param)
                        rewriter.insert_op(const_op, InsertPoint.before(xdsl_op))
                        const_ssa = const_op.result
                        self.params_to_ssa_params[xdsl_param] = const_ssa

                    params_xdsl.append(const_ssa)

                wires_xdsl: list[SSAValue] = []
                for wire in qml_decomp_op.wires:
                    # Each decomposed operator should act on a subset of wires
                    # of the original operator, and these wires should have been registered before.
                    qubit_ssa = self.wire_to_ssa_qubits[wire]
                    wires_xdsl.append(qubit_ssa)

                custom_op = CustomOp(
                    params=params_xdsl,
                    in_qubits=wires_xdsl,
                    gate_name=qml_decomp_op.name,
                )

                for idx, wire in enumerate(qml_decomp_op.wires):
                    self.wire_to_ssa_qubits[wire] = custom_op.out_qubits[idx]
                    self.ssa_qubits_to_wires[custom_op.out_qubits[idx]] = wire

                rewriter.insert_op(custom_op, InsertPoint.before(xdsl_op))

            # After all decompositions, we replace the results of the original CustomOp
            # with the last SSA values of the decomposed CustomOps for each wire.
            for idx, wire in enumerate(qml_op.wires):
                xdsl_op.results[idx].replace_by(self.wire_to_ssa_qubits[wire])

            rewriter.erase_op(xdsl_op)


@dataclass(frozen=True)
class DecompositionTransformPass(passes.ModulePass):
    """A pass that applies the Transform pattern to a module."""

    name = "decomposition-transform"

    # pylint: disable=arguments-renamed,no-self-use, arguments-differ
    def apply(self, _ctx: context.MLContext, module: builtin.ModuleOp) -> None:
        pattern = DecompositionTransform(module)
        pattern_rewriter.PatternRewriteWalker(
            pattern_rewriter.GreedyRewritePatternApplier([pattern]), apply_recursively=False
        ).rewrite_module(module)

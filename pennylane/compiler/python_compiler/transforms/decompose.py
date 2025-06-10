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

import inspect
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Union

from xdsl import context, passes, pattern_rewriter
from xdsl.dialects import builtin, func
from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.builtin import Float64Type, FloatAttr, IntegerAttr, IntegerType, BoolAttr
from xdsl.ir import SSAValue
from xdsl.rewriter import InsertPoint

import pennylane as qml
from pennylane import ops
from pennylane.compiler.python_compiler.quantum_dialect import AllocOp, CustomOp, ExtractOp
from pennylane.operation import Operator
from pennylane.ops import __all__ as ops_all
from pennylane.transforms.decompose import _operator_decomposition_gen

# pylint: disable=missing-function-docstring


# This is just a preliminary structure for mapping of PennyLane gates to xDSL operations.
# Support for all PennyLane gates is not implemented yet.
from_str_to_PL_gate = {
    name: getattr(ops, name)
    for name in ops_all
    if inspect.isclass(getattr(ops, name, None))
    and issubclass(getattr(ops, name), qml.operation.Operator)
}


def resolve_gate(name: str) -> Operator:
    """Resolve the gate from the name."""
    try:
        return from_str_to_PL_gate[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported gate: {name}") from exc


def qml_to_xdsl_param(param: Union[int, float, bool]) -> Union[IntegerAttr, FloatAttr, BoolAttr]:
    """Convert a PennyLane parameter (int or float) into an xDSL constant-attr."""
    val = param.item() if hasattr(param, "item") else param
    if isinstance(val, int):
        return IntegerAttr(val, IntegerType(64))
    if isinstance(val, float):
        return FloatAttr(val, Float64Type())
    if isinstance(val, bool):
        return BoolAttr(val, IntegerType(1))
    raise TypeError(f"Unsupported parameter type: {type(val)}")


# pylint: disable=too-many-instance-attributes
class DecompositionTransform(pattern_rewriter.RewritePattern):
    """A rewrite pattern that replaces every ``quantum.custom``
    operation (i.e. CustomOp) with its PennyLane decomposition."""

    # Several TODO for this pass/transform:
    #
    # - Add support for qml.ctrl (WIP)
    # - Add support for qml.adjoint (WIP)
    # - Add support for other operations (e.g., QubitUnaryOp, GlobalPhaseOp, etc.),
    #   both in the decomposition and in the gate set.
    # - Add support for the decomp_graph
    # - Move the logic for qubit mapping and operator conversion to a separate class
    #   if this turns out to be useful in other transforms.
    # - Add support for complex parameters (e.g., complex numbers, arrays, etc.)
    # - Add support for dynamic wires and parameters

    def __init__(
        self,
        module: builtin.ModuleOp,
        gate_set=None,
        max_expansion=None,
    ):
        super().__init__()
        self.module = module
        self.gate_set = gate_set if gate_set is not None else set(ops_all)
        self.max_expansion = max_expansion

        self.wire_to_ssa_qubits: Dict[int, SSAValue] = {}
        self.ssa_qubits_to_wires: Dict[SSAValue, int] = {}
        self.params_to_ssa_params: Dict[Union[FloatAttr, IntegerAttr], SSAValue] = {}

        self.quantum_register: Union[SSAValue, None] = None

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
        decomposition = list(
            _operator_decomposition_gen(
                op,
                self.stopping_condition,
                max_expansion=self.max_expansion,
                current_depth=0,
                decomp_graph=None,
            )
        )
        return decomposition

    def resolve_constant_params(self, param: SSAValue) -> Union[float, int]:
        """Resolve the constant parameter from the SSA value."""
        op = param.owner
        if not isinstance(op, ConstantOp):
            raise NotImplementedError(f"Expected ConstantOp but got {type(op)}")
        val = op.value
        if not isinstance(val, (FloatAttr, IntegerAttr)):
            raise NotImplementedError(f"Constant has unexpected attr type: {type(val)}")
        if val not in self.params_to_ssa_params:
            self.params_to_ssa_params[val] = param
        return val.value.data

    def ssa_to_qml_params(self, op: CustomOp, control: bool = False) -> list[float | int]:
        """Get the parameters from the operation."""
        attr = "in_ctrl_values" if control else "params"
        values = getattr(op, attr, None)
        if not values:
            return []
        return [self.resolve_constant_params(p) for p in values if p is not None]

    def ssa_to_qml_wires(self, op: CustomOp, control: bool = False) -> list[int]:
        """Get the wires from the operation."""
        attr = "in_ctrl_qubits" if control else "in_qubits"
        qubits = getattr(op, attr, None)
        if not qubits:
            return []
        return [self.resolve_constant_wire(q) for q in qubits if q is not None]

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
        raise NotImplementedError(f"Cannot resolve wire for operation {op}")

    def xdsl_to_qml_op(self, op: CustomOp) -> Operator:
        """Given a ``quantum.custom`` xDSL op, convert it to a PennyLane operator."""
        gate_name = op.properties["gate_name"].data
        parameters = self.ssa_to_qml_params(op)
        wires = self.ssa_to_qml_wires(op)
        gate = resolve_gate(gate_name)(*parameters, wires=wires)
        if op.properties.get("adjoint") is not None:
            gate = qml.adjoint(gate)
        if len(op.in_ctrl_qubits) != 0:
            in_ctrl_qubits = self.ssa_to_qml_wires(op, control=True)
            in_ctrl_values = self.ssa_to_qml_params(op, control=True)
            gate = qml.ctrl(gate, control=in_ctrl_qubits, control_values=in_ctrl_values)
        return gate

    def initialize_qubit_mapping(self, funcOp: func.FuncOp):
        """Scan the function to populate the quantum register and wire mappings."""
        saw_alloc = False
        for op in funcOp.body.walk():
            if isinstance(op, AllocOp):
                if saw_alloc:
                    raise ValueError("Found more than one AllocOp for this FuncOp.")
                saw_alloc = True
                self.quantum_register = op.qreg
            elif isinstance(op, ExtractOp):
                wire = op.idx_attr.parameters[0].data
                if wire not in self.wire_to_ssa_qubits:
                    self.wire_to_ssa_qubits[wire] = op.qubit
                    # We update the reverse mapping as well for completeness,
                    # but this should not be used in practice for decompositions.
                    self.ssa_qubits_to_wires[op.qubit] = wire

    # pylint:disable=arguments-differ, too-many-branches
    @pattern_rewriter.op_type_rewrite_pattern
    def match_and_rewrite(self, funcOp: func.FuncOp, rewriter: pattern_rewriter.PatternRewriter):
        """Rewrite the function by replacing CustomOps with their decomposition."""

        self.initialize_qubit_mapping(funcOp)

        for xdsl_op in funcOp.body.walk():

            if not isinstance(xdsl_op, CustomOp):
                continue
            if self.quantum_register is None:
                raise ValueError("Quantum register (AllocOp) not found in the function.")
            if len(self.wire_to_ssa_qubits) == 0:
                raise NotImplementedError("No wires extracted from the register have been found. ")

            qml_op = self.xdsl_to_qml_op(xdsl_op)
            qml_decomp_ops = self.decompose_operation(qml_op)

            print(f"xdsl_op: {xdsl_op}")
            print(f"qml_op: {qml_op}")
            print(f"qml_decomp_ops: {qml_decomp_ops}")

            for qml_decomp_op in qml_decomp_ops:
                if is_adjoint := isinstance(qml_decomp_op, qml.ops.op_math.Adjoint):
                    qml_decomp_op = qml_decomp_op.base

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

                control_wires_xdsl: list[SSAValue] = []
                control_values_xdsl: list[SSAValue] = []
                # TODO: The IR does not seem to be consistent with the
                # representation of controlled gates.
                if isinstance(qml_decomp_op, qml.ops.op_math.Controlled) and not isinstance(
                    qml_decomp_op, qml.CNOT
                ):
                    for wire in qml_decomp_op.control_wires:
                        qubit_ssa = self.wire_to_ssa_qubits[wire]
                        control_wires_xdsl.append(qubit_ssa)

                    # TODO: this repetition could be avoided
                    for value in qml_decomp_op.control_values:
                        xdsl_param = qml_to_xdsl_param(value)
                        if xdsl_param in self.params_to_ssa_params:
                            const_ssa = self.params_to_ssa_params[xdsl_param]
                        else:
                            const_op = ConstantOp(value=xdsl_param)
                            rewriter.insert_op(const_op, InsertPoint.before(xdsl_op))
                            const_ssa = const_op.result
                            self.params_to_ssa_params[xdsl_param] = const_ssa
                        control_values_xdsl.append(const_ssa)

                custom_op = CustomOp(
                    params=params_xdsl,
                    in_qubits=wires_xdsl,
                    gate_name=qml_decomp_op.name,
                    adjoint=is_adjoint,
                    in_ctrl_qubits=control_wires_xdsl,
                    in_ctrl_values=control_values_xdsl,
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

    # We use type annotations since these are (configurable) dataclass fields
    gate_set: Optional[
        Union[Iterable[Union[str, type]], Dict[Union[str, type], float], Callable]
    ] = None
    max_expansion: Optional[int] = None

    # pylint: disable=arguments-renamed,no-self-use, arguments-differ
    def apply(self, _ctx: context.MLContext, module: builtin.ModuleOp) -> None:
        pattern = DecompositionTransform(
            module,
            gate_set=self.gate_set,
            max_expansion=self.max_expansion,
        )
        pattern_rewriter.PatternRewriteWalker(
            pattern_rewriter.GreedyRewritePatternApplier([pattern]), apply_recursively=False
        ).rewrite_module(module)

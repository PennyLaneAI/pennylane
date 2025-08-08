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
"""This file contains the implementation of the visualize transform,
written using xDSL."""

import inspect
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Union

from xdsl import context, passes, pattern_rewriter
from xdsl.dialects import builtin, func
from xdsl.dialects.builtin import (
    DenseIntOrFPElementsAttr,
    Float64Type,
    FloatAttr,
    IntegerAttr,
    IntegerType,
)
from xdsl.dialects.stablehlo import ConstantOp as StableHLOConstantOp
from xdsl.dialects.tensor import ExtractOp as TensorExtractOp
from xdsl.ir import SSAValue

import pennylane as qml
from pennylane import ops
from pennylane.operation import Operator
from pennylane.ops import __all__ as ops_all
from pennylane import measurements
from pennylane.measurements import MeasurementProcess

from ..dialects.quantum import AllocOp as AllocOpPL
from ..dialects.quantum import CustomOp
from ..dialects.quantum import ExtractOp as ExtractOpPL
from ..dialects.quantum import StateOp
from .api import compiler_transform

# This is just a preliminary structure for mapping of PennyLane gates to xDSL operations.
# Support for all PennyLane gates is not implemented yet.
from_str_to_PL_gate = {
    name: getattr(ops, name)
    for name in ops_all
    if inspect.isclass(getattr(ops, name, None)) and issubclass(getattr(ops, name), Operator)
}

# TODO: only support state measurements for now
from_str_to_PL_measurement = {
    "quantum.state": qml.state,
}


def resolve_gate(name: str) -> Operator:
    """Resolve the gate from the name."""
    try:
        return from_str_to_PL_gate[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported gate: {name}") from exc


def resolve_measurement(name: str) -> MeasurementProcess:
    """Resolve the measurement from the name."""
    try:
        return from_str_to_PL_measurement[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported measurement: {name}") from exc


def qml_to_xdsl_param(param: Union[int, float]) -> Union[IntegerAttr, FloatAttr]:
    """Convert a PennyLane parameter (int or float) into an xDSL constant-attr."""
    val = param.item() if hasattr(param, "item") else param
    if isinstance(val, int):
        return IntegerAttr(val, IntegerType(64))
    if isinstance(val, float):
        return FloatAttr(val, Float64Type())
    raise TypeError(f"Unsupported parameter type: {type(val)}")


def _extract_dense_constant_value(op) -> float | int:
    """Extract the first value from a stablehlo.constant op."""
    attr = op.properties.get("value")
    if isinstance(attr, DenseIntOrFPElementsAttr):
        # TODO: handle multi-value cases if needed
        return attr.get_values()[0]
    raise NotImplementedError(f"Unexpected attr type in constant: {type(attr)}")


# pylint: disable=too-many-instance-attributes
class VisualizationTransform(pattern_rewriter.RewritePattern):
    """A pattern that matches a function and rewrites it to collect
    the quantum operations into a list of PennyLane operators."""

    # Several TODO for this pass/transform:
    #
    # - Add support for qml.ctrl (WIP)
    # - Add support for other operations (e.g., QubitUnaryOp, GlobalPhaseOp, etc.),
    #   both in the visualization and in the gate set.
    # - Add support for the decomp_graph
    # - Move the logic for qubit mapping and operator conversion to a separate class
    #   if this turns out to be useful in other transforms.
    # - Add support for complex parameters (e.g., complex numbers, arrays, etc.)
    # - Add support for dynamic wires and parameters

    def __init__(
        self,
        module: builtin.ModuleOp,
    ):
        super().__init__()
        self.module = module

        self.wire_to_ssa_qubits: Dict[int, SSAValue] = {}
        self.ssa_qubits_to_wires: Dict[SSAValue, int] = {}
        self.params_to_ssa_params: Dict[Union[FloatAttr, IntegerAttr], SSAValue] = {}

        self.quantum_register: Union[SSAValue, None] = None

    def resolve_constant_params(self, ssa: SSAValue) -> float | int:
        """Resolve a constant parameter SSA value to a Python float or int."""
        op = ssa.owner
        if isinstance(op, TensorExtractOp):
            return self.resolve_constant_params(op.tensor)
        if op.name == "builtin.unregistered":
            if hasattr(op, "attributes"):
                if op.attributes["op_name__"].data == "stablehlo.convert":
                    return self.resolve_constant_params(op.operands[0])
        if op.name == "stablehlo.constant":
            return _extract_dense_constant_value(op)
        raise NotImplementedError(f"Cannot resolve parameters for op: {op}")

    def resolve_constant_wire(self, ssa: SSAValue) -> int:
        """Resolve the wire for the given SSA qubit."""
        op = ssa.owner
        if isinstance(op, TensorExtractOp):
            return self.resolve_constant_wire(op.tensor)
        if op.name == "builtin.unregistered":
            if hasattr(op, "attributes"):
                if op.attributes["op_name__"].data == "stablehlo.convert":
                    return self.resolve_constant_wire(op.operands[0])
        if op.name == "stablehlo.constant":
            return _extract_dense_constant_value(op)
        if isinstance(op, CustomOp):
            return self.resolve_constant_wire(op.in_qubits[ssa.index])
        if isinstance(op, ExtractOpPL):
            return self.resolve_constant_wire(op.idx)
        raise NotImplementedError(f"Cannot resolve wire for op: {op}")

    def ssa_to_qml_params(self, op: CustomOp) -> list[float | int]:
        """Get the parameters from the operation."""
        if not hasattr(op, "params"):
            return []
        return [self.resolve_constant_params(p) for p in op.params if p is not None]

    def ssa_to_qml_wires(self, op: CustomOp) -> list[int]:
        """Get the wires from the operation."""
        if not hasattr(op, "in_qubits"):
            return []
        return [self.resolve_constant_wire(q) for q in op.in_qubits if q is not None]

    def xdsl_to_qml_op(self, op: CustomOp) -> Operator:
        """Given a ``quantum.custom`` xDSL op, convert it to a PennyLane operator."""
        gate_name = op.properties["gate_name"].data
        parameters = self.ssa_to_qml_params(op)
        wires = self.ssa_to_qml_wires(op)
        gate = resolve_gate(gate_name)(*parameters, wires=wires)
        if op.properties.get("adjoint") is not None:
            gate = qml.adjoint(gate)
        return gate

    def xdsl_to_qml_meas(self, meas) -> Operator:
        """Given a measurement in xDSL, convert it to a PennyLane measurement."""
        meas_name = meas.name
        # So far, only `qml.state` is supported
        measurement = resolve_measurement(meas_name)
        return measurement()

    # TODO: this will probably no longer be needed once PR #7937 is merged
    def initialize_qubit_mapping(self, funcOp: func.FuncOp):
        """Scan the function to populate the quantum register and wire mappings."""
        saw_alloc = False
        for op in funcOp.body.walk():
            if isinstance(op, AllocOpPL):
                if saw_alloc:
                    raise ValueError("Found more than one AllocOp for this FuncOp.")
                saw_alloc = True
                self.quantum_register = op.qreg
            elif isinstance(op, ExtractOpPL):
                wire = self.resolve_constant_wire(op.idx)
                if wire not in self.wire_to_ssa_qubits:
                    self.wire_to_ssa_qubits[wire] = op.qubit
                    # We update the reverse mapping as well for completeness,
                    # but this should not be used in practice for visualization.
                    self.ssa_qubits_to_wires[op.qubit] = wire

    # pylint:disable=arguments-differ, too-many-branches
    @pattern_rewriter.op_type_rewrite_pattern
    def match_and_rewrite(self, funcOp: func.FuncOp, rewriter: pattern_rewriter.PatternRewriter):

        self.initialize_qubit_mapping(funcOp)
        collected_ops = []
        collected_meas = []

        for xdsl_op in funcOp.body.walk():

            if isinstance(xdsl_op, StateOp):
                qml_meas = self.xdsl_to_qml_meas(xdsl_op)
                collected_meas.append(qml_meas)
                print(f"collected meas: {collected_meas}")
            if not isinstance(xdsl_op, CustomOp):
                continue
            if self.quantum_register is None:
                raise ValueError("Quantum register (AllocOp) not found in the function.")
            if len(self.wire_to_ssa_qubits) == 0:
                raise NotImplementedError("No wires extracted from the register have been found.")

            qml_op = self.xdsl_to_qml_op(xdsl_op)
            collected_ops.append(qml_op)
            print(f"collected ops: {collected_ops}")


@dataclass(frozen=True)
class VisualizationTransformPass(passes.ModulePass):
    """A pass that applies the Transform pattern to a module."""

    name = "visualization-transform"

    # We use type annotations since these are (configurable) dataclass fields
    gate_set: Optional[
        Union[Iterable[Union[str, type]], Dict[Union[str, type], float], Callable]
    ] = None
    max_expansion: Optional[int] = None

    # pylint: disable=arguments-renamed,no-self-use, arguments-differ
    def apply(self, _ctx: context.MLContext, module: builtin.ModuleOp) -> None:
        pattern = VisualizationTransform(module)
        pattern_rewriter.PatternRewriteWalker(
            pattern_rewriter.GreedyRewritePatternApplier([pattern]), apply_recursively=False
        ).rewrite_module(module)


visualization_pass = compiler_transform(VisualizationTransformPass)

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
"""This file contains the implementation of the QMLCollector class,
which collects and maps PennyLane operations and measurements from xDSL."""

import inspect

from xdsl.dialects.builtin import (
    DenseIntOrFPElementsAttr,
    IntegerAttr,
)
from xdsl.dialects.tensor import ExtractOp as TensorExtractOp
from xdsl.ir import SSAValue

import pennylane as qml
from pennylane import ops
from pennylane.compiler.python_compiler.dialects.quantum import CustomOp
from pennylane.compiler.python_compiler.dialects.quantum import ExtractOp as ExtractOpPL
from pennylane.measurements import MeasurementProcess
from pennylane.operation import Operator
from pennylane.ops import __all__ as ops_all

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

######################################################
### Gate/Measurement resolution
######################################################


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


######################################################
### Parameters/Wires traceback
######################################################


def _extract_dense_constant_value(op) -> float | int:
    """Extract the first value from a stablehlo.constant op."""
    attr = op.properties.get("value")
    if isinstance(attr, DenseIntOrFPElementsAttr):
        # TODO: handle multi-value cases if needed
        return attr.get_values()[0]
    raise NotImplementedError(f"Unexpected attr type in constant: {type(attr)}")


def resolve_constant_params(ssa: SSAValue) -> float | int:
    """Resolve a constant parameter SSA value to a Python float or int."""
    op = ssa.owner
    if isinstance(op, TensorExtractOp):
        return resolve_constant_params(op.tensor)
    if op.name == "stablehlo.constant":
        return _extract_dense_constant_value(op)
    if op.name == "arith.addf":
        return resolve_constant_params(op.operands[0]) + resolve_constant_params(op.operands[1])
    if op.name == "arith.constant":
        return op.value.value.data  # used by Catalyst
    if op.name == "stablehlo.convert":
        return resolve_constant_params(op.operands[0])
    raise NotImplementedError(f"Cannot resolve parameters for op: {op}")


def dispatch_wires_extract(op: ExtractOpPL):
    """Dispatch the wire resolution for the given extract operation."""
    if op.idx_attr is not None:  # used by Catalyst
        return resolve_constant_wire(op.idx_attr)
    return resolve_constant_wire(op.idx)  # used by xDSL


def resolve_constant_wire(ssa: SSAValue) -> int:
    """Resolve the wire for the given SSA qubit."""
    if isinstance(ssa, IntegerAttr):  # used by Catalyst
        return ssa.value.data
    op = ssa.owner
    if isinstance(op, TensorExtractOp):
        return resolve_constant_wire(op.tensor)
    if op.name == "builtin.unregistered":
        if hasattr(op, "attributes"):
            if op.attributes["op_name__"].data == "stablehlo.convert":
                return resolve_constant_wire(op.operands[0])
    if op.name == "stablehlo.constant":
        return _extract_dense_constant_value(op)
    if isinstance(op, CustomOp):
        return resolve_constant_wire(op.in_qubits[ssa.index])
    if isinstance(op, ExtractOpPL):
        return dispatch_wires_extract(op)
    raise NotImplementedError(f"Cannot resolve wire for op: {op}")


######################################################
### Parameters/Wires Conversion
######################################################


def ssa_to_qml_params(op: CustomOp, control: bool = False) -> list[float | int]:
    """Get the parameters from the operation."""
    attr = "in_ctrl_values" if control else "params"
    values = getattr(op, attr, None)
    if not values:
        return []
    return [resolve_constant_params(p) for p in values if p is not None]


def ssa_to_qml_wires(op: CustomOp, control: bool = False) -> list[int]:
    """Get the wires from the operation."""
    attr = "in_ctrl_qubits" if control else "in_qubits"
    qubits = getattr(op, attr, None)
    if not qubits:
        return []
    return [resolve_constant_wire(q) for q in qubits if q is not None]


############################################################
### xDSL ---> PennyLane Operators/Measurements conversion
############################################################


def xdsl_to_qml_op(op: CustomOp) -> Operator:
    """Convert a ``quantum.custom`` xDSL op to a PennyLane operator."""
    gate_name = op.properties.get("gate_name").data
    parameters = ssa_to_qml_params(op)
    wires = ssa_to_qml_wires(op)
    gate_cls = resolve_gate(gate_name)
    gate = gate_cls(*parameters, wires=wires)

    if op.properties.get("adjoint"):
        gate = qml.adjoint(gate)

    ctrls = ssa_to_qml_wires(op, control=True)
    if ctrls:
        cvals = ssa_to_qml_params(op, control=True)
        gate = qml.ctrl(gate, control=ctrls, control_values=cvals)

    return gate


def xdsl_to_qml_meas(meas) -> Operator:
    """Given a measurement in xDSL, convert it to a PennyLane measurement."""
    meas_name = meas.name
    measurement = resolve_measurement(meas_name)
    return measurement()

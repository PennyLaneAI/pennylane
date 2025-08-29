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
from pennylane.compiler.python_compiler.dialects.quantum import (
    ComputationalBasisOp,
    CustomOp,
)
from pennylane.compiler.python_compiler.dialects.quantum import ExtractOp as ExtractOpPL
from pennylane.compiler.python_compiler.dialects.quantum import (
    MeasureOp,
    NamedObsOp,
    TensorOp,
)
from pennylane.measurements import MeasurementProcess
from pennylane.operation import Operator
from pennylane.ops import __all__ as ops_all
from pennylane.typing import Callable

# This is just a preliminary structure for mapping of PennyLane gates to xDSL operations.
# Support for all PennyLane gates is not implemented yet.
from_str_to_PL_gate = {
    name: getattr(ops, name)
    for name in ops_all
    if inspect.isclass(getattr(ops, name, None)) and issubclass(getattr(ops, name), Operator)
}

from_str_to_PL_measurement = {
    "quantum.state": qml.state,
    "quantum.probs": qml.probs,
    "quantum.sample": qml.sample,
    "quantum.expval": qml.expval,
    "quantum.var": qml.var,
    "quantum.measure": qml.measure,
}

######################################################
### Gate/Measurement resolution
######################################################


def _resolve(name: str, mapping: dict, kind: str):
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported {kind}: {name}") from exc


def resolve_gate(name: str) -> Operator:
    """Resolve the gate from the name."""
    return _resolve(name, from_str_to_PL_gate, "gate")


def resolve_measurement(name: str) -> MeasurementProcess:
    """Resolve the measurement from the name."""
    return _resolve(name, from_str_to_PL_measurement, "measurement")


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
    match op.name:
        case "stablehlo.constant":
            return _extract_dense_constant_value(op)
        case "arith.addf":
            return resolve_constant_params(op.operands[0]) + resolve_constant_params(op.operands[1])
        case "arith.constant":
            return op.value.value.data  # Catalyst
        case "stablehlo.convert":
            return resolve_constant_params(op.operands[0])
        case _:
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
    if op.name == "stablehlo.convert":
        return resolve_constant_wire(op.operands[0])
    if op.name == "stablehlo.constant":
        return _extract_dense_constant_value(op)
    if isinstance(op, CustomOp):
        return resolve_constant_wire(op.in_qubits[ssa.index])
    if isinstance(op, ExtractOpPL):
        return dispatch_wires_extract(op)
    if isinstance(op, MeasureOp):
        return resolve_constant_wire(op.in_qubit)
    raise NotImplementedError(f"Cannot resolve wire for op: {op}")


######################################################
### Parameters/Wires Conversion
######################################################


def _extract(op, attr: str, resolver: Callable, single: bool = False):
    """Helper to extract and resolve attributes."""
    values = getattr(op, attr, None)
    if not values:
        return [] if not single else None
    if single:
        return resolver(values)
    return [resolver(v) for v in values if v is not None]


def ssa_to_qml_params(op: CustomOp, control: bool = False) -> list[float | int]:
    """Get the parameters from the operation."""
    return _extract(op, "in_ctrl_values" if control else "params", resolve_constant_params)


def ssa_to_qml_wires(op: CustomOp, control: bool = False) -> list[int]:
    """Get the wires from the operation."""
    return _extract(op, "in_ctrl_qubits" if control else "in_qubits", resolve_constant_wire)


def ssa_to_qml_wires_named(op: NamedObsOp) -> int:
    """Get the wire from the named observable operation."""
    if not op.qubit:
        raise ValueError("No qubit found for named observable operation.")
    return resolve_constant_wire(op.qubit)


############################################################
### xDSL ---> PennyLane Operators/Measurements conversion
############################################################


def xdsl_to_qml_custom_op(op: CustomOp) -> Operator:
    """Convert a ``quantum.custom`` xDSL op to a PennyLane operator."""
    gate_cls = resolve_gate(op.properties.get("gate_name").data)
    gate = gate_cls(*ssa_to_qml_params(op), wires=ssa_to_qml_wires(op))

    if op.properties.get("adjoint"):
        gate = qml.adjoint(gate)

    ctrls = ssa_to_qml_wires(op, control=True)
    if ctrls:
        cvals = ssa_to_qml_params(op, control=True)
        gate = qml.ctrl(gate, control=ctrls, control_values=cvals)

    return gate


def xdsl_to_qml_measure_op(op: MeasureOp) -> MeasurementProcess:
    """Convert a ``quantum.measure`` xDSL op to a PennyLane measurement."""
    wire = _extract(op, "in_qubit", resolve_constant_wire, single=True)
    return resolve_measurement(op.name)(wires=wire)


def xdsl_to_qml_named_op(op: NamedObsOp | TensorOp) -> Operator:
    """Convert a ``quantum.namedobs`` xDSL op to a PennyLane operator."""

    if op.name == "quantum.tensor":
        ops_list = [xdsl_to_qml_named_op(operand.owner) for operand in op.operands]
        return qml.prod(*ops_list)

    if op.name == "quantum.namedobs":
        return resolve_gate(op.type.data.value)(wires=ssa_to_qml_wires_named(op))


def xdsl_to_qml_compbasis_op(op: ComputationalBasisOp) -> list[int] | None:
    """Convert a ``quantum.compbasis`` xDSL op to a PennyLane operator."""
    return _extract(op, "qubits", resolve_constant_wire)


def xdsl_to_qml_meas(meas, *args, **kwargs) -> MeasurementProcess:
    """Given a measurement in xDSL, convert it to a PennyLane measurement."""
    return resolve_measurement(meas.name)(*args, **kwargs)

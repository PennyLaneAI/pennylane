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

from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, IntegerAttr
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
    GlobalPhaseOp,
    HamiltonianOp,
    MeasureOp,
    NamedObsOp,
    TensorOp,
)
from pennylane.measurements import MeasurementProcess
from pennylane.operation import Operator
from pennylane.ops import __all__ as ops_all
from pennylane.typing import Callable

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

# pylint: disable=too-many-return-statements

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
### Helpers
######################################################


def _extract(op, attr: str, resolver: Callable, single: bool = False):
    """Helper to extract and resolve attributes."""
    values = getattr(op, attr, None)
    if not values:
        return [] if not single else None
    return resolver(values) if single else [resolver(v) for v in values if v is not None]


def _extract_dense_constant_value(op) -> float | int:
    """Extract the first value from a stablehlo.constant op."""
    attr = op.properties.get("value")
    if isinstance(attr, DenseIntOrFPElementsAttr):
        # TODO: handle multi-value cases if needed
        return attr.get_values()[0]
    raise NotImplementedError(f"Unexpected attr type in constant: {type(attr)}")


def _apply_adjoint_and_ctrls(qml_op: Operator, xdsl_op) -> Operator:
    """Apply adjoint and control modifiers to a gate if needed."""
    if xdsl_op.properties.get("adjoint"):
        qml_op = qml.adjoint(qml_op)
    ctrls = ssa_to_qml_wires(xdsl_op, control=True)
    if ctrls:
        cvals = ssa_to_qml_params(xdsl_op, control=True)
        qml_op = qml.ctrl(qml_op, control=ctrls, control_values=cvals)
    return qml_op


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
        case "builtin.unregistered":
            if hasattr(op, "attributes"):
                match op.attributes["op_name__"].data:
                    case "stablehlo.concatenate":
                        return [resolve_constant_params(operand) for operand in op.operands]
                    case "stablehlo.broadcast_in_dim":
                        return resolve_constant_params(op.operands[0])
        case _:
            raise NotImplementedError(f"Cannot resolve parameters for op: {op}")


def dispatch_wires_extract(op: ExtractOpPL):
    """Dispatch the wire resolution for the given extract operation."""
    if op.idx_attr is not None:  # used by Catalyst
        return resolve_constant_wire(op.idx_attr)
    return resolve_constant_wire(op.idx)  # used by xDSL


def resolve_constant_wire(ssa: SSAValue) -> float | int:
    """Resolve the wire for the given SSA qubit."""
    if isinstance(ssa, IntegerAttr):  # Catalyst
        return ssa.value.data

    op = ssa.owner

    match op:

        case TensorExtractOp(tensor=tensor):
            return resolve_constant_wire(tensor)
        case _ if op.name == "stablehlo.convert":
            return resolve_constant_wire(op.operands[0])
        case _ if op.name == "stablehlo.constant":
            return _extract_dense_constant_value(op)
        case CustomOp() | GlobalPhaseOp():
            all_qubits = list(getattr(op, "in_qubits", [])) + list(
                getattr(op, "in_ctrl_qubits", [])
            )
            return resolve_constant_wire(all_qubits[ssa.index])
        case ExtractOpPL():
            return dispatch_wires_extract(op)
        case MeasureOp(in_qubit=in_qubit):
            return resolve_constant_wire(in_qubit)
        case _:
            raise NotImplementedError(f"Cannot resolve wire for op: {op}")


######################################################
### Parameters/Wires Conversion
######################################################


def ssa_to_qml_params(
    op: CustomOp, control: bool = False, single: bool = False
) -> list[float | int] | float | int | None:
    """Get the parameters from the operation."""
    return _extract(op, "in_ctrl_values" if control else "params", resolve_constant_params, single)


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


def xdsl_to_qml_op(op: CustomOp | GlobalPhaseOp) -> Operator:
    """Convert an xDSL op to a PennyLane operator."""

    match op.name:

        case "quantum.gphase":
            gate = qml.GlobalPhase(ssa_to_qml_params(op, single=True), wires=ssa_to_qml_wires(op))

        case _:
            gate_cls = resolve_gate(op.properties.get("gate_name").data)
            gate = gate_cls(*ssa_to_qml_params(op), wires=ssa_to_qml_wires(op))

    return _apply_adjoint_and_ctrls(gate, op)


def xdsl_to_qml_obs_op(
    op: NamedObsOp | TensorOp | HamiltonianOp | ComputationalBasisOp,
) -> Operator | list[int] | None:
    """Convert an xDSL observable operation to a PennyLane operator."""

    match op.name:

        case "quantum.namedobs":
            return resolve_gate(op.type.data.value)(wires=ssa_to_qml_wires_named(op))
        case "quantum.tensor":
            return qml.prod(*(xdsl_to_qml_obs_op(operand.owner) for operand in op.operands))
        case "quantum.hamiltonian":
            coeffs = _extract(op, "coeffs", resolve_constant_params, single=True)
            ops_list = [xdsl_to_qml_obs_op(term.owner) for term in op.terms]
            return qml.Hamiltonian(coeffs, ops_list)
        case "quantum.compbasis":
            return _extract(op, "qubits", resolve_constant_wire)
        case _:
            raise NotImplementedError(f"Cannot resolve named op: {op}")


def xdsl_to_qml_meas(meas, *args, **kwargs) -> MeasurementProcess:
    """Given a measurement in xDSL, convert it to a PennyLane measurement."""
    return resolve_measurement(meas.name)(*args, **kwargs)

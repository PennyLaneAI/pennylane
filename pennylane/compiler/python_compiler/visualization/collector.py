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

from typing import Any, Callable, Union

from xdsl.dialects import builtin, func
from xdsl.dialects.builtin import (
    FloatAttr,
    IntegerAttr,
)
from xdsl.ir import SSAValue

from pennylane.compiler.python_compiler.dialects.quantum import AllocOp as AllocOpPL
from pennylane.compiler.python_compiler.dialects.quantum import CustomOp
from pennylane.compiler.python_compiler.dialects.quantum import ExtractOp as ExtractOpPL
from pennylane.compiler.python_compiler.dialects.quantum import StateOp
from pennylane.measurements import MeasurementProcess
from pennylane.operation import Operator

from .xdsl_conversion import (
    dispatch_wires_extract,
    xdsl_to_qml_meas,
    xdsl_to_qml_op,
)


class QMLCollector:
    """Collects PennyLane ops and measurements from an xDSL module.

    Walks all `FuncOp`s in the given module, building a mapping of SSA qubits to wire indices,
    and converting supported xDSL operations and measurements to PennyLane objects.
    """

    # Several TODO for this pass/collector:
    # - Add support for other operations (e.g., QubitUnaryOp, GlobalPhaseOp, etc.),
    # - Add support for measurement operations (e.g., qml.probs, qml.expval, etc.)
    # - Add support for complex parameters (e.g., complex numbers, arrays, etc.)
    # - Add support for dynamic wires and parameters

    SUPPORTED_OPS: dict[type, Callable[[Any], Operator | MeasurementProcess]]

    def __init__(self, module: builtin.ModuleOp):
        self.module = module
        self.wire_to_ssa_qubits: dict[int, SSAValue] = {}
        self.ssa_qubits_to_wires: dict[SSAValue, int] = {}
        self.params_to_ssa_params: dict[Union[FloatAttr, IntegerAttr], SSAValue] = {}
        self.quantum_register: SSAValue | None = None

    # pylint: disable=protected-access
    SUPPORTED_OPS = {
        StateOp: lambda self, xdsl_obj: self._handle_measurement(xdsl_obj),
        CustomOp: lambda self, xdsl_obj: self._handle_custom_op(xdsl_obj),
    }

    def _handle_measurement(self, xdsl_state) -> MeasurementProcess:
        return xdsl_to_qml_meas(xdsl_state)

    def _handle_custom_op(self, xdsl_custom_op) -> Operator:
        if self.quantum_register is None:
            raise ValueError("Quantum register (AllocOp) not found.")
        if not self.wire_to_ssa_qubits:
            raise NotImplementedError("No wires extracted from the register found.")
        return xdsl_to_qml_op(xdsl_custom_op)

    # TODO: this will probably no longer be needed once PR #7937 is merged
    def _process_qubit_mapping(self, op):
        """Populate wire mappings from AllocOp and ExtractOp."""

        if isinstance(op, AllocOpPL):
            if self.quantum_register is not None:
                raise ValueError("Found more than one AllocOp for this FuncOp.")
            self.quantum_register = op.qreg

        elif isinstance(op, ExtractOpPL):
            wire = dispatch_wires_extract(op)
            if wire not in self.wire_to_ssa_qubits:
                self.wire_to_ssa_qubits[wire] = op.qubit
                # We update the reverse mapping as well for completeness,
                # but this should not be used in practice for visualization.
                self.ssa_qubits_to_wires[op.qubit] = wire

    def clear_mappings(self):
        """Clear all wire and parameter mappings."""

        self.wire_to_ssa_qubits.clear()
        self.ssa_qubits_to_wires.clear()
        self.params_to_ssa_params.clear()
        self.quantum_register = None

    def collect(self, reset: bool = True) -> tuple[list[Operator], list[MeasurementProcess]]:
        """Collect PennyLane ops and measurements from the module."""

        if reset:
            self.clear_mappings()

        collected_ops: list[Operator] = []
        collected_meas: list[MeasurementProcess] = []

        for func_op in self.module.walk():

            if not isinstance(func_op, func.FuncOp):
                continue

            for op in func_op.body.walk():

                self._process_qubit_mapping(op)
                handler = self.SUPPORTED_OPS.get(type(op), None)

                if handler:
                    qml_obj = handler(self, op)
                    (collected_meas if isinstance(op, StateOp) else collected_ops).append(qml_obj)

        return collected_ops, collected_meas

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

from functools import singledispatchmethod
from typing import Any, Union

from xdsl.dialects import builtin, func
from xdsl.ir import SSAValue

from pennylane.compiler.python_compiler.dialects.quantum import AllocOp as AllocOpPL
from pennylane.compiler.python_compiler.dialects.quantum import (
    CustomOp,
    ExpvalOp,
)
from pennylane.compiler.python_compiler.dialects.quantum import ExtractOp as ExtractOpPL
from pennylane.compiler.python_compiler.dialects.quantum import (
    GlobalPhaseOp,
    ProbsOp,
    SampleOp,
    StateOp,
    VarianceOp,
)
from pennylane.measurements import MeasurementProcess
from pennylane.operation import Operator

from .xdsl_conversion import (
    dispatch_wires_extract,
    xdsl_to_qml_compbasis_op,
    xdsl_to_qml_meas,
    xdsl_to_qml_obs_op,
    xdsl_to_qml_op,
)


class QMLCollector:
    """Collects PennyLane ops and measurements from an xDSL module.

    Walks all `FuncOp`s in the given module, building a mapping of SSA qubits to wire indices,
    and converting supported xDSL operations and measurements to PennyLane objects.
    """

    def __init__(self, module: builtin.ModuleOp):
        self.module = module
        self.wire_to_ssa_qubits: dict[int, SSAValue] = {}
        self.quantum_register: SSAValue | None = None

    # pylint: disable=unused-argument
    @singledispatchmethod
    def handle(self, _: Any) -> Union[Operator, MeasurementProcess, None]:
        """Default handler for unsupported operations. If the operation is not recognized, return None."""
        return None

    ############################################################
    ### Measurements
    ############################################################

    @handle.register
    def _(self, xdsl_meas: StateOp | SampleOp) -> MeasurementProcess:
        return xdsl_to_qml_meas(xdsl_meas)

    @handle.register
    def _(self, xdsl_probs: ProbsOp) -> MeasurementProcess:
        compbasis_op = xdsl_probs.obs.owner
        return xdsl_to_qml_meas(xdsl_probs, xdsl_to_qml_compbasis_op(compbasis_op))

    @handle.register
    def _(self, xdsl_meas_op: ExpvalOp | VarianceOp) -> MeasurementProcess:
        obs_op = xdsl_meas_op.obs.owner
        return xdsl_to_qml_meas(xdsl_meas_op, xdsl_to_qml_obs_op(obs_op))

    ############################################################
    ### Operators
    ############################################################

    @handle.register
    def _(self, xdsl_op: CustomOp | GlobalPhaseOp) -> Operator:
        if self.quantum_register is None:
            raise ValueError("Quantum register (AllocOp) not found.")
        if not self.wire_to_ssa_qubits:
            raise NotImplementedError("No wires extracted from the register found.")
        return xdsl_to_qml_op(xdsl_op)

    ############################################################
    ### Internal Methods
    ############################################################

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

    def clear_mappings(self):
        """Clear all wire and parameter mappings."""

        self.wire_to_ssa_qubits.clear()
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
                result = self.handle(op)

                if isinstance(result, MeasurementProcess):
                    collected_meas.append(result)

                if isinstance(result, Operator):
                    collected_ops.append(result)

        return collected_ops, collected_meas

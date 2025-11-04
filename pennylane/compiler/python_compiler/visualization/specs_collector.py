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

"""
This file contains the implementation of the a collector method for qml.specs,
which collects and maps PennyLane operations and measurements from xDSL.
"""

import enum
from functools import singledispatch

import xdsl
from xdsl.dialects import func
from xdsl.dialects.scf import ForOp, IfOp

from pennylane.compiler.python_compiler.dialects.catalyst import CallbackOp
from pennylane.compiler.python_compiler.dialects.quantum import (
    CustomOp,
    ExpvalOp,
    GlobalPhaseOp,
    MeasureOp,
    MultiRZOp,
    ProbsOp,
    QubitUnitaryOp,
    SampleOp,
    SetBasisStateOp,
    SetStateOp,
    StateOp,
    VarianceOp,
)
from pennylane.compiler.python_compiler.visualization.xdsl_conversion import *


class ResourceType(enum.Enum):
    """Enum for what kind of resource corresponds to a given xDSL operation type."""

    GATE = "gate"
    MEASUREMENT = "measurement"
    PPM = "ppm"


class ResourcesResult:
    """Class to hold the result of resource collection for a given operation."""

    def __init__(self):
        self.quantum_operations: dict[str, int] = {}
        self.quantum_measurements: dict[str, int] = {}
        self.ppm_operations: dict[str, int] = {}

    def merge_with(self, other: "ResourcesResult") -> None:
        """Merge another ResourcesResult into this one."""
        for name, count in other.quantum_operations.items():
            self.quantum_operations[name] = self.quantum_operations.get(name, 0) + count
        for name, count in other.quantum_measurements.items():
            self.quantum_measurements[name] = self.quantum_measurements.get(name, 0) + count
        for name, count in other.ppm_operations.items():
            self.ppm_operations[name] = self.ppm_operations.get(name, 0) + count

    def multiply_by_scalar(self, scalar: int) -> None:
        """Multiply all counts by a scalar."""
        for name in self.quantum_operations:
            self.quantum_operations[name] *= scalar
        for name in self.quantum_measurements:
            self.quantum_measurements[name] *= scalar
        for name in self.ppm_operations:
            self.ppm_operations[name] *= scalar


@singledispatch
def handle(xdsl_op: xdsl.ir.Operation) -> tuple[ResourceType, str] | tuple[None, None]:
    # breakpoint()
    # print(f"Unsupported xDSL op: {xdsl_op}")
    # raise NotImplementedError(f"Unsupported xDSL op: {xdsl_op}")
    return None, None


############################################################
### Measurements
############################################################


@handle.register
def _(xdsl_meas: StateOp) -> tuple[ResourceType, str]:
    return ResourceType.MEASUREMENT, xdsl_to_qml_measurement_type(xdsl_meas)


@handle.register
def _(xdsl_meas_op: ExpvalOp | VarianceOp | ProbsOp | SampleOp) -> tuple[ResourceType, str]:
    obs_op = xdsl_meas_op.obs.owner
    return ResourceType.MEASUREMENT, xdsl_to_qml_measurement_type(
        xdsl_meas_op, xdsl_to_qml_measurement_type(obs_op)
    )


@handle.register
def _(xdsl_measure: MeasureOp) -> tuple[ResourceType, str]:
    return ResourceType.MEASUREMENT, xdsl_to_qml_measurement_type(xdsl_measure)


############################################################
### Operators
############################################################


@handle.register
def _(
    xdsl_op: CustomOp | GlobalPhaseOp | QubitUnitaryOp | SetStateOp | MultiRZOp | SetBasisStateOp,
) -> tuple[ResourceType, str]:
    return ResourceType.GATE, xdsl_to_qml_op_type(xdsl_op)


def _collect_region(region) -> ResourcesResult:
    """Collect PennyLane ops and measurements from a region."""

    collected_ops = ResourcesResult()

    for op in region.ops:
        if isinstance(op, ForOp):
            iters = count_static_loop_iterations(op)
            body_ops = _collect_region(op.body)
            body_ops.multiply_by_scalar(iters)
            collected_ops.merge_with(body_ops)
            continue

        if isinstance(op, IfOp):
            # NOTE: For now we count operations from both branches
            collected_ops.merge_with(_collect_region(op.true_region))
            collected_ops.merge_with(_collect_region(op.false_region))
            continue

        resource_type, resource = handle(op)

        if resource_type is None or resource is None:
            # xDSL op type is not a PennyLane resource to be tracked
            continue

        match resource_type:
            case ResourceType.GATE:
                collected_ops.quantum_operations[resource] = (
                    collected_ops.quantum_operations.get(resource, 0) + 1
                )
            case ResourceType.MEASUREMENT:
                collected_ops.quantum_measurements[resource] = (
                    collected_ops.quantum_measurements.get(resource, 0) + 1
                )
            # TODO: PPM specs
            case _:
                raise NotImplementedError(
                    f"Unsupported resource type {resource_type} for resource {resource}."
                )

    return collected_ops


def specs_collect(module) -> dict[str, int]:
    """Collect PennyLane ops and measurements from the module."""

    collected_ops = ResourcesResult()

    for func_op in module.body.ops:

        if isinstance(func_op, CallbackOp):
            print("Skipping CallbackOp in collector.")
            continue

        if not isinstance(func_op, func.FuncOp):
            raise ValueError("Expected FuncOp in module body.")

        collected_ops.merge_with(_collect_region(func_op.body))

    # print("Measurements:", collected_ops.quantum_measurements)
    return collected_ops.quantum_operations

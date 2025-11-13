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
import warnings
from collections import defaultdict
from functools import singledispatch

import xdsl
from xdsl.dialects import func
from xdsl.dialects.scf import ForOp, IfOp, WhileOp

from pennylane.compiler.python_compiler.dialects.catalyst import CallbackOp
from pennylane.compiler.python_compiler.dialects.qec import (
    PPMeasurementOp,
    PPRotationOp,
    SelectPPMeasurementOp,
)
from pennylane.compiler.python_compiler.dialects.quantum import (
    AllocOp,
    AllocQubitOp,
    CustomOp,
    DeviceInitOp,
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
from pennylane.compiler.python_compiler.inspection.xdsl_conversion import *


class ResourceType(enum.Enum):
    """Enum for what kind of resource corresponds to a given xDSL operation type."""

    # Circuit resources
    GATE = "gate"
    MEASUREMENT = "measurement"
    PPM = "ppm"

    # Extra circuit info
    OTHER = "other"


class ResourcesResult:
    """Class to hold the result of resource collection for a given operation."""

    def __init__(self):
        self.quantum_operations: dict[str, int] = defaultdict(int)
        self.quantum_measurements: dict[str, int] = defaultdict(int)
        self.ppm_operations: dict[str, int] = defaultdict(int)

        self.resource_sizes: dict[int, int] = defaultdict(int)

        self.device_name = None
        self.num_wires = 0  # More accurately, the number of NEW allocations in this region

    def merge_with(self, other: "ResourcesResult") -> None:
        """Merge another ResourcesResult into this one."""
        for name, count in other.quantum_operations.items():
            self.quantum_operations[name] += count
        for name, count in other.quantum_measurements.items():
            self.quantum_measurements[name] += count
        for name, count in other.ppm_operations.items():
            self.ppm_operations[name] += count

        for size, count in other.resource_sizes.items():
            self.resource_sizes[size] += count

        self.device_name = self.device_name or other.device_name
        self.num_wires += other.num_wires

    def multiply_by_scalar(self, scalar: int) -> None:
        """Multiply all counts by a scalar."""
        for name in self.quantum_operations:
            self.quantum_operations[name] *= scalar
        for name in self.quantum_measurements:
            self.quantum_measurements[name] *= scalar
        for name in self.ppm_operations:
            self.ppm_operations[name] *= scalar

        for size in self.resource_sizes:
            self.resource_sizes[size] *= scalar

        # This is the number of allocations WITHIN this region, should be scaled
        self.num_wires *= scalar

    def __repr__(self) -> str:
        return f"ResourcesResult(device_name: {self.device_name}, num_wires: {self.num_wires}, operations: {self.quantum_operations}, measurements: {self.quantum_measurements}, PPMs: {self.ppm_operations})"

    __str__ = __repr__


@singledispatch
def handle_resource(
    xdsl_op: xdsl.ir.Operation,
) -> tuple[ResourceType, str, int] | tuple[None, None]:
    # Default handler for unsupported xDSL op types
    return None, None


############################################################
### Measurements
############################################################


@handle_resource.register
def _(xdsl_meas: StateOp) -> tuple[ResourceType, str]:
    return ResourceType.MEASUREMENT, xdsl_to_qml_measurement_type(xdsl_meas)


@handle_resource.register
def _(xdsl_meas_op: ExpvalOp | VarianceOp | ProbsOp | SampleOp) -> tuple[ResourceType, str]:
    obs_op = xdsl_meas_op.obs.owner
    return ResourceType.MEASUREMENT, xdsl_to_qml_measurement_type(
        xdsl_meas_op, xdsl_to_qml_measurement_type(obs_op)
    )


@handle_resource.register
def _(xdsl_measure: MeasureOp) -> tuple[ResourceType, str]:
    return ResourceType.MEASUREMENT, xdsl_to_qml_measurement_type(xdsl_measure)


############################################################
### Quantum Gates
############################################################


@handle_resource.register
def _(
    xdsl_op: CustomOp | GlobalPhaseOp | QubitUnitaryOp | SetStateOp | MultiRZOp | SetBasisStateOp,
) -> tuple[ResourceType, str]:
    return ResourceType.GATE, xdsl_to_qml_op_type(xdsl_op)


############################################################
### PPM Operations
############################################################


@handle_resource.register
def _(xdsl_op: PPRotationOp) -> tuple[ResourceType, str]:
    if xdsl_op.rotation_kind.value.data == 0:
        # Sanity check that this is an identity
        assert all(
            pauli_op.data == "I" for pauli_op in xdsl_op.pauli_product
        ), "Found non-identity PPRotation with pi/0 rotation!"
        s = "PPR-identity"
    else:
        s = f"PPR-pi/{abs(xdsl_op.rotation_kind.value.data)}-w{len(xdsl_op.in_qubits)}"
    return ResourceType.PPM, s


@handle_resource.register
def _(xdsl_op: PPMeasurementOp | SelectPPMeasurementOp) -> tuple[ResourceType, str]:
    return ResourceType.PPM, f"PPM-w{len(xdsl_op.in_qubits)}"


############################################################
### Other Specs Info
############################################################


@handle_resource.register
def _(
    xdsl_op: DeviceInitOp | AllocOp | AllocQubitOp,
) -> tuple[None, None]:
    # If these types are matched, parse them with the extra specs handler
    return ResourceType.OTHER, None


@singledispatch
def handle_extra(xdsl_op: xdsl.ir.Operation, resources: ResourcesResult) -> None:
    raise NotImplementedError(f"Unsupported xDSL op: {xdsl_op}")


@handle_extra.register
def _(xdsl_op: DeviceInitOp, resources: ResourcesResult) -> None:
    resources.device_name = xdsl_op.device_name.data


@handle_extra.register
def _(xdsl_op: AllocQubitOp | AllocOp, resources: ResourcesResult) -> None:
    # TODO: Should be able to handle deallocs as well
    if isinstance(xdsl_op, AllocQubitOp):
        nallocs = 1
    else:
        nallocs = xdsl_op.nqubits_attr.value.data
    resources.num_wires += nallocs


############################################################
### Main Routines
############################################################


def _collect_region(region, loop_warning=False, cond_warning=False) -> ResourcesResult:
    """Collect PennyLane ops and measurements from a region."""

    resources = ResourcesResult()

    for op in region.ops:
        if isinstance(op, ForOp):
            body_ops = _collect_region(
                op.body, loop_warning=loop_warning, cond_warning=cond_warning
            )
            try:
                iters = count_static_loop_iterations(op)
                body_ops.multiply_by_scalar(iters)
            except NotImplementedError:
                # Unable to statically determine loop iterations
                if not loop_warning:
                    warnings.warn(
                        "Specs was unable to determine the number of loop iterations. "
                        "The results will assume the loop runs only once. "
                        "This may be fixed in some cases by inlining dynamic arguments."
                    )
                loop_warning = True
            resources.merge_with(body_ops)
            continue

        if isinstance(op, WhileOp):
            if not loop_warning:
                warnings.warn(
                    "Specs was unable to determine the number of loop iterations. "
                    "The results will assume the loop runs only once. "
                    "This may be fixed in some cases by inlining dynamic arguments."
                )
                loop_warning = True
            body_ops = _collect_region(
                op.after_region, loop_warning=loop_warning, cond_warning=cond_warning
            )
            resources.merge_with(body_ops)
            continue

        if isinstance(op, IfOp):
            if not cond_warning:
                # NOTE: For now we count operations from both branches
                warnings.warn(
                    "Specs was unable to determine the branch of a conditional. "
                    "The results will assume both branches of the conditional are run."
                )
                cond_warning = True
            resources.merge_with(
                _collect_region(
                    op.true_region, loop_warning=loop_warning, cond_warning=cond_warning
                )
            )
            resources.merge_with(
                _collect_region(
                    op.false_region, loop_warning=loop_warning, cond_warning=cond_warning
                )
            )
            continue

        resource_type, resource = handle_resource(op)

        if resource_type is None:
            # xDSL op type is not a PennyLane resource to be tracked
            continue

        match resource_type:
            case ResourceType.GATE:
                resources.quantum_operations[resource] += 1
                n_qubits = 0
                if hasattr(op, "in_qubits"):
                    n_qubits += len(op.in_qubits)
                if hasattr(op, "in_ctrl_qubits"):
                    n_qubits += len(op.in_ctrl_qubits)
                resources.resource_sizes[n_qubits] += 1
            case ResourceType.MEASUREMENT:
                resources.quantum_measurements[resource] += 1
            case ResourceType.PPM:
                resources.ppm_operations[resource] += 1
                resources.resource_sizes[len(op.in_qubits)] += 1
            case ResourceType.OTHER:
                # Parse out extra circuit information
                handle_extra(op, resources)
            case _:
                raise NotImplementedError(
                    f"Unsupported resource type {resource_type} for resource {resource}."
                )

    return resources


def specs_collect(module) -> ResourcesResult:
    """Collect PennyLane resources from the module."""

    resources = ResourcesResult()

    for func_op in module.body.ops:
        # TODO: May need to determine how many times a function is called for functions other than the main circuit

        if isinstance(func_op, CallbackOp):
            # Skip callback ops, which are not part of the quantum circuit itself
            continue

        if not isinstance(func_op, func.FuncOp):
            raise ValueError("Expected FuncOp in module body.")

        resources.merge_with(_collect_region(func_op.body))

    return resources

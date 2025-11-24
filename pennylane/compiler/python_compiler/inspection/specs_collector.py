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
from xdsl.dialects.scf import ForOp, IfOp, IndexSwitchOp, WhileOp

from pennylane.compiler.python_compiler.dialects.catalyst import CallbackOp
from pennylane.compiler.python_compiler.dialects.mbqc import GraphStatePrepOp, MeasureInBasisOp
from pennylane.compiler.python_compiler.dialects.qec import (
    PPMeasurementOp,
    PPRotationOp,
    SelectPPMeasurementOp,
)
from pennylane.compiler.python_compiler.dialects.quantum import (
    AllocOp,
    AllocQubitOp,
    CountsOp,
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

# A list of all custom dialect names used by Catalyst for MLIR ops
# Note that this isn't a complete list, just one where specs *must* support every op
_CUSTOM_DIALECT_NAMES = frozenset(
    {
        "quantum",
        "qec",
        "mbqc",
    }
)

# TODO: How to handle `gradient` and `mitigation` dialects

# Ops to skip counting as "classical ops", only relevant if these ops would otherwise be
# counted as classical ops (i.e. they are not already quantum ops, QECs, etc.)
_SKIPPED_CLASSICAL_OPS = frozenset(
    {
        "quantum.compbasis",
        "quantum.dealloc",
        "quantum.dealloc_qb",
        "quantum.device_release",
        "quantum.extract",
        "quantum.finalize",
        "quantum.init",
        "quantum.insert",
        "quantum.num_qubits",
        "quantum.yield",
        "qec.yield",
    }
)

# TODO: Handle these somehow
_TODO_OPS = [
    "quantum.adjoint",
    "quantum.hamiltonian",
    "quantum.hermitian",
    "quantum.namedobs",
    "quantum.tensor",
    "qec.fabricate",
    "qec.layer",
    "qec.prepare",
]


class ResourceType(enum.Enum):
    """Enum for what kind of resource corresponds to a given xDSL operation type."""

    # Circuit resources
    GATE = "gate"
    MEASUREMENT = "measurement"
    QEC = "qec"

    # Extra circuit info
    METADATA = "meta"

    # MLIR data
    FUNC_CALL = "fcall"
    OTHER = "other"


class ResourcesResult:
    """Class to hold the result of resource collection for a given operation."""

    def __init__(self):
        self.quantum_operations: dict[str, int] = defaultdict(int)
        self.quantum_measurements: dict[str, int] = defaultdict(int)
        self.qec_operations: dict[str, int] = defaultdict(int)

        self.resource_sizes: dict[int, int] = defaultdict(int)

        self.classical_instructions: dict[str, int] = defaultdict(int)
        self.function_calls: dict[str, int] = defaultdict(int)

        self.device_name = None
        self.num_wires = 0  # More accurately, the number of NEW allocations in this region

    def merge_with(self, other: "ResourcesResult", method: str = "sum") -> None:
        """Merge another ResourcesResult into this one."""

        if method == "max":
            merge_func = max
        elif method == "min":
            merge_func = min
        elif method == "sum":
            merge_func = lambda a, b: a + b
        else:
            raise ValueError(f"Unsupported merge method: '{method}'. Use 'sum', 'max', or 'min'.")

        for name, count in other.quantum_operations.items():
            self.quantum_operations[name] = merge_func(self.quantum_operations[name], count)
        for name, count in other.quantum_measurements.items():
            self.quantum_measurements[name] = merge_func(self.quantum_measurements[name], count)
        for name, count in other.qec_operations.items():
            self.qec_operations[name] = merge_func(self.qec_operations[name], count)

        for size, count in other.resource_sizes.items():
            self.resource_sizes[size] = merge_func(self.resource_sizes[size], count)

        for name, count in other.classical_instructions.items():
            self.classical_instructions[name] = merge_func(self.classical_instructions[name], count)
        for name, count in other.function_calls.items():
            self.function_calls[name] = merge_func(self.function_calls[name], count)

        self.device_name = self.device_name or other.device_name
        self.num_wires = merge_func(self.num_wires, other.num_wires)

    def multiply_by_scalar(self, scalar: int) -> None:
        """Multiply all counts by a scalar."""
        for name in self.quantum_operations:
            self.quantum_operations[name] *= scalar
        for name in self.quantum_measurements:
            self.quantum_measurements[name] *= scalar
        for name in self.qec_operations:
            self.qec_operations[name] *= scalar

        for size in self.resource_sizes:
            self.resource_sizes[size] *= scalar

        for name in self.classical_instructions:
            self.classical_instructions[name] *= scalar
        for name in self.function_calls:
            self.function_calls[name] *= scalar

        # This is the number of allocations WITHIN this region, should be scaled
        self.num_wires *= scalar

    def __repr__(self) -> str:
        return (
            f"ResourcesResult(device: {self.device_name}, "
            f"wires: {self.num_wires}, "
            f"quantum_ops: {sum(self.quantum_operations.values())}, "
            f"measurements: {sum(self.quantum_measurements.values())}, "
            f"QECs: {sum(self.qec_operations.values())}, "
            f"classical_instructions: {sum(self.classical_instructions.values())}, "
            f"fn_calls: {sum(self.function_calls.values())})"
        )

    __str__ = __repr__


@singledispatch
def handle_resource(
    xdsl_op: xdsl.ir.Operation,
) -> tuple[ResourceType, str] | tuple[None, None]:
    # Default handler for unsupported xDSL op types
    return ResourceType.OTHER, xdsl_op.name


############################################################
### Measurements
############################################################


@handle_resource.register
def _(xdsl_meas: MeasureOp) -> tuple[ResourceType, str]:
    return ResourceType.MEASUREMENT, xdsl_to_qml_measurement_type(xdsl_meas)


@handle_resource.register
def _(
    xdsl_meas: CountsOp | ExpvalOp | ProbsOp | SampleOp | StateOp | VarianceOp,
) -> tuple[ResourceType, str]:
    obs_op = xdsl_meas.obs.owner
    return ResourceType.MEASUREMENT, xdsl_to_qml_measurement_type(
        xdsl_meas, xdsl_to_qml_measurement_type(obs_op)
    )


############################################################
### Quantum Gates
############################################################


@handle_resource.register
def _(
    xdsl_op: CustomOp | GlobalPhaseOp | MultiRZOp | SetBasisStateOp | SetStateOp | QubitUnitaryOp,
) -> tuple[ResourceType, str]:
    return ResourceType.GATE, xdsl_to_qml_op_type(xdsl_op)


############################################################
### QEC Operations
############################################################


@handle_resource.register
def _(xdsl_op: GraphStatePrepOp | MeasureInBasisOp) -> tuple[ResourceType, str]:
    return ResourceType.QEC, xdsl_op.name


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
    return ResourceType.QEC, s


@handle_resource.register
def _(xdsl_op: PPMeasurementOp | SelectPPMeasurementOp) -> tuple[ResourceType, str]:
    return ResourceType.QEC, f"PPM-w{len(xdsl_op.in_qubits)}"


############################################################
### Subroutine calls
############################################################


@handle_resource.register
def _(
    xdsl_op: func.CallOp,
) -> tuple[ResourceType, str]:
    # If these types are matched, parse them with the extra specs handler
    return ResourceType.FUNC_CALL, xdsl_op.callee.string_value()


############################################################
### Circuit Metadata
############################################################


@handle_resource.register
def _(
    xdsl_op: DeviceInitOp | AllocOp | AllocQubitOp,
) -> tuple[None, None]:
    # If these types are matched, parse them with the specs metadata handler
    return ResourceType.METADATA, None


@singledispatch
def handle_metadata(xdsl_op: xdsl.ir.Operation, resources: ResourcesResult) -> None:
    raise NotImplementedError(f"Unsupported xDSL op: {xdsl_op}")


@handle_metadata.register
def _(xdsl_op: DeviceInitOp, resources: ResourcesResult) -> None:
    resources.device_name = xdsl_op.device_name.data


@handle_metadata.register
def _(xdsl_op: AllocQubitOp | AllocOp, resources: ResourcesResult) -> None:
    if isinstance(xdsl_op, AllocQubitOp):
        nallocs = 1
    else:
        nallocs = xdsl_op.nqubits_attr.value.data
    resources.num_wires += nallocs


############################################################
### Main Routines
############################################################


def _resolve_function_calls(
    func: str, func_to_resources: dict[str, ResourcesResult]
) -> ResourcesResult:
    """Resolve subroutine function calls within the collected resources.

    Note that this function is recursive and modifies the ResourcesResult associated with `func`
    in place. After running, the ResourcesResult object associated with `func` will include the
    resources of any subroutine functions and its function call table will be empty
    """
    resources = func_to_resources[func]

    for called_func in list(resources.function_calls.keys()):
        count = resources.function_calls.pop(called_func)

        if called_func not in func_to_resources:
            # External function, cannot resolve
            continue

        called_resources = _resolve_function_calls(called_func, func_to_resources)

        # Merge the called function's resources into this one
        called_resources_scaled = ResourcesResult()
        called_resources_scaled.merge_with(called_resources)
        called_resources_scaled.multiply_by_scalar(count)
        resources.merge_with(called_resources_scaled)

    return resources


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
                        "This may be fixed in some cases by inlining dynamic arguments.",
                        UserWarning,
                    )
                loop_warning = True
            resources.merge_with(body_ops)
            continue

        if isinstance(op, WhileOp):
            if not loop_warning:
                warnings.warn(
                    "Specs was unable to determine the number of loop iterations. "
                    "The results will assume the loop runs only once. "
                    "This may be fixed in some cases by inlining dynamic arguments.",
                    UserWarning,
                )
                loop_warning = True
            body_ops = _collect_region(
                op.after_region, loop_warning=loop_warning, cond_warning=cond_warning
            )
            resources.merge_with(body_ops)
            continue

        if isinstance(op, IfOp):
            if not cond_warning:
                warnings.warn(
                    "Specs was unable to determine the branch of a conditional or switch statement. "
                    "The results will take the maximum resources across all possible branches.",
                    UserWarning,
                )
                cond_warning = True

            used_resources = _collect_region(
                op.true_region, loop_warning=loop_warning, cond_warning=cond_warning
            )
            used_resources.merge_with(
                _collect_region(
                    op.false_region, loop_warning=loop_warning, cond_warning=cond_warning
                ),
                method="max",
            )

            resources.merge_with(used_resources)
            continue

        if isinstance(op, IndexSwitchOp):
            if not cond_warning:
                warnings.warn(
                    "Specs was unable to determine the branch of a conditional or switch statement. "
                    "The results will take the maximum resources across all possible branches.",
                    UserWarning,
                )
                cond_warning = True

            used_resources = _collect_region(
                op.case_regions[0], loop_warning=loop_warning, cond_warning=cond_warning
            )

            for region in op.case_regions[1:]:
                used_resources.merge_with(
                    _collect_region(region, loop_warning=loop_warning, cond_warning=cond_warning),
                    method="max",
                )

            resources.merge_with(used_resources)
            continue

        resource_type, resource = handle_resource(op)

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

            case ResourceType.QEC:
                resources.qec_operations[resource] += 1
                if op.name.startswith("ppm."):
                    resources.resource_sizes[len(op.in_qubits)] += 1

            case ResourceType.METADATA:
                # Parse out extra circuit information
                handle_metadata(op, resources)

            case ResourceType.OTHER:
                if op.name in _SKIPPED_CLASSICAL_OPS:
                    continue

                if op.dialect_name() in _CUSTOM_DIALECT_NAMES:
                    # Unknown custom dialect op, warn the user
                    warnings.warn(
                        f"Specs encountered an unknown operation '{op.name}' from the "
                        f"'{op.dialect_name()}' dialect. Some resource data may be missing.",
                        UserWarning,
                    )
                    continue

                resources.classical_instructions[resource] += 1

            case ResourceType.FUNC_CALL:
                resources.function_calls[resource] += 1

            case _:
                # Should be unreachable
                raise NotImplementedError(
                    f"Unsupported resource type {resource_type} for resource {resource}."
                )

    return resources


def specs_collect(module) -> ResourcesResult:
    """Collect PennyLane resources from the module."""

    func_to_resources = {}
    entry_func = None

    func_decl_warning = False

    for func_op in module.body.ops:
        # TODO: May need to determine how many times a function is called for functions other than the main circuit

        if isinstance(func_op, CallbackOp):
            # Skip callback ops, which are not part of the quantum circuit itself
            continue

        if not isinstance(func_op, func.FuncOp):
            raise ValueError("Expected FuncOp in module body.")

        if func_op.is_declaration:
            if not func_decl_warning:
                warnings.warn(
                    f"Specs encountered an external function declaration, and could not analyze its contents. "
                    "Some resource data may be missing.",
                    UserWarning,
                )
                func_decl_warning = True
            continue  # Skip external function declarations

        resources = _collect_region(func_op.body)
        func_to_resources[func_op.sym_name.data] = resources

        if "qnode" in func_op.attributes:
            # The main entrypoint for a qnode is always marked by the `qnode` attribute
            entry_func = func_op.sym_name.data

    if entry_func not in func_to_resources:
        raise ValueError("Entry function not found in module.")

    a = _resolve_function_calls(entry_func, func_to_resources)
    return a

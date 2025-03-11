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

"""Defines the data structure that stores resource estimates for each decomposition."""

from __future__ import annotations

import functools
from dataclasses import dataclass, field

import pennylane as qml


@dataclass(frozen=True)
class Resources:
    r"""Stores resource estimates.

    Args:
        num_gates (int): the total number of gates.
        gate_counts (dict): dictionary mapping compressed ops to number of occurrences.

    """

    num_gates: int = 0
    gate_counts: dict[CompressedResourceOp, int] = field(default_factory=dict)

    def __post_init__(self):
        """Remove zero-count gates and verify that num_gates is correct."""
        assert all(v > 0 for v in self.gate_counts.values())
        assert self.num_gates == sum(self.gate_counts.values())

    def __add__(self, other: Resources):
        return Resources(
            self.num_gates + other.num_gates,
            _combine_dict(self.gate_counts, other.gate_counts),
        )

    def __mul__(self, scalar: int):
        return Resources(self.num_gates * scalar, _scale_dict(self.gate_counts, scalar))

    __rmul__ = __mul__


def _combine_dict(dict1: dict, dict2: dict):
    r"""Combines two dictionaries and adds values of common keys."""

    combined_dict = dict1.copy()

    for k, v in dict2.items():
        combined_dict[k] = combined_dict.get(k, 0) + v

    return combined_dict


def _scale_dict(dict1: dict, scalar: int):
    r"""Scales the values in a dictionary with a scalar."""

    scaled_dict = dict1.copy()

    for k in scaled_dict:
        scaled_dict[k] *= scalar

    return scaled_dict


class CompressedResourceOp:
    """A lightweight class representing an operator to be decomposed.

    An object of this class represents an operator in the decomposition graph. If the decomposition
    of this operator is independent of its parameters, e.g., ``Rot`` can be decomposed into two
    ``RZ`` and an ``RY`` regardless of the angles, then every occurrence of this operator in the
    circuit is represented by the same ``CompressedOp``.

    On the other hand, for more complex ops such as ``PauliRot``, for which the numbers of each
    gate depend on its pauli word, each occurrence of this operator with a different pauli word
    will be a different ``CompressedOp`` object, thus a new node in the decomposition graph.

    Args:
        op_type: the operator type
        params (dict): the parameters of the operator relevant to the resource estimation of
            its decompositions. This should only include parameters that affect the gate counts.

    """

    def __init__(self, op_type, params: dict = None):
        if not isinstance(op_type, type):
            raise TypeError(f"op_type must be a type, got {op_type}")
        if not issubclass(op_type, qml.operation.Operator):
            raise TypeError(f"op_type must be a subclass of Operator, got {op_type}")
        self.op_type = op_type
        self.params = params or {}
        self._hashable_params = _make_hashable(params) if params else ()

    def __hash__(self) -> int:
        return hash((self.op_type, self._hashable_params))

    def __eq__(self, other: CompressedResourceOp) -> bool:
        return (
            isinstance(other, CompressedResourceOp)
            and (self.op_type == other.op_type)
            and (self.params == other.params)
        )

    def __repr__(self):
        return self.op_type.__name__


def _make_hashable(d) -> tuple:
    return tuple((k, _make_hashable(v)) for k, v in d.items()) if isinstance(d, dict) else d


def _validate_resource_rep(op_type, params):
    """Validates the resource representation of an operator."""

    if not issubclass(op_type, qml.operation.Operator):
        raise TypeError(f"op_type must be a type of Operator, got {op_type}")

    if op_type.resource_param_keys is None:
        raise NotImplementedError(f"resource_param_keys undefined for {op_type.__name__}")

    missing_params = set(op_type.resource_param_keys) - set(params.keys())
    if missing_params:
        raise TypeError(
            f"Missing resource parameters for {op_type.__name__}: {list(missing_params)}. "
            f"Expected: {op_type.resource_param_keys}"
        )

    invalid_params = set(params.keys()) - set(op_type.resource_param_keys)
    if invalid_params:
        raise TypeError(
            f"Invalid resource parameters for {op_type.__name__}: {list(invalid_params)}. "
            f"Expected: {op_type.resource_param_keys}"
        )


def resource_rep(op_type, **params) -> CompressedResourceOp:
    """Creates a ``CompressedResourceOp`` representation of an operator.

    When defining the resource function associated with a decomposition rule. The keys of the
    returned dictionary should be created using this function. The resource rep of an operator
    is a lightweight data structure containing the minimal information needed to determine the
    resource estimate of a decomposition.

    Args:
        op_type: the operator type
        **params: parameters that are relevant to the resource estimation of the operator's
            decompositions. This should be consistent with ``op_type.resource_param_keys``.

    """
    _validate_resource_rep(op_type, params)
    if op_type is qml.ops.Controlled or op_type is qml.ops.ControlledOp:
        return controlled_resource_rep(**params)
    if issubclass(op_type, qml.ops.Adjoint):
        return adjoint_resource_rep(**params)
    return CompressedResourceOp(op_type, params)


def controlled_resource_rep(
    base_class,
    base_params: dict,
    num_control_wires: int,
    num_zero_control_values: int = 0,
    num_work_wires: int = 0,
):
    """Creates a ``CompressedResourceOp`` representation of a general controlled operator.

    Args:
        base_class: the base operator type
        base_params (dict): the resource params of the base operator
        num_control_wires (int): the number of control wires
        num_zero_control_values (int): the number of control values that are 0
        num_work_wires (int): the number of work wires

    """

    _validate_resource_rep(base_class, base_params)

    # Flatten nested controlled operations
    if base_class is qml.ops.Controlled or base_class is qml.ops.ControlledOp:
        base_resource_rep = controlled_resource_rep(**base_params)
        base_class = base_resource_rep.params["base_class"]
        base_params = base_resource_rep.params["base_params"]
        num_control_wires += base_resource_rep.params["num_control_wires"]
        num_zero_control_values += base_resource_rep.params["num_zero_control_values"]
        num_work_wires += base_resource_rep.params["num_work_wires"]

    elif base_class in custom_ctrl_op_to_base():
        num_control_wires += base_class.num_wires - 1
        base_class = custom_ctrl_op_to_base()[base_class]

    elif base_class is qml.MultiControlledX:
        base_class = qml.X
        num_control_wires += base_params["num_control_wires"]
        num_zero_control_values += base_params["num_zero_control_values"]
        num_work_wires += base_params["num_work_wires"]
        base_params = {}

    elif base_class is qml.ControlledQubitUnitary:
        base_class = qml.QubitUnitary
        num_control_wires += base_params["num_control_wires"]
        num_zero_control_values += base_params["num_zero_control_values"]
        num_work_wires += base_params["num_work_wires"]
        base_params = base_params["base"].resource_params

    return CompressedResourceOp(
        qml.ops.Controlled,
        {
            "base_class": base_class,
            "base_params": base_params,
            "num_control_wires": num_control_wires,
            "num_zero_control_values": num_zero_control_values,
            "num_work_wires": num_work_wires,
        },
    )


def adjoint_resource_rep(base_class, base_params):
    """Creates a ``CompressedResourceOp`` representation of the adjoint of an operator.

    Args:
        base_class: the base operator type
        base_params (dict): the resource params of the base operator

    """
    base_resource_rep = resource_rep(base_class, **base_params)  # flattens any nested structures
    return CompressedResourceOp(
        qml.ops.Adjoint,
        {"base_class": base_resource_rep.op_type, "base_params": base_resource_rep.params},
    )


@functools.lru_cache()
def custom_ctrl_op_to_base():
    """The set of custom controlled operations."""

    return {
        qml.CNOT: qml.X,
        qml.Toffoli: qml.X,
        qml.CZ: qml.Z,
        qml.CCZ: qml.Z,
        qml.CY: qml.Y,
        qml.CSWAP: qml.SWAP,
        qml.CH: qml.H,
        qml.CRX: qml.RX,
        qml.CRY: qml.RY,
        qml.CRZ: qml.RZ,
        qml.CRot: qml.Rot,
        qml.ControlledPhaseShift: qml.PhaseShift,
    }

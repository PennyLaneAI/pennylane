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
from functools import cached_property
from typing import Optional, Type

import numpy as np

import pennylane as qml
from pennylane.operation import Operator


@dataclass(frozen=True)
class Resources:
    r"""Stores resource estimates.

    Args:
        gate_counts (dict): dictionary mapping operator types to their number of occurrences.

    """

    gate_counts: dict[CompressedResourceOp, int] = field(default_factory=dict)

    def __post_init__(self):
        """Verify that all gate counts are non-zero."""
        assert all(v > 0 for v in self.gate_counts.values())

    @cached_property
    def num_gates(self) -> int:
        """The total number of gates."""
        return sum(self.gate_counts.values())

    def __add__(self, other: Resources):
        return Resources(
            _combine_dict(self.gate_counts, other.gate_counts),
        )

    def __mul__(self, scalar: int):
        return Resources(_scale_dict(self.gate_counts, scalar))

    __rmul__ = __mul__

    def __repr__(self):
        return f"<num_gates={self.num_gates}, gate_counts={self.gate_counts}>"


def _combine_dict(dict1: dict, dict2: dict):
    r"""Combines two dictionaries and adds values of common keys."""

    combined_dict = dict1.copy()

    for k, v in dict2.items():
        combined_dict[k] = combined_dict.get(k, 0) + v

    return combined_dict


def _scale_dict(dict1: dict, scalar: int):
    r"""Scales the values in a dictionary with a scalar."""

    return {key: scalar * value for key, value in dict1.items()}


class CompressedResourceOp:
    """A lightweight representation of an operator to be decomposed.

    .. note::

        This class is only relevant when the new experimental graph-based decomposition system
        (introduced in v0.41) is enabled via ``qml.decomposition.enable_graph()``. This new way of
        doing decompositions is generally more resource efficient and accommodates multiple alternative
        decomposition rules for an operator. In this new system, custom decomposition rules are
        defined as quantum functions, and it is currently required that every decomposition rule
        declares its required resources using :func:`~pennylane.register_resources`.

    The ``CompressedResourceOp` is a lightweight data structure that contains an operator type
    and a set of parameters that affects the resource requirement of this operator. If the
    decomposition of an operator is independent of its parameters, e.g., ``Rot`` can be decomposed
    into two ``RZ`` gates and an ``RY`` regardless of the angles, then every occurrence of this
    operator in the circuit is represented by the same ``CompressedResourceOp`` which only
    specifies the operator type, i.e., ``Rot``.

    On the other hand, for some operators such as ``MultiRZ``, for which the numbers of ``CNOT``
    gates in its decomposition depends on the number of wires, the resource representation of
    a ``MultiRZ`` must include this information. To create a ``CompressedResourceOp`` object for
    an operator, use the :func:`~pennylane.resource_rep` function.

    Args:
        op_type: the operator type
        params (dict): the parameters of the operator relevant to the resource estimation of
            its decompositions. This should only include parameters that affect the gate counts.

    .. seealso:: :func:`~pennylane.resource_rep`

    """

    def __init__(self, op_type: Type[Operator], params: Optional[dict] = None):
        if not isinstance(op_type, type):
            raise TypeError(f"op_type must be an Operator type, got {type(op_type)}")
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
            and self.op_type == other.op_type
            and self.params == other.params
        )

    def __repr__(self):
        params = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.op_type.__name__}({params})" if self.params else self.op_type.__name__


def _make_hashable(d):
    if isinstance(d, dict):
        return tuple(
            sorted(((str(k), _make_hashable(v)) for k, v in d.items()), key=lambda x: x[0])
        )
    if hasattr(d, "tolist"):
        return d.tolist()
    return d


def _validate_resource_rep(op_type, params):
    """Validates the resource representation of an operator."""

    if not issubclass(op_type, qml.operation.Operator):
        raise TypeError(f"op_type must be a type of Operator, got {op_type}")

    if not isinstance(op_type.resource_keys, set):
        raise TypeError(
            f"{op_type.__name__}.resource_keys must be a set, not a {type(op_type.resource_keys)}"
        )

    unexpected_arguments = set(params.keys()) - op_type.resource_keys
    if unexpected_arguments:
        raise TypeError(
            f"Unexpected keyword arguments for resource_rep({op_type.__name__}): "
            f"{list(unexpected_arguments)}). Expected: {list(op_type.resource_keys)}"
        )

    missing_arguments = op_type.resource_keys - set(params.keys())
    if missing_arguments:
        raise TypeError(
            f"Missing keyword arguments for resource_rep({op_type.__name__}): "
            f"{list(missing_arguments)}. Expected: {list(op_type.resource_keys)}"
        )


def resource_rep(op_type: Type[Operator], **params) -> CompressedResourceOp:
    """Binds an operator type with additional resource parameters.

    .. note::

        This function is only relevant when the new experimental graph-based decomposition system
        (introduced in v0.41) is enabled via ``qml.decomposition.enable_graph()``. This new way of
        doing decompositions is generally more resource efficient and accommodates multiple alternative
        decomposition rules for an operator. In this new system, custom decomposition rules are
        defined as quantum functions, and it is currently required that every decomposition rule
        declares its required resources using :func:`~pennylane.register_resources`.

    Args:
        op_type: the operator class to create a resource representation for.
        **params: parameters relevant to the resource estimate of the operator's decompositions.
            This should be consistent with ``op_type.resource_keys``.

    Returns:
        CompressedResourceOp: a lightweight representation of the operator.

    **Example**

    The resource parameters of an operator are a minimal set of information required to determine
    the resource estimate of its decompositions. To check the required set of keyword arguments
    for an operator type, refer to the ``resource_keys`` attribute of the operator class:

    >>> qml.MultiRZ.resource_keys
    {'num_wires'}

    When calling ``resource_rep`` for ``MultiRZ``, ``num_wires`` must be provided as a keyword argument.

    >>> rep = resource_rep(qml.MultiRZ, num_wires=3)
    >>> rep
    MultiRZ, {'num_wires': 3}
    >>> type(rep)
    <class 'pennylane.decomposition.resources.CompressedResourceOp'>

    .. seealso:: See how this function is used in the context of defining a decomposition rule using :func:`~pennylane.register_resources`

    .. details::
        :title: Usage Details

        The same approach applies also to symbolic operators. For example, if the decomposition
        of an operator contains a controlled operation:

        .. code-block:: python

            def my_decomp(wires):
                qml.ctrl(
                    qml.MultiRZ(wires=wires[:3]),
                    control=wires[3:5],
                    control_values=[0, 1],
                    work_wires=wires[5]
                )

        To declare this controlled operator in the resource function, we find the resource keys
        of ``qml.ops.Controlled``:

        >>> qml.ops.Controlled.resource_keys
        {
            'base_class',
            'base_params',
            'num_control_wires',
            'num_zero_control_values',
            'num_work_wires'
        }

        Then the resource representation can be created as follows:

        >>> qml.resource_rep(
        ...     qml.ops.Controlled,
        ...     base_class=qml.ops.MultiRZ,
        ...     base_params={'num_wires': 3},
        ...     num_control_wires=2,
        ...     num_zero_control_values=1,
        ...     num_work_wires=1
        ... )
        Controlled, {'base_class': <class 'pennylane.ops.qubit.parametric_ops_multi_qubit.MultiRZ'>, 'base_params': {'num_wires': 3}, 'num_control_wires': 2, 'num_zero_control_values': 1, 'num_work_wires': 1}

        Alternatively, use the utility function :func:`~pennylane.decomposition.controlled_resource_rep`:

        >>> qml.decomposition.controlled_resource_rep(
        ...     base_class=qml.ops.MultiRZ,
        ...     base_params={'num_wires': 3},
        ...     num_control_wires=2,
        ...     num_zero_control_values=1,
        ...     num_work_wires=1
        ... )
        Controlled, {'base_class': <class 'pennylane.ops.qubit.parametric_ops_multi_qubit.MultiRZ'>, 'base_params': {'num_wires': 3}, 'num_control_wires': 2, 'num_zero_control_values': 1, 'num_work_wires': 1}

        .. seealso:: :func:`~pennylane.decomposition.controlled_resource_rep` and :func:`~pennylane.decomposition.adjoint_resource_rep`

    """
    _validate_resource_rep(op_type, params)
    if op_type in (qml.ops.Controlled, qml.ops.ControlledOp):
        return controlled_resource_rep(**params)
    if issubclass(op_type, qml.ops.Adjoint):
        return adjoint_resource_rep(**params)
    if issubclass(op_type, qml.ops.Pow):
        return pow_resource_rep(**params)
    return CompressedResourceOp(op_type, params)


def controlled_resource_rep(
    base_class: Type[Operator],
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


def adjoint_resource_rep(base_class: Type[Operator], base_params: dict = None):
    """Creates a ``CompressedResourceOp`` representation of the adjoint of an operator.

    Args:
        base_class: the base operator type
        base_params (dict): the resource params of the base operator

    """
    base_params = base_params or {}
    base_resource_rep = resource_rep(base_class, **base_params)  # flattens any nested structures
    return CompressedResourceOp(
        qml.ops.Adjoint,
        {"base_class": base_resource_rep.op_type, "base_params": base_resource_rep.params},
    )


def _is_integer(x):
    """Checks if x is an integer."""
    return isinstance(x, int) or np.issubdtype(getattr(x, "dtype", None), np.integer)


def pow_resource_rep(base_class, base_params, z):
    """Creates a ``CompressedResourceOp`` representation of the power of an operator.

    Args:
        base_class: the base operator type
        base_params (dict): the resource params of the base operator
        z (int or float): the power

    """
    if (not qml.math.is_abstract(z)) and (not _is_integer(z) or z < 0):
        raise NotImplementedError("Non-integer powers or negative powers are not supported yet.")
    base_resource_rep = resource_rep(base_class, **base_params)
    return CompressedResourceOp(
        qml.ops.Pow,
        {"base_class": base_resource_rep.op_type, "base_params": base_resource_rep.params, "z": z},
    )


@functools.lru_cache(maxsize=1)
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

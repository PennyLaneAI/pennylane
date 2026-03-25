# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines an internal helper function to reconstruct an operator from a resource rep."""

from collections.abc import Callable

import pennylane as qml
from pennylane.decomposition.resources import resource_rep
from pennylane.operation import Operator
from pennylane.wires import Wires

_reconstructors = {}


def decomps_use_reconstructor(op_type, op_params):
    """Checks for special cases that has_reconstructor is not yet prepared to handle."""
    # TODO: Controlled to be implemented in a follow-up PR [sc-110068]
    # TODO: Adjoint to be implemented in a follow-up PR [sc-110066]
    if op_type in (qml.ops.Controlled, qml.ops.Adjoint):
        return False
    if issubclass(op_type, qml.ops.Pow):
        base_class, base_params = op_params["base_class"], op_params["base_params"]
        return decomps_use_reconstructor(base_class, base_params)
    return has_reconstructor(op_type, op_params)


def get_decomp_kwargs(op):
    """Returns the kwargs needed for a decomposition rule."""
    rep = resource_rep(op.__class__, **op.resource_params)
    return (
        op.resource_params
        if decomps_use_reconstructor(rep.op_type, rep.params)
        else op.hyperparameters
    )


def register_reconstructor(op_type: type[Operator]):
    """A decorator that registers a function as the reconstructor of op_type.

    A reconstructor is expected to take ``(*op.data, wires=op.wires, **op.resource_params)``
    as input and return an instance of the original op.

    """

    def _decorator(func: Callable):
        _reconstructors[op_type] = func

    return _decorator


def has_reconstructor(op_class: type[Operator], op_params: dict):
    """Checks whether a reconstructor exists for the resource rep."""

    if op_class.resource_params is Operator.resource_params:
        # If the operator inherited its resource_params from Operator, it means
        # that this operator is never compatible with the graph system.
        return False

    if op_class is qml.templates.TemporaryAND:
        # TODO: Special case of an controlled-like operator which takes control_values,
        # to be handled in a follow-up PR. [sc-110068]
        return False

    if op_class not in _reconstructors and not op_class.resource_keys - {"num_wires"}:
        return True

    # TODO: Controlled to be implemented in a follow-up PR [sc-110068]
    if op_class in (qml.ops.Adjoint, qml.ops.Pow):
        base_class, base_params = op_params["base_class"], op_params["base_params"]
        return has_reconstructor(base_class, base_params)

    return op_class in _reconstructors


def reconstruct(data: tuple, wires: Wires, op_type: type[Operator], op_params: dict) -> Operator:
    """Reconstruct an instance of op_type with resource params."""

    if op_type not in _reconstructors and not op_type.resource_keys - {"num_wires"}:
        # Assume the default for simple gates. Since we don't actually have a Gate
        # class to use in an issubclass check, we use the resource_keys as a proxy.
        # A simple Gate wouldn't have any resource_keys defined. Another special case
        # is when an operator only has a single resource param that is the number of
        # wires. Such an operator also doesn't take anything beyond data and wires
        # in its constructor. The number of wires is typically redundant since this
        # information is apparant from the shape of the wires array already.
        return op_type(*data, wires=wires)

    if op_type is qml.ops.Adjoint:
        base_class, base_params = op_params["base_class"], op_params["base_params"]
        return qml.adjoint(reconstruct)(data, wires, base_class, base_params)

    if op_type is qml.ops.Controlled:
        # TODO: to be implemented in a follow-up PR [sc-110068]
        raise NotImplementedError  # pragma: no cover

    if issubclass(op_type, qml.ops.Pow):
        base_class, base_params = op_params["base_class"], op_params["base_params"]
        return qml.pow(reconstruct(data, wires, base_class, base_params), z=op_params["z"])

    if op_type in _reconstructors:
        return _reconstructors[op_type](*data, wires=wires, **op_params)

    raise NotImplementedError  # pragma: no cover

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

"""Defines the ``DecompositionRule`` class to represent a decomposition rule."""

from __future__ import annotations

from typing import Callable

from .resources import Resources


def decomposition(qfunc: Callable, resource_fn: Callable = None) -> DecompositionRule:
    """Creates a DecompositionRule from a quantum function.

    Args:
        qfunc (Callable): the quantum function that represents the decomposition.
        resource_fn (Callable): a function that computes a gate count of this decomposition rule.

    Returns:
        DecompositionRule: the decomposition rule.

    **Example**

    A decomposition rule should be defined as a qfunc:

    .. code-block:: python

        import pennylane as qml

        def _multi_rz_decomposition(theta, wires, **__):
            for w0, w1 in zip(wires[-1:0:-1], wires[-2::-1]):
                qml.CNOT(wires=(w0, w1))
            qml.RZ(theta, wires=wires[0])
            for w0, w1 in zip(wires[1:], wires[:-1]):
                qml.CNOT(wires=(w0, w1))

    The signature of this qfunc should be ``(*params, wires, **hyperparams)``.

    Along with the qfunc implementation of a decomposition, a resource function should be defined
    that computes a resource estimate for this decomposition rule from a set of parameters:

    .. code-block:: python

        def _multi_rz_decomposition_resources(num_wires):
            return {
                qml.RZ.make_resource_rep(): 1,
                qml.CNOT.make_resource_rep(): 2 * (num_wires - 1)
            }

    The signature of this function should be ``(**resource_params)``, where ``resource_params``
    should agree with the ``resource_params`` property of the operator.
    
    The two functions can be combined to create a ``DecompositionRule`` object:

    .. code-block:: python

        multi_rz_decomposition = decomposition(
            _multi_rz_decomposition,
            resource_fn=_multi_rz_decomposition_resources
        )

    Alternatively, use the decorator syntax:

    .. code-block:: python

        @qml.decomposition
        def multi_rz_decomposition(theta, wires, **__):
            for w0, w1 in zip(wires[-1:0:-1], wires[-2::-1]):
                qml.CNOT(wires=(w0, w1))
            qml.RZ(theta, wires=wires[0])
            for w0, w1 in zip(wires[1:], wires[:-1]):
                qml.CNOT(wires=(w0, w1))

        @multi_rz_decomposition.resources
        def _(num_wires):
            return {
                qml.RZ.make_resource_rep(): 1,
                qml.CNOT.make_resource_rep(): 2 * (num_wires - 1)
            }

    Now ``multi_rz_decomposition`` is a ``DecompositionRule`` object.

    """
    decomposition_rule = DecompositionRule(qfunc)
    if resource_fn is not None:
        decomposition_rule._compute_resources = resource_fn  # pylint: disable=protected-access
    return decomposition_rule


class DecompositionRule:
    """Represents a decomposition rule for an operator.

    Attributes:
        impl (Callable): the quantum function implementing the decomposition rule

    """

    def __init__(self, func: Callable):
        self.impl = func
        self._compute_resources = None

    def compute_resources(self, *args, **kwargs) -> Resources:
        """Computes the resources required to implement this decomposition rule."""
        if self._compute_resources is None:
            raise NotImplementedError("No resource estimation found for this decomposition rule.")
        gate_counts: dict = self._compute_resources(*args, **kwargs)
        gate_counts = {op: count for op, count in gate_counts.items() if count > 0}
        num_gates = sum(gate_counts.values())
        return Resources(num_gates, gate_counts)

    @property
    def resources(self):
        """Registers a function as the resource estimator of this decomposition rule."""

        def _compute_resources_decorator(resource_func):
            self._compute_resources = resource_func

        return _compute_resources_decorator

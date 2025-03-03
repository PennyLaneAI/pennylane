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

from collections import defaultdict
from typing import Callable

from pennylane.operation import Operator

from .resources import CompressedResourceOp, Resources, resource_rep


def register_resources(
    resources: Callable | dict, qfunc: Callable = None
) -> DecompositionRule | Callable[[Callable], DecompositionRule]:
    """Registers a resource function with a decomposition rule.

    Args:
        resources (Callable): the resource estimate of a decomposition rule, in the form of a
            dictionary of gate counts or a function that returns this dictionary.
        qfunc (Callable): the quantum function that implements the decomposition.

    Returns:
        DecompositionRule: a data structure that represents a decomposition rule.

    **Example**

    A decomposition rule should be defined as a qfunc:

    .. code-block:: python

        import pennylane as qml

        def cnot_decomp(wires):
            qml.H(wires=wires[1])
            qml.CZ(wires=wires)
            qml.H(wires=wires[1])

    This qfunc is expected to take ``(*op.params, op.wires, **op.hyperparameters)`` as arguments,
    where ``op`` is an instance of the operator type that this decomposition is for.

    Along with the qfunc implementation of a decomposition, a dictionary mapping operator types to
    their number of occurrences must be registered:

    .. code-block:: python

        cnot_decomp = qml.register_resources({qml.H: 2, qml.CZ: 1}, cnot_decomp)

    This decomposition rule can then be registered with an operator type:

    .. code-block:: python

        qml.add_decomposition(qml.CNOT, cnot_decomp)

    or specified as a fixed decomposition rule to the decompose transform:

    .. code-block:: python

        @partial(qml.transforms.decompose, fixed_decomps={qml.CNOT: cnot_decomp})
        def circuit():
            ...

    Alternatively, use the decorator syntax:

    .. code-block:: python

        @qml.register_resources({qml.H: 2, qml.CZ: 1})
        def cnot_decomp(wires):
            qml.H(wires=wires[1])
            qml.CZ(wires=wires)
            qml.H(wires=wires[1])

        qml.add_decomposition(qml.CNOT, cnot_decomp)

    .. details::
        :title: Usage Details

        In many cases, the resource function of an operator's decomposition is not static. For
        example, consider ``MultiRZ``, whose decomposition function looks like:

        .. code-block:: python

            import pennylane as qml

            def multi_rz_decomposition(theta, wires, **__):
                for w0, w1 in zip(wires[-1:0:-1], wires[-2::-1]):
                    qml.CNOT(wires=(w0, w1))
                qml.RZ(theta, wires=wires[0])
                for w0, w1 in zip(wires[1:], wires[:-1]):
                    qml.CNOT(wires=(w0, w1))

        We notice that the number of ``CNOT`` gates required to compose a ``MultiRZ`` depends on
        the number of wires. For each operator class, we can find the set of parameters that
        affect the gate count of its decompositions in the ``resource_param_keys`` attribute:

        >>> qml.MultiRZ.resource_param_keys
        ('num_wires',)

        Any decomposition of ``MultiRZ`` is expected to depend on its number of wires, so we define
        a function that takes ``num_wires`` as an argument:

        .. code-block:: python

            def _multi_rz_resources(num_wires):
                return {
                    qml.CNOT: 2 * (num_wires - 1),
                    qml.RZ: 1
                }

        and register it with the decomposition rule:

        .. code-block:: python

            multi_rz_decomposition = qml.register_resources(
                _multi_rz_resources,
                multi_rz_decomposition
            )

        We also notice that two ``MultiRZ`` gates acting on different numbers of wires will have
        different decompositions. Consequently, when defining a decomposition rule that contains
        ``MultiRZ`` gates, we require that more information is provided.

        Consider a fictitious operator with the following decomposition:

        .. code-block:: python

            def my_decomposition(thata, wires):
                qml.MultiRZ(theta, wires=wires[:-1])
                qml.MultiRZ(theta, wires=wires)
                qml.MultiRZ(theta, wires=wires[1:])

        It contains two ``MultiRZ`` gates on ``num_wires - 1`` wires and one ``MultiRZ`` gate on
        ``num_wires`` wires, which should be reflected in its corresponding resource function:

        .. code-block:: python

            def _my_decomposition_resources(num_wires):
                return {
                    qml.resource_rep(qml.MultiRZ, num_wires=num_wires - 1): 2,
                    qml.resource_rep(qml.MultiRZ, num_wires=num_wires): 1
                }

            my_decomposition = qml.register_resources(
                _my_decomposition_resources,
                my_decomposition
            )

        where ``qml.resource_rep`` is a utility function that wraps an operator type and
        any additional information relavent to its resource estimate into a data structure.

        .. seealso::

            :func:`~pennylane.resource_rep`

    """

    if qfunc:  # enables the normal syntax
        return DecompositionRule(qfunc, resources)

    def _decorator(_qfunc) -> DecompositionRule:
        return DecompositionRule(_qfunc, resources)

    return _decorator  # enables the decorator syntax


class DecompositionRule:  # pylint: disable=too-few-public-methods
    """Represents a decomposition rule for an operator.

    Attributes:
        impl (Callable): the quantum function implementing the decomposition rule

    """

    def __init__(self, func: Callable, resources: Callable | dict = None):
        self.impl = func
        if isinstance(resources, dict):
            resource_fn = lambda: resources
            self._compute_resources = resource_fn
        else:
            self._compute_resources = resources

    def compute_resources(self, *args, **kwargs) -> Resources:
        """Computes the resources required to implement this decomposition rule."""
        if self._compute_resources is None:
            raise NotImplementedError("No resource estimation found for this decomposition rule.")
        gate_counts: dict = self._compute_resources(*args, **kwargs)
        assert isinstance(gate_counts, dict), "Resource function must return a dictionary."
        gate_counts = {_auto_wrap(op): count for op, count in gate_counts.items() if count > 0}
        num_gates = sum(gate_counts.values())
        return Resources(num_gates, gate_counts)


def _auto_wrap(op_type):
    """Conveniently wrap an operator type in a resource representation."""
    if isinstance(op_type, CompressedResourceOp):
        return op_type
    if not issubclass(op_type, Operator):
        raise TypeError(
            "The keys of the dictionary returned by the resource function must be a subclass of "
            "Operator or a CompressedResourceOp constructed with qml.resource_rep"
        )
    try:
        return resource_rep(op_type)
    except TypeError as e:
        raise TypeError(
            f"Operator {op_type.__name__} has non-empty resource_param_keys. A resource "
            f"representation must be explicitly constructed using qml.resource_rep"
        ) from e


_decompositions = defaultdict(list)


def add_decomposition(op_type, decomposition_rule: DecompositionRule) -> None:
    """Register a decomposition rule with an operator class."""
    _decompositions[op_type].append(decomposition_rule)


def get_decompositions(op_type) -> list[DecompositionRule]:
    """Get all known decomposition rules for an operator class."""
    return _decompositions[op_type][:]


def has_decomposition(op_type) -> bool:
    """Check whether an operator has decomposition rules defined."""
    return op_type in _decompositions and len(_decompositions[op_type]) > 0

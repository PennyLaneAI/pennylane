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
    """Bind a quantum function to its required resources.

    .. note::

        This function is only relevant when the new experimental graph-based decomposition system
        (introduced in v0.41) is enabled via ``qml.decompositions.enable_graph()``. This new way of
        doing decompositions is generally more performant and accomodates multiple alternative
        decomposition rules for an operator. In this new system, custom decomposition rules are
        defined as quantum functions, and it is currently required that every decomposition rule
        declares its required resources using ``qml.register_resources``

    Args:
        resources (dict or Callable): a dictionary mapping unique operators within the given
            ``qfunc`` to their number of occurrences therein. If a function is provided instead
            of a static dictionary, a dictionary must be returned from the function. For more
            information, see Usage details.
        qfunc (Callable): the quantum function that implements the decomposition.

    Returns:
        DecompositionRule:
            a data structure that represents a decomposition rule, which contains a PennyLane
            quantum function representing the decomposition, and its resource function.


    **Example**

    This function can be used as a decorator to bind a quantum function to its required resources
    so that it can be used as a decomposition rule within the new decomposition system.

    .. code-block:: python

        import pennylane as qml

        qml.decompositions.enable_graph()

        @qml.register_resources({qml.H: 2, qml.CZ: 1})
        def my_cnot(wires):
            qml.H(wires=wires[1])
            qml.CZ(wires=wires)
            qml.H(wires=wires[1])

        @partial(qml.transforms.decompose, fixed_decomps={qml.CNOT: my_cnot})
        @qml.qnode(qml.device("default.qubit"))
        def circuit():
            qml.CNOT(wires=[0, 1])
            return qml.state()


    >>> print(qml.draw(circuit, level="device")())
    0: ────╭●────┤  State
    1: ──H─╰Z──H─┤  State

    Alternatively, the decomposition rule can be created in-line:

    >>> my_cnot = qml.register_resources({qml.H: 2, qml.CZ: 1}, my_cnot)

    .. details::
        :title: Usage Details

        Quantum functions representing custom decompositions within the new decomposition system
        are expected to take ``(*op.parameters, op.wires, **op.hyperparameters)`` as arguments,
        where ``op`` is an instance of the operator type that the decomposition is for.

        In many cases, the resource requirement of an operator's decomposition is not static; some
        operators have properties that directly affect the resource estimate of its decompositions,
        i.e., the types of gates that exists in the decomposition and their number of occurrences.
        For example, the number of gates in the decomposition for ``qml.MultiRZ`` changes based
        on the number of wires it acts on.

        For each operator class, the set of parameters that affects the type of gates and their
        number of occurrences in its decompositions is given by the ``resource_keys`` attribute:

        >>> qml.CNOT.resource_keys
        {}
        >>> qml.MultiRZ.resource_keys
        {'num_wires'}

        The output of ``resource_keys`` indicates that custom decompositions for the operator
        should be registered to a resource function (as opposed to a static dictionary) that
        accepts those exact arguments and returns a dictionary.

        .. code-block:: python

            def _multi_rz_resources(num_wires):
                return {
                    qml.CNOT: 2 * (num_wires - 1),
                    qml.RZ: 1
                }

            @qml.register_resources(_multi_rz_resources)
            def multi_rz_decomposition(theta, wires, **__):
                for w0, w1 in zip(wires[-1:0:-1], wires[-2::-1]):
                    qml.CNOT(wires=(w0, w1))
                qml.RZ(theta, wires=wires[0])
                for w0, w1 in zip(wires[1:], wires[:-1]):
                    qml.CNOT(wires=(w0, w1))

        Additionally, if a custom decomposition for an operator contains gates that, in turn,
        have properties that affect their own decompositions, each unique instance must be
        counted separately in the resources function.

        Consider a fictitious operator with the following decomposition:

        .. code-block:: python

            def my_decomp(thata, wires):
                qml.MultiRZ(theta, wires=wires[:-1])
                qml.MultiRZ(theta, wires=wires)
                qml.MultiRZ(theta, wires=wires[1:])

        It contains two ``MultiRZ`` gates acting on ``num_wires - 1`` wires (the first and last
        ``MultiRZ``) and one ``MultiRZ`` gate acting on exactly ``num_wires`` wires. This
        distinction must be reflected in the resource function:

        .. code-block:: python

            def my_resources(num_wires):
                return {
                    qml.resource_rep(qml.MultiRZ, num_wires=num_wires - 1): 2,
                    qml.resource_rep(qml.MultiRZ, num_wires=num_wires): 1
                }

            my_decomp = qml.register_resources(my_resources, my_decomp)

        where ``qml.resource_rep`` is a utility function that wraps an operator type and any
        additional information relevant to its resource estimate into a compressed data structure.

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

    def __call__(self, *args, **kwargs):
        return self.impl(*args, **kwargs)

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


def add_decomps(op_type, decomps: DecompositionRule | list[DecompositionRule]) -> None:
    """Register a list of decomposition rules with an operator class."""
    if isinstance(decomps, list) and all(isinstance(d, DecompositionRule) for d in decomps):
        _decompositions[op_type].extend(decomps)
    elif isinstance(decomps, DecompositionRule):
        _decompositions[op_type].append(decomps)
    else:
        raise TypeError(
            "A decomposition rule must be a qfunc with a resource estimate "
            "registered using qml.register_resources"
        )


def list_decomps(op_type) -> list[DecompositionRule]:
    """Lists all known decomposition rules for an operator class."""
    return _decompositions[op_type][:]


def has_decomp(op_type) -> bool:
    """Check whether an operator has decomposition rules defined."""
    return op_type in _decompositions and len(_decompositions[op_type]) > 0

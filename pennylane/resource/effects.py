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
"""Declared resource effects of compilation passes.

A resource effect describes how a compilation pass changes the resources of a program, so that
:func:`~.estimate` can report the resources after the pass without applying it. Effects act on
the resources of each function, as reported by Catalyst's resource analysis, before the
functions are combined over the call graph.

.. warning::

    This module is a prototype and may change or be removed without deprecation.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# Resources of one function, in the JSON format of Catalyst's resource analysis
FunctionData = dict[str, Any]


@dataclass(frozen=True)
class Op:
    """An operator as counted by the resource analysis.

    Args:
        name (str): Name of the operator, e.g. ``"Hadamard"`` or ``"PPR-pi/8"``.
        wires (int): Number of wires it acts on, or ``-1`` if unknown.
    """

    name: str
    wires: int


@dataclass(frozen=True)
class Wires:
    """Effect of an operator on the qubits of the function it is in.

    Args:
        alloc (float): Number of qubits allocated, added to the total number of allocations.
        peak (float): Number of extra qubits live at once while the operator is applied, which
            may raise the peak number of live qubits of the function.
    """

    alloc: float = 0
    peak: float = 0


class FunctionResources:
    """The resources of a function, which a resource effect reads and updates in place.

    Args:
        data (dict): The resources of the function in the JSON format of Catalyst's resource
            analysis.
    """

    def __init__(self, data: FunctionData):
        self.data = data

    @property
    def ops(self) -> dict[Op, float]:
        """The number of each operator in the function, excluding its callees."""
        return {
            Op(name, int(wires)): count
            for wires, ops in self.data["quantum_operations"].items()
            for name, count in ops.items()
        }

    @ops.setter
    def ops(self, ops: dict[Op, float]):
        grouped = {}
        for op, count in ops.items():
            if count:
                grouped.setdefault(str(op.wires), {})[op.name] = count
        self.data["quantum_operations"] = grouped

    def add_allocations(self, num_qubits: float):
        """Add ``num_qubits`` allocated qubits."""
        self.data["num_qubits"]["alloc"] += num_qubits
        self.data["num_qubits"]["total"] += num_qubits

    def add_peak(self, num_qubits: float):
        """Raise the peak number of live qubits by ``num_qubits``."""
        if "peak" in self.data["num_qubits"]:
            self.data["num_qubits"]["peak"] += num_qubits


# A resource effect updates the resources of a function in place, for the given pass options
ResourceEffect = Callable[[FunctionResources, dict], None]


def identity(function: FunctionResources, options: dict) -> None:
    """The resource effect of a pass that does not change the resources."""
    del function, options


# What a substitution rule returns for an operator: nothing to keep it, or what replaces it,
# optionally with its effect on the qubits
SubstitutionResult = None | dict[Op, float] | tuple[dict[Op, float], Wires]


def substitution_effect(rule: Callable[[Op, dict], SubstitutionResult]) -> ResourceEffect:
    """The resource effect of a pass that replaces each operator independently of the others.

    Args:
        rule (Callable[[Op, dict], SubstitutionResult]): Returns what replaces an operator for
            the given pass options: ``None`` to keep the operator, a mapping from operators to
            their expected numbers, or such a mapping with the :class:`Wires` effect of the
            replacement. It is called once per distinct operator of each function.
    """

    def effect(function: FunctionResources, options: dict) -> None:
        new_ops = {}
        extra_peak = 0
        for op, count in function.ops.items():
            replacement = rule(op, options)
            if replacement is None:
                new_ops[op] = new_ops.get(op, 0) + count
                continue
            replacement, wires = (
                replacement if isinstance(replacement, tuple) else (replacement, Wires())
            )
            for new_op, new_count in replacement.items():
                new_ops[new_op] = new_ops.get(new_op, 0) + count * new_count
            function.add_allocations(wires.alloc * count)
            # The extra qubits of an operator are released after it, so they do not add up
            if count:
                extra_peak = max(extra_peak, wires.peak)
        function.ops = new_ops
        function.add_peak(extra_peak)

    return effect


_EFFECTS: dict[str, ResourceEffect] = {}


def register_effect(pass_name: str, effect: ResourceEffect) -> None:
    """Register the resource effect of the compilation pass ``pass_name``."""
    _EFFECTS[pass_name] = effect


def substitution(pass_name: str):
    """Decorator registering a substitution rule as the resource effect of ``pass_name``. See
    :func:`substitution_effect`."""

    def decorator(rule):
        register_effect(pass_name, substitution_effect(rule))
        return rule

    return decorator


def transfer(pass_name: str):
    """Decorator registering a function that updates the resources of each function in place as
    the resource effect of ``pass_name``."""

    def decorator(effect):
        register_effect(pass_name, effect)
        return effect

    return decorator


def get_effect(pass_name: str) -> ResourceEffect | None:
    """The resource effect registered for ``pass_name``, if any."""
    return _EFFECTS.get(pass_name)


def apply_effect(data: dict[str, FunctionData], effect: ResourceEffect, options: dict) -> dict:
    """Return the resources of all functions after a pass with the given effect and options.

    Args:
        data (dict[str, dict]): The resources of each function, in the JSON format of Catalyst's
            resource analysis. It is not modified.
        effect (ResourceEffect): The effect of the pass.
        options (dict): The options of the pass.
    """
    data = copy.deepcopy(data)
    for function in data.values():
        effect(FunctionResources(function), options)
    return data

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

import pennylane as qml

from .resources import CompressedResourceOp, Resources


def decomposition(qfunc: Callable) -> DecompositionRule:
    """Decorator that wraps a qfunc in a ``DecompositionRule``."""
    return DecompositionRule(qfunc)


class DecompositionRule:
    """Represents a decomposition rule for an operator."""

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


class ControlledDecompositionRule(DecompositionRule):
    """A decomposition rule for a controlled operation with a decomposition."""

    def __init__(self, base_decomposition: DecompositionRule):
        self._base_decomposition = base_decomposition
        super().__init__(self._impl)

    def compute_resources(
        self, base_params, num_control_wires, num_zero_control_values, num_work_wires
    ) -> Resources:
        base_resource_decomp = self._base_decomposition.compute_resources(**base_params)
        controlled_resources = {
            CompressedResourceOp(
                qml.ops.Controlled,
                {
                    "base_class": base_op_compressed.op_type,
                    "base_params": base_op_compressed.params,
                    "num_control_wires": num_control_wires,
                    "num_zero_control_values": num_zero_control_values,
                    "num_work_wires": num_work_wires,
                },
            ): count
            for base_op_compressed, count in base_resource_decomp.gate_counts.items()
            if count > 0
        }
        gate_count = sum(controlled_resources.values())
        return Resources(gate_count, controlled_resources)

    def _impl(self, base, control_wires, control_values, work_wires, **__):
        """The default implementation of a controlled decomposition."""

        for w, val in zip(control_wires, control_values):
            if not val:
                qml.PauliX(w)
        qml.ctrl(
            self._base_decomposition.impl,
            control=control_wires,
            control_values=control_values,
            work_wires=work_wires,
        )(*base.params, wires=base.wires, **base.hyperparameters)
        for w, val in zip(control_wires, control_values):
            if not val:
                qml.PauliX(w)

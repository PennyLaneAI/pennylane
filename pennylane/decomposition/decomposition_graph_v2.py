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

"""Implements the class DecompositionGraph

This module implements a graph-based decomposition algorithm that constructs a graph of operators
connected by decomposition rules, and then traverses it using Dijkstra's algorithm to find the best
decomposition for every operator.

The architecture of this module utilizes design patterns similar to those present in Qiskit's
implementation of the basis translator, the Boost Graph library, and RustworkX.

"""

from __future__ import annotations

from dataclasses import dataclass

from pennylane.decomposition import CompressedResourceOp, DecompositionRule, Resources, WorkWireSpec


@dataclass(frozen=True)
class _OpNode:
    """A node that represents an operator."""

    op: CompressedResourceOp
    """The resource rep of the operator."""

    num_work_wires: int
    """The number of work wires available to decompose this operator."""


@dataclass(frozen=True)
class _DecompNode:
    """A node that represents a decomposition rule."""

    rule: DecompositionRule
    resource: Resources
    work_wire_spec: WorkWireSpec

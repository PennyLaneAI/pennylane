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
"""Shared ``Operator2`` subclasses used across the ``tests/core/`` suite."""

# pylint: disable=too-few-public-methods

from pennylane.core.operator import Operator2
from pennylane.wires import Wires


class DynOp(Operator2):
    """Operator with one dynamic parameter and wires."""

    dynamic_argnames = ("phi",)

    def __init__(self, phi, wires):
        super().__init__(phi, wires=wires)


class TwoDynOp(Operator2):
    """Operator with two dynamic parameters."""

    dynamic_argnames = ("phi", "theta")

    def __init__(self, phi, theta, wires):
        super().__init__(phi, theta, wires=wires)


class StaticOp(Operator2):
    """Operator with a static argument."""

    static_argnames = ("label",)

    def __init__(self, label, wires):
        super().__init__(label, wires=wires)


class CompOp(Operator2):
    """Operator with a compilable static argument."""

    compilable_argnames = ("n",)

    def __init__(self, n, wires):
        super().__init__(n, wires=wires)


class MultiWireOp(Operator2):
    """Operator with two wire arguments."""

    wire_argnames = ("wires", "ctrl_wires")

    def __init__(self, wires, ctrl_wires):
        super().__init__(wires=wires, ctrl_wires=ctrl_wires)


class HybridOp(Operator2):
    """Operator with a hybrid argument that can contain Operator2 leaves."""

    hybrid_argnames = ("ops",)

    def __init__(self, ops, wires):
        super().__init__(ops, wires=wires)


class HybridWireOp(Operator2):
    """Operator with a wire argument that is also a hybrid argument."""

    wire_argnames = ("pytree_wires",)
    hybrid_argnames = ("pytree_wires",)

    def __init__(self, pytree_wires):
        super().__init__([Wires(w) for w in pytree_wires])


class FullOp(Operator2):
    """Operator using dynamic, static, and hybrid argument groups."""

    dynamic_argnames = ("phi",)
    static_argnames = ("static",)
    hybrid_argnames = ("hybrid",)

    def __init__(self, phi, static, hybrid, wires):
        super().__init__(phi, static, hybrid, wires=wires)

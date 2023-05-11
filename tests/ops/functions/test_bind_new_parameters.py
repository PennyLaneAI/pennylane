# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This module contains unit tests for ``qml.bind_parameters``.
"""

import pytest

import pennylane as qml
from pennylane.ops.functions import bind_new_parameters


class TestCompositeOps:
    """Tests for using `bind_new_parameters` with `CompositeOp`."""

    def test_vanilla_operands(self, operands, new_params):
        """Test that `bind_new_parameters` with `CompositeOp` that only has
        vanilla operands binds the new parameters without mutating the original
        operator."""

    def test_nested_composite_ops(self, operands, new_params):
        """Test that `bind_new_parameters` with `CompositeOp` where the operands
        are also `CompositeOp`'s binds the new parameters without mutating the
        original operator."""

    def test_nested_symbolic_ops(self, operands, new_params):
        """Test that `bind_new_parameters` with `CompositeOp` where the operands
        are `SymbolicOp`'s binds the new parameters without mutating the original
        operator."""

    def test_nested_mixed_ops(self, operands, new_params):
        """Test that `bind_new_parameters` with `CompositeOp` where the operands
        are of mixed types binds the new parameters without mutating the original
        operator."""


class TestScalarSymbolicOps:
    """Tests for using `bind_new_parameters` with `ScalarSymbolicOp`."""

    def test_vanilla_operands(self, operands, new_params):
        """Test that `bind_new_parameters` with `ScalarSymbolicOp` that only has
        a vanilla operator base binds the new parameters without mutating the
        original operator."""

    def test_nested_composite_ops(self, operands, new_params):
        """Test that `bind_new_parameters` with `ScalarSymbolicOp` where the base
        operator is a `CompositeOp`'s binds the new parameters without mutating the
        original operator."""

    def test_nested_symbolic_ops(self, operands, new_params):
        """Test that `bind_new_parameters` with `ScalarSymbolicOp` where the base
        operator is a `SymbolicOp`'s binds the new parameters without mutating the
        original operator."""


class TestSymbolicOps:
    """Tests for using `bind_new_parameters` with `SymbolicOp`."""

    def test_vanilla_operands(self, operands, new_params):
        """Test that `bind_new_parameters` with `SymbolicOp` that only has
        a vanilla operator base binds the new parameters without mutating the
        original operator."""

    def test_nested_composite_ops(self, operands, new_params):
        """Test that `bind_new_parameters` with `SymbolicOp` where the base
        operator is a `CompositeOp`'s binds the new parameters without mutating the
        original operator."""

    def test_nested_symbolic_ops(self, operands, new_params):
        """Test that `bind_new_parameters` with `SymbolicOp` where the base
        operator is a `SymbolicOp`'s binds the new parameters without mutating the
        original operator."""


class TestHamiltonian:
    """Tests for using `bind_new_parameters` with `Hamiltonian`."""
    pass


class TestTensor:
    """Tests for using `bind_new_parameters` with `Tensor`."""
    pass


class TestVanillaOperators:
    """Tests for using `bind_new_parameters` with PennyLane vanilla operators."""
    pass

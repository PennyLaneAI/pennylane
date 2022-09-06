# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
Unit tests for the composite operator class of qubit operations
"""
from copy import copy
import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.ops.op_math import CompositeOp


class OpMissingProperties(CompositeOp):
    @property
    def is_hermitian(self):
        return False

    def matrix(self, wire_order=None):
        return np.eye(4)

    def sparse_matrix(self, wire_order=None):
        return np.eye(4)


class OpMissingSymbol(OpMissingProperties):
    _name = "NoSymbol"


class ValidOp(OpMissingProperties):
    _name = "ValidOp"
    _op_symbol = "#"


class TestConstruction:
    """Test the construction of composite ops."""

    base = (qml.S(0), qml.T(1))

    def test_direct_initialization_fails(self):
        with pytest.raises(
            TypeError, match="Can't instantiate abstract class CompositeOp with abstract methods"
        ):
            op = CompositeOp(*self.base)

    def test_class_missing_properties_fails(self):
        with pytest.raises(NotImplementedError, match="Child class must specify _name"):
            op = OpMissingProperties(*self.base)

    def test_class_missing_symbol_fails(self):
        with pytest.raises(NotImplementedError, match="Child class must specify _op_symbol"):
            op = OpMissingSymbol(*self.base)

    def test_raise_error_fewer_than_2_operands(self):
        """Test that initializing a composite operator with less than 2 operands raises a ValueError."""
        with pytest.raises(ValueError, match="Require at least two operators to combine;"):
            ValidOp(qml.PauliX(0))

    def test_initialization(self):
        op = ValidOp(*self.base)
        assert op.name == "ValidOp"
        assert op.op_symbol == "#"

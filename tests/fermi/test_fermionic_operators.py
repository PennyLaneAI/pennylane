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
"""Unit Tests for the Fermionic representation classes."""
import pickle
from copy import copy, deepcopy

import pytest

from pennylane.fermi.fermionic import FermiWord, FermiC, FermiA

# pylint: disable=too-few-public-methods


class TestFermiC:
    """Test the methods of the creation operator FermiC"""

    @pytest.mark.parametrize("wire", [1, 3])
    def test_initialization(self, wire):
        """Test __init__ function returns the expected FermiWord object"""
        op = FermiC(wire)
        op_dict = {(0, wire): "+"}

        assert isinstance(op, FermiWord)
        assert len(op) == 1
        assert list(op.keys()) == [(0, wire)]
        assert list(op.values()) == ["+"]

        assert op.sorted_dic == op_dict

    @pytest.mark.parametrize("wire", ["a", -2, [0, 1], 1.2])
    def test_bad_wire_raises_error(self, wire):
        """Test that passing a value for wire that is not a positive integer raises an error"""
        with pytest.raises(ValueError, match="expected a single, positive integer value for wire"):
            _ = FermiC(wire)


class TestFermiA:
    """Test the methods of the annihilation operator FermiA"""

    @pytest.mark.parametrize("wire", [1, 3])
    def test_initialization(self, wire):
        """Test __init__ function returns the expected FermiWord object"""
        op = FermiA(wire)
        op_dict = {(0, wire): "-"}

        assert isinstance(op, FermiWord)
        assert len(op) == 1
        assert list(op.keys()) == [(0, wire)]
        assert list(op.values()) == ["-"]

        assert op.sorted_dic == op_dict

    @pytest.mark.parametrize("wire", ["a", -2, [0, 1], 1.2])
    def test_bad_wire_raises_error(self, wire):
        """Test that passing a value for wire that is not a positive integer raises an error"""
        with pytest.raises(ValueError, match="expected a single, positive integer value for wire"):
            _ = FermiA(wire)

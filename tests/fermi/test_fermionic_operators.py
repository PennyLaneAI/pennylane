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

import pytest

from pennylane.fermi.fermionic import FermiA, FermiC, FermiWord

# pylint: disable=too-few-public-methods


class TestFermiC:
    """Test the methods of the creation operator FermiC"""

    @pytest.mark.parametrize("orbital", [1, 3])
    def test_initialization(self, orbital):
        """Test __init__ function returns the expected FermiWord object"""
        op = FermiC(orbital)
        op_dict = {(0, orbital): "+"}

        assert isinstance(op, FermiWord)
        assert len(op) == 1
        assert list(op.keys()) == [(0, orbital)]
        assert list(op.values()) == ["+"]

        assert dict(op) == op_dict

    @pytest.mark.parametrize("orbital", ["a", -2, [0, 1], 1.2])
    def test_bad_orbital_raises_error(self, orbital):
        """Test that passing a value for orbital that is not a positive integer raises an error"""
        with pytest.raises(
            ValueError, match="expected a single, positive integer value for orbital"
        ):
            _ = FermiC(orbital)


class TestFermiA:
    """Test the methods of the annihilation operator FermiA"""

    @pytest.mark.parametrize("orbital", [1, 3])
    def test_initialization(self, orbital):
        """Test __init__ function returns the expected FermiWord object"""
        op = FermiA(orbital)
        op_dict = {(0, orbital): "-"}

        assert isinstance(op, FermiWord)
        assert len(op) == 1
        assert list(op.keys()) == [(0, orbital)]
        assert list(op.values()) == ["-"]

        assert dict(op) == op_dict

    @pytest.mark.parametrize("orbital", ["a", -2, [0, 1], 1.2])
    def test_bad_orbital_raises_error(self, orbital):
        """Test that passing a value for orbital that is not a positive integer raises an error"""
        with pytest.raises(
            ValueError, match="expected a single, positive integer value for orbital"
        ):
            _ = FermiA(orbital)

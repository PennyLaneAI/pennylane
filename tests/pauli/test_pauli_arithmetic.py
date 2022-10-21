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
"""Unit Tests for the PauliWord and PauliSentence classes"""
import pytest

# import numpy as np
# from scipy import sparse
from pennylane.pauli.pauli_arithmetic import PauliWord, PauliSentence, I, X, Y, Z

pw1 = PauliWord({0: I, 1: X, 2: Y})
pw2 = PauliWord({"a": X, "b": X, "c": Z})
pw3 = PauliWord({0: Z, "b": Z})
pw4 = PauliWord({})


class TestPauliWord:
    def test_missing(self):
        """Test the result when a missing key is indexed"""
        pw = PauliWord({0: I, 1: X, 2: Y})
        assert 3 not in pw.keys()
        assert pw[3] == I

    def test_set_items(self):
        """Test that setting items raises an error"""
        pw = PauliWord({0: I, 1: X, 2: Y})
        with pytest.raises(NotImplementedError):
            pw[3] = Z  # trying to add to a pw after instantiation is prohibited

    def test_hash(self):
        """Test that a unique hash exists for different PauliWords."""
        pw1 = PauliWord({0: I, 1: X, 2: Y})
        pw2 = PauliWord({0: I, 1: X, 2: Y})  # same as 1
        pw3 = PauliWord({1: X, 2: Y, 0: I})  # same as 1 but reordered
        pw4 = PauliWord({1: Z, 2: Z})  # distinct from above

        assert pw1.__hash__() == pw2.__hash__()
        assert pw1.__hash__() == pw3.__hash__()
        assert pw1.__hash__() != pw4.__hash__()

    tup_pws_wires = ((pw1, {0, 1, 2}), (pw2, {"a", "b", "c"}), (pw3, {0, "b"}), (pw4, set()))

    @pytest.mark.parametrize("pw, wires", tup_pws_wires)
    def test_wires(self, pw, wires):
        """Test that the wires are tracked correctly."""
        assert pw.wires == wires

    def test_mul(self):
        assert True

    def test_to_mat(self):
        assert True

    def test_to_mat_wire_order(self):
        assert True

    def test_to_mat_sparse(self):
        assert True


class TestPauliSentence:
    def test_missing(self):
        """Test the result when a missing key is indexed"""
        pw = PauliWord({0: X})
        new_pw = PauliWord({"a": Z})
        ps = PauliSentence({pw: 1.0})

        assert new_pw not in ps.keys()
        assert ps[new_pw] == 0.0

    def test_set_items(self):
        """Test that we can add to a PauliSentence"""
        pw = PauliWord({0: X})
        ps = PauliSentence({pw: 1.0})

        new_pw = {"a": Z}
        assert new_pw not in ps.keys()

        ps[new_pw] = 3.45
        assert new_pw in ps.keys() and ps[new_pw] == 3.45

    def test_str(self):
        assert True

    def test_wires(self):
        assert True

    def test_mul(self):
        assert True

    def test_add(self):
        assert True

    def test_to_mat(self):
        assert True

    def test_simplify(self):
        assert True

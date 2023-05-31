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

from pennylane.fermi.fermionic import FermiWord

fw1 = FermiWord({(0, 0): "+", (1, 1): "-"})
fw2 = FermiWord({(0, 0): "+", (1, 0): "-"})
fw3 = FermiWord({(0, 0): "+", (1, 3): "-", (2, 0): "+", (3, 4): "-"})
fw4 = FermiWord({})


class TestFermiWord:
    def test_missing(self):
        """Test that empty string is returned for missing key."""
        fw = FermiWord({(0, 0): "+", (1, 1): "-"})
        assert (2, 3) not in fw.keys()
        assert fw[(2, 3)] == ""

    def test_set_items(self):
        """Test that setting items raises an error"""
        fw = FermiWord({(0, 0): "+", (1, 1): "-"})
        with pytest.raises(TypeError, match="FermiWord object does not support assignment"):
            fw[(2, 2)] = "+"

    def test_update_items(self):
        """Test that updating items raises an error"""
        fw = FermiWord({(0, 0): "+", (1, 1): "-"})
        with pytest.raises(TypeError, match="FermiWord object does not support assignment"):
            fw.update({(2, 2): "+"})

    def test_hash(self):
        """Test that a unique hash exists for different FermiWords."""
        fw_1 = FermiWord({(0, 0): "+", (1, 1): "-"})
        fw_2 = FermiWord({(0, 0): "+", (1, 1): "-"})  # same as 1
        fw_3 = FermiWord({(1, 1): "-", (0, 0): "+"})  # same as 1 but reordered
        fw_4 = FermiWord({(0, 0): "+", (2, 2): "-"})  # distinct from above

        assert fw_1.__hash__() == fw_2.__hash__()
        assert fw_1.__hash__() == fw_3.__hash__()
        assert fw_1.__hash__() != fw_4.__hash__()

    @pytest.mark.parametrize("fw", (fw1, fw2, fw3, fw4))
    def test_copy(self, fw):
        """Test that the copy is identical to the original."""
        copy_fw = copy(fw)
        deep_copy_fw = deepcopy(fw)

        assert copy_fw == fw
        assert deep_copy_fw == fw
        assert copy_fw is not fw
        assert deep_copy_fw is not fw

    tup_fws_wires = ((fw1, [0, 1]), (fw2, [0, 0]), (fw3, [0, 3, 0, 4]), (fw4, []))

    @pytest.mark.parametrize("fw, wires", tup_fws_wires)
    def test_wires(self, fw, wires):
        """Test that the wires are tracked correctly."""
        assert fw.wires == wires

    tup_fw_compact = (
        (fw1, "0+ 1-"),
        (fw2, "0+ 0-"),
        (fw3, "0+ 3- 0+ 4-"),
        (fw4, ""),
    )

    @pytest.mark.parametrize("fw, str_rep", tup_fw_compact)
    def test_compact(self, fw, str_rep):
        assert fw.to_string() == str_rep

    tup_fw_str = (
        (fw1, "<FermiWord = '0+ 1-'>"),
        (fw2, "<FermiWord = '0+ 0-'>"),
        (fw3, "<FermiWord = '0+ 3- 0+ 4-'>"),
        (fw4, "<FermiWord = ''>"),
    )

    @pytest.mark.parametrize("fw, str_rep", tup_fw_str)
    def test_str(self, fw, str_rep):
        assert str(fw) == str_rep
        assert repr(fw) == str_rep

    tup_fw_mult = (
        (fw1, fw1, FermiWord({(0, 0): "+", (1, 1): "-", (2, 0): "+", (3, 1): "-"})),
        (
            fw1,
            fw3,
            FermiWord(
                {(0, 0): "+", (1, 1): "-", (2, 0): "+", (3, 3): "-", (4, 0): "+", (5, 4): "-"}
            ),
        ),
        (fw2, fw1, FermiWord({(0, 0): "+", (1, 0): "-", (2, 0): "+", (3, 1): "-"})),
        (fw1, fw4, fw1),
        (fw4, fw3, fw3),
        (fw4, fw4, fw4),
    )

    @pytest.mark.parametrize("f1, f2, result_fw", tup_fw_mult)
    def test_mul(self, f1, f2, result_fw):
        assert f1 * f2 == result_fw

    tup_fw_mult_error = (
        (fw1, [1.5]),
        (fw4, "string"),
    )

    @pytest.mark.parametrize("f1, f2", tup_fw_mult_error)
    def test_mul_error(self, f1, f2):
        with pytest.raises(TypeError, match=f"Cannot multiply FermiWord by {type(f2)}."):
            f1 * f2  # pylint: disable=pointless-statement

    tup_fw_pow = (
        (fw1, 0, FermiWord({})),
        (fw1, 1, FermiWord({(0, 0): "+", (1, 1): "-"})),
        (fw1, 2, FermiWord({(0, 0): "+", (1, 1): "-", (2, 0): "+", (3, 1): "-"})),
        (
            fw2,
            3,
            FermiWord(
                {(0, 0): "+", (1, 0): "-", (2, 0): "+", (3, 0): "-", (4, 0): "+", (5, 0): "-"}
            ),
        ),
    )

    @pytest.mark.parametrize("f1, pow, result_fw", tup_fw_pow)
    def test_pow(self, f1, pow, result_fw):
        assert f1**pow == result_fw

    tup_fw_pow_error = ((fw1, -1), (fw3, 1.5))

    @pytest.mark.parametrize("f1, pow", tup_fw_pow_error)
    def test_pow_error(self, f1, pow):
        with pytest.raises(ValueError, match="The exponent must be a positive integer."):
            f1**pow  # pylint: disable=pointless-statement

    def test_pickling(self):
        """Check that FermiWords can be pickled and unpickled."""
        fw = FermiWord({(0, 0): "+", (1, 1): "-"})
        serialization = pickle.dumps(fw)
        new_fw = pickle.loads(serialization)
        assert fw == new_fw

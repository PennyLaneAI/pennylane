# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
Unit tests for pennylane.registers.
"""

import pytest

import pennylane as qp
from pennylane.core.wires import Wires


class TestRegisters:
    @pytest.mark.parametrize(
        "wire_dict, expected_register",
        [
            (
                {"alice": 3, "bob": {"nest1": 3, "nest2": 3}, "cleo": 3},
                {
                    "alice": Wires([0, 1, 2]),
                    "nest1": Wires([3, 4, 5]),
                    "nest2": Wires([6, 7, 8]),
                    "bob": Wires([3, 4, 5, 6, 7, 8]),
                    "cleo": Wires([9, 10, 11]),
                },
            ),
            (
                {"alice": 3, "bob": {"nest1": 3}, "cleo": 3},
                {
                    "alice": Wires([0, 1, 2]),
                    "nest1": Wires([3, 4, 5]),
                    "bob": Wires([3, 4, 5]),
                    "cleo": Wires([6, 7, 8]),
                },
            ),
            (
                {"alice": 3, "bob": {"nest1": {"nest2": {"nest3": 1}}}, "cleo": 3},
                {
                    "alice": Wires([0, 1, 2]),
                    "nest3": Wires([3]),
                    "nest2": Wires([3]),
                    "nest1": Wires([3]),
                    "bob": Wires([3]),
                    "cleo": Wires([4, 5, 6]),
                },
            ),
            (
                {
                    "alice": 3,
                    "bob": {"nest1": {"nest2": {"nest3": 1}, "nest2a": 2}, "nest1a": 3},
                    "cleo": 3,
                },
                {
                    "alice": Wires([0, 1, 2]),
                    "nest3": Wires([3]),
                    "nest2": Wires([3]),
                    "nest2a": Wires([4, 5]),
                    "nest1": Wires([3, 4, 5]),
                    "nest1a": Wires([6, 7, 8]),
                    "bob": Wires([3, 4, 5, 6, 7, 8]),
                    "cleo": Wires([9, 10, 11]),
                },
            ),
        ],
    )
    def test_build_registers(self, wire_dict, expected_register):
        """Test that the registers function returns the correct dictionary of Wires instances.
        The expected result is a dictionary that has elements ordered first by appearance from left
        to right, then by nestedness (also known as DFS traversal order)."""

        wire_dict = qp.registers(wire_dict)

        assert wire_dict == expected_register

    @pytest.mark.parametrize(
        "wire_dict, expected_error_msg",
        [
            (
                {"alice": 3, "bob": {"nest1": 3, "nest2": {}}, "cleo": 3},
                "Got an empty dictionary",
            ),
            (
                {"alice": 3, "bob": {"nest1": {"nest2": {"nest3": {1}}}}, "cleo": 3},
                r"Expected '\{1\}' to be either a dict or an int",
            ),
        ],
    )
    def test_errors_for_registers(self, wire_dict, expected_error_msg):
        """Test that the registers function raises the right error for given Wires dictionaries"""
        with pytest.raises(ValueError, match=expected_error_msg):
            qp.registers(wire_dict)

    def test_with_empty_register(self):
        """Test that empty wire registers are allowed and processed correctly."""

        out_0 = qp.registers({"a": 2, "b": 0})
        assert out_0 == {"a": Wires([0, 1]), "b": Wires([])}

        out_1 = qp.registers({"a": 2, "b": {"first b": 0, "second b": 4}})
        assert out_1 == {
            "a": Wires([0, 1]),
            "b": Wires([2, 3, 4, 5]),
            "first b": Wires([]),
            "second b": Wires([2, 3, 4, 5]),
        }

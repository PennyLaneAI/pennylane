# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit Tests for the Boseonic representation classes."""
import pickle
from copy import copy, deepcopy

import numpy as np
import pytest
from scipy import sparse

import pennylane as qml
from pennylane import numpy as pnp
from pennylane.labs.vibrational.bosonic import (
    BoseWord,
    BoseSentence,
)


class TestBoseWord:
    # Expected bose sentences were computed manually or with openfermion
    @pytest.mark.parametrize(
        ("bose_sentence", "expected"),
        [
            (
                BoseSentence(
                    {
                        BoseWord({(0, 0): "+", (1, 0): "+"}): 5.051e-06,
                        BoseWord({(0, 0): "+", (1, 0): "-"}): 5.051e-06,
                        BoseWord({(0, 0): "-", (1, 0): "+"}): 5.051e-06,
                        BoseWord({(0, 0): "-", (1, 0): "-"}): 5.051e-06,
                    }
                ),
                BoseSentence(
                    {
                        BoseWord({(0, 0): "+", (1, 0): "+"}): 5.051e-06,
                        BoseWord({(0, 0): "+", (1, 0): "-"}): 1.0102e-05,
                        BoseWord({}): 5.051e-06,
                        BoseWord({(0, 0): "-", (1, 0): "-"}): 5.051e-06,
                    }
                ),
            ),
            (
                BoseWord(
                    {(0, 0): "-", (1, 1): "-", (2, 0): "+", (3, 1): "+", (4, 0): "+", (5, 2): "+"}
                ),
                BoseSentence(
                    {
                        BoseWord({(0, 0): "+", (1, 2): "+"}): 2.0,
                        BoseWord({(0, 0): "+", (1, 1): "+", (2, 2): "+", (3, 1): "-"}): 2.0,
                        BoseWord({(0, 0): "+", (1, 0): "+", (2, 2): "+", (3, 0): "-"}): 1.0,
                        BoseWord(
                            {
                                (0, 0): "+",
                                (1, 0): "+",
                                (2, 1): "+",
                                (3, 2): "+",
                                (4, 0): "-",
                                (5, 1): "-",
                            }
                        ): 1.0,
                    }
                ),
            ),
        ],
    )
    def test_normal_order(self, bose_sentence, expected):
        assert bose_sentence.normal_order() == expected

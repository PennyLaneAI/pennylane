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
"""Tests for pennylane/dla/lie_closure.py functionality"""
import pytest
import numpy as np
from pennylane.dla import VSpace
from pennylane.pauli import PauliWord, PauliSentence

ops1 = [
    PauliSentence({
        PauliWord({0:"X", 1:"X"}) : 1.,
        PauliWord({0:"Y", 1:"Y"}) : 1.
    }),
    PauliSentence({
        PauliWord({0:"X", 1:"X"}) : 1.,
    }),
    PauliSentence({
        PauliWord({0:"Y", 1:"Y"}) : 1.,
    }),
]

class TestVSpace:
    """Unit and integration tests for VSpace class"""

    def test_init(self):
        """Unit tests for initialization"""
        vspace = VSpace(ops1)
        assert all(isinstance(op, PauliSentence) for op in vspace.basis)
        assert np.allclose(vspace.M, [[1., 1.], [1., 0.]])
        assert vspace.basis == ops1[:-1]
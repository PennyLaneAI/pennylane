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
"""Tests for multi_dispatch keyword-argument handling (GH-9140)."""

import numpy as onp
import pytest

from pennylane import math as fn


def test_stack_keyword_values():
    """stack(values=[...]) must not raise IndexError and must match positional stack."""
    tensor1 = onp.array([[1, 2], [3, 4]])
    tensor2 = onp.array([[5, 6], [7, 8]])

    result_kw = fn.stack(values=[tensor1, tensor2], axis=0)
    result_pos = fn.stack([tensor1, tensor2], axis=0)

    assert fn.allequal(result_kw, result_pos)
    assert fn.allequal(result_kw, onp.stack([tensor1, tensor2], axis=0))


def test_stack_keyword_empty_no_index_error():
    """Empty values= list should fail like NumPy, not with IndexError from multi_dispatch."""
    with pytest.raises(ValueError, match="need at least one array to stack"):
        fn.stack(values=[], axis=0)

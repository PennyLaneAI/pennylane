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

"""Tests for the decomposition rules for MultiX"""

import pytest

from pennylane.labs.transforms.multix import MultiX
from pennylane.ops.functions.assert_valid import _check_decomposition_new


@pytest.mark.parametrize("n", [2, 3, 5])
def test_valid_decomp(n):
    """Test that the decomposition rule from make_selectpaulirot_to_phase_gradient_decomp works as expected
    as a fixed decomposition and yields the correct resources"""
    wires = range(n)
    op = MultiX(wires)
    _check_decomposition_new(op)

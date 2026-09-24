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
"""
Tests for qp.decompositions.all_decomps
"""

import pennylane as qp
from pennylane.decomposition import all_decomps
from pennylane.typing import Float, Wire


def test_skip_ops():
    """Test that skip_ops will prevent that op and all children from being added."""

    out = all_decomps(qp.Y(Wire[1]), skip_ops={qp.Y(Wire[1])})
    assert len(out) == 0


def test_concrete_input_abstractified():
    """Test that if a concrete input is provided, it is abstractified."""

    out = all_decomps(qp.RX(0.5, 0))
    assert qp.RX(Float, Wire[1]) in out

    assert all(op.is_fully_abstract for op in out)

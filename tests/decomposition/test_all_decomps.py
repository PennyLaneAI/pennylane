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


def test_skip_ops_root():
    """Test that skip_ops will prevent that op and all children from being added."""

    out = all_decomps(qp.Y(Wire[1]), skip_ops={qp.Y(Wire[1])})
    assert len(out) == 0


def test_skip_ops_children():
    """Test the children of a root operator can still be skipped."""

    out = all_decomps(qp.X(Wire[1]), skip_ops={qp.GlobalPhase(Float)})
    assert qp.GlobalPhase(Float) not in out


def test_concrete_input_abstractified():
    """Test that if a concrete input is provided, it is abstractified."""

    out = all_decomps(qp.RX(0.5, 0))
    assert qp.RX(Float, Wire[1]) in out

    assert all(op.is_fully_abstract for op in out)


def test_non_applicable_rules_ignored():
    """Test that non-applicable rules are ignore for a given abstract operator.
    3-qubit unitary chosen as qubit unitary has rules for one and two qubit versions.
    """
    op = qp.QubitUnitary(Float[8, 8], Wire[3])
    rules_map = qp.decomposition.all_decomps(op)

    rules = rules_map[op]
    assert all(r.is_applicable(**op.arguments) for r in rules)


def test_higher_order_operator():
    """Test providing a more complicated root node."""

    rules_map = qp.decomposition.all_decomps(qp.Select([qp.X(0), qp.Y(0)], (1, 2)))

    assert qp.Select([qp.X(Wire[1]), qp.Y(Wire[1])], Wire[2], Wire[0]) in rules_map

    assert qp.H(Wire[1]) in rules_map

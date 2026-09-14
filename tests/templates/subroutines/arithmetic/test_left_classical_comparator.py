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
"""Tests for LeftClassicalComparator."""

import pytest

import pennylane as qp
from pennylane.decomposition import list_decomps
from pennylane.ops.functions.assert_valid import _test_decomposition_rule, assert_valid


@pytest.mark.usefixtures("enable_and_disable_capture")
def test_assert_valid_and_decomposition():
    """Standard Operator2 checks, with capture enabled and disabled."""
    op = qp.LeftClassicalComparator(
        x_wires=[0, 1, 2], L=2, target_wire=3, work_wires=[4, 5], comparator=">="
    )
    assert_valid(op, skip_differentiation=True)
    for rule in list_decomps(qp.LeftClassicalComparator):
        _test_decomposition_rule(op, rule)


@pytest.mark.usefixtures("enable_and_disable_capture")
def test_adjoint_decomposition():
    """Adjoint decomposition is capture compatible."""
    op = qp.adjoint(
        qp.LeftClassicalComparator(
            x_wires=[0, 1, 2], L=2, target_wire=3, work_wires=[4, 5], comparator="<"
        )
    )
    for rule in list_decomps("Adjoint(LeftClassicalComparator)"):
        _test_decomposition_rule(op, rule)

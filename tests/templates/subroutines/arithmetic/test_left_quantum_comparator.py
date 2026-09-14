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
"""Tests for LeftQuantumComparator."""

import pytest

import pennylane as qp
from pennylane.decomposition import list_decomps
from pennylane.ops.functions.assert_valid import _test_decomposition_rule, assert_valid


@pytest.mark.usefixtures("enable_and_disable_capture")
def test_assert_valid_and_decomposition():
    """Standard Operator2 checks, with capture enabled and disabled."""
    op = qp.LeftQuantumComparator(
        x_wires=[0, 1, 2],
        y_wires=[3, 4, 5],
        target_wire=6,
        work_wires=[7, 8],
        comparator="<=",
    )
    assert_valid(op, skip_differentiation=True)
    for rule in list_decomps(qp.LeftQuantumComparator):
        _test_decomposition_rule(op, rule)


@pytest.mark.usefixtures("enable_and_disable_capture")
def test_adjoint_decomposition():
    """Adjoint decomposition is capture compatible."""
    op = qp.adjoint(
        qp.LeftQuantumComparator(
            x_wires=[0, 1, 2],
            y_wires=[3, 4, 5],
            target_wire=6,
            work_wires=[7, 8],
            comparator="<",
        )
    )
    for rule in list_decomps("Adjoint(LeftQuantumComparator)"):
        _test_decomposition_rule(op, rule)

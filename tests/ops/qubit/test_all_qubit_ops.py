# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
Unit tests for the available built-in discrete-variable quantum operations. Only tests over
multiple types of operations should exist in this file. Type-specific tests should go in
the more specific file.
"""

import pytest

import pennylane as qp

# pylint: disable=too-few-public-methods


@pytest.mark.parametrize(
    "op, basis",
    [
        (qp.X(0), "X"),
        (qp.Y(0), "Y"),
        (qp.Z(0), "Z"),
        (qp.S(0), "Z"),
        (qp.T(0), "Z"),
        (qp.SX(0), "X"),
        (qp.RX(0.5, 0), "X"),
        (qp.RY(0.5, 0), "Y"),
        (qp.RZ(0.5, 0), "Z"),
        (qp.PhaseShift(0.5, 0), "Z"),
        (qp.PCPhase(1.23, 7, (1, 2, 3)), "Z"),
        (qp.X(0) + qp.Y(0), None),
        (qp.X(0) @ qp.Y(1), None),
    ],
)
def test_basis_deprecation(op, basis):
    """Test Operation.basis raises a deprecation warning."""
    with pytest.warns(
        qp.exceptions.PennyLaneDeprecationWarning,
        match="Operation.basis is deprecated in v0.46 and will be removed in v0.47.",
    ):
        assert op.basis == basis

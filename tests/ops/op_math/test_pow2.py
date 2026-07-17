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

"""Tests for the Pow2 class."""

from pennylane.ops import ISWAP, Identity, PhaseShift, T, Z
from pennylane.ops.op_math.pow2 import Pow2, pow2
from tests.core.operator.operator2_utils import DynOp


def test_initialization():
    """Tests initializing a Pow2 operator."""

    base = DynOp(0.5, wires=0)

    # lazy
    op = pow2(base, z=0.5, lazy=True)
    assert isinstance(op, Pow2)
    assert op.static_args["z"] == 0.5
    assert op.base == base

    # eager
    op = pow2(base, z=0.5, lazy=False)
    assert isinstance(op, Pow2)
    assert op.static_args["z"] == 0.5
    assert op.base == base

    # has a custom pow()
    op = pow2(T(0), z=3, lazy=False)
    assert isinstance(op, PhaseShift)

    # produces no ops
    op = pow2(Identity(0), z=2, lazy=False)
    assert isinstance(op, Identity)

    # produces multiple ops
    ops = pow2(ISWAP((0, 1)), z=6, lazy=False)
    assert ops == Z(0) @ Z(1)

    # we call Pow2 directly
    op = Pow2(base, z=1.5)
    assert isinstance(op, Pow2)
    assert op.static_args["z"] == 1.5
    assert op.base == base

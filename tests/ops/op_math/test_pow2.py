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
from pennylane.core import AnnotatedQueue
from pennylane.ops.op_math.pow2 import pow2, Pow2
from tests.core.operator.operator2_utils import DynOp


def test_initialization():
    """Tests initializing a Pow2 operator."""

    base = DynOp(0.5, wires=0)

    op = pow2(base, z=0.5)
    assert isinstance(op, Pow2)
    assert op.static_args["z"] == 0.5
    assert op.base == base

    op = Pow2(base, z=1.5)
    assert isinstance(op, Pow2)
    assert op.static_args["z"] == 1.5
    assert op.base == base

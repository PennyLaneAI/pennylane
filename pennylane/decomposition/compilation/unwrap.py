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
"""Just a simple utility for extracting nested adjoint and controlled info."""

from pennylane.core import Operator2
from pennylane.ops import Adjoint, Controlled


def unwrap(
    op: Operator2, is_adjoint: bool = False, n_ctrls: int = 0
) -> tuple[Operator2, bool, int]:
    """A simple utility for extracting out the adjoint and control information from an operator."""

    if not isinstance(op, (Adjoint, Controlled)):
        return op, is_adjoint, n_ctrls
    if isinstance(op, Adjoint):
        return unwrap(op.base, not is_adjoint, n_ctrls)
    # is controlled
    return unwrap(op.base, is_adjoint, n_ctrls + len(op.control_wires))

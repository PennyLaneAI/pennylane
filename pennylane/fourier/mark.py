# Copyright 2018-2026 Xanadu Quantum Technologies Inc.

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
Contains the 'label' function for customizing operator labels.
"""

from pennylane.operation import Operator
from pennylane.ops.op_math import SymbolicOp


class MarkedOp(SymbolicOp):
    """Creates a marked operator."""


def mark(op: Operator, tag: str) -> MarkedOp:
    """Marks an operator with a custom tag."""
    return MarkedOp(op, tag)

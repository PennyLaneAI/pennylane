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
"""Shared assertions for structured decomposition rules."""

import pennylane as qp
from pennylane.core.operator import Operator
from pennylane.decomposition import DecompositionRule


def assert_basis_change_decomposition(op: Operator, rule: DecompositionRule) -> None:
    """Check a rule and the decompositions of its structured basis-change operands."""
    qp.testing.assert_valid_decomp_rule(op, rule)
    for gate in qp.testing.decomp_rule_to_tape(op, rule).operations:
        if isinstance(gate, qp.ops.ChangeOpBasis2):
            for block in gate.operands:
                assert not isinstance(block, qp.ops.Prod2)
                qp.testing.assert_decomposable(block)

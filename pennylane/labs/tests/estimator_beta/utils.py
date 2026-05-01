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
Shared helpers for testing estimator_beta test_suite
"""

import pennylane.labs.estimator_beta as qre


def decomp_equal(decomp1, decomp2):
    """Tests the equality of two decompositions"""
    if len(decomp1) != len(decomp2):
        return False

    for op1, op2 in zip(decomp1, decomp2):
        if isinstance(op1, (qre.Allocate, qre.Deallocate)):
            ops_equal = op1.equal(op2)
        else:
            ops_equal = op1 == op2

        if not ops_equal:
            return False

    return True

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
Former location of :func:`~.testing.assert_valid`, kept as an alias of
:mod:`pennylane.testing.operators` until Catalyst and Lightning use the new location.
"""

from pennylane.testing.decompositions import (
    assert_valid_decomposition_rule as _test_decomposition_rule,
)
from pennylane.testing.operators import _check_eigendecomposition, assert_valid

__all__ = ["assert_valid", "_check_eigendecomposition", "_test_decomposition_rule"]

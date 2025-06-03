# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit test module for the merge rotations transform"""

import pytest

pytestmark = pytest.mark.external

xdsl = pytest.importorskip("xdsl")

import pennylane as qml
from pennylane.compiler.python_compiler.transforms import MergeRotationsPass


class MergeRotationsPass:
    """Unit tests for MergeRotationsPass."""

    def test_no_composable_ops(self):
        """Test that nothing changes when there are no composable gates."""

    def test_composable_ops(self):
        """Test that composable gates are cancelled."""

    def test_cond(self):
        """Test that composable gates are merged correctly when conditions
        are present."""

    def test_for_loop(self):
        """Test that composable gates are merged correctly when for loops
        are present."""

    def test_while_loop(self):
        """Test that composable gates are merged correctly when while loops
        are present."""

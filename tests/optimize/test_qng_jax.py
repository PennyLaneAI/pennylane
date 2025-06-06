# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test Jax-based Catalyst-compatible QNG optimizer"""

import pennylane as qml


class TestBasics:
    """Test basic properties of the QNGOptimizerJax."""

    def test_initialization_default(self):
        """Test that initializing QNGOptimizerJax with default values works."""
        opt = qml.QNGOptimizerJax()
        assert opt.stepsize == 0.01
        assert opt.approx == "block-diag"
        assert opt.lam == 0

    def test_initialization_custom(self):
        """Test that initializing QNGOptimizerJax with custom values works."""
        opt = qml.QNGOptimizerJax(stepsize=0.05, approx="diag", lam=1e-9)
        assert opt.stepsize == 0.05
        assert opt.approx == "diag"
        assert opt.lam == 1e-9

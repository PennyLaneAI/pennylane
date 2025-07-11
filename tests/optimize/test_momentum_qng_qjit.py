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
"""Test Jax-based Catalyst-compatible Momentum-QNG optimizer"""

import numpy as np

import pennylane as qml

dev_names = (
    "default.qubit",
    "lightning.qubit",
)


class TestBasics:
    """Test basic properties of the MomentumQNGOptimizerQJIT."""

    def test_initialization_default(self):
        """Test that initializing MomentumQNGOptimizerQJIT with default values works."""
        opt = qml.MomentumQNGOptimizerQJIT()
        assert opt.stepsize == 0.01
        assert opt.momentum == 0.9
        assert opt.approx == "block-diag"
        assert opt.lam == 0

    def test_initialization_custom(self):
        """Test that initializing MomentumQNGOptimizerQJIT with custom values works."""
        opt = qml.MomentumQNGOptimizerQJIT(stepsize=0.05, momentum=0.8, approx="diag", lam=1e-9)
        assert opt.stepsize == 0.05
        assert opt.momentum == 0.8
        assert opt.approx == "diag"
        assert opt.lam == 1e-9

    def test_init_zero_state(self):
        """Test that the MomentumQNGOptimizerQJIT state is initialized to an array of zeros."""
        opt = qml.MomentumQNGOptimizerQJIT()
        params = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        state = opt.init(params)
        assert np.all(state == np.zeros_like(params))

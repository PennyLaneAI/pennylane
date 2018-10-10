# Copyright 2018 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Adagrad optimizer"""

import autograd
import autograd.numpy as np

from .gradient_descent import GradientDescentOptimizer


class AdagradOptimizer(GradientDescentOptimizer):
    """Gradient-descent optimizer with past-gradient-dependent
    learning rate in each dimension."""

    def __init__(self, stepsize=0.01):
        super().__init__(stepsize)
        self.accumulation = None

    def apply_grad(self, grad, x):
        """Update x to take a single optimization step."""
        if self.accumulation is None:
            self.accumulation = grad * grad
        else:
            self.accumulation += grad * grad

        # elementwise multiplication
        return x - (self.stepsize / np.sqrt(self.accumulation + 1e-8)) * grad 

    def reset(self):
        """Reset optimizer by erasing memory of past steps."""
        self.accumulation = None

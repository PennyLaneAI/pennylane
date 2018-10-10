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
"""Adam optimizer"""

import autograd
import autograd.numpy as np

from .gradient_descent import GradientDescentOptimizer


class AdamOptimizer(GradientDescentOptimizer):
    """Gradient-descent optimizer with adaptive learning rate, first and second moment."""

    def __init__(self, stepsize=0.01, beta1=0.9, beta2=0.99):
        super().__init__(stepsize)
        self.beta1 = beta1
        self.beta2 = beta2
        self.stepsize = stepsize
        self.firstmoment = None
        self.secondmoment = None
        self.t = 0

    def apply_grad(self, grad, x):
        """Update x to take a single optimization step."""

        self.t += 1

        # Update first moment
        if self.firstmoment is None:
            self.firstmoment = grad
        else:
            self.firstmoment = self.beta1 * self.firstmoment + (1 - self.beta1) * grad

        # Update second moment
        if self.secondmoment is None:
            self.secondmoment = grad * grad
        else:
            self.secondmoment = self.beta2 * self.secondmoment + (1 - self.beta2) * (grad * grad)

        # Update step size (instead of correcting for bias)
        adapted_stepsize = self.stepsize * np.sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t)

        return x - adapted_stepsize * self.firstmoment / (np.sqrt(self.secondmoment) + 1e-8)

    def reset(self):
        """Reset optimizer by erasing memory of past steps."""
        self.firstmoment = None
        self.secondmoment = None
        self.t = 0

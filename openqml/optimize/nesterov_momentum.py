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
"""Nesterov momentum optimizer"""

import autograd
import autograd.numpy as np

from .momentum import MomentumOptimizer


class NesterovMomentumOptimizer(MomentumOptimizer):
    """Gradient-descent optimizer with Nesterov momentum."""

    def __init__(self, stepsize=0.01, momentum=0.9):
        super().__init__(stepsize, momentum)

    def compute_grad(self, objective_fn, x, grad_fn=None):
        """Compute gradient of objective_fn at the shifted point (x - momentum*accumulation) """

        if self.accumulation is None:
            shifted_x = x
        else:
            shifted_x = x - self.momentum * self.accumulation

        if grad_fn is not None:
            g = grad_fn(shifted_x)  # just call the supplied grad function
        else:
            g = autograd.grad(objective_fn)(shifted_x)  # default is autograd
        return g

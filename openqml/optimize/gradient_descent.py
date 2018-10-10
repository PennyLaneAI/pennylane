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
"""Gradient descent optimizer"""

import autograd
import autograd.numpy as np


class GradientDescentOptimizer(object):
    """Base class for gradient-descent-based optimizers."""

    def __init__(self, stepsize=0.01):
        self.stepsize = stepsize

    def step(self, objective_fn, x, grad_fn=None):
        """Update x with one step of the optimizer."""

        x_shape = x.shape

        g = self.compute_grad(objective_fn, x, grad_fn=grad_fn)

        if len(x_shape) > 1:  # reshape gradient after grad() flattened it
            g = g.reshape(x_shape)

        x_out = self.apply_grad(g, x)

        return x_out

    def compute_grad(self, objective_fn, x, grad_fn=None):
        """Compute gradient of objective_fn at the point x"""
        if grad_fn is not None:
            g = grad_fn(x)  # just call the supplied grad function
        else:
            g = autograd.grad(objective_fn)(x)  # default is autograd
        return g

    def apply_grad(self, grad, x):
        """Update x to take a single optimization step"""
        return x - self.stepsize * grad

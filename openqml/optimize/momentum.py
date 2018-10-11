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
"""Momentum optimizer"""
from .gradient_descent import GradientDescentOptimizer


class MomentumOptimizer(GradientDescentOptimizer):
    r"""Gradient-descent optimizer with momentum.

    The momentum optimizer adds a "momentum" term to gradient descent
    which considers the past gradients:

    .. math:: x^{(t+1)} = x^{(t)} - a^{(t+1)}.

    The accumulator term :math:`a` is updated as follows:

    .. math:: a^{(t+1)} = m a^{(t)} + \eta \nabla f(x^{(t)}),

    with user defined parameters:

    * :math:`\eta`: the step size
    * :math:`m`: the momentum

    Args:
        stepsize (float): user-defined hyperparameter :math:`\eta`
        momentum (float): user-defined hyperparameter :math:`m`
    """
    def __init__(self, stepsize=0.01, momentum=0.9):
        super().__init__(stepsize)
        self.momentum = momentum
        self.accumulation = None

    def apply_grad(self, grad, x):
        # docstring is inherited from GradientDescentOptimizer
        if self.accumulation is None:
            self.accumulation = self.stepsize * grad
        else:
            self.accumulation = self.momentum * self.accumulation + self.stepsize * grad
        return x - self.accumulation

    def reset(self):
        """Reset optimizer by erasing memory of past steps."""
        self.accumulation = None

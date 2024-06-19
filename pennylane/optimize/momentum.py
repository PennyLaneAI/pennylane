# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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

    .. note::

        When using ``torch``, ``tensorflow`` or ``jax`` interfaces, refer to :doc:`Gradients and training </introduction/interfaces>` for suitable optimizers.

    """

    def __init__(self, stepsize=0.01, momentum=0.9):
        super().__init__(stepsize)
        self.momentum = momentum
        self.accumulation = None

    def apply_grad(self, grad, args):
        r"""Update the trainable args to take a single optimization step. Flattens and unflattens
        the inputs to maintain nested iterables as the parameters of the optimization.

        Args:
            grad (tuple [array]): the gradient of the objective
                function at point :math:`x^{(t)}`: :math:`\nabla f(x^{(t)})`.
            args (tuple): the current value of the variables :math:`x^{(t)}`.

        Returns:
            list [array]: the new values :math:`x^{(t+1)}`.
        """
        args_new = list(args)

        if self.accumulation is None:
            self.accumulation = [0.0] * len(args)

        trained_index = 0
        for index, arg in enumerate(args):
            if getattr(arg, "requires_grad", False):
                self._update_accumulation(index, grad[trained_index])
                args_new[index] = arg - self.accumulation[index]

                trained_index += 1

        return args_new

    def _update_accumulation(self, index, grad):
        r"""Update the accumulation.

        Args:
            index (int): index of argument to update.
            grad (ndarray): gradient at index
        """
        self.accumulation[index] = self.momentum * self.accumulation[index] + self.stepsize * grad

    def reset(self):
        """Reset optimizer by erasing memory of past steps."""
        self.accumulation = None

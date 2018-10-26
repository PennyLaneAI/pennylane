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
from .optimizer_utilities import _flatten, _unflatten


class GradientDescentOptimizer(object):
    r"""Base class for gradient-descent-based optimizers.

    A step of the gradient descent optimizer computes the new weights via the rule

    .. math::

        x^{(t+1)} = x^{(t)} - \eta \nabla f(x^{(t)}).

    where :math:`\eta` is a user-defined hyperparameter corresponding to step size.

    Args:
        stepsize (float): the user-defined hyperparameter :math:`\eta`
    """
    def __init__(self, stepsize=0.01):
        self.stepsize = stepsize

    def step(self, objective_fn, x, grad_fn=None):
        """Update x with one step of the optimizer.

        Args:
            objective_fn (function): the objective function for optimization
            x (array): NumPy array containing the weights
            grad_fn (function): Optional gradient function of the
                objective function with respect to the weights ``x``.
                If ``None``, the gradient function is computed automatically.

        Returns:
            array: the new weights :math:`x^{(t+1)}`
        """

        g = self.compute_grad(objective_fn, x, grad_fn=grad_fn)

        x_out = self.apply_grad(g, x)

        return x_out

    @staticmethod
    def compute_grad(objective_fn, x, grad_fn=None):
        r"""Compute gradient of the objective_fn at the point x.

        Args:
            objective_fn (function): the objective function for optimization
            x (array): NumPy array containing the weights
            grad_fn (function): Optional gradient function of the
                objective function with respect to the weights ``x``.
                If ``None``, the gradient function is computed automatically.

        Returns:
            array: NumPy array containing the gradient :math:`\nabla f(x^{(t)})`
        """
        if grad_fn is not None:
            g = grad_fn(x)  # just call the supplied grad function
        else:
            # default is autograd
            g = autograd.grad(objective_fn)(x)  # pylint: disable=no-value-for-parameter
        return g

    def apply_grad(self, grad, x):
        r"""Update the weights x to take a single optimization step. Flattens and unflattens
        the inputs to maintain nested iterables as the parameters of the optimization.

        Args:
            grad (array): The gradient of the objective
                function at point :math:`x^{(t)}`: :math:`\nabla f(x^{(t)})`
            x (array): the current value of the weights :math:`x^{(t)}`

        Returns:
            array: the new weights :math:`x^{(t+1)}`
        """

        x_flat = _flatten(x)
        grad_flat = _flatten(grad)

        new_x_flat = [a - self.stepsize * b for a, b in zip(x_flat, grad_flat)]

        return _unflatten(new_x_flat, x)[0]

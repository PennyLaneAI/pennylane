# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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

from pennylane._grad import grad as get_gradient
from pennylane.utils import _flatten, unflatten


class GradientDescentOptimizer:
    r"""Basic gradient-descent optimizer.

    Base class for other gradient-descent-based optimizers.

    A step of the gradient descent optimizer computes the new values via the rule

    .. math::

        x^{(t+1)} = x^{(t)} - \eta \nabla f(x^{(t)}).

    where :math:`\eta` is a user-defined hyperparameter corresponding to step size.

    Args:
        stepsize (float): the user-defined hyperparameter :math:`\eta`
    """

    def __init__(self, stepsize=0.01):
        self._stepsize = stepsize

    def update_stepsize(self, stepsize):
        r"""Update the initialized stepsize value :math:`\eta`.

        This allows for techniques such as learning rate scheduling.

        Args:
            stepsize (float): the user-defined hyperparameter :math:`\eta`
        """
        self._stepsize = stepsize

    def step_and_cost(self, objective_fn, *args, grad_fn=None, **kwargs):
        """Update x with one step of the optimizer and return the corresponding objective
        function value prior to the step.

        Args:
            objective_fn (function): the objective function for optimization
            *args : Variable length argument list for objective function
            grad_fn (function): Optional gradient function of the
                objective function with respect to the variables ``x``.
                If ``None``, the gradient function is computed automatically.
            **kwargs : Variable length dictionary of keywords for the cost function

        Returns:
            tuple: the new variable values :math:`x^{(t+1)}` and the objective function output
                prior to the step
        """
        trainable_indexes = self._process_args(args)

        g, forward = self.compute_grad(objective_fn, args, kwargs, grad_fn=grad_fn)
        new_args = self.apply_grad(g, args, trainable_indexes)

        if forward is None:
            forward = objective_fn(*args, **kwargs)

        new_args.append(forward)
        return new_args

    def step(self, objective_fn, *args, grad_fn=None, **kwargs):
        """Update x with one step of the optimizer.

        Args:
            objective_fn (function): the objective function for optimization
            *args : Variable length argument list for objective function
            grad_fn (function): Optional gradient function of the
                objective function with respect to the variables ``x``.
                If ``None``, the gradient function is computed automatically.
            **kwargs : Variable length dictionary of keywords for the cost function

        Returns:
            array: the new variable values :math:`x^{(t+1)}`
        """
        trainable_indexes = self._process_args(args)

        g, _ = self.compute_grad(objective_fn, args, kwargs, grad_fn=grad_fn)
        new_args = self.apply_grad(g, args, trainable_indexes)

        return new_args

    @staticmethod
    def _process_args(args):
        r"""Determine which arguments are optimized and which are constant

        Args:
            args (tuple(array)): Variables for the objective function

        Returns:
            List: Indices of args for parameters that will be updated
            List: Parameters that will be updated
        """
        indexes = []
        for index, value in enumerate(args):
            if getattr(value, "requires_grad", True):
                indexes.append(index)
        return indexes

    @staticmethod
    def compute_grad(objective_fn, args, kwargs, grad_fn=None):
        r"""Compute gradient of the objective_fn at the point x and return it along with the
            objective function forward pass (if available).

        Args:
            objective_fn (function): the objective function for optimization
            args (tuple(array)): Tuple of NumPy arrays containing the current values for the
                objection function
            kwargs (dict): Keywords for the cost function
            grad_fn (function): Optional gradient function of the objective function with respect to
                the variables ``x``. If ``None``, the gradient function is computed automatically.

        Returns:
            tuple: The NumPy array containing the gradient :math:`\nabla f(x^{(t)})` and the
                objective function output. If ``grad_fn`` is provided, the objective function
                will not be evaluted and instead ``None`` will be returned.
        """
        g = get_gradient(objective_fn) if grad_fn is None else grad_fn
        grad = g(*args, **kwargs)
        forward = getattr(g, "forward", None)

        return grad, forward

    def apply_grad(self, grad_array, args, trainable_indexes):
        r"""Update the variables to take a single optimization step. Flattens and unflattens
        the inputs to maintain nested iterables as the parameters of the optimization.

        Args:
            grad (array): The gradient of the objective
                function at point :math:`x^{(t)}`: :math:`\nabla f(x^{(t)})`
            x (tuple(array)): the current value of the variables :math:`x^{(t)}`

        Returns:
            array: the new values :math:`x^{(t+1)}`
        """
        args_new = list(args)
        for index_args, grad in zip(trainable_indexes, grad_array):
            x_flat = _flatten(args[index_args])
            grad_flat = _flatten(grad)

            x_new_flat = [e - self._stepsize * g for g, e in zip(grad_flat, x_flat)]

            args_new[index_args] = unflatten(x_new_flat, args[index_args])

        return args_new
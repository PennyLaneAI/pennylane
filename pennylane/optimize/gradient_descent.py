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
"""Gradient descent optimizer"""

import warnings

from pennylane._grad import grad as get_gradient
from pennylane.utils import _flatten, unflatten
from pennylane.numpy import ndarray, tensor


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
        self.stepsize = stepsize

    @property
    def _stepsize(self):
        warnings.warn(
            "'_stepsize' is deprecated. Please use 'stepsize' instead.",
            UserWarning,
            stacklevel=2,
        )

        return self.stepsize

    @_stepsize.setter
    def _stepsize(self, stepsize):
        warnings.warn(
            "'_stepsize' is deprecated. Please use 'stepsize' instead.",
            UserWarning,
            stacklevel=2,
        )

        self.stepsize = stepsize

    def update_stepsize(self, stepsize):
        r"""Update the initialized stepsize value :math:`\eta`.

        This allows for techniques such as learning rate scheduling.

        Args:
            stepsize (float): the user-defined hyperparameter :math:`\eta`
        """
        warnings.warn(
            "'update_stepsize' is deprecated. Stepsize value can be updated using "
            "the 'stepsize' attribute.",
            UserWarning,
            stacklevel=2,
        )

        self.stepsize = stepsize

    def step_and_cost(self, objective_fn, *args, grad_fn=None, **kwargs):
        """Update trainable arguments with one step of the optimizer and return the corresponding
        objective function value prior to the step.

        Args:
            objective_fn (function): the objective function for optimization
            *args : variable length argument list for objective function
            grad_fn (function): optional gradient function of the
                objective function with respect to the variables ``*args``.
                If ``None``, the gradient function is computed automatically.
                Must return a ``tuple[array]`` with the same number of elements as ``*args``.
                Each array of the tuple should have the same shape as the corresponding argument.
            **kwargs : variable length of keyword arguments for the objective function

        Returns:
            tuple[list [array], float]: the new variable values :math:`x^{(t+1)}` and the objective
            function output prior to the step.
            If single arg is provided, list [array] is replaced by array.
        """

        g, forward = self.compute_grad(objective_fn, args, kwargs, grad_fn=grad_fn)
        new_args = self.apply_grad(g, args)

        if forward is None:
            forward = objective_fn(*args, **kwargs)

        # unwrap from list if one argument, cleaner return
        if len(new_args) == 1:
            return new_args[0], forward
        return new_args, forward

    def step(self, objective_fn, *args, grad_fn=None, **kwargs):
        """Update trainable arguments with one step of the optimizer.

        Args:
            objective_fn (function): the objective function for optimization
            *args : Variable length argument list for objective function
            grad_fn (function): optional gradient function of the
                objective function with respect to the variables ``x``.
                If ``None``, the gradient function is computed automatically.
                Must return a ``tuple[array]`` with the same number of elements as ``*args``.
                Each array of the tuple should have the same shape as the corresponding argument.
            **kwargs : variable length of keyword arguments for the objective function

        Returns:
            list [array]: the new variable values :math:`x^{(t+1)}`.
            If single arg is provided, list [array] is replaced by array.
        """

        g, _ = self.compute_grad(objective_fn, args, kwargs, grad_fn=grad_fn)
        new_args = self.apply_grad(g, args)

        # unwrap from list if one argument, cleaner return
        if len(new_args) == 1:
            return new_args[0]

        return new_args

    @staticmethod
    def compute_grad(objective_fn, args, kwargs, grad_fn=None):
        r"""Compute gradient of the objective function at the given point and return it along with
        the objective function forward pass (if available).

        Args:
            objective_fn (function): the objective function for optimization
            args (tuple): tuple of NumPy arrays containing the current parameters for the
                objection function
            kwargs (dict): keyword arguments for the objective function
            grad_fn (function): optional gradient function of the objective function with respect to
                the variables ``args``. If ``None``, the gradient function is computed automatically.
                Must return the same shape of tuple [array] as the autograd derivative.

        Returns:
            tuple (array): NumPy array containing the gradient :math:`\nabla f(x^{(t)})` and the
            objective function output. If ``grad_fn`` is provided, the objective function
            will not be evaluted and instead ``None`` will be returned.
        """
        g = get_gradient(objective_fn) if grad_fn is None else grad_fn
        grad = g(*args, **kwargs)
        forward = getattr(g, "forward", None)

        num_trainable_args = 0
        for arg in args:
            if getattr(arg, "requires_grad", True):
                num_trainable_args += 1

        if num_trainable_args == 1:
            grad = (grad,)

        return grad, forward

    def apply_grad(self, grad, args):
        r"""Update the variables to take a single optimization step. Flattens and unflattens
        the inputs to maintain nested iterables as the parameters of the optimization.

        Args:
            grad (tuple [array]): the gradient of the objective
                function at point :math:`x^{(t)}`: :math:`\nabla f(x^{(t)})`
            args (tuple): the current value of the variables :math:`x^{(t)}`

        Returns:
            list [array]: the new values :math:`x^{(t+1)}`
        """
        args_new = list(args)

        trained_index = 0
        for index, arg in enumerate(args):
            if getattr(arg, "requires_grad", True):
                x_flat = _flatten(arg)
                grad_flat = _flatten(grad[trained_index])
                trained_index += 1

                x_new_flat = [e - self.stepsize * g for g, e in zip(grad_flat, x_flat)]

                args_new[index] = unflatten(x_new_flat, args[index])

                if isinstance(arg, ndarray):
                    # Due to a bug in unflatten, input PennyLane tensors
                    # are being unwrapped. Here, we cast them back to PennyLane
                    # tensors.
                    # TODO: remove when the following is fixed:
                    # https://github.com/PennyLaneAI/pennylane/issues/966
                    args_new[index] = args_new[index].view(tensor)
                    args_new[index].requires_grad = True

        return args_new

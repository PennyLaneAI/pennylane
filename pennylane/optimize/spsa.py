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
"""Simultaneous perturbation stochastic approximation (SPSA) optimizer"""

import numpy as np
from pennylane._grad import grad as get_gradient


class SPSAOptimizer:
    r"""Simultaneous perturbation stochastic approximation (SPSA) optimizer.

    Gradient descent based optimization method using a random pertubation vector in gradient
    approximation.

    A step of the SPSA optimizer computes the new values via the rule

    .. math::

        x^{(t+1)} = x^{(t)} - a_k \hat{g}_k(x^{(t)}).

    with estimated gradient

    .. math::

        \hat{g}_{ki}(x^{(t)}) = \frac{f(x^{(t)}+c_k\Delta_k)-f(x^{(t)}-c_k\Delta_k)}{2c_k\Delta_k}

        a_k = \frac{a}{(A+k)^\alpha}
        c_k = \frac{c}{k^\gamma}
    
    for each step :math:`k`

    Args:
        a (float): scaling parameter for step size
        c (float): scaling parameter for evaluation step size
        A (float): stability constant
        alpha (float): scaling exponent for step size
        gamma (float): scaling exponent for evaluation step size
    """

    def __init__(self, p, a, c, A, alpha, gamma):
        self.a = a
        self.c = c
        self.A = A
        self.alpha = alpha
        self.gamma = gamma
        a.k = 1

    def step_and_cost(self, objective_fn, *args, grad_fn=None, **kwargs):
        """Update trainable arguments with one step of the optimizer and return the corresponding
        objective function value prior to the step.

        Args:
            objective_fn (function): the objective function for optimization
            *args : variable length argument list for objective function
            **kwargs : variable length of keyword arguments for the objective function

        Returns:
            tuple[list [array], float]: the new variable values :math:`x^{(t+1)}` and the objective
            function output prior to the step.
            If single arg is provided, list [array] is replaced by array.
        """

        g = self.compute_grad(objective_fn, args, kwargs)
        new_args = self.apply_grad(g, args)

        forward = objective_fn(*args, **kwargs)

        self.k += 1

        # unwrap from list if one argument, cleaner return
        if len(new_args) == 1:
            return new_args[0], forward
        return new_args, forward

    def step(self, objective_fn, *args, **kwargs):
        """Update trainable arguments with one step of the optimizer.

        Args:
            objective_fn (function): the objective function for optimization
            *args : Variable length argument list for objective function
            **kwargs : variable length of keyword arguments for the objective function

        Returns:
            list [array]: the new variable values :math:`x^{(t+1)}`.
            If single arg is provided, list [array] is replaced by array.
        """

        g, _ = self.compute_grad(objective_fn, args, kwargs)
        new_args = self.apply_grad(g, args)

        self.k += 1

        # unwrap from list if one argument, cleaner return
        if len(new_args) == 1:
            return new_args[0]

        return new_args

    def compute_grad(self, objective_fn, args, kwargs):
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
            list[array]: NumPy array containing the gradient :math:`\nabla f(x^{(t)})`.
        """

        ck = self.c / self.k^self.gamma

        delta = []
        args_minus_pert = list(args)
        args_plus_pert = list(args)

        for index, arg in enumerate(args):
            if getattr(arg, "requires_grad", False):
                delta.append(np.random.binomial(1, 1/2, arg.size) * 2 - 1)
                args_minus_pert[index] = arg - ck * delta[index]
                args_plus_pert[index] = arg + ck * delta[index]

        diff_sc = (objective_fn(*args_plus_pert, **kwargs) - objective_fn(*args_minus_pert, **kwargs)) / (2 * ck)

        grad = []
        for delta_i in delta:
            grad.append(diff_sc / delta_i)

        num_trainable_args = sum(getattr(arg, "requires_grad", False) for arg in args)
        grad = (grad,) if num_trainable_args == 1 else grad

        return grad

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

        ak = self.a / (self.k + self.A)^self.alpha

        args_new = list(args)

        trained_index = 0
        for index, arg in enumerate(args):
            if getattr(arg, "requires_grad", False):
                args_new[index] = arg - ak * grad[trained_index]

                trained_index += 1

        return args_new

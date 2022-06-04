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

from pennylane import numpy as np


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

    for each step :math:`k`, where :math:`\Delta_k` is a random pertubation vector drawn from a
    +-1 Bernoulli distribution with probability of :math:`1/2` for both `-1` and `+1`.

    Args:
        a (float): scaling parameter for step size
        c (float): scaling parameter for evaluation step size
        A (float): stability constant
        alpha (float): scaling exponent for step size
        gamma (float): scaling exponent for evaluation step size
    """

    # pylint: disable-msg=too-many-arguments
    def __init__(self, maxit=100, a=None, c=0.1, A=None, alpha=0.602, gamma=0.101):
        self.a = a
        self.c = c
        if A is None:
            self.A = maxit / 10
        else:
            self.A = A
        self.alpha = alpha
        self.gamma = gamma
        self.k = 1

    # pylint: disable-msg=too-many-arguments
    def step_and_cost(
        self, objective_fn, *args, a=None, c=None, A=None, alpha=None, gamma=None, **kwargs
    ):
        """Update trainable arguments with one step of the optimizer and return the corresponding
        objective function value prior to the step.

        Args:
            objective_fn (function): the objective function for optimization
            a (float): scaling parameter for step size
            c (float): scaling parameter for evaluation step size
            A (float): stability constant
            alpha (float): scaling exponent for step size
            gamma (float): scaling exponent for evaluation step size
            *args : variable length argument list for objective function
            **kwargs : variable length of keyword arguments for the objective function

        Returns:
            tuple[list [array], float]: the new variable values :math:`x^{(t+1)}` and the objective
            function output prior to the step.
            If single arg is provided, list [array] is replaced by array.
        """

        new_args = self.step(objective_fn, *args, a=a, c=c, A=A, alpha=alpha, gamma=gamma, **kwargs)

        forward = objective_fn(*args, **kwargs)

        return new_args, forward

    # pylint: disable-msg=too-many-arguments
    def step(self, objective_fn, *args, a=None, c=None, A=None, alpha=None, gamma=None, **kwargs):
        """Update trainable arguments with one step of the optimizer.

        Args:
            objective_fn (function): the objective function for optimization
            a (float): scaling parameter for step size
            c (float): scaling parameter for evaluation step size
            A (float): stability constant
            alpha (float): scaling exponent for step size
            gamma (float): scaling exponent for evaluation step size
            *args : Variable length argument list for objective function
            **kwargs : variable length of keyword arguments for the objective function

        Returns:
            list [array]: the new variable values :math:`x^{(t+1)}`.
            If single arg is provided, list [array] is replaced by array.
        """

        if a is None:
            a = self.a
        if c is None:
            c = self.c
        if A is None:
            A = self.A
        if alpha is None:
            alpha = self.alpha
        if gamma is None:
            gamma = self.gamma

        g = self.compute_grad(objective_fn, args, kwargs, c=c, k=self.k, gamma=gamma)
        new_args = self.apply_grad(g, args, a=a, A=A, k=self.k, alpha=alpha)

        self.k += 1

        # unwrap from list if one argument, cleaner return
        if len(new_args) == 1:
            return new_args[0]

        return new_args

    # pylint: disable-msg=too-many-arguments
    @staticmethod
    def compute_grad(objective_fn, args, kwargs, c, k, gamma):
        r"""Compute gradient of the objective function at the given point and return it along with
        the objective function forward pass (if available).

        Args:
            objective_fn (function): the objective function for optimization
            args (tuple): tuple of NumPy arrays containing the current parameters for the
                objection function
            kwargs (dict): keyword arguments for the objective function
            c (float): scaling parameter for evaluation step size
            k (int): iteration step
            gamma (float): scaling exponent for evaluation step size

        Returns:
            list[array]: NumPy array containing the gradient :math:`\hat{g}(x^{(t)})`.
        """

        ck = c / (k + 1) ** gamma

        delta = []
        args_minus_pert = list(args)
        args_plus_pert = list(args)

        for index, arg in enumerate(args):
            if getattr(arg, "requires_grad", False):
                di = np.random.binomial(1, 1 / 2, arg.size) * 2 - 1
                args_minus_pert[index] = arg - ck * di
                args_plus_pert[index] = arg + ck * di
                delta.append(di)

        grad = (
            objective_fn(*args_plus_pert, **kwargs) - objective_fn(*args_minus_pert, **kwargs)
        ) / (2 * ck * np.array(delta, dtype=object))

        num_trainable_args = sum(getattr(arg, "requires_grad", False) for arg in args)
        grad = (grad,) if num_trainable_args == 1 else grad

        return grad

    # pylint: disable-msg=too-many-arguments
    def apply_grad(self, grad, args, a, A, k, alpha):
        r"""Update the variables to take a single optimization step. Flattens and unflattens
        the inputs to maintain nested iterables as the parameters of the optimization.

        Args:
            grad (tuple [array]): the gradient of the objective
                function at point :math:`x^{(t)}`: :math:`\nabla f(x^{(t)})`
            args (tuple): the current value of the variables :math:`x^{(t)}`
            a (float): scaling parameter for step size
            A (float): stability constant
            k (int): iteration step
            alpha (float): scaling exponent for step size

        Returns:
            list [array]: the new values :math:`x^{(t+1)}`
        """

        if a is None:
            a = (A + 1) ** alpha * 0.1 / (np.abs(np.concatenate(grad)).max() + 1)
            self.a = a

        ak = a / (k + A + 1) ** alpha

        args_new = list(args)

        trained_index = 0
        for index, arg in enumerate(args):
            if getattr(arg, "requires_grad", False):
                args_new[index] = arg - ak * grad[trained_index]
                trained_index += 1

        return args_new

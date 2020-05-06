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
"""Adam optimizer"""
import math

from pennylane.utils import _flatten, unflatten
from .gradient_descent import GradientDescentOptimizer


class AdamOptimizer(GradientDescentOptimizer):
    r"""Gradient-descent optimizer with adaptive learning rate, first and second moment.

    Adaptive Moment Estimation uses a step-dependent learning rate,
    a first moment :math:`a` and a second moment :math:`b`, reminiscent of
    the momentum and velocity of a particle:

    .. math::
        x^{(t+1)} = x^{(t)} - \eta^{(t+1)} \frac{a^{(t+1)}}{\sqrt{b^{(t+1)}} + \epsilon },

    where the update rules for the three values are given by

    .. math::
        a^{(t+1)} &= \frac{\beta_1 a^{(t)} + (1-\beta_1)\nabla f(x^{(t)})}{(1- \beta_1)},\\
        b^{(t+1)} &= \frac{\beta_2 b^{(t)} + (1-\beta_2) ( \nabla f(x^{(t)}))^{\odot 2} }{(1- \beta_2)},\\
        \eta^{(t+1)} &= \eta^{(t)} \frac{\sqrt{(1-\beta_2)}}{(1-\beta_1)}.

    Above, :math:`( \nabla f(x^{(t-1)}))^{\odot 2}` denotes the element-wise square operation,
    which means that each element in the gradient is multiplied by itself. The
    hyperparameters :math:`\beta_1` and :math:`\beta_2` can also be step-dependent.
    Initially, the first and second moment are zero.

    The shift :math:`\epsilon` avoids division by zero.

    For more details, see `arXiv:1412.6980 <https://arxiv.org/abs/1412.6980>`_.

    Args:
        stepsize (float): the user-defined hyperparameter :math:`\eta`
        beta1 (float): hyperparameter governing the update of the first and second moment
        beta2 (float): hyperparameter governing the update of the first and second moment
        eps (float): offset :math:`\epsilon` added for numerical stability

    """

    def __init__(self, stepsize=0.01, beta1=0.9, beta2=0.99, eps=1e-8):
        super().__init__(stepsize)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.fm = None
        self.sm = None
        self.t = 0

    def apply_grad(self, grad, x):
        r"""Update the variables x to take a single optimization step. Flattens and unflattens
        the inputs to maintain nested iterables as the parameters of the optimization.

        Args:
            grad (array): The gradient of the objective
                function at point :math:`x^{(t)}`: :math:`\nabla f(x^{(t)})`
            x (array): the current value of the variables :math:`x^{(t)}`

        Returns:
            array: the new values :math:`x^{(t+1)}`
        """

        self.t += 1

        grad_flat = list(_flatten(grad))
        x_flat = _flatten(x)

        # Update first moment
        if self.fm is None:
            self.fm = grad_flat
        else:
            self.fm = [self.beta1 * f + (1 - self.beta1) * g for f, g in zip(self.fm, grad_flat)]

        # Update second moment
        if self.sm is None:
            self.sm = [g * g for g in grad_flat]
        else:
            self.sm = [
                self.beta2 * f + (1 - self.beta2) * g * g for f, g in zip(self.sm, grad_flat)
            ]

        # Update step size (instead of correcting for bias)
        new_stepsize = (
            self._stepsize * math.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        )

        x_new_flat = [
            e - new_stepsize * f / (math.sqrt(s) + self.eps)
            for f, s, e in zip(self.fm, self.sm, x_flat)
        ]

        return unflatten(x_new_flat, x)

    def reset(self):
        """Reset optimizer by erasing memory of past steps."""
        self.fm = None
        self.sm = None
        self.t = 0

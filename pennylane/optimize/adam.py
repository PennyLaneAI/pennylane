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
"""Adam optimizer"""

import autograd.numpy as np
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

    The shift :math:`\epsilon` avoids division by zero and is set to :math:`10^{-8}` in PennyLane.

    For more details, see https://arxiv.org/pdf/1412.6980.pdf :cite:`kingma2014adam`.

    Args:
        stepsize (float): the user-defined hyperparameter :math:`\eta`
        beta1 (float): hyperparameter governing the update of the first and second moment
        beta2 (float): hyperparameter governing the update of the first and second moment
    """
    def __init__(self, stepsize=0.01, beta1=0.9, beta2=0.99):
        super().__init__(stepsize)
        self.beta1 = beta1
        self.beta2 = beta2
        self.stepsize = stepsize
        self.fm = None
        self.sm = None
        self.t = 0

    def apply_grad(self, grad, x):
        # docstring is inherited from GradientDescentOptimizer

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
            self.sm = [self.beta2 * f + (1 - self.beta2) * g * g for f, g in zip(self.sm, grad_flat)]

        # Update step size (instead of correcting for bias)
        new_stepsize = self.stepsize*np.sqrt(1-self.beta2**self.t)/(1-self.beta1**self.t)

        x_new_flat = [e - new_stepsize * f / (np.sqrt(s)+1e-8) for f, s, e in zip(self.fm, self.sm, x_flat)]

        return unflatten(x_new_flat, x)

    def reset(self):
        """Reset optimizer by erasing memory of past steps."""
        self.fm = None
        self.sm = None
        self.t = 0

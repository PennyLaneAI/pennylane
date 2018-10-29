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
"""Adagrad optimizer"""

import autograd.numpy as np

from openqml.utils import _flatten, _unflatten

from .gradient_descent import GradientDescentOptimizer


class AdagradOptimizer(GradientDescentOptimizer):
    r"""Gradient-descent optimizer with past-gradient-dependent
    learning rate in each dimension.

    Adagrad adjusts the learning rate for each parameter :math:`x_i`
    in :math:`x` based on past gradients. We therefore have to consider
    each parameter update individually,

    .. math::
        x^{(t+1)}_i = x^{(t)}_i - \eta_i^{(t+1)} \partial_{w_i} f(x^{(t)}),

    where the gradient was replaced by a (scalar) partial derivative.

    The learning rate in step :math:`t` is given by

    .. math::
        \eta_i^{(t+1)} = \frac{ \eta_{\mathrm{init}} }{ \sqrt{a_i^{(t+1)} + \epsilon } },
        ~~~ a_i^{(t+1)} = \sum_{k=1}^t (\partial_{x_i} f(x^{(k)}))^2.

    The shift :math:`\epsilon` avoids division by zero and is set to
    :math:`10^{-8}` by default.

    :math:`\eta`: is the step size, a user defined parameter.

    Args:
        stepsize (float): the user-defined hyperparameter :math:`\eta`
    """
    def __init__(self, stepsize=0.01):
        super().__init__(stepsize)
        self.accumulation = None

    def apply_grad(self, grad, x):
        # docstring is inherited from GradientDescentOptimizer

        x_flat = list(_flatten(x))
        grad_flat = list(_flatten(grad))

        if self.accumulation is None:
            self.accumulation = [g*g for g in grad_flat]

        else:
            self.accumulation = [a + g*g for a, g in zip(self.accumulation, grad_flat)]

        x_new_flat = [e - (self.stepsize / np.sqrt(a + 1e-8)) * g for a, g, e in zip(self.accumulation, grad_flat, x_flat)]

        return _unflatten(x_new_flat, x)[0]

    def reset(self):
        """Reset optimizer by erasing memory of past steps."""
        self.accumulation = None

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
"""Root mean square propagation optimizer"""

import autograd.numpy as np

from openqml.utils import _flatten, _unflatten

from .adagrad import AdagradOptimizer


class RMSPropOptimizer(AdagradOptimizer):
    r"""Root mean squared propagation optimizer.

    The root mean square progation optimizer is a modified
    :class:`Adagrad optimizer <openqml.optmimize.AdagradOptimizer>`,
    with a decay of learning rate adaptation.

    Extensions of the Adagrad optimization method generally
    start the sum :math:`a` over past gradients in the denominator
    of the learning rate at a finite :math:`t'` with :math:`0 < t' < t`,
    or decay past gradients to avoid an ever-decreasing learning rate.

    Root Mean Square propagation is such an adaptation, where

    .. math::
        a_i^{(t+1)} = \gamma a_i^{(t)} + (1-\gamma) (\partial_{x_i} f(x^{(t)}))^2.

    Args:
        stepsize (float): the user-defined hyperparameter :math:`\eta`
            used in the Adagrad optmization
        gamma (float): the learning rate decay
    """
    def __init__(self, stepsize=0.01, decay=0.9):
        super().__init__(stepsize)
        self.decay = decay

    def apply_grad(self, grad, x):
        # docstring is inherited from AdagradOptimizer

        grad_flat = list(_flatten(grad))
        x_flat = list(_flatten(x))

        if self.accumulation is None:
            self.accumulation = [(1 - self.decay) * g*g for g in grad_flat]
        else:
            self.accumulation = [self.decay*a + (1-self.decay)*g*g for a, g in zip(self.accumulation, grad_flat)]

        x_new_flat = [e - (self.stepsize / np.sqrt(a + 1e-8)) * g for a, g, e in zip(self.accumulation, grad_flat, x_flat)]

        return _unflatten(x_new_flat, x)[0]

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
"""Root mean square propagation optimizer"""
import math

from pennylane.utils import _flatten, unflatten
from pennylane.numpy import ndarray, tensor
from .adagrad import AdagradOptimizer


class RMSPropOptimizer(AdagradOptimizer):
    r"""Root mean squared propagation optimizer.

    The root mean square progation optimizer is a modified
    :class:`Adagrad optimizer <pennylane.optmimize.AdagradOptimizer>`,
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
        decay (float): the learning rate decay :math:`\gamma`
        eps (float): offset :math:`\epsilon` added for numerical stability
            (see :class:`Adagrad <pennylane.optmimize.AdagradOptimizer>`)

    """

    def __init__(self, stepsize=0.01, decay=0.9, eps=1e-8):
        super().__init__(stepsize)
        self.decay = decay
        self.eps = eps

    def apply_grad(self, grad, args):
        r"""Update the variables args to take a single optimization step. Flattens and unflattens
        the inputs to maintain nested iterables as the parameters of the optimization.

        Args:
            grad (tuple [array]): the gradient of the objective function at
                point :math:`x^{(t)}`: :math:`\nabla f(x^{(t)})`.
            args (tuple): the current value of the variables :math:`x^{(t)}`.

        Returns:
            list [array]: the new values :math:`x^{(t+1)}`
        """
        args_new = list(args)

        if self.accumulation is None:
            self.accumulation = [None] * len(args)

        trained_index = 0
        for index, arg in enumerate(args):
            if getattr(arg, "requires_grad", True):
                x_flat = _flatten(arg)
                grad_flat = list(_flatten(grad[trained_index]))
                trained_index += 1

                self._update_accumulation(index, grad_flat)

                x_new_flat = [
                    e - (self.stepsize / math.sqrt(a + self.eps)) * g
                    for a, g, e in zip(self.accumulation[index], grad_flat, x_flat)
                ]

                args_new[index] = unflatten(x_new_flat, arg)

                if isinstance(arg, ndarray):
                    # Due to a bug in unflatten, input PennyLane tensors
                    # are being unwrapped. Here, we cast them back to PennyLane
                    # tensors.
                    # TODO: remove when the following is fixed:
                    # https://github.com/PennyLaneAI/pennylane/issues/966
                    args_new[index] = args_new[index].view(tensor)
                    args_new[index].requires_grad = True

        return args_new

    def _update_accumulation(self, index, grad_flat):
        r"""Update the accumulation with the flattened gradient.

        Args:
            index (int): index of argument to update.
            grad_flat (list): flattened form of the gradient.
        """
        if self.accumulation[index] is None:
            self.accumulation[index] = [(1 - self.decay) * g * g for g in grad_flat]
        else:
            self.accumulation[index] = [
                self.decay * a + (1 - self.decay) * g * g
                for a, g in zip(self.accumulation[index], grad_flat)
            ]

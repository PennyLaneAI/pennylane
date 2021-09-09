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
"""Nesterov momentum optimizer"""
from pennylane._grad import grad as get_gradient
from pennylane.utils import _flatten, unflatten
from pennylane.numpy import ndarray, tensor
from .momentum import MomentumOptimizer


class NesterovMomentumOptimizer(MomentumOptimizer):
    r"""Gradient-descent optimizer with Nesterov momentum.

    Nesterov Momentum works like the
    :class:`Momentum optimizer <.pennylane.optimize.MomentumOptimizer>`,
    but shifts the current input by the momentum term when computing the gradient
    of the objective function:

    .. math:: a^{(t+1)} = m a^{(t)} + \eta \nabla f(x^{(t)} - m a^{(t)}).

    The user defined parameters are:

    * :math:`\eta`: the step size
    * :math:`m`: the momentum

    Args:
        stepsize (float): user-defined hyperparameter :math:`\eta`
        momentum (float): user-defined hyperparameter :math:`m`
    """

    def compute_grad(self, objective_fn, args, kwargs, grad_fn=None):
        r"""Compute gradient of the objective function at at the shifted point :math:`(x -
        m\times\text{accumulation})` and return it along with the objective function forward pass
        (if available).

        Args:
            objective_fn (function): the objective function for optimization.
            args (tuple): tuple of NumPy arrays containing the current values for the
                objection function.
            kwargs (dict): keyword arguments for the objective function.
            grad_fn (function): optional gradient function of the objective function with respect to
                the variables ``x``. If ``None``, the gradient function is computed automatically.
                Must return the same shape of tuple [array] as the autograd derivative.

        Returns:
            tuple [array]: the NumPy array containing the gradient :math:`\nabla f(x^{(t)})` and the
            objective function output. If ``grad_fn`` is provided, the objective function
            will not be evaluted and instead ``None`` will be returned.
        """
        shifted_args = list(args)

        trainable_args = []
        for arg in args:
            if getattr(arg, "requires_grad", True):
                trainable_args.append(arg)

        if self.accumulation:
            for index, arg in enumerate(trainable_args):
                if self.accumulation[index]:
                    x_flat = _flatten(arg)
                    acc = _flatten(self.accumulation[index])

                    shifted_x_flat = [e - self.momentum * a for a, e in zip(acc, x_flat)]

                    shifted_args[index] = unflatten(shifted_x_flat, arg)

                    if isinstance(shifted_args[index], ndarray):
                        # Due to a bug in unflatten, input PennyLane tensors
                        # are being unwrapped. Here, we cast them back to PennyLane
                        # tensors.
                        # TODO: remove when the following is fixed:
                        # https://github.com/PennyLaneAI/pennylane/issues/966
                        shifted_args[index] = shifted_args[index].view(tensor)
                        shifted_args[index].requires_grad = True

        g = get_gradient(objective_fn) if grad_fn is None else grad_fn
        grad = g(*shifted_args, **kwargs)
        forward = getattr(g, "forward", None)

        if len(trainable_args) == 1:
            grad = (grad,)

        return grad, forward

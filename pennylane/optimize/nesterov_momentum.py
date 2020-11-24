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
"""Nesterov momentum optimizer"""
from pennylane._grad import grad as get_gradient
from pennylane.utils import _flatten, unflatten
from .momentum import MomentumOptimizer


class NesterovMomentumOptimizer(MomentumOptimizer):
    r"""Gradient-descent optimizer with Nesterov momentum.

    Nesterov Momentum works like the :class:`Momentum optimizer <.pennylane.optimize.MomentumOptimizer>`,
    but shifts the current input by the momentum term when computing the gradient of the objective function:

    .. math:: a^{(t+1)} = m a^{(t)} + \eta \nabla f(x^{(t)} - m a^{(t)}).

    The user defined parameters are:

    * :math:`\eta`: the step size
    * :math:`m`: the momentum

    Args:
        stepsize (float): user-defined hyperparameter :math:`\eta`
        momentum (float): user-defined hyperparameter :math:`m`
    """

    def compute_grad(self, objective_fn, x, grad_fn=None):
        r"""Compute gradient of the objective_fn at at the shifted point :math:`(x -
        m\times\text{accumulation})` and return it along with the objective function
        forward pass (if available).

        Args:
            objective_fn (function): the objective function for optimization
            x (array): NumPy array containing the current values of the variables to be updated
            grad_fn (function): Optional gradient function of the objective function with respect to
                the variables ``x``. If ``None``, the gradient function is computed automatically.

        Returns:
            tuple: The NumPy array containing the gradient :math:`\nabla f(x^{(t)})` and the
                objective function output. If ``grad_fn`` is provided, the objective function
                will not be evaluted and instead ``None`` will be returned.
        """

        x_flat = _flatten(x)

        if self.accumulation is None:
            shifted_x_flat = list(x_flat)
        else:
            shifted_x_flat = [e - self.momentum * a for a, e in zip(self.accumulation, x_flat)]

        shifted_x = unflatten(shifted_x_flat, x)

        g = get_gradient(objective_fn) if grad_fn is None else grad_fn
        grad = g(shifted_x)
        forward = getattr(g, "forward", None)

        return grad, forward

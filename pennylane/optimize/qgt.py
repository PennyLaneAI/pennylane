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
"""QGT optimizer"""
#pylint: disable=too-many-branches
from collections import Sequence

import autograd
import autograd.numpy as np

from pennylane.utils import _flatten, unflatten

from .gradient_descent import GradientDescentOptimizer


class QGTOptimizer(GradientDescentOptimizer):
    r"""Optimizer with adaptive learning rate.

    Because this optimizer performs quantum evaluations to determine
    the learning rate, the objective function must be a QNode.

    Args:
        stepsize (float): the user-defined stepsize parameter :math:`\eta`
        tol (float): tolerance used for inverse
    """
    def __init__(self, stepsize=0.01, tol=1e-6):
        self._stepsize = stepsize
        self.tol = tol

    def step(self, qnode, x):
        """Update x with one step of the optimizer.

        Args:
            qnode (QNode): the QNode for optimization
            x (array): NumPy array containing the current values of the variables to be updated

        Returns:
            array: the new variable values :math:`x^{(t+1)}`
        """
        if not hasattr(qnode, "metric_tensor"):
            raise ValueError("Objective function must be a QNode")

        g = self.compute_grad(qnode, x)
        self.metric_tensor = qnode.metric_tensor(x)
        x_out = self.apply_grad(g, x)
        return x_out

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
        grad_flat = list(_flatten(grad))
        x_flat = _flatten(x)

        # inverse metric tensor
        # note: in the cases where np.abs(self.metric_tensor) > self.tol, we
        # should raise a warning to let the user know that tol should be reduced
        G_inv = np.where(np.abs(self.metric_tensor) > self.tol, 1 / self.metric_tensor, 0)

        x_new_flat = [e - self._stepsize * g * d for e, g, d in zip(x_flat, G_inv, grad_flat)]

        return unflatten(x_new_flat, x)

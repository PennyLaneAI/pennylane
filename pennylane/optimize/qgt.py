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
    r"""Optimizer with adaptive learning rate, via calculation
    of the quantum gradient tensor.

    The QGT optimizer uses a step-dependent learning rate,
    with the learning rate dependent on the pseudo-inverse
    of the quantum gradient tensor :math:`G`:

    .. math::
        x^{(t+1)} = x^{(t)} - \eta G(f(x^{(t)})^{-1} \nabla f(x^{(t)}).

    Consider a quantum node represented by the variational quantum circuit

    .. math::

        U(\mathbf{\theta}) = W(\theta_{i+1}, \dots, \theta_{N})X(\theta_{i})
        V(\theta_1, \dots, \theta_{i-1}),

    where :math:`X(\theta_{i}) = e^{i\theta_i K_i}` (i.e., the gate :math:`K_i`
    is the *generator* of the parametrized operation :math:`X(\theta_i)` corresponding
    to the :math:`i`-th parameter).

    The quantum gradient tensor element is thus given by:

    .. math::

        G_{ij} = \langle 0 \mid V^{-1} K_i K_j V \mid 0\rangle
        - \langle 0 \mid V^{-1} K_i V \mid 0\rangle\right
        \langle 0 \mid V^{-1} K_j V \mid 0\rangle\right

    For parametric layer :math:`\ell` in the variational quantum circuit
    containing $n$ parameters, an :math:`n\times n` block diagonal submatrix
    of the quantum gradient tensor :math:`G_{ij}^{(\ell)}` is computed
    by directly querying the quantum device.

    .. note::

        As the quantum gradient tensor is computed by evaluating subcircuits
        of the original quantum node, the QGT optimizer
        **only supports single QNodes** as objective functions.

    Args:
        stepsize (float): the user-defined hyperparameter :math:`\eta`
        diag_approx (bool): if ``True``, forces a diagonal approximation
            where the calculated metric tensor only contains diagonal
            elements :math:`G_{ii}`. In some cases, this may reduce the
            time taken per optimization step.
        tol (float): tolerance used when finding the inverse of the
            quantum gradient tensor
    """
    def __init__(self, stepsize=0.01, diag_approx=False):
        self._stepsize = stepsize
        self.diag_approx = diag_approx
        self.metric_tensor = None

    def step(self, qnode, x, recompute_tensor=True):
        """Update x with one step of the optimizer.

        Args:
            qnode (QNode): the QNode for optimization
            x (array): NumPy array containing the current values of the variables to be updated
            recompute_tensor (bool): whether or not the metric tensor should
                be recomputed. If not, the metric tensor from the previous
                optimization step is used.

        Returns:
            array: the new variable values :math:`x^{(t+1)}`
        """
        if not hasattr(qnode, "metric_tensor"):
            raise ValueError("Objective function must be a QNode")

        if recompute_tensor or self.metric_tensor is None:
            self.metric_tensor = qnode.metric_tensor(x, diag_approx=self.diag_approx)

        g = self.compute_grad(qnode, x)
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
        grad_flat = np.array(list(_flatten(grad)))
        x_flat = np.array(list(_flatten(x)))

        # pseudo-inverse metric tensor
        G_inv = np.linalg.pinv(self.metric_tensor)

        x_new_flat = x_flat - self._stepsize * G_inv @ grad_flat

        return unflatten(x_new_flat, x)

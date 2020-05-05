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
"""Quantum natural gradient optimizer"""
# pylint: disable=too-many-branches

import numpy as np

from pennylane.utils import _flatten, unflatten
from .gradient_descent import GradientDescentOptimizer


class QNGOptimizer(GradientDescentOptimizer):
    r"""Optimizer with adaptive learning rate, via calculation
    of the diagonal or block-diagonal approximation to the Fubini-Study metric tensor.
    A quantum generalization of natural gradient descent.

    The QNG optimizer uses a step- and parameter-dependent learning rate,
    with the learning rate dependent on the pseudo-inverse
    of the Fubini-Study metric tensor :math:`g`:

    .. math::
        x^{(t+1)} = x^{(t)} - \eta g(f(x^{(t)}))^{-1} \nabla f(x^{(t)}),

    where :math:`f(x^{(t)}) = \langle 0 | U(x^{(t)})^\dagger \hat{B} U(x^{(t)}) | 0 \rangle`
    is an expectation value of some observable measured on the variational
    quantum circuit :math:`U(x^{(t)})`.

    Consider a quantum node represented by the variational quantum circuit

    .. math::

        U(\mathbf{\theta}) = W(\theta_{i+1}, \dots, \theta_{N})X(\theta_{i})
        V(\theta_1, \dots, \theta_{i-1}),

    where all parametrized gates can be written of the form :math:`X(\theta_{i}) = e^{i\theta_i K_i}`.
    That is, the gate :math:`K_i` is the *generator* of the parametrized operation :math:`X(\theta_i)`
    corresponding to the :math:`i`-th parameter.

    For each parametric layer :math:`\ell` in the variational quantum circuit
    containing :math:`n` parameters, the :math:`n\times n` block-diagonal submatrix
    of the Fubini-Study tensor :math:`g_{ij}^{(\ell)}` is calculated directly on the
    quantum device in a single evaluation:

    .. math::

        g_{ij}^{(\ell)} = \langle \psi_\ell | K_i K_j | \psi_\ell \rangle
        - \langle \psi_\ell | K_i | \psi_\ell\rangle
        \langle \psi_\ell |K_j | \psi_\ell\rangle

    where :math:`|\psi_\ell\rangle =  V(\theta_1, \dots, \theta_{i-1})|0\rangle`
    (that is, :math:`|\psi_\ell\rangle` is the quantum state prior to the application
    of parameterized layer :math:`\ell`).

    Combining the quantum natural gradient optimizer with the analytic parameter-shift
    rule to optimize a variational circuit with :math:`d` parameters and :math:`L` layers,
    a total of :math:`2d+L` quantum evaluations are required per optimization step.

    For more details, see:

        James Stokes, Josh Izaac, Nathan Killoran, Giuseppe Carleo.
        "Quantum Natural Gradient." `arXiv:1909.02108 <https://arxiv.org/abs/1909.02108>`_, 2019.

    .. note::

        The QNG optimizer **only supports single QNodes** as objective functions.

        In particular:

        * For hybrid classical-quantum models, the "mixed geometry" of the model
          makes it unclear which metric should be used for which parameter.
          For example, parameters of quantum nodes are better suited to
          one metric (such as the QNG), whereas others (e.g., parameters of classical nodes)
          are likely better suited to another metric.

        * For multi-QNode models, we don't know what geometry is appropriate
          if a parameter is shared amongst several QNodes.

    .. seealso::

        See the :ref:`quantum natural gradient example <quantum_natural_gradient>`
        for more details on Fubini-Study metric tensor and this optimization class.

    Args:
        stepsize (float): the user-defined hyperparameter :math:`\eta`
        diag_approx (bool): If ``True``, forces a diagonal approximation
            where the calculated metric tensor only contains diagonal
            elements :math:`G_{ii}`. In some cases, this may reduce the
            time taken per optimization step.
        lam (float): metric tensor regularization :math:`G_{ij}+\lambda I`
            to be applied at each optimization step
    """

    def __init__(self, stepsize=0.01, diag_approx=False, lam=0):
        super().__init__(stepsize)
        self.diag_approx = diag_approx
        self.metric_tensor = None
        self.lam = lam

    def step(self, qnode, x, recompute_tensor=True, metric_tensor_fn=None):
        """Update x with one step of the optimizer.

        Args:
            qnode (QNode): the QNode for optimization
            x (array): NumPy array containing the current values of the variables to be updated
            recompute_tensor (bool): Whether or not the metric tensor should
                be recomputed. If not, the metric tensor from the previous
                optimization step is used.
            metric_tensor_fn (function): Optional metric tensor function
                with respect to the variables ``x``.
                If ``None``, the metric tensor function is computed automatically.

        Returns:
            array: the new variable values :math:`x^{(t+1)}`
        """
        # pylint: disable=arguments-differ
        if not hasattr(qnode, "metric_tensor") and not metric_tensor_fn:
            raise ValueError(
                "The objective function must either be encoded as a single QNode or "
                "a VQECost object for the natural gradient to be automatically computed. "
                "Otherwise, metric_tensor_fn must be explicitly provided to the optimizer."
            )

        if recompute_tensor or self.metric_tensor is None:
            if not metric_tensor_fn:
                # pseudo-inverse metric tensor
                self.metric_tensor = qnode.metric_tensor([x], diag_approx=self.diag_approx)
            else:
                self.metric_tensor = metric_tensor_fn([x], diag_approx=self.diag_approx)
            self.metric_tensor += self.lam * np.identity(self.metric_tensor.shape[0])

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
        x_new_flat = x_flat - self._stepsize * np.linalg.solve(self.metric_tensor, grad_flat)
        return unflatten(x_new_flat, x)

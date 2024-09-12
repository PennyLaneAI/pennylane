# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Quantum natural gradient optimizer with momentum"""

# pylint: disable=too-many-branches
# pylint: disable=too-many-arguments
from pennylane import numpy as pnp
from pennylane.utils import _flatten, unflatten

from .qng import QNGOptimizer


class MomentumQNGOptimizer(QNGOptimizer):
    r"""A generalization of the Quantum Natural Gradient (QNG) optimizer by considering a discrete-time Langevin equation
    with QNG force. For details of the theory and derivation of Momentum-QNG, please, see:

        Oleksandr Borysenko, Mykhailo Bratchenko, Ilya Lukin, Mykola Luhanko, Ihor Omelchenko,
        Andrii Sotnikov and Alessandro Lomi.
        "Application of Langevin Dynamics to Advance the Quantum Natural Gradient Optimization Algorithm"
        `arXiv:2409.01978 <https://arxiv.org/abs/2409.01978>`__

    We are grateful to David Wierichs for his generous help with the multi-argument variant of the MomentumQNGOptimizer class.

    ``MomentumQNGOptimizer`` is a subclass of the ``QNGOptimizer`` class and requires one additional
    hyperparameter (the momentum coefficient) :math:`0 \leq \rho < 1`, the default value being :math:`\rho=0.9`. For :math:`\rho=0` Momentum-QNG
    reduces to the basic QNG.
    In this way, the parameter update rule in Momentum-QNG reads:

    .. math::
        x^{(t+1)} = x^{(t)} + \rho (x^{(t)} - x^{(t-1)}) - \eta g(f(x^{(t)}))^{-1} \nabla f(x^{(t)}),

    where :math:`\eta` is a stepsize (learning rate) value, :math:`g(f(x^{(t)}))^{-1}` is the pseudo-inverse
    of the Fubini-Study metric tensor and :math:`f(x^{(t)}) = \langle 0 | U(x^{(t)})^\dagger \hat{B} U(x^{(t)}) | 0 \rangle`
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
        "Quantum Natural Gradient."
        `Quantum 4, 269 <https://doi.org/10.22331/q-2020-05-25-269>`_, 2020.

    .. note::

        The QNG optimizer supports using a single :class:`~.QNode` as the objective function. Alternatively,
        the metric tensor can directly be provided to the :func:`step` method of the optimizer,
        using the ``metric_tensor_fn`` keyword argument.

        For the following cases, providing ``metric_tensor_fn`` may be useful:

        * For hybrid classical-quantum models, the "mixed geometry" of the model
          makes it unclear which metric should be used for which parameter.
          For example, parameters of quantum nodes are better suited to
          one metric (such as the QNG), whereas others (e.g., parameters of classical nodes)
          are likely better suited to another metric.

        * For multi-QNode models, we don't know what geometry is appropriate
          if a parameter is shared amongst several QNodes.

    **Examples:**

    TODO

    .. seealso::

        Also see the examples from the reference above, benchmarking the Momentum-QNG optimizer
        against the basic QNG, Momentum and Adam:
        - `QAOA <https://github.com/borbysh/Momentum-QNG/blob/main/QAOA_depth4.ipynb>`__
        - `VQE <https://github.com/borbysh/Momentum-QNG/blob/main/portfolio_optimization.ipynb>`__

    Keyword Args:
        stepsize=0.01 (float): the user-defined hyperparameter :math:`\eta`
        momentum=0.9 (float): the user-defined hyperparameter :math:`\rho`
        approx (str): Which approximation of the metric tensor to compute.

            - If ``None``, the full metric tensor is computed

            - If ``"block-diag"``, the block-diagonal approximation is computed, reducing
              the number of evaluated circuits significantly.

            - If ``"diag"``, only the diagonal approximation is computed, slightly
              reducing the classical overhead but not the quantum resources
              (compared to ``"block-diag"``).

        lam=0 (float): metric tensor regularization :math:`G_{ij}+\lambda I`
            to be applied at each optimization step
    """

    def __init__(self, stepsize=0.01, momentum=0.9, approx="block-diag", lam=0):
        super().__init__(stepsize)
        self.momentum = momentum
        self.accumulation = None

    def apply_grad(self, grad, args):
        r"""Update the parameter array :math:`x` for a single optimization step. Flattens and
        unflattens the inputs to maintain nested iterables as the parameters of the optimization.

        Args:
            grad (array): The gradient of the objective
                function at point :math:`x^{(t)}`: :math:`\nabla f(x^{(t)})`
            args (array): the current value of the variables :math:`x^{(t)}`

        Returns:
            array: the new values :math:`x^{(t+1)}`
        """
        args_new = list(args)

        if self.accumulation is None:
            self.accumulation = [pnp.zeros_like(g) for g in grad]

        mt = self.metric_tensor if isinstance(self.metric_tensor, tuple) else (self.metric_tensor,)

        trained_index = 0

        for index, arg in enumerate(args):
            if getattr(arg, "requires_grad", False):
                grad_flat = pnp.array(list(_flatten(grad[trained_index])))
                # self.metric_tensor has already been reshaped to 2D, matching flat gradient.
                qng_update = pnp.linalg.solve(mt[trained_index], grad_flat)

                self.accumulation[trained_index] *= self.momentum
                self.accumulation[trained_index] += self.stepsize * unflatten(
                    qng_update, grad[trained_index]
                )
                args_new[index] = arg - self.accumulation[trained_index]

                trained_index += 1

        return tuple(args_new)

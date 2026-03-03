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


from pennylane import numpy as pnp

from .qng import QNGOptimizer, _flatten_np, _unflatten_np


class MomentumQNGOptimizer(QNGOptimizer):
    r"""A generalization of the Quantum Natural Gradient (QNG) optimizer by considering a discrete-time Langevin equation
    with QNG force. For details of the theory and derivation of Momentum-QNG, please see:

        Oleksandr Borysenko, Mykhailo Bratchenko, Ilya Lukin, Mykola Luhanko, Ihor Omelchenko,
        Andrii Sotnikov and Alessandro Lomi.
        "Application of Langevin Dynamics to Advance the Quantum Natural Gradient Optimization Algorithm"
        `arXiv:2409.01978 <https://arxiv.org/abs/2409.01978>`__

    We are grateful to David Wierichs for his generous help with the multi-argument variant of the ``MomentumQNGOptimizer`` class.

    ``MomentumQNGOptimizer`` is a subclass of ``QNGOptimizer`` that requires one additional
    hyperparameter (the momentum coefficient) :math:`0 \leq \rho < 1`, the default value being :math:`\rho=0.9`. For :math:`\rho=0` Momentum-QNG
    reduces to the basic QNG.
    In this way, the parameter update rule in Momentum-QNG reads:

    .. math::
        x^{(t+1)} = x^{(t)} + \rho (x^{(t)} - x^{(t-1)}) - \eta g(f(x^{(t)}))^{-1} \nabla f(x^{(t)}),

    where :math:`\eta` is a stepsize (learning rate) value, :math:`g(f(x^{(t)}))^{-1}` is the pseudo-inverse
    of the Fubini-Study metric tensor and :math:`f(x^{(t)}) = \langle 0 | U(x^{(t)})^\dagger \hat{B} U(x^{(t)}) | 0 \rangle`
    is an expectation value of some observable measured on the variational
    quantum circuit :math:`U(x^{(t)})`.

    Args:
        stepsize (float): the user-defined hyperparameter :math:`\eta` (default value: 0.01).
        momentum (float): the user-defined hyperparameter :math:`\rho` (default value: 0.9).
        approx (str): approximation method for the metric tensor (default value: "block-diag").

            - If ``None``, the full metric tensor is computed.

            - If ``"block-diag"``, the block-diagonal approximation is computed, reducing
              the number of evaluated circuits significantly.

            - If ``"diag"``, only the diagonal approximation is computed, slightly
              reducing the classical overhead but not the quantum resources
              (compared to ``"block-diag"``).

        lam (float): metric tensor regularization :math:`G_{ij}+\lambda I`
            to be applied at each optimization step (default value: 0).

    **Examples:**

    Consider an objective function realized as a :class:`~.QNode` that returns the
    expectation value of a Hamiltonian.

    >>> dev = qml.device("default.qubit", wires=(0, 1, "aux"))
    >>> @qml.qnode(dev)
    ... def circuit(params):
    ...     qml.RX(params[0], wires=0)
    ...     qml.RY(params[1], wires=0)
    ...     return qml.expval(qml.X(0))

    Once constructed, the cost function can be passed directly to the
    optimizer's :meth:`~.step` function. In addition to the standard learning
    rate, the ``MomentumQNGOptimizer`` takes a ``momentum`` parameter:

    >>> eta = 0.01
    >>> rho = 0.93
    >>> init_params = qml.numpy.array([0.5, 0.23], requires_grad=True)
    >>> opt = qml.MomentumQNGOptimizer(stepsize=eta, momentum=rho)
    >>> theta_new = opt.step(circuit, init_params)
    >>> theta_new
    tensor([0.50437193, 0.18562052], requires_grad=True)

    An alternative function to calculate the metric tensor of the QNode can be provided to ``step``
    via the ``metric_tensor_fn`` keyword argument, see :class:`~.pennylane.QNGOptimizer` for
    details.

    .. seealso::

        For details on quantum natural gradient, see :class:`~.pennylane.QNGOptimizer`.
        See :class:`~.pennylane.MomentumOptimizer` for a first-order optimizer with momentum.
        Also see the examples from the reference above, benchmarking the Momentum-QNG optimizer
        against the basic QNG, Momentum and Adam:

        - `QAOA <https://github.com/borbysh/Momentum-QNG/blob/main/QAOA_depth4.ipynb>`__
        - `VQE <https://github.com/borbysh/Momentum-QNG/blob/main/portfolio_optimization.ipynb>`__

        See :class:`~.MomentumQNGOptimizerQJIT` for an Optax-like and ``jax.jit``/``qml.qjit``-compatible implementation.

    """

    def __init__(self, stepsize=0.01, momentum=0.9, approx="block-diag", lam=0):
        super().__init__(stepsize, approx, lam)
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

        metric_tensor = (
            self.metric_tensor if isinstance(self.metric_tensor, tuple) else (self.metric_tensor,)
        )

        trained_index = 0

        for index, arg in enumerate(args):
            if getattr(arg, "requires_grad", False):
                grad_flat = pnp.array(list(_flatten_np(grad[trained_index])))
                # self.metric_tensor has already been reshaped to 2D, matching flat gradient.
                qng_update = pnp.linalg.pinv(metric_tensor[trained_index]) @ grad_flat

                self.accumulation[trained_index] *= self.momentum
                self.accumulation[trained_index] += self.stepsize * _unflatten_np(
                    qng_update, grad[trained_index]
                )
                args_new[index] = arg - self.accumulation[trained_index]

                trained_index += 1

        return tuple(args_new)

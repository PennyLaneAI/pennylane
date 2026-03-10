# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Quantum natural gradient optimizer with momentum for Jax/Catalyst interface"""

from pennylane import math

from .qng_qjit import QNGOptimizerQJIT


class MomentumQNGOptimizerQJIT(QNGOptimizerQJIT):
    r"""Optax-like and ``jax.jit``/``qml.qjit``-compatible implementation of the :class:`~.MomentumQNGOptimizer`,
    a generalized Quantum Natural Gradient (QNG) optimizer considering a discrete-time Langevin equation
    with QNG force.

    For more theoretical details, see the :class:`~.MomentumQNGOptimizer` documentation.

    .. note::

        Please be aware of the following:

        - As with ``MomentumQNGOptimizer``, ``MomentumQNGOptimizerQJIT`` supports a single QNode to encode the objective function.

        - ``MomentumQNGOptimizerQJIT`` does not support any QNode with multiple arguments. A potential workaround
          would be to combine all parameters into a single objective function argument.

        - ``MomentumQNGOptimizerQJIT`` does not work correctly if there is any classical processing in the QNode circuit
          (e.g., ``2 * theta`` as a gate parameter).

    Parameters:
        stepsize (float): the stepsize hyperparameter (default value: 0.01).
        momentum (float): the momentum coefficient hyperparameter (default value: 0.9).
        approx (str): approximation method for the metric tensor (default value: "block-diag").

            - If ``None``, the full metric tensor is computed

            - If ``"block-diag"``, the block-diagonal approximation is computed, reducing
              the number of evaluated circuits significantly

            - If ``"diag"``, the diagonal approximation is computed, slightly
              reducing the classical overhead but not the quantum resources
              (compared to ``"block-diag"``)

        lam (float): metric tensor regularization to be applied at each optimization step (default value: 0).

    **Example:**

    Consider a hybrid workflow to optimize an objective function defined by a quantum circuit.
    To make the entire workflow faster, the update step and the whole optimization
    can be just-in-time compiled using the :func:`~.qjit` decorator:

    .. code-block:: python

        import pennylane as qml
        import jax.numpy as jnp

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(params):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=1)
            return qml.expval(qml.Z(0) + qml.X(1))

        opt = qml.MomentumQNGOptimizerQJIT(stepsize=0.1, momentum=0.2)

        @qml.qjit
        def update_step_qjit(i, args):
            params, state = args
            return opt.step(circuit, params, state)

        @qml.qjit
        def optimization_qjit(params, iters):
            state = opt.init(params)
            args = (params, state)
            params, state = qml.for_loop(iters)(update_step_qjit)(args)
            return params

    >>> params = jnp.array([0.1, 0.2])
    >>> iters = 1000
    >>> optimization_qjit(params=params, iters=iters)
    Array([ 3.14159265, -1.57079633], dtype=float64)

    Make sure you are using the ``lightning.qubit`` device along with ``qml.qjit``.
    """

    def __init__(self, stepsize=0.01, momentum=0.9, approx="block-diag", lam=0):
        super().__init__(stepsize, approx, lam)
        self.momentum = momentum

    def init(self, params):
        """Return the initial state of the optimizer. This state is always initialized as an
        array of zeros with the same shape and type of the given array of parameters.

        Args:
            params (array): QNode parameters

        Returns:
            array: initial state of the optimizer
        """
        # pylint:disable=no-self-use
        return math.zeros_like(params)

    def _apply_grad(self, mt, grad, params, state):
        """Update the optimizer's state and the array of parameters for a single optimization
        step according to the Quantum Natural Gradient algorithm with momentum.
        """
        shape = math.shape(grad)
        grad_flat = math.flatten(grad)
        update_flat = math.linalg.pinv(mt) @ grad_flat
        update = math.reshape(update_flat, shape)
        state = self.momentum * state + self.stepsize * update
        new_params = params - state
        return new_params, state

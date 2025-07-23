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
"""Quantum natural gradient optimizer for Jax/Catalyst interface"""

from pennylane import math
from pennylane.compiler import active_compiler
from pennylane.gradients.metric_tensor import metric_tensor
from pennylane.workflow import QNode

has_jax = True
try:
    import jax
except ModuleNotFoundError:
    has_jax = False


class QNGOptimizerQJIT:
    r"""Optax-like and ``jax.jit``/``qml.qjit``-compatible implementation of the :class:`~.QNGOptimizer`,
    a step- and parameter-dependent learning rate optimizer, leveraging a reparameterization of
    the optimization space based on the Fubini-Study metric tensor.

    For more theoretical details, see the :class:`~.QNGOptimizer` documentation.

    .. note::

        Please be aware of the following:

            - As with ``QNGOptimizer``, ``QNGOptimizerQJIT`` supports a single QNode to encode the objective function.

            - ``QNGOptimizerQJIT`` does not support any QNode with multiple arguments. A potential workaround
              would be to combine all parameters into a single objective function argument.

            - ``QNGOptimizerQJIT`` does not work correctly if there is any classical processing in the QNode circuit
              (e.g., ``2 * theta`` as a gate parameter).

    Args:
        stepsize (float): the user-defined stepsize hyperparameter (default value: 0.01).
        approx (str): approximation method for the metric tensor (default value: "block-diag").

            - If ``None``, the full metric tensor is computed.

            - If ``"block-diag"``, the block-diagonal approximation is computed, reducing
              the number of evaluated circuits significantly.

            - If ``"diag"``, the diagonal approximation is computed, slightly
              reducing the classical overhead but not the quantum resources
              (compared to ``"block-diag"``).

        lam (float): metric tensor regularization to be applied at each optimization step (default value: 0).

    **Example:**

    Consider a hybrid workflow to optimize an objective function defined by a quantum circuit.
    To make the optimization faster, the entire workflow can be just-in-time compiled using
    the ``qml.qjit`` decorator:

    .. code-block:: python

        import pennylane as qml
        import jax.numpy as jnp

        @qml.qjit(autograph=True)
        def workflow():
            dev = qml.device("lightning.qubit", wires=2)

            @qml.qnode(dev)
            def circuit(params):
                qml.RX(params[0], wires=0)
                qml.RY(params[1], wires=1)
                return qml.expval(qml.Z(0) + qml.X(1))

            opt = qml.QNGOptimizerQJIT(stepsize=0.2)

            params = jnp.array([0.1, 0.2])
            state = opt.init(params)
            for _ in range(100):
                params, state = opt.step(circuit, params, state)

            return params

    >>> workflow()
    Array([ 3.14159265, -1.57079633], dtype=float64)

    Make sure you are using the ``lightning.qubit`` device along with ``qml.qjit`` with ``autograph`` enabled.
    Using ``qml.qjit`` on the whole workflow with ``autograph`` not enabled may lead to a substantial increase
    in compilation time and no runtime benefits.

    The ``jax.jit`` decorator should not be used on the entire workflow.
    However, it can be used with the ``default.qubit`` device to just-in-time
    compile the ``step`` (or ``step_and_cost``) method of the optimizer, leading
    to a significative increase in runtime performance:

    .. code-block:: python

        import pennylane as qml
        import jax.numpy as jnp
        import jax
        from functools import partial

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(params):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=1)
            return qml.expval(qml.Z(0) + qml.X(1))

        opt = qml.QNGOptimizerQJIT(stepsize=0.2)
        step = jax.jit(partial(opt.step, circuit))

        params = jnp.array([0.1, 0.2])
        state = opt.init(params)
        for _ in range(100):
            params, state = step(params, state)

    >>> params
    Array([ 3.14159265, -1.57079633], dtype=float64)
    """

    def __init__(self, stepsize=0.01, approx="block-diag", lam=0):
        self.stepsize = stepsize
        self.approx = approx
        self.lam = lam

    def init(self, params):
        """Return the initial state of the optimizer.

        Args:
            params (array): QNode parameters

        Returns:
            None

        .. note::

            Since the Quantum Natural Gradient (QNG) algorithm doesn't actually require any particular state,
            this method always returns an empty ``None`` state. However, it is provided to match
            the ``optax``-like interface for all Jax-based quantum-specific optimizers.

        """
        # pylint:disable=unused-argument
        # pylint:disable=no-self-use
        return None

    def step(self, qnode, params, state, **kwargs):
        """Update the QNode parameters and the optimizer's state for a single optimization step.

        Args:
            qnode (QNode): QNode objective function to be optimized
            params (array): QNode parameters to be updated
            state: current state of the optimizer
            **kwargs : variable-length keyword arguments for the QNode

        Returns:
            tuple: (new parameters values, new optimizer's state)
        """
        mt = self._get_metric_tensor(qnode, params, **kwargs)
        grad = self._get_grad(qnode, params, **kwargs)
        new_params, new_state = self._apply_grad(mt, grad, params, state)
        return new_params, new_state

    def step_and_cost(self, qnode, params, state, **kwargs):
        """Update the QNode parameters and the optimizer's state for a single optimization step
        and return the corresponding objective function value prior to the step.

        Args:
            qnode (QNode): QNode objective function to be optimized
            params (array): QNode parameters to be updated
            state: current state of the optimizer
            **kwargs : variable-length keyword arguments for the QNode

        Returns:
            tuple: (new parameters values, new optimizer's state, objective function value)
        """
        mt = self._get_metric_tensor(qnode, params, **kwargs)
        cost, grad = self._get_value_and_grad(qnode, params, **kwargs)
        new_params, new_state = self._apply_grad(mt, grad, params, state)
        return new_params, new_state, cost

    @staticmethod
    def _get_grad(qnode, params, **kwargs):
        """Return the gradient of the QNode objective function at the given point. The method is implemented to dispatch
        to Catalyst when it is required (e.g. when using ``qml.qjit``) or to fall back to Jax otherwise.

        Raise an ``ModuleNotFoundError`` if the required package is not installed.
        """
        if active_compiler() == "catalyst":
            import catalyst  # pylint: disable=import-outside-toplevel

            return catalyst.grad(qnode)(params, **kwargs)
        if has_jax:
            return jax.grad(qnode)(params, **kwargs)
        raise ModuleNotFoundError("Jax is required.")  # pragma: no cover

    @staticmethod
    def _get_value_and_grad(qnode, params, **kwargs):
        """Return the value and the gradient of the QNode objective function at the given point. The method is implemented
        to dispatch to Catalyst when it is required (e.g. when using ``qml.qjit``) or to fall back to Jax otherwise.

        Raise an ``ModuleNotFoundError`` if the required package is not installed.
        """
        if active_compiler() == "catalyst":
            import catalyst  # pylint: disable=import-outside-toplevel

            return catalyst.value_and_grad(qnode)(params, **kwargs)
        if has_jax:
            return jax.value_and_grad(qnode)(params, **kwargs)
        raise ModuleNotFoundError("Jax is required.")  # pragma: no cover

    def _get_metric_tensor(self, qnode, params, **kwargs):
        """Compute the metric tensor of the QNode objective function at the given point using the method specified
        by the optimizer's ``approx`` attribute. It returns the reshaped matrix after applying the regularization
        given by the optimizer's ``lam`` attribute.

        Raise a ``ValueError`` if the given objective function is not encoded as a QNode.
        """
        # pylint: disable=not-callable
        if not isinstance(qnode, QNode):
            raise ValueError(
                "The objective function must be encoded as a single QNode to use the Quantum Natural Gradient optimizer."
            )
        mt = metric_tensor(qnode, approx=self.approx)(params, **kwargs)
        # reshape tensor into a matrix (acting on the flat grad vector)
        shape = math.shape(mt)
        size = 1 if shape == () else math.prod(shape[: len(shape) // 2])
        mt_matrix = math.reshape(mt, (size, size))
        # apply regularization for matrix inversion
        if self.lam != 0:
            mt_matrix += self.lam * math.eye(size, like=mt_matrix)
        return mt_matrix

    def _apply_grad(self, mt, grad, params, state):
        """Update the parameter array ``params`` for a single optimization step according to the Quantum
        Natural Gradient algorithm. The method doesn't perform any transformation on ``state`` since the QNG
        optimizer doesn't actually require any particular state.
        """
        shape = math.shape(grad)
        grad_flat = math.flatten(grad)
        update_flat = math.linalg.pinv(mt) @ grad_flat
        update = math.reshape(update_flat, shape)
        new_params = params - self.stepsize * update
        return new_params, state

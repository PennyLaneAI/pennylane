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
"""Quantum natural gradient optimizer"""
import numbers
from collections.abc import Iterable

from pennylane import math
from pennylane import numpy as pnp
from pennylane.gradients.metric_tensor import metric_tensor
from pennylane.wires import Wires
from pennylane.workflow import QNode

from .gradient_descent import GradientDescentOptimizer


def _reshape_and_regularize(tensor, lam):
    shape = math.shape(tensor)
    size = 1 if shape == () else math.prod(shape[: len(shape) // 2])
    tensor = math.reshape(tensor, (size, size))
    # Add regularization
    tensor += lam * math.eye(size, like=tensor)
    return tensor


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
    of parametrized layer :math:`\ell`).

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

    Args:
        stepsize (float): the user-defined hyperparameter :math:`\eta` (default value: 0.01).
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

    For VQE/VQE-like problems, the objective function for the optimizer can be
    realized as a :class:`~.QNode` that returns the expectation value of a Hamiltonian.

    >>> dev = qml.device("default.qubit", wires=(0, 1, "aux"))
    >>> @qml.qnode(dev)
    ... def circuit(params):
    ...     qml.RX(params[0], wires=0)
    ...     qml.RY(params[1], wires=0)
    ...     return qml.expval(qml.X(0) + qml.X(1))

    Once constructed, the cost function can be passed directly to the
    optimizer's :meth:`~.step` function:

    >>> from pennylane import numpy as np
    >>> eta = 0.01
    >>> init_params = np.array([0.011, 0.012])
    >>> opt = qml.QNGOptimizer(eta)
    >>> theta_new = opt.step(circuit, init_params)
    >>> theta_new
    tensor([ 0.01100528, -0.02799954], requires_grad=True)

    An alternative function to calculate the metric tensor of the QNode can be provided to ``step``
    via the ``metric_tensor_fn`` keyword argument. For example, we can provide a function
    to calculate the metric tensor via the adjoint method.

    >>> adj_metric_tensor = qml.adjoint_metric_tensor(circuit)
    >>> opt.step(circuit, init_params, metric_tensor_fn=adj_metric_tensor)
    tensor([ 0.01100528, -0.02799954], requires_grad=True)

    .. note::

        If the objective function takes multiple trainable arguments, ``QNGOptimizer`` applies the
        metric tensor for each argument individually. This means that "correlations" between
        parameters from different arguments are not taken into account. In order to take all
        correlations into account within the optimization, consider combining all parameters into
        one objective function argument.

    .. seealso::

        See the `quantum natural gradient example <demo:demos/tutorial_quantum_natural_gradient>`_
        for more details on the Fubini-Study metric tensor and this optimization class.

        See :class:`~.QNGOptimizerQJIT` for an Optax-like and ``jax.jit``/``qml.qjit``-compatible implementation.
    """

    def __init__(self, stepsize=0.01, approx="block-diag", lam=0):
        super().__init__(stepsize)

        self.approx = approx

        self.metric_tensor = None
        self.lam = lam

    def step_and_cost(
        self, qnode, *args, grad_fn=None, recompute_tensor=True, metric_tensor_fn=None, **kwargs
    ):
        """Update the parameter array :math:`x` with one step of the optimizer and return the
        corresponding objective function value prior to the step.

        Args:
            qnode (QNode): the QNode for optimization
            *args : variable length argument list for qnode
            grad_fn (function): optional gradient function of the
                qnode with respect to the variables ``*args``.
                If ``None``, the gradient function is computed automatically.
                Must return a ``tuple[array]`` with the same number of elements as ``*args``.
                Each array of the tuple should have the same shape as the corresponding argument.
            recompute_tensor (bool): Whether or not the metric tensor should
                be recomputed. If not, the metric tensor from the previous
                optimization step is used.
            metric_tensor_fn (function): Optional metric tensor function
                with respect to the variables ``args``.
                If ``None``, the metric tensor function is computed automatically.
            **kwargs : variable length of keyword arguments for the qnode

        Returns:
            tuple: the new variable values :math:`x^{(t+1)}` and the objective function output
            prior to the step
        """
        # pylint: disable=arguments-differ
        if not isinstance(qnode, QNode) and metric_tensor_fn is None:
            raise ValueError(
                "The objective function must be encoded as a single QNode for the natural gradient "
                "to be automatically computed. Otherwise, metric_tensor_fn must be explicitly "
                "provided to the optimizer."
            )

        if recompute_tensor or self.metric_tensor is None:
            if metric_tensor_fn is None:
                metric_tensor_fn = metric_tensor(qnode, approx=self.approx)

            mt = metric_tensor_fn(*args, **kwargs)
            if isinstance(mt, tuple):
                self.metric_tensor = tuple(_reshape_and_regularize(_mt, self.lam) for _mt in mt)
            else:
                self.metric_tensor = _reshape_and_regularize(mt, self.lam)

        g, forward = self.compute_grad(qnode, args, kwargs, grad_fn=grad_fn)
        new_args = self.apply_grad(g, args)

        if forward is None:
            forward = qnode(*args, **kwargs)

        if len(new_args) == 1:
            new_args = new_args[0]

        return new_args, forward

    # pylint: disable=arguments-differ
    def step(
        self, qnode, *args, grad_fn=None, recompute_tensor=True, metric_tensor_fn=None, **kwargs
    ):
        """Update the parameter array :math:`x` with one step of the optimizer.

        Args:
            qnode (QNode): the QNode for optimization
            *args : variable length argument list for qnode
            grad_fn (function): optional gradient function of the
                qnode with respect to the variables ``*args``.
                If ``None``, the gradient function is computed automatically.
                Must return a ``tuple[array]`` with the same number of elements as ``*args``.
                Each array of the tuple should have the same shape as the corresponding argument.
            recompute_tensor (bool): Whether or not the metric tensor should
                be recomputed. If not, the metric tensor from the previous
                optimization step is used.
            metric_tensor_fn (function): Optional metric tensor function
                with respect to the variables ``args``.
                If ``None``, the metric tensor function is computed automatically.
            **kwargs : variable length of keyword arguments for the qnode

        Returns:
            array: the new variable values :math:`x^{(t+1)}`
        """
        new_args, _ = self.step_and_cost(
            qnode,
            *args,
            grad_fn=grad_fn,
            recompute_tensor=recompute_tensor,
            metric_tensor_fn=metric_tensor_fn,
            **kwargs,
        )
        return new_args

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
        mt = self.metric_tensor if isinstance(self.metric_tensor, tuple) else (self.metric_tensor,)

        trained_index = 0
        for index, arg in enumerate(args):
            if getattr(arg, "requires_grad", False):
                grad_flat = pnp.array(list(_flatten_np(grad[trained_index])))
                # self.metric_tensor has already been reshaped to 2D, matching flat gradient.
                update = pnp.linalg.pinv(mt[trained_index]) @ grad_flat
                args_new[index] = arg - self.stepsize * _unflatten_np(update, grad[trained_index])

                trained_index += 1

        return tuple(args_new)


def _flatten_np(x):
    """Iterate recursively through an arbitrarily nested structure in depth-first order.

    See also :func:`_unflatten`.

    Args:
        x (array, Iterable, Any): each element of an array or an Iterable may itself be any of these types

    Yields:
        Any: elements of x in depth-first order
    """
    if isinstance(x, pnp.ndarray):
        yield from _flatten_np(
            x.flat
        )  # should we allow object arrays? or just "yield from x.flat"?
    elif isinstance(x, Wires):
        # Reursive calls to flatten `Wires` will cause infinite recursion (`Wires` atoms are `Wires`).
        # Since Wires are always flat, just yield.
        yield from x
    elif isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
        for item in x:
            yield from _flatten_np(item)
    else:
        yield x


def _unflatten_np_dispatch(flat, model):
    """Restores an arbitrary nested structure to a flattened iterable.

    See also :func:`_flatten`.

    Args:
        flat (array): 1D array of items
        model (array, Iterable, Number): model nested structure

    Raises:
        TypeError: if ``model`` contains an object of unsupported type

    Returns:
        Union[array, list, Any], array: first elements of flat arranged into the nested
        structure of model, unused elements of flat
    """
    if isinstance(model, (numbers.Number, str)):
        return flat[0], flat[1:]

    if isinstance(model, pnp.ndarray):
        idx = model.size
        res = pnp.array(flat)[:idx].reshape(model.shape)
        return res, flat[idx:]

    if isinstance(model, Iterable):
        res = []
        for x in model:
            val, flat = _unflatten_np_dispatch(flat, x)
            res.append(val)
        return res, flat

    raise TypeError(f"Unsupported type in the model: {type(model)}")


def _unflatten_np(flat, model):
    """Wrapper for :func:`_unflatten`.

    Args:
        flat (array): 1D array of items
        model (array, Iterable, Number): model nested structure

    Raises:
        ValueError: if ``flat`` has more elements than ``model``
    """
    res, tail = _unflatten_np_dispatch(pnp.asarray(flat), model)
    if len(tail) != 0:
        raise ValueError("Flattened iterable has more elements than the model.")
    return res

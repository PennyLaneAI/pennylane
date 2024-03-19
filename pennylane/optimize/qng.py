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
# pylint: disable=too-many-branches
# pylint: disable=too-many-arguments
from pennylane import numpy as pnp

import pennylane as qml
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

    For VQE/VQE-like problems, the objective function for the optimizer can be
    realized as :class:`~.QNode` that returns the expectation value of a Hamiltonian.

    >>> dev = qml.device("default.qubit", wires=(0, 1, "aux"))
    >>> @qml.qnode(dev)
    ... def circuit(params):
    ...     qml.RX(params[0], wires=0)
    ...     qml.RY(params[1], wires=0)
    ...     return qml.expval(qml.X(0) + qml.X(1))

    Once constructed, the cost function can be passed directly to the
    optimizer's ``step`` function:

    >>> eta = 0.01
    >>> init_params = np.array([0.011, 0.012])
    >>> opt = qml.QNGOptimizer(eta)
    >>> theta_new = opt.step(circuit, init_params)
    >>> theta_new
    tensor([ 0.01100528, -0.02799954], requires_grad=True)

    An alternative function to calculate the metric tensor of the QNode
    can be provided to :meth:`~.step`
    via the ``metric_tensor_fn`` keyword argument.  For example, we can provide a function
    to calculate the metric tensor via the adjoint method.

    >>> adj_metric_tensor = qml.adjoint_metric_tensor(circuit, circuit.device)
    >>> opt.step(circuit, init_params, metric_tensor_fn=adj_metric_tensor)
    tensor([ 0.01100528, -0.02799954], requires_grad=True)

    .. seealso::

        See the :doc:`quantum natural gradient example <demo:demos/tutorial_quantum_natural_gradient>`
        for more details on Fubini-Study metric tensor and this optimization class.

    Keyword Args:
        stepsize=0.01 (float): the user-defined hyperparameter :math:`\eta`
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
        if not isinstance(qnode, qml.QNode) and metric_tensor_fn is None:
            raise ValueError(
                "The objective function must be encoded as a single QNode for the natural gradient "
                "to be automatically computed. Otherwise, metric_tensor_fn must be explicitly "
                "provided to the optimizer."
            )

        if recompute_tensor or self.metric_tensor is None:
            if metric_tensor_fn is None:
                metric_tensor_fn = qml.metric_tensor(qnode, approx=self.approx)

            _metric_tensor = metric_tensor_fn(*args, **kwargs)
            # Reshape metric tensor to be square
            shape = qml.math.shape(_metric_tensor)
            size = qml.math.prod(shape[: len(shape) // 2])
            self.metric_tensor = qml.math.reshape(_metric_tensor, (size, size))
            # Add regularization
            self.metric_tensor = self.metric_tensor + self.lam * qml.math.eye(
                size, like=_metric_tensor
            )

        g, forward = self.compute_grad(qnode, args, kwargs, grad_fn=grad_fn)
        new_args = pnp.array(self.apply_grad(g, args), requires_grad=True)

        if forward is None:
            forward = qnode(*args, **kwargs)

        # Note: for now, we only have single element lists as the new
        # arguments, but this might change, see TODO below.
        # Once the other approach is implemented, we need to unwrap from list
        # if one argument for a cleaner return.
        # if len(new_args) == 1:
        return new_args[0], forward

        # TODO: The scenario of the following return statement is not implemented
        # yet, as currently only a single metric tensor can be processed.
        # An optimizer refactor is needed to accomodate for this (similar to other
        # optimizers for which `apply_grad` will have to be patched to allow for
        # tuple-valued gradients to be processed)
        #
        # For multiple QNode arguments, `qml.jacobian` and `qml.metric_tensor`
        # return a tuple of arrays. Each of the gradient arrays has to be processed
        # together with the corresponding array in the metric tensor tuple.
        # This requires modifications of the `GradientDescentOptimizer` base class
        # as none of the optimizers accomodate for this use case.
        # return new_args, forward

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
        grad_flat = pnp.array(list(_flatten(grad)))
        x_flat = pnp.array(list(_flatten(args)))
        x_new_flat = x_flat - self.stepsize * pnp.linalg.solve(self.metric_tensor, grad_flat)
        return unflatten(x_new_flat, args)

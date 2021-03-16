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
"""Rosalin optimizer"""
# pylint: disable=too-many-instance-attributes
from scipy.stats import multinomial

import pennylane as qml
from pennylane import numpy as np


from .gradient_descent import GradientDescentOptimizer


class ShotAdaptiveOptimizer(GradientDescentOptimizer):
    r"""Optimizer with adaptive shot rate, via calculation
    of the variances of the parameter-shift gradient.

    Args:
        stepsize (float): the user-defined hyperparameter :math:`\eta`
        diag_approx (bool): If ``True``, forces a diagonal approximation
            where the calculated metric tensor only contains diagonal
            elements :math:`G_{ii}`. In some cases, this may reduce the
            time taken per optimization step.
        lam (float): metric tensor regularization :math:`G_{ij}+\lambda I`
            to be applied at each optimization step

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

        The QNG optimizer supports single QNodes or :class:`~.ExpvalCost` objects as objective functions.
        Alternatively, the metric tensor can directly be provided to the :func:`step` method of the optimizer,
        using the ``metric_tensor_fn`` argument.

        For the following cases, providing metric_tensor_fn may be useful:

        * For hybrid classical-quantum models, the "mixed geometry" of the model
          makes it unclear which metric should be used for which parameter.
          For example, parameters of quantum nodes are better suited to
          one metric (such as the QNG), whereas others (e.g., parameters of classical nodes)
          are likely better suited to another metric.

        * For multi-QNode models, we don't know what geometry is appropriate
          if a parameter is shared amongst several QNodes.

        If the objective function is VQE/VQE-like, i.e., a function of a group
        of QNodes that share an ansatz, there are two ways to use the optimizer:

        * Realize the objective function as an :class:`~.ExpvalCost` object, which has
          a ``metric_tensor`` method.

        * Manually provide the ``metric_tensor_fn`` corresponding to the metric tensor of
          of the QNode(s) involved in the objective function.

    **Examples:**

    For VQE/VQE-like problems, the objective function for the optimizer can be
    realized as an ExpvalCost object.

    >>> dev = qml.device("default.qubit", wires=1)
    >>> def circuit(params, wires=0):
    ...     qml.RX(params[0], wires=wires)
    ...     qml.RY(params[1], wires=wires)
    >>> coeffs = [1, 1]
    >>> obs = [qml.PauliX(0), qml.PauliZ(0)]
    >>> H = qml.Hamiltonian(coeffs, obs)
    >>> cost_fn = qml.ExpvalCost(circuit, H, dev)

    Once constructed, the cost function can be passed directly to the
    optimizer's ``step`` function:

    >>> eta = 0.01
    >>> init_params = [0.011, 0.012]
    >>> opt = qml.QNGOptimizer(eta)
    >>> theta_new = opt.step(cost_fn, init_params)
    >>> print(theta_new)
    [0.011445239214543481, -0.027519522461477233]

    Alternatively, the same objective function can be used for the optimizer
    by manually providing the ``metric_tensor_fn``.

    >>> qnodes = qml.map(circuit, obs, dev, 'expval')
    >>> cost_fn = qml.dot(coeffs, qnodes)
    >>> eta = 0.01
    >>> init_params = [0.011, 0.012]
    >>> opt = qml.QNGOptimizer(eta)
    >>> theta_new = opt.step(cost_fn, init_params, metric_tensor_fn=qnodes.qnodes[0].metric_tensor)
    >>> print(theta_new)
    [0.011445239214543481, -0.027519522461477233]
    """
    def __init__(self, min_shots, mu=0.99, b=1e-6, stepsize=0.07):
        # hyperparameters
        self.min_shots = min_shots
        self.mu = mu  # running average constant
        self.b = b  # regularization bias
        self.lipschitz = None

        # keep track of the total number of shots used
        self.shots_used = 0
        # total number of iterations
        self.k = 0
        # Number of shots per parameter
        self.s = None

        # Running average of the parameter gradients
        self.chi = None
        # Running average of the variance of the parameter gradients
        self.xi = None

        super().__init__(stepsize=stepsize)

    @staticmethod
    def estimate_hamiltonian(qnodes, coeffs, shots, argnums, *args, **kwargs):
        """Returns an array containing length ``shots`` single-shot estimates
        of the Hamiltonian. The shots are distributed randomly over
        the terms in the Hamiltonian, as per a Multinomial distribution.

        Args:
            qnodes (Sequence[.tape.QNode]): Sequence of QNodes, each one when evaluated
                returning the corresponding expectation value of a term in the Hamiltonian.
            coeffs (Sequence[float]): Sequences of coefficients corresponding to
                each term in the Hamiltonian.
            shots (int): The number of shots used to estimate the Hamiltonian expectation
                value. These shots are distributed over the terms in the Hamiltonian,
                as per a Multinomial distribution.
            argnums (Sequence[int]): the QNode argument indices which are trainable
            *args: Arguments to the QNodes
            **kwargs: Keyword arguments to the QNodes

        Returns:
            array[float]: the single-shot gradients of the Hamiltonian expectation value
        """

        # determine the shot probability per term
        prob_shots = np.abs(coeffs) / np.sum(np.abs(coeffs))

        # construct the multinomial distribution, and sample
        # from it to determine how many shots to apply per term
        si = multinomial(n=shots, p=prob_shots)
        shots_per_term = si.rvs()[0]

        grads = []

        for h, c, p, s in zip(qnodes, coeffs, prob_shots, shots_per_term):

            # if the number of shots is 0, do nothing
            if s == 0:
                continue

            # set the QNode device shots
            h.device.shots = [(1, s)]

            jacs = []
            for i in argnums:
                j = qml.jacobian(h, argnum=i)(*args, **kwargs)

                if s == 1:
                    j = np.expand_dims(j, 0)

                # Divide each term by the probability per shot. This is
                # because we are sampling one at a time.
                jacs.append(c * j / p)

            grads.append(jacs)

        return [np.concatenate(i) for i in zip(*grads)]

    @staticmethod
    def check_device(dev):
        r"""Verifies that the device used by the objective function is non-analytic.

        Args:
            dev (.Device): the device to verify

        Raises:
            ValueError: if the device is analytic
        """
        if dev.analytic:
            raise ValueError("The Rosalin optimizer can only be used with non-analytic devices")

    def check_learning_rate(self, coeffs):
        r"""Verifies that the learning rate is less than 2 over the Lipschitz constant,
        where the Lipschitz constant is given by :math:`\sum |c_i|` for Hamiltonian
        coefficients :math:`c_i`.

        Args:
            coeffs (Sequence[float]): the coefficients of the terms in the Hamiltonian

        Raises:
            ValueError: if the learning rate is large than :math:`2/\sum |c_i|`
        """
        self.lipschitz = np.sum(np.abs(coeffs))

        if self._stepsize > 2 / self.lipschitz:
            raise ValueError("The learning rate must be less than ", 2 / self.lipschitz)

    def compute_grad(self, objective_fn, *args, **kwargs):
        r"""Compute gradient of the objective function, as well as the variance of the gradient,
        at the given point.

        Args:
            objective_fn (function): the objective function for optimization
            *args: arguments to the objective function
            **kwargs: keyword arguments to the objective function

        Returns:
            tuple[array[float]]: a tuple of NumPy arrays containing the gradient
            :math:`\nabla f(x^{(t)})` and the variance of the gradient.
        """
        max_s = max(np.concatenate([i.flatten() for i in self.s]))

        if isinstance(objective_fn, qml.ExpvalCost):
            self.check_device(objective_fn.qnodes[0].device)

            qnodes = objective_fn.qnodes
            coeffs = objective_fn.hamiltonian.coeffs
            original_shots = qnodes[0].device.shots

            if self.lipschitz is None:
                self.check_learning_rate(coeffs)

            try:
                grads = self.estimate_hamiltonian(
                    qnodes, coeffs, max_s, self.trainable_args, *args, **kwargs
                )
            finally:
                qnodes[0].device.shots = original_shots

        elif isinstance(objective_fn, qml.tape.QNode):
            self.check_device(objective_fn.device)
            original_shots = objective_fn.device.shots

            if self.lipschitz is None:
                self.check_learning_rate(1)

            try:
                objective_fn.device.shots = [(1, max_s)]
                grads = [
                    qml.jacobian(objective_fn, argnum=i)(*args, **kwargs)
                    for i in self.trainable_args
                ]
            finally:
                objective_fn.device.shots = original_shots
        else:
            raise ValueError(
                "The objective function must either be encoded as a single QNode or "
                "an ExpvalCost object for the Rosalin optimizer. "
            )

        # grads will have dimension [max(self.s), *params.shape]
        # For each parameter, we want to truncate the number of shots to self.s[idx],
        # and take the mean and the variance.
        gradients = []
        gradient_variances = []

        for i, grad in enumerate(grads):
            p_ind = list(np.ndindex(*grad.shape[1:]))

            g = np.zeros_like(grad[0])
            s = np.zeros_like(grad[0])

            for idx in p_ind:
                g[idx] = np.mean(grad[(slice(0, self.s[i][idx]),) + idx])
                s[idx] = np.var(grad[(slice(0, self.s[i][idx]),) + idx], ddof=1)

            gradients.append(g)
            gradient_variances.append(s)

        return gradients, gradient_variances

    def step(self, objective_fn, *args, **kwargs):
        """Update trainable arguments with one step of the optimizer.

        Args:
            objective_fn (function): the objective function for optimization
            *args : Variable length argument list for objective function
            **kwargs : variable length of keyword arguments for the objective function

        Returns:
            list[array]: the new variable values :math:`x^{(t+1)}`.
            If single arg is provided, list [array] is replaced by array.
        """

        self.trainable_args = set()

        for index, arg in enumerate(args):
            if getattr(arg, "requires_grad", True):
                self.trainable_args |= {index}

        if self.s is None:
            # Number of shots per parameter
            self.s = [
                np.zeros_like(a, dtype=np.int64) + self.min_shots
                for i, a in enumerate(args)
                if i in self.trainable_args
            ]

        # keep track of the number of shots run
        sum_s = np.sum(np.concatenate([i.flatten() for i in self.s]))
        self.shots_used += int(2 * sum_s)

        # compute the gradient, as well as the variance in the gradient,
        # using the number of shots determined by the array s.
        grads, grad_variances = self.compute_grad(objective_fn, *args, **kwargs)
        new_args = self.apply_grad(grads, args)

        if self.xi is None:
            self.chi = [np.zeros_like(g, dtype=np.float64) for g in grads]
            self.xi = [np.zeros_like(g, dtype=np.float64) for g in grads]

        # running average of the gradient
        self.chi = [self.mu * c + (1 - self.mu) * g for c, g in zip(self.chi, grads)]

        # running average of the gradient variance
        self.xi = [self.mu * x + (1 - self.mu) * v for x, v in zip(self.xi, grad_variances)]

        for idx, (c, x, v) in enumerate(zip(self.chi, self.xi, grad_variances)):
            xi = x / (1 - self.mu ** (self.k + 1))
            chi = c / (1 - self.mu ** (self.k + 1))

            # determine the new optimum shots distribution for the next
            # iteration of the optimizer
            s = np.ceil(
                (2 * self.lipschitz * self._stepsize * xi)
                / (
                    (2 - self.lipschitz * self._stepsize)
                    * (chi ** 2 + self.b * (self.mu ** self.k))
                )
            )

            # apply an upper and lower bound on the new shot distributions,
            # to avoid the number of shots reducing below min(2, min_shots),
            # or growing too significantly.
            gamma = (
                (self._stepsize - self.lipschitz * self._stepsize ** 2 / 2) * chi ** 2
                - xi * self.lipschitz * self._stepsize ** 2 / (2 * s)
            ) / s

            argmax_gamma = np.unravel_index(np.argmax(gamma), gamma.shape)
            smax = s[argmax_gamma]
            self.s[idx] = np.int64(np.clip(s, min(2, self.min_shots), smax))

        self.k += 1

        # unwrap from list if one argument, cleaner return
        if len(new_args) == 1:
            return new_args[0]

        return new_args

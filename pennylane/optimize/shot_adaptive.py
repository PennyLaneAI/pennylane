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
"""Shot adaptive optimizer"""
# pylint: disable=too-many-instance-attributes,too-many-arguments


from copy import copy

import numpy as np
import scipy as sp

from pennylane import math, measurements
from pennylane._grad import jacobian
from pennylane.ops import LinearCombination
from pennylane.queuing import apply
from pennylane.tape import make_qscript
from pennylane.workflow import QNode, construct_tape, set_shots

from .gradient_descent import GradientDescentOptimizer


class ShotAdaptiveOptimizer(GradientDescentOptimizer):
    r"""Optimizer where the shot rate is adaptively calculated using the variances of the
    parameter-shift gradient.

    By keeping a running average of the parameter-shift gradient and the *variance*
    of the parameter-shift gradient, this optimizer frugally distributes a shot
    budget across the partial derivatives of each parameter.

    In addition, weighted random sampling can be used to further distribute the
    shot budget across the local terms from which the Hamiltonian is constructed.

    .. note::

        The shot adaptive optimizer only supports single QNode objects as objective functions.
        The bound device must also be instantiated with a finite number of shots.

    Args:
        min_shots (int): the minimum number of shots used to estimate the expectations
            of each term in the Hamiltonian. Note that this must be larger than 2 for the variance
            of the gradients to be computed.
        term_sampling (str): the random sampling algorithm to multinomially distribute the shot
            budget across terms in the Hamiltonian expectation value. Currently, only
            ``"weighted_random_sampling"`` is supported. The default value is ``None``, which
            disables the random sampling behaviour.
        mu (float): the running average constant :math:`\mu \in [0, 1]`. Used to control how
            quickly the number of shots recommended for each gradient component changes (default value: 0.99).
        b (float): the regularization bias. The bias should be kept small, but non-zero (default value: 1e-06).
        stepsize (float): the learning rate :math:`\eta` (default value: 0.07). The learning rate *must* be such
            that :math:`\eta < 2/L = 2/\sum_i|c_i|`, where:

            * :math:`L \leq \sum_i|c_i|` is the bound on the `Lipschitz constant
              <https://en.wikipedia.org/wiki/Lipschitz_continuity>`__ of the variational quantum
              algorithm objective function, and

            * :math:`c_i` are the coefficients of the Hamiltonian used in the objective function.

    **Example**

    For VQE/VQE-like problems, the objective function for the optimizer can be realized
    as a :class:`~.QNode` object measuring the expectation of a :class:`~.ops.LinearCombination`.

    >>> from pennylane import numpy as np
    >>> from functools import partial
    >>> coeffs = [2, 4, -1, 5, 2]
    >>> obs = [
    ...   qml.X(1),
    ...   qml.Z(1),
    ...   qml.X(0) @ qml.X(1),
    ...   qml.Y(0) @ qml.Y(1),
    ...   qml.Z(0) @ qml.Z(1)
    ... ]
    >>> H = qml.Hamiltonian(coeffs, obs)
    >>> dev = qml.device("default.qubit", wires=2)
    >>> @partial(qml.set_shots, shots=100)

    ... @qml.qnode(dev)
    ... def cost(weights):
    ...     qml.StronglyEntanglingLayers(weights, wires=range(2))
    ...     return qml.expval(H)

    Once constructed, the cost function can be passed directly to the
    optimizer's ``step`` method. The attributes ``opt.shots_used`` and
    ``opt.total_shots_used`` can be used to track the number of shots per
    iteration, and across the life of the optimizer, respectively.

    >>> shape = qml.templates.StronglyEntanglingLayers.shape(n_layers=2, n_wires=2)
    >>> params = np.random.random(shape)
    >>> opt = qml.ShotAdaptiveOptimizer(min_shots=10, term_sampling="weighted_random_sampling")
    >>> for i in range(60):
    ...    params = opt.step(cost, params)
    ...    print(f"Step {i}: cost = {cost(params):.2f}, shots_used = {opt.total_shots_used}")
    Step 0: cost = -5.69, shots_used = 240
    Step 1: cost = -2.98, shots_used = 336
    Step 2: cost = -4.97, shots_used = 624
    Step 3: cost = -5.53, shots_used = 1054
    Step 4: cost = -6.50, shots_used = 1798
    Step 5: cost = -6.68, shots_used = 2942
    Step 6: cost = -6.99, shots_used = 4350
    Step 7: cost = -6.97, shots_used = 5814
    Step 8: cost = -7.00, shots_used = 7230
    Step 9: cost = -6.69, shots_used = 9006
    Step 10: cost = -6.85, shots_used = 11286
    Step 11: cost = -6.63, shots_used = 14934
    Step 12: cost = -6.86, shots_used = 17934
    Step 13: cost = -7.19, shots_used = 22950
    Step 14: cost = -6.99, shots_used = 28302
    Step 15: cost = -7.38, shots_used = 34134
    Step 16: cost = -7.66, shots_used = 41022
    Step 17: cost = -7.21, shots_used = 48918
    Step 18: cost = -7.53, shots_used = 56286
    Step 19: cost = -7.46, shots_used = 63822
    Step 20: cost = -7.31, shots_used = 72534
    Step 21: cost = -7.23, shots_used = 82014
    Step 22: cost = -7.31, shots_used = 92838

    .. details::
        :title: Usage Details

        The shot adaptive optimizer is based on the iCANS1 optimizer by
        `Kübler et al. (2020) <https://quantum-journal.org/papers/q-2020-05-11-263/>`__, and works
        as follows:

        1. The initial step of the optimizer is performed with some specified minimum
           number of shots, :math:`s_{min}`, for all partial derivatives.

        2. The parameter-shift rule is then used to estimate the gradient :math:`g_i` with :math:`s_i` shots
           for each parameter :math:`\theta_i`, parameters, as well as the variances
           :math:`v_i` of the estimated gradients.

        3. Gradient descent is performed for each parameter :math:`\theta_i`, using
           the pre-defined learning rate :math:`\eta` and the gradient information :math:`g_i`:
           :math:`\theta_i \rightarrow \theta_i - \eta g_i`.

        4. A maximum shot number is set by maximizing the improvement in the expected gain per shot.
           For a specific parameter value, the improvement in the expected gain per shot
           is then calculated via

           .. math::
               \gamma_i = \frac{1}{s_i} \left[ \left(\eta - \frac{1}{2} L\eta^2\right)
                           g_i^2 - \frac{L\eta^2}{2s_i}v_i \right],

           where:

           * :math:`L \leq \sum_i|c_i|` is the bound on the `Lipschitz constant
             <https://en.wikipedia.org/wiki/Lipschitz_continuity>`__ of the variational quantum algorithm objective function,

           * :math:`c_i` are the coefficients of the Hamiltonian, and

           * :math:`\eta` is the learning rate, and *must* be bound such that :math:`\eta < 2/L`
             for the above expression to hold.

        5. Finally, the new values of :math:`s_{i+1}` (shots for partial derivative of parameter
           :math:`\theta_i`) is given by:

           .. math::

               s_{i+1} = \frac{2L\eta}{2-L\eta}\left(\frac{v_i}{g_i^2}\right)\propto
                     \frac{v_i}{g_i^2}.

        In addition to the above, to counteract the presence of noise in the system, a
        running average of :math:`g_i` and :math:`s_i` (:math:`\chi_i` and :math:`\xi_i` respectively)
        are used when computing :math:`\gamma_i` and :math:`s_i`.

        For more details, see:

        * Andrew Arrasmith, Lukasz Cincio, Rolando D. Somma, and Patrick J. Coles. "Operator Sampling
          for Shot-frugal Optimization in Variational Algorithms." `arXiv:2004.06252
          <https://arxiv.org/abs/2004.06252>`__ (2020).

        * Jonas M. Kübler, Andrew Arrasmith, Lukasz Cincio, and Patrick J. Coles. "An Adaptive Optimizer
          for Measurement-Frugal Variational Algorithms." `Quantum 4, 263
          <https://quantum-journal.org/papers/q-2020-05-11-263/>`__ (2020).
    """

    # pylint: disable = too-many-positional-arguments
    def __init__(self, min_shots, term_sampling=None, mu=0.99, b=1e-6, stepsize=0.07):
        self.term_sampling = term_sampling
        self.trainable_args = set()

        # hyperparameters
        self.min_shots = min_shots
        self.mu = mu  # running average constant
        self.b = b  # regularization bias
        self.lipschitz = None

        self.shots_used = 0
        """int: number of shots used on the current iteration"""

        self.total_shots_used = 0
        """int: total number of shots used across all iterations"""

        # total number of iterations
        self.k = 0
        # Number of shots per parameter
        self.s = None
        # maximum number of shots required to evaluate across all parameters
        self.max_shots = None

        # Running average of the parameter gradients
        self.chi = None
        # Running average of the variance of the parameter gradients
        self.xi = None

        super().__init__(stepsize=stepsize)

    @staticmethod
    def qnode_weighted_random_sampling(qnode, coeffs, observables, shots, argnums, *args, **kwargs):
        """Returns an array of length ``shots`` containing single-shot estimates
        of the Hamiltonian gradient. The shots are distributed randomly over
        the terms in the Hamiltonian, as per a multinomial distribution.

        Args:
            qnode (.QNode): A QNode that returns the expectation value of a Hamiltonian.
            coeffs (List[float]): The coefficients of the Hamiltonian being measured
            observables (List[Operator]]): The terms of the Hamiltonian being measured
            shots (int): The number of shots used to estimate the Hamiltonian expectation
                value. These shots are distributed over the terms in the Hamiltonian,
                as per a Multinomial distribution.
            argnums (Sequence[int]): the QNode argument indices which are trainable
            *args: Arguments to the QNode
            **kwargs: Keyword arguments to the QNode

        Returns:
            array[float]: the single-shot gradients of the Hamiltonian expectation value
        """
        qnode = copy(qnode)
        base_func = qnode.func

        # determine the shot probability per term
        prob_shots = np.abs(coeffs) / np.sum(np.abs(coeffs))

        # construct the multinomial distribution, and sample
        # from it to determine how many shots to apply per term
        si = sp.stats.multinomial(n=shots, p=prob_shots)
        shots_per_term = si.rvs()[0]

        grads = []

        for o, c, p, s in zip(observables, coeffs, prob_shots, shots_per_term, strict=True):
            # if the number of shots is 0, do nothing
            if s == 0:
                continue

            def func(*qnode_args, **qnode_kwargs):
                qs = make_qscript(base_func)(*qnode_args, **qnode_kwargs)
                for op in qs.operations:
                    apply(op)
                return measurements.expval(o)  # pylint:disable=cell-var-from-loop

            new_shots = 1 if s == 1 else [(1, int(s))]

            qnode.func = func
            qnode = set_shots(qnode, shots=new_shots)

            if s > 1:

                def cost(*args, **kwargs):

                    return math.stack(qnode(*args, **kwargs))

            else:
                cost = qnode

            jacs = jacobian(cost, argnum=argnums)(*args, **kwargs)

            if s == 1:
                jacs = [np.expand_dims(j, 0) for j in jacs]

            # Divide each term by the probability per shot. This is
            # because we are sampling one at a time.
            grads.append([(c * j / p) for j in jacs])

        return [np.concatenate(i) for i in zip(*grads, strict=True)]

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

        if self.stepsize > 2 / self.lipschitz:
            raise ValueError(f"The learning rate must be less than {2 / self.lipschitz}")

    def _single_shot_qnode_gradients(self, qnode, args, kwargs):
        """Compute the single shot gradients of a QNode."""
        tape = construct_tape(qnode)(*args, **kwargs)
        if not tape.shots:
            raise ValueError(
                "The Rosalin optimizer can only be used with qnodes "
                "that estimate expectation values with a finite number of shots."
            )
        [expval] = tape.measurements
        coeffs, observables = (
            expval.obs.terms()
            if isinstance(expval.obs, LinearCombination)
            else ([1.0], [expval.obs])
        )

        if self.lipschitz is None:
            self.check_learning_rate(coeffs)

        if self.term_sampling == "weighted_random_sampling":
            return self.qnode_weighted_random_sampling(
                qnode, coeffs, observables, self.max_shots, self.trainable_args, *args, **kwargs
            )
        if self.term_sampling is not None:
            raise ValueError(
                f"Unknown Hamiltonian term sampling method {self.term_sampling}. "
                "Only term_sampling='weighted_random_sampling' and "
                "term_sampling=None currently supported."
            )

        new_shots = [(1, int(self.max_shots))]

        def cost(*args, **kwargs):
            return math.stack(set_shots(qnode, shots=new_shots)(*args, **kwargs))

        grads = [jacobian(cost, argnum=i)(*args, **kwargs) for i in self.trainable_args]

        return grads

    def compute_grad(self, objective_fn, args, kwargs):  # pylint: disable=arguments-renamed
        r"""Compute the gradient of the objective function, as well as the variance of the gradient,
        at the given point.

        Args:
            objective_fn (function): the objective function for optimization
            args: arguments to the objective function
            kwargs: keyword arguments to the objective function

        Returns:
            tuple[array[float], array[float]]: a tuple of NumPy arrays containing the gradient
            :math:`\nabla f(x^{(t)})` and the variance of the gradient
        """
        if isinstance(objective_fn, QNode) or hasattr(objective_fn, "device"):
            grads = self._single_shot_qnode_gradients(objective_fn, args, kwargs)
        else:
            raise ValueError(
                "The objective function must be encoded as a single QNode object for the shot "
                "adaptive optimizer. "
            )

        # grads will have dimension [max(self.s), *params.shape]
        # For each parameter, we want to truncate the number of shots to self.s[idx],
        # and take the mean and the variance.
        gradients = []
        gradient_variances = []

        for i, grad in enumerate(grads):
            p_ind = np.ndindex(*grad.shape[1:])

            g = np.zeros_like(grad[0])
            s = np.zeros_like(grad[0])

            for idx in p_ind:
                grad_slice = grad[(slice(0, self.s[i][idx]),) + idx]
                g[idx] = np.mean(grad_slice)
                s[idx] = np.var(grad_slice, ddof=1)

            gradients.append(g)
            gradient_variances.append(s)

        return gradients, gradient_variances

    def step(self, objective_fn, *args, **kwargs):
        """Update trainable arguments with one step of the optimizer.

        Args:
            objective_fn (function): the objective function for optimization
            *args: variable length argument list for objective function
            **kwargs: variable length of keyword arguments for the objective function

        Returns:
            list[array]: The new variable values :math:`x^{(t+1)}`.
            If single arg is provided, list[array] is replaced by array.
        """
        self.trainable_args = set()

        for index, arg in enumerate(args):
            if getattr(arg, "requires_grad", False):
                self.trainable_args |= {index}

        if self.s is None:
            # Number of shots per parameter
            self.s = [
                np.zeros_like(a, dtype=np.int64) + self.min_shots
                for i, a in enumerate(args)
                if i in self.trainable_args
            ]

        # keep track of the number of shots run
        s = np.concatenate([i.flatten() for i in self.s])
        self.max_shots = max(s)
        self.shots_used = int(2 * np.sum(s))
        self.total_shots_used += self.shots_used

        # compute the gradient, as well as the variance in the gradient,
        # using the number of shots determined by the array s.
        grads, grad_variances = self.compute_grad(objective_fn, args, kwargs)
        new_args = self.apply_grad(grads, args)

        if self.xi is None:
            self.chi = [np.zeros_like(g, dtype=np.float64) for g in grads]
            self.xi = [np.zeros_like(g, dtype=np.float64) for g in grads]

        # running average of the gradient
        self.chi = [self.mu * c + (1 - self.mu) * g for c, g in zip(self.chi, grads, strict=True)]

        # running average of the gradient variance
        self.xi = [
            self.mu * x + (1 - self.mu) * v for x, v in zip(self.xi, grad_variances, strict=True)
        ]

        for idx, (c, x) in enumerate(zip(self.chi, self.xi, strict=True)):
            xi = x / (1 - self.mu ** (self.k + 1))
            chi = c / (1 - self.mu ** (self.k + 1))

            # determine the new optimum shots distribution for the next
            # iteration of the optimizer
            s = np.ceil(
                (2 * self.lipschitz * self.stepsize * xi)
                / ((2 - self.lipschitz * self.stepsize) * (chi**2 + self.b * (self.mu**self.k)))
            )

            # apply an upper and lower bound on the new shot distributions,
            # to avoid the number of shots reducing below min(2, min_shots),
            # or growing too significantly.
            gamma = (
                (self.stepsize - self.lipschitz * self.stepsize**2 / 2) * chi**2
                - xi * self.lipschitz * self.stepsize**2 / (2 * s)
            ) / s

            argmax_gamma = np.unravel_index(np.argmax(gamma), gamma.shape)
            smax = max(s[argmax_gamma], 2)

            self.s[idx] = np.int64(np.clip(s, max(2, self.min_shots), smax))

        self.k += 1

        # unwrap from list if one argument, cleaner return
        if len(new_args) == 1:
            return new_args[0]

        return new_args

    def step_and_cost(self, objective_fn, *args, **kwargs):
        """Update trainable arguments with one step of the optimizer and return the corresponding
        objective function value prior to the step.

        The objective function will be evaluated using the maximum number of shots
        across all parameters as determined by the optimizer during the
        optimization step.

        .. warning::

            Unlike other gradient descent optimizers, the objective function will be evaluated
            **separately** to the gradient computation, and will result in extra
            device evaluations.

        Args:
            objective_fn (function): the objective function for optimization
            *args : variable length argument list for objective function
            **kwargs : variable length of keyword arguments for the objective function

        Returns:
            tuple[list [array], float]: the new variable values :math:`x^{(t+1)}` and the objective
            function output prior to the step.
            If single arg is provided, list [array] is replaced by array.
        """
        new_args = self.step(objective_fn, *args, **kwargs)
        forward = set_shots(objective_fn, shots=int(self.max_shots))(*args, **kwargs)
        return new_args, forward

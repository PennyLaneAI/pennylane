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


class RosalinOptimizer(GradientDescentOptimizer):

    def __init__(self, min_shots, mu=0.99, b=1e-6, stepsize=0.07):
        # hyperparameters
        self.min_shots = min_shots
        self.mu = mu  # running average constant
        self.b = b    # regularization bias
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
    def estimate_hamiltonian(qnodes, coeffs, shots, *args, **kwargs):
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

            # Divide each term by the probability per shot. This is
            # because we are sampling one at a time.
            grads.append(c * qml.jacobian(h)(*args, **kwargs) / p)

        return np.concatenate(grads)

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
        if isinstance(objective_fn, qml.ExpvalCost):
            qnodes = objective_fn.qnodes
            coeffs = objective_fn.hamiltonian.coeffs
            original_shots = qnodes[0].device.shots

            if self.lipschitz is None:
                self.check_learning_rate(coeffs)

            try:
                grads = self.estimate_hamiltonian(qnodes, coeffs, np.max(self.s), *args, **kwargs)
            finally:
                qnodes[0].device.shots = original_shots

        elif isinstance(objective_fn, qml.tape.QNode):
            original_shots = objective_fn.device.shots

            if self.lipschitz is None:
                self.check_learning_rate(1)

            try:
                objective_fn.device.shots = [(1, np.max(self.s))]
                grads = qml.jacobian(objective_fn)(*args, **kwargs)
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

        p_ind = list(np.ndindex(*grads[0].shape))

        g = np.zeros_like(grads[0])
        s = np.zeros_like(grads[0])

        for idx in p_ind:
            g[idx] = np.mean(grads[(slice(0, self.s[idx]),) + idx])
            s[idx] = np.var(grads[(slice(0, self.s[idx]),) + idx], ddof=1)

        return g, s

    def step(self, objective_fn, *args, **kwargs):
        """Perform a single step of the Rosalin optimizer."""

        if self.s is None:
            # Number of shots per parameter
            self.s = np.zeros_like(args[0], dtype=np.int64) + self.min_shots

        # keep track of the number of shots run
        self.shots_used += int(2 * np.sum(self.s))

        # compute the gradient, as well as the variance in the gradient,
        # using the number of shots determined by the array s.
        grad, S = self.compute_grad(objective_fn, *args, **kwargs)

        # gradient descent update
        if not isinstance(grad, tuple):
            grads = (grad,)

        new_args = self.apply_grad(grads, args)

        if self.xi is None:
            self.chi = np.zeros_like(grad, dtype=np.float64)
            self.xi = np.zeros_like(grad, dtype=np.float64)

        # running average of the gradient variance
        self.xi = self.mu * self.xi + (1 - self.mu) * S
        xi = self.xi / (1 - self.mu ** (self.k + 1))

        # running average of the gradient
        self.chi = self.mu * self.chi + (1 - self.mu) * grad
        chi = self.chi / (1 - self.mu ** (self.k + 1))

        # determine the new optimum shots distribution for the next
        # iteration of the optimizer
        s = np.ceil(
            (2 * self.lipschitz * self._stepsize * xi)
            / ((2 - self.lipschitz * self._stepsize) * (chi ** 2 + self.b * (self.mu ** self.k)))
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
        self.s = np.clip(s, min(2, self.min_shots), smax)

        self.k += 1

        # unwrap from list if one argument, cleaner return
        if len(new_args) == 1:
            return new_args[0]

        return new_args

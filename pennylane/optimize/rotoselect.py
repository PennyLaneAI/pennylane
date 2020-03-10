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
"""Rotoselect gradient free optimizer"""

import pennylane as qml
from pennylane import numpy as np


class RotoselectOptimizer:
    r"""Rotoselect gradient free optimizer.

    The Rotoselect optimizer minimizes an objective function with respect to the parameters and
    gates of a quantum circuit without the need of calculating the gradient of the function. The
    algorithm works by updating the parameters :math:`\theta = \theta_1, \dots, \theta_D` and gate
    choices :math:`R = R_1,\dots,R_D` one at a time according to a closed-form expression for the
    optimal value of the :math:`d^{th}` parameter :math:`\theta^*_d` when the other parameters and
    gate choices are fixed:

    .. math:: \theta^*_d = \underset{\theta_d}{\text{argmin}}\left<H\right>_{\theta_d}
              = -\frac{\pi}{2} - \text{arctan2}\left(2\left<H\right>_{\theta_d=0}
              - \left<H\right>_{\theta_d=\pi/2} - \left<H\right>_{\theta_d=-\pi/2},
              \left<H\right>_{\theta_d=\pi/2} - \left<H\right>_{\theta_d=-\pi/2}\right),

    where :math:`\left<H\right>_{\theta_d}` is the expectation value of the objective function
    optimized over the parameter :math:`\theta_d`. :math:`\text{arctan2}(x, y)` computes the
    element-wise arc tangent of :math:`x/y` choosing the quadrant correctly, avoiding, in
    particular, division-by-zero when :math:`y = 0`.

    The algorithm is described in further detail in `Ostaszewski et al. (2019) <https://arxiv.org/abs/1905.09692>`_.

    Keyword Args:
        possible_generators (list[~.Operation]): List containing the possible ``pennylane.ops.qubit`` operators
            that are allowed in the circuit. Default is the set of Pauli matrices :math:`\{X, Y, Z\}`.

    **Example:**

    Initialize the optimizer, set the initial values of ``x``, choose the initial generators to
    start with, and set the number of steps to optimize over.

    >>> opt = qml.optimize.RotoselectOptimizer()
    >>> x = [0.3, 0.7]
    >>> generators = [qml.RX, qml.RY]
    >>> n_steps = 1000

    Define a cost function based on a PennyLane circuit

    >>> def cost(x, generators):
    >>>     X_1, Y_2 = circuit(x, generators=generators)
    >>>     return 0.2 * X_1 + 0.5 * Y_2

    Run the optimization step-by-step for ``n_steps`` steps.

    >>> cost_rotosel = []
    >>> for _ in range(n_steps):
    >>>     cost_rotosel.append(cost(x, generators))
    >>>     x, generators = opt.step(cost, x, generators)

    The optimized values for x should now be stored in ``x`` together with the optimal gates for
    the circuit, while steps-vs-cost can be seen by plotting ``cost_rotosel``.
    """
    # pylint: disable=too-few-public-methods

    def __init__(self, **kwargs):
        self.possible_generators = kwargs.get("possible_generators", [qml.RX, qml.RY, qml.RZ])

    def step(self, objective_fn, x, generators):
        """Update x with one step of the optimizer.

        Args:
            objective_fn (function): The objective function for optimization. It should take two lists
                of the values ``x`` and the gates ``generators`` as inputs, and return a single value.
            x (array[float]): NumPy array containing the initial values of the variables to be optimized over.
            generators (list[~.Operation]): List containing the initial ``pennylane.ops.qubit``
                operators to be used in the circuit and optimized over.

        Returns:
            array: The new variable values :math:`x^{(t+1)}` as well as the new generators.
        """
        # make sure that x is an array
        if np.ndim(x) == 0:
            x = np.array([x])

        for d, _ in enumerate(x):
            x[d], generators[d] = self._find_optimal_params(objective_fn, x, generators, d)

        return x, generators

    def _find_optimal_params(self, objective_fn, x, generators, d):
        """Optimizer for the generators.

        Optimizes for the best generator at position ``d``.

        Args:
            objective_fn (function): The objective function for optimization. It should take two lists
                of the values ``x`` and the gates ``generators`` as inputs, and return a single value.
            x (array[float]): NumPy array containing the initial values of the variables to be optimized over.
            generators (list[~.Operation]): List containing the initial ``pennylane.ops.qubit``
                operators to be used in the circuit and optimized over.
            d (int): The position in the input array ``x`` containing the value to be optimized.

        Returns:
            tuple: Tuple containing the parameter value and generator that, at position ``d`` in their
            respective lists ``x`` and ``generators``, optimizes the objective function.
        """
        params_opt_d = x[d]
        generators_opt_d = generators[d]
        params_opt_cost = objective_fn(x, generators)

        for generator in self.possible_generators:
            generators[d] = generator

            x = self._rotosolve(objective_fn, x, generators, d)
            params_cost = objective_fn(x, generators)

            # save the best paramter and generator for position d
            if params_cost <= params_opt_cost:
                params_opt_d = x[d]
                params_opt_cost = params_cost
                generators_opt_d = generator
        return params_opt_d, generators_opt_d

    @staticmethod
    def _rotosolve(objective_fn, x, generators, d):
        """The rotosolve step for one parameter and one set of generators.

        Args:
            objective_fn (function): The objective function for optimization. It should take two lists
                of the values ``x`` and the gates ``generators`` as inputs, and return a single value.
            x (array[float]): NumPy array containing the initial values of the variables to be optimized over.
            generators (list[~.Operation]): List containing the initial ``pennylane.ops.qubit``
                operators to be used in the circuit and optimized over.
            d (int): The position in the input array ``x`` containing the value to be optimized.

        Returns:
            array: The input array ``x`` with the value at position ``d`` optimized.
        """
        # helper function for x[d] = theta
        def insert(x, d, theta):
            x[d] = theta
            return x

        H_0 = float(objective_fn(insert(x, d, 0), generators))
        H_p = float(objective_fn(insert(x, d, np.pi / 2), generators))
        H_m = float(objective_fn(insert(x, d, -np.pi / 2), generators))

        a = np.arctan2(2 * H_0 - H_p - H_m, H_p - H_m)

        x[d] = -np.pi / 2 - a

        if x[d] <= -np.pi:
            x[d] += 2 * np.pi
        return x

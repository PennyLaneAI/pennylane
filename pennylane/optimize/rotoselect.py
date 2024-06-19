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
"""Rotoselect gradient free optimizer"""

import numpy as np

import pennylane as qml
from pennylane.utils import _flatten, unflatten


class RotoselectOptimizer:
    r"""Rotoselect gradient-free optimizer.

    The Rotoselect optimizer minimizes an objective function with respect to the rotation gates and
    parameters of a quantum circuit without the need for calculating the gradient of the function.
    The algorithm works by updating the parameters :math:`\theta = \theta_1, \dots, \theta_D`
    and rotation gate choices :math:`R = R_1,\dots,R_D` one at a time according to a closed-form
    expression for the optimal value of the :math:`d^{th}` parameter :math:`\theta^*_d` when the
    other parameters and gate choices are fixed:

    .. math:: \theta^*_d = \underset{\theta_d}{\text{argmin}}\left<H\right>_{\theta_d}
              = -\frac{\pi}{2} - \text{arctan2}\left(2\left<H\right>_{\theta_d=0}
              - \left<H\right>_{\theta_d=\pi/2} - \left<H\right>_{\theta_d=-\pi/2},
              \left<H\right>_{\theta_d=\pi/2} - \left<H\right>_{\theta_d=-\pi/2}\right),

    where :math:`\left<H\right>_{\theta_d}` is the expectation value of the objective function
    optimized over the parameter :math:`\theta_d`. :math:`\text{arctan2}(x, y)` computes the
    element-wise arc tangent of :math:`x/y` choosing the quadrant correctly, avoiding, in
    particular, division-by-zero when :math:`y = 0`.

    Which parameters and gates that should be optimized over is decided in the user-defined cost
    function, where :math:`R` is a list of parametrized rotation gates in a quantum circuit, along
    with their respective parameters :math:`\theta` for the circuit and its gates. Note that the
    number of generators should match the number of parameters.

    The algorithm is described in further detail in
    `Ostaszewski et al. (2021) <https://doi.org/10.22331/q-2021-01-28-391>`_.

    Args:
        possible_generators (list[~.Operation]): List containing the possible
            ``pennylane.ops.qubit`` operators that are allowed in the circuit.
            Default is the set of Pauli rotations :math:`\{R_x, R_y, R_z\}`.

    **Example:**

    Initialize the Rotoselect optimizer, set the initial values of  the weights ``x``,
    choose the initial generators, and set the number of steps to optimize over.

    >>> opt = qml.optimize.RotoselectOptimizer()
    >>> x = [0.3, 0.7]
    >>> generators = [qml.RX, qml.RY]
    >>> n_steps = 10

    Set up the PennyLane circuit using the ``default.qubit`` simulator device.

    >>> dev = qml.device("default.qubit", shots=None, wires=2)
    >>> @qml.qnode(dev)
    ... def circuit(params, generators=None):  # generators will be passed as a keyword arg
    ...     generators[0](params[0], wires=0)
    ...     generators[1](params[1], wires=1)
    ...     qml.CNOT(wires=[0, 1])
    ...     return qml.expval(qml.Z(0)), qml.expval(qml.X(1))

    Define a cost function based on the above circuit.

    >>> def cost(x, generators):
    ...     Z_1, X_2 = circuit(x, generators=generators)
    ...     return 0.2 * Z_1 + 0.5 * X_2

    Run the optimization step-by-step for ``n_steps`` steps.

    >>> cost_rotosel = []
    >>> for _ in range(n_steps):
    ...     cost_rotosel.append(cost(x, generators))
    ...     x, generators = opt.step(cost, x, generators)

    The optimized values for x should now be stored in ``x`` together with the optimal gates for
    the circuit, while steps-vs-cost can be seen by plotting ``cost_rotosel``.
    """

    # pylint: disable=too-few-public-methods

    def __init__(self, possible_generators=None):
        self.possible_generators = possible_generators or [qml.RX, qml.RY, qml.RZ]

    def step_and_cost(self, objective_fn, x, generators, **kwargs):
        """Update trainable arguments with one step of the optimizer and return the corresponding
        objective function value prior to the step.

        Args:
            objective_fn (function): The objective function for optimization. It must have the
                signature ``objective_fn(x, generators=None)`` with a sequence of the values ``x``
                and a list of the gates ``generators`` as inputs, returning a single value.
            x (Union[Sequence[float], float]): sequence containing the initial values of the
                variables to be optimized over or a single float with the initial value
            generators (list[~.Operation]): list containing the initial ``pennylane.ops.qubit``
                operators to be used in the circuit and optimized over
            **kwargs : variable length of keyword arguments for the objective function.

        Returns:
            tuple: the new variable values :math:`x^{(t+1)}`, the new generators, and the objective
            function output prior to the step
        """
        x_new, generators = self.step(objective_fn, x, generators, **kwargs)

        return x_new, generators, objective_fn(x, generators, **kwargs)

    def step(self, objective_fn, x, generators, **kwargs):
        r"""Update trainable arguments with one step of the optimizer.

        Args:
            objective_fn (function): The objective function for optimization. It must have the
                signature ``objective_fn(x, generators=None)`` with a sequence of the values ``x``
                and a list of the gates ``generators`` as inputs, returning a single value.
            x (Union[Sequence[float], float]): sequence containing the initial values of the
                variables to be optimized over or a single float with the initial value
            generators (list[~.Operation]): list containing the initial ``pennylane.ops.qubit``
                operators to be used in the circuit and optimized over
            **kwargs : variable length of keyword arguments for the objective function.

        Returns:
            array: The new variable values :math:`x^{(t+1)}` as well as the new generators.
        """
        x_flat = np.fromiter(_flatten(x), dtype=float)
        # wrap the objective function so that it accepts the flattened parameter array
        # pylint:disable=unnecessary-lambda-assignment
        objective_fn_flat = lambda x_flat, gen: objective_fn(
            unflatten(x_flat, x), generators=gen, **kwargs
        )

        try:
            assert len(x_flat) == len(generators)
        except AssertionError as e:
            raise ValueError(
                f"Number of parameters {x} must be equal to the number of generators."
            ) from e

        for d, _ in enumerate(x_flat):
            x_flat[d], generators[d] = self._find_optimal_generators(
                objective_fn_flat, x_flat, generators, d
            )

        return unflatten(x_flat, x), generators

    def _find_optimal_generators(self, objective_fn, x, generators, d):
        r"""Optimizer for the generators.

        Optimizes for the best generator at position ``d``.

        Args:
            objective_fn (function): The objective function for optimization. It must have the
                signature ``objective_fn(x, generators=None)`` with a sequence of the values ``x``
                and a list of the gates ``generators`` as inputs, returning a single value.
            x (Union[Sequence[float], float]): sequence containing the initial values of the
                variables to be optimized over or a single float with the initial value
            generators (list[~.Operation]): list containing the initial ``pennylane.ops.qubit``
                operators to be used in the circuit and optimized over
            d (int): the position in the input sequence ``x`` containing the value to be optimized

        Returns:
            tuple: tuple containing the parameter value and generator that, at position ``d`` in
            ``x`` and ``generators``, optimizes the objective function
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
        r"""The rotosolve step for one parameter and one set of generators.

        Updates the parameter :math:`\theta_d` based on Equation 1 in
        `Ostaszewski et al. (2021) <https://doi.org/10.22331/q-2021-01-28-391>`_.

        Args:
            objective_fn (function): The objective function for optimization. It must have the
                signature ``objective_fn(x, generators=None)`` with a sequence of the values ``x``
                and a list of the gates ``generators`` as inputs, returning a single value.
            x (Union[Sequence[float], float]): sequence containing the initial values of the
                variables to be optimized overs or a single float with the initial value
            generators (list[~.Operation]): list containing the initial ``pennylane.ops.qubit``
                operators to be used in the circuit and optimized over
            d (int): the position in the input sequence ``x`` containing the value to be optimized

        Returns:
            array: the input sequence ``x`` with the value at position ``d`` optimized
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

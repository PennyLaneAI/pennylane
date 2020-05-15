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
"""Rotosolve gradient free optimizer"""

from pennylane import numpy as np
from pennylane.utils import _flatten, unflatten


class RotosolveOptimizer:
    r"""Rotosolve gradient free optimizer.

    The Rotosolve optimizer minimizes an objective function with respect to the parameters of a
    quantum circuit without the need for calculating the gradient of the function. The algorithm
    works by updating the parameters :math:`\theta = \theta_1, \dots, \theta_D` one at a time
    according to a closed-form expression for the optimal value of the :math:`d^{th}` parameter
    :math:`\theta^*_d` when the other parameters are fixed:

    .. math:: \theta^*_d = \underset{\theta_d}{\text{argmin}}\left<H\right>_{\theta_d}
              = -\frac{\pi}{2} - \text{arctan2}\left(2\left<H\right>_{\theta_d=0}
              - \left<H\right>_{\theta_d=\pi/2} - \left<H\right>_{\theta_d=-\pi/2},
              \left<H\right>_{\theta_d=\pi/2} - \left<H\right>_{\theta_d=-\pi/2}\right),

    where :math:`\left<H\right>_{\theta_d}` is the expectation value of the objective function
    optimized over the parameter :math:`\theta_d`. :math:`\text{arctan2}(x, y)` computes the
    element-wise arc tangent of :math:`x/y` choosing the quadrant correctly, avoiding, in
    particular, division-by-zero when :math:`y = 0`.

    The algorithm is described in further detail in `Ostaszewski et al. (2019) <https://arxiv.org/abs/1905.09692>`_

    **Example:**

    Initialize the optimizer, set the initial values of ``x`` to be used and set the number of
    steps to optimize over.

    >>> opt = qml.optimize.RotosolveOptimizer()
    >>> x = [0.3, 0.7]
    >>> n_steps = 1000

    Set up the PennyLane circuit using the ``default.qubit`` as simulator device.

    >>> dev = qml.device("default.qubit", analytic=True, wires=2)
    ... @qml.qnode(dev)
    ... def circuit(params):
    ...     qml.RX(params[0], wires=0)
    ...     qml.RY(params[1], wires=1)
    ...     qml.CNOT(wires=[0, 1])
    ...     return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliY(1))

    Define a cost function (that takes a list of values as input and return a single value) based
    on the above circuit.

    >>> def cost(x):
    ...     X_1, Y_2 = circuit(x)
    ...     return 0.2 * X_1 + 0.5 * Y_2

    Run the optimization step-by-step for ``n_steps`` steps.

    >>> cost_rotosel = []
    >>> for _ in range(n_steps):
    ...     cost_rotosel.append(cost(x))
    ...     x = opt.step(cost, x)

    The optimized values for x should now be stored in ``x`` and steps-vs-cost can be seen by
    plotting ``cost_rotosel``.
    """
    # pylint: disable=too-few-public-methods

    def step(self, objective_fn, x):
        r"""Update x with one step of the optimizer.

        Args:
            objective_fn (function): The objective function for optimization. It should take a
                sequence of the values ``x`` and a list of the gates ``generators`` as inputs, and
                return a single value.
            x (Union[Sequence[float], float]): Sequence containing the initial values of the
                variables to be optimized over, or a single float with the initial value.

        Returns:
            array: The new variable values :math:`x^{(t+1)}`.
        """
        x_flat = np.fromiter(_flatten(x), dtype=float)
        objective_fn_flat = lambda x_flat: objective_fn(unflatten(x_flat, x))

        for d, _ in enumerate(x_flat):
            x_flat = self._rotosolve(objective_fn_flat, x_flat, d)

        return unflatten(x_flat, x)

    @staticmethod
    def _rotosolve(objective_fn, x, d):
        r"""The rotosolve step for one parameter.

        Updates the parameter :math:`\theta_d` based on Equation 1 in
        `Ostaszewski et al. (2019) <https://arxiv.org/abs/1905.09692>`_.

        Args:
            objective_fn (function): The objective function for optimization. It should take a
                sequence of the values ``x`` and a list of the gates ``generators`` as inputs, and
                return a single value.
            x (Union[Sequence[float], float]): Sequence containing the initial values of the
                variables to be optimized over, or a single float with the initial value.
            d (int): The position in the input sequence ``x`` containing the value to be optimized.

        Returns:
            array: The input sequence ``x`` with the value at position ``d`` optimized.
        """
        # helper function for x[d] = theta
        def insert(x, d, theta):
            x[d] = theta
            return x

        H_0 = float(objective_fn(insert(x, d, 0)))
        H_p = float(objective_fn(insert(x, d, np.pi / 2)))
        H_m = float(objective_fn(insert(x, d, -np.pi / 2)))

        a = np.arctan2(2 * H_0 - H_p - H_m, H_p - H_m)

        x[d] = -np.pi / 2 - a

        if x[d] <= -np.pi:
            x[d] += 2 * np.pi
        return x

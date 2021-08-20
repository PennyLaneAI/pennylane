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
"""Rotosolve gradient free optimizer"""

import numpy as np
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
    >>> n_steps = 10

    Set up the PennyLane circuit using the ``default.qubit`` as simulator device.

    >>> dev = qml.device("default.qubit", shots=None, wires=2)
    ... @qml.qnode(dev)
    ... def circuit(params):
    ...     qml.RX(params[0], wires=0)
    ...     qml.RY(params[1], wires=1)
    ...     qml.CNOT(wires=[0, 1])
    ...     return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(1))

    Define a cost function (that takes a list of values as input and return a single value) based
    on the above circuit.

    >>> def cost(x):
    ...     Z_1, X_2 = circuit(x)
    ...     return 0.2 * Z_1 + 0.5 * X_2

    Run the optimization step-by-step for ``n_steps`` steps.

    >>> cost_rotosolve = []
    >>> for _ in range(n_steps):
    ...     cost_rotosolve.append(cost(x))
    ...     x = opt.step(cost, x)

    The optimized values for x should now be stored in ``x`` and steps-vs-cost can be seen by
    plotting ``cost_rotosel``.
    """
    # pylint: disable=too-few-public-methods

    def step_and_cost(self, objective_fn, *args, **kwargs):
        r"""Update args with one step of the optimizer and return the corresponding objective
        function value prior to the step.

        Args:
            objective_fn (function): The objective function for optimization. It should take a
                sequence of the values ``*args``  as inputs, and return a single value.
            *args : variable length argument list containing the initial values of the
                variables to be optimized over or a single float with the initial value
            **kwargs : variable length dictionary of keywords for the objective function

        Returns:
            tuple[list [array], float]: the new variable values :math:`x^{(t+1)}` and the objective
            function output prior to the step.
            If single arg is provided, list [array] is replaced by array.
        """
        x_new = self.step(objective_fn, *args, **kwargs)

        return x_new, objective_fn(*args, **kwargs)

    def step(self, objective_fn, *args, **kwargs):
        r"""Update args with one step of the optimizer.

        Args:
            objective_fn (function): the objective function for optimization. It should take a
                sequence of the values ``*args`` and a list of the gates ``generators`` as inputs, and
                return a single value.
            *args : variable length sequence containing the initial
                values of the variables to be optimized over or a single float with the initial
                value.
            **kwargs : variable length keyword arguments for the objective function.

        Returns:
            list [array]: the new variable values :math:`x^{(t+1)}`.
            If single arg is provided, list [array] is replaced by array.
        """
        # will single out one variable to change at a time
        # these hold the arguments not getting updated
        before_args = []
        after_args = list(args)

        # mutable version of args to get updated
        args_new = list(args)

        for index, arg in enumerate(args):
            # removing current arg from after_args
            del after_args[0]

            if getattr(arg, "requires_grad", True):
                x_flat = np.fromiter(_flatten(arg), dtype=float)

                # version of objective function that depends on a flattened version of
                # just the one argument.  All others held constant.
                objective_fn_flat = lambda x_flat, arg_kw=arg: objective_fn(
                    *before_args, unflatten(x_flat, arg_kw), *after_args, **kwargs
                )

                # updating each parameter in current arg
                for d, _ in enumerate(x_flat):
                    x_flat = self._rotosolve(objective_fn_flat, x_flat, d)

                args_new[index] = unflatten(x_flat, arg)

            # updating before_args for next loop
            before_args.append(args_new[index])

        # unwrap arguments if only one, backward compatible and cleaner
        if len(args_new) == 1:
            return args_new[0]
        return args_new

    @staticmethod
    def _rotosolve(objective_fn, x, d):
        r"""The rotosolve step for one parameter.

        Updates the parameter :math:`\theta_d` based on Equation 1 in
        `Ostaszewski et al. (2019) <https://arxiv.org/abs/1905.09692>`_.

        Args:
            objective_fn (function): the objective function for optimization. It should take a
                sequence of the values ``x`` and a list of the gates ``generators`` as inputs, and
                return a single value.
            x (Union[Sequence[float], float]): sequence containing the initial values of the
                variables to be optimized over or a single float with the initial value.
            d (int): the position in the input sequence ``x`` containing the value to be optimized.

        Returns:
            array: the input sequence ``x`` with the value at position ``d`` optimized.
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

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
from scipy.optimize import brute, shgo
from pennylane.utils import _flatten, unflatten


def _brute_optimizer(fun, num_steps, **kwargs):
    r"""Brute force optimizer, wrapper of scipy.optimizer.brute that repeats it
    ``num_steps`` times."""
    width = 2 * np.pi
    x_min = 0.0
    Ns = kwargs.pop("Ns")
    for _ in range(num_steps):
        _range = (x_min - width / 2, x_min + width / 2)
        x_min, y_min, *_ = brute(fun, ranges=(_range,), full_output=True, Ns=Ns, **kwargs)
        width /= Ns

    return x_min, y_min


def _shgo_optimizer(fun, **kwargs):
    r"""Wrapper for ``scipy.optimize.shgo`` (Simplicial Homology global optimizer)."""
    opt_res = shgo(fun, **kwargs)
    return opt_res.x, opt_res.fun


class RotosolveOptimizer:
    r"""Rotosolve gradient-free optimizer.

    The Rotosolve optimizer minimizes an objective function with respect to the parameters of a
    quantum circuit without the need for calculating the gradient of the function. The algorithm
    updates the parameters :math:`\boldsymbol{\theta} = \theta_1, \dots, \theta_D` by
    reconstructing the cost function with respect to one of the parameters at a time
    while keeping all other parameters fixed.
    This requires a purely classical one-dimensional global optimization over the range
    :math:`(-\pi,\pi]` in general, which can be replaced by a closed-form expression for the
    optimal value if the :math:`d^{th}` parametrized gate has only two eigenvalues. In this case
    the optimal value :math:`\theta^*_d` is given by

    .. math:: \theta^*_d = \underset{\theta_d}{\text{argmin}}\left<H\right>_{\theta_d}
              = -\frac{\pi}{2} - \text{arctan2}\left(2\left<H\right>_{\theta_d=0}
              - \left<H\right>_{\theta_d=\pi/2} - \left<H\right>_{\theta_d=-\pi/2},
              \left<H\right>_{\theta_d=\pi/2} - \left<H\right>_{\theta_d=-\pi/2}\right),

    where :math:`\left<H\right>_{\theta_d}` is the expectation value of the objective function
    restricted to only depend on the parameter :math:`\theta_d`. :math:`\text{arctan2}(x, y)`
    computes the element-wise arc tangent of :math:`x/y` choosing the quadrant correctly and
    avoiding, in particular, division by zero when :math:`y = 0`.

    The algorithm is described in further detail in
    `Gil Vidal and Theis (2018) <https://arxiv.org/abs/1812.06323>`_ and
    `Ostaszewski et al. (2019) <https://arxiv.org/abs/1905.09692>`_.

    **Example:**

    Initialize the optimizer, set the initial values ``x`` of the parameters :math:`\theta` and set
    the number of steps to optimize over.

    >>> opt = qml.optimize.RotosolveOptimizer()
    >>> num_steps = 10

    Set up the PennyLane circuit using the ``default.qubit`` as simulator device. The first
    argument controls three Pauli rotations with three parameters (one frequency each), the
    second a layer of rotations with a single parameter (three frequencies), and the third
    argument feeds three parameters into three controled Pauli rotations (two frequencies
    each). We also initialize a set of parameters for all these operations and summarize
    the numbers of frequencies in ``num_frequencies``.
    The ``cost_function` is defined simply by measuring the expectation value of the tensor
    product of ``PauliZ`` operators on all qubits.

    .. code-block :: python

        dev = qml.device('default.qubit', wires=3, shots=None)
        @qml.qnode(dev)
        def cost_function(rot_param, layer_par, crot_param):
            for i, par in enumerate(rot_param):
                qml.RX(par, wires=i)
            for w in dev.wires:
                qml.RX(layer_par, wires=w)
            for i, par in enumerate(crot_param):
                qml.CRY(par, wires=[i, (i+1)%3])

            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2))

        init_param = [
            np.array([0.3, 0.2, 0.67], requires_grad=True),
            np.array(1.1, requires_grad=True),
            np.array([-0.2, 0.1, -2.5], requires_grad=True),
        ]

        num_frequencies = [[1, 1, 1], 3, [2, 2, 2]]

    Run the optimization step-by-step for ``n_steps`` steps.

    .. code-block :: python

        cost_rotosolve = []

        opt = qml.RotosolveOptimizer()
        param = init_param.copy()
        for _ in range(num_steps):
            param, cost = opt.step_and_cost(
                cost_function,
                *param,
                num_frequencies=num_frequencies,
            )
            cost_rotosolve.append(cost)

    The optimized values for x are now stored in ``param`` and steps-vs-cost can be
    assessed by plotting ``cost_rotosolve``.
    The keyword argument `requires_grad` can be used to determine whether the respective
    parameter should be optimized or not, following the behaviour of gradient computations and
    gradient-based optimizers.

    In addition, the optimization technique for the Rotosolve substeps can be chosen via the
    ``optimizer`` and ``optimizer_kwargs`` keyword arguments and the minimized cost of the
    intermediate univariate reconstructions can be read out via ``full_output``, including the
    cost _after_ the full Rotosolve step:

    .. code-block :: python

        param = init_param.copy()
        for step in range(num_steps):
            param, cost, sub_cost = opt.step_and_cost(
                cost_function,
                *param,
                num_frequencies=num_frequencies,
                full_output=True,
            )
            print(f"Cost before step: {cost}")
            print(f"Minimization substeps: {np.round(sub_cost, 6)}")

    The ``full_output`` feature is available for both, ``step`` and ``step_and_cost``.
    """
    # pylint: disable=too-few-public-methods

    def step_and_cost(
        self,
        objective_fn,
        *args,
        num_frequencies=None,
        optimizer=None,
        optimizer_kwargs=None,
        full_output=False,
        **kwargs,
    ):
        r"""Update args with one step of the optimizer and return the corresponding objective
        function value prior to the step.

        Args:
            objective_fn (function): the objective function for optimization. It should take a
                sequence of the values ``*args`` and a list of the gates ``generators`` as inputs,
                and return a single value.
            *args : variable length sequence containing the initial
                values of the variables to be optimized over or a single float with the initial
                value.
            num_frequencies (int|array[int]): The number of frequencies in the ``objective_fn`` per
                parameter. If an ``int``, the same number is used for all parameters; if
                ``array[int]``, the shape of ``args`` and ``num_frequencies`` has to coincide.
                Defaults to ``num_frequencies=1``, corresponding to Pauli rotation gates.
            optimizer (callable|"brute"|"shgo"): the optimization method used for the univariate
                minimization. If a callable, should have the signature
                ``(fun, **kwargs) -> x_min, y_min``, where ``y_min`` is tracked and returned
                if ``full_output==True`` but is not relevant to the optimization.
                If ``"brute"`` or ``"shgo"``, the corresponding global optimizer of SciPy is used.
                Defaults to ``"brute"``.
            optimizer_kwargs : keyword arguments for the ``optimizer``. For ``"brute"`` and
                ``"shgo"``, these kwargs are passed to the respective SciPy implementation.
                Has to be given as one dictionary, *not variable length*.
            full_output (bool): whether to return the intermediate minimized energy values from
                the univariate optimization steps.
            **kwargs : variable length keyword arguments for the objective function.

        Returns:
            list [array]: the new variable values :math:`x^{(t+1)}`.
            If single arg is provided, list [array] is replaced by array.
        """
        if num_frequencies is None:
            num_frequencies = [1] * len(args)
        elif np.isscalar(num_frequencies) and np.isclose(int(num_frequencies), num_frequencies):
            num_frequencies = [num_frequencies] * len(args)
        else:
            if len(num_frequencies) != len(args):
                raise ValueError(
                    "The number of the provided numbers of frequencies "
                    f"({len(num_frequencies)}) does not match the number of function arguments "
                    f"({len(args)})."
                )

        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        if optimizer in [None, "brute"]:
            optimizer = _brute_optimizer
            optimizer_kwargs.setdefault("num_steps", 4)
            optimizer_kwargs.setdefault("Ns", 100)
        elif optimizer == "shgo":
            optimizer = _shgo_optimizer
            optimizer_kwargs.setdefault("bounds", ((-np.pi, np.pi),))

        # will single out one arg to change at a time
        # these hold the arguments not getting updated
        before_args = []
        after_args = list(args)
        # mutable version of args to get updated
        args_new = list(args)

        # Prepare intermediate minimization results cache
        if full_output:
            y_output = []
        # Compute the very first evaluation in order to be able to cache it
        H_0 = objective_fn(*args, **kwargs)

        for arg_index, (arg, num_frequency) in enumerate(zip(args, num_frequencies)):
            del after_args[0]

            if getattr(arg, "requires_grad", True):
                x_flat = np.fromiter(_flatten(arg), dtype=float)
                num_params = len(x_flat)
                if np.isscalar(num_frequency):
                    num_frequency_flat = [num_frequency] * num_params
                else:
                    num_frequency_flat = np.fromiter(_flatten(num_frequency), dtype=int)
                    if len(num_frequency_flat) != num_params:
                        raise ValueError(
                            "The number of the numbers of frequencies "
                            f"({len(num_frequency_flat)}) for the {arg_index}th argument does "
                            f"not match the number of parameters in that argument ({num_params})."
                        )
                shift_vecs = np.eye(num_params)

                # Iterate over current arg:
                for par_index, (shift_vec, _num_frequency) in enumerate(
                    zip(shift_vecs, num_frequency_flat)
                ):
                    # univariate objective function
                    univariate_obj_fn = lambda x: objective_fn(
                        *before_args,
                        unflatten(x_flat + shift_vec * x, arg),
                        *after_args,
                        **kwargs,
                    )
                    x_min, y_min = self._rotosolve(
                        univariate_obj_fn,
                        _num_frequency,
                        optimizer,
                        optimizer_kwargs,
                        full_output,
                        H_0=(H_0 if arg_index + par_index == 0 else None),
                    )
                    x_flat += shift_vec * x_min
                    if full_output:
                        y_output.append(y_min)

                args_new[arg_index] = unflatten(x_flat, arg)

            # updating before_args for next loop
            before_args.append(args_new[arg_index])

        # unwrap arguments if only one, backward compatible and cleaner
        if len(args_new) == 1:
            args_new = args_new[0]

        if full_output:
            return args_new, H_0, y_output
        else:
            return args_new, H_0

    def step(
        self,
        objective_fn,
        *args,
        num_frequencies=None,
        optimizer=None,
        optimizer_kwargs=None,
        full_output=False,
        **kwargs,
    ):
        r"""Update args with one step of the optimizer.

        Args:
            objective_fn (function): the objective function for optimization. It should take a
                sequence of the values ``*args`` and a list of the gates ``generators`` as inputs,
                and return a single value.
            *args : variable length sequence containing the initial
                values of the variables to be optimized over or a single float with the initial
                value.
            num_frequencies (int|array[int]): The number of frequencies in the ``objective_fn`` per
                parameter. If an ``int``, the same number is used for all parameters; if
                ``array[int]``, the shape of ``args`` and ``num_frequencies`` has to coincide.
                Defaults to ``num_frequencies=1``, corresponding to Pauli rotation gates.
            optimizer (callable|"brute"|"shgo"): the optimization method used for the univariate
                minimization. If a callable, should have the signature
                ``(fun, **kwargs) -> x_min, y_min``, where ``y_min`` is tracked but not essential
                to the functionality of the optimization. If ``"brute"`` or ``"shgo"``, the
                corresponding global optimizer of SciPy is used. Defaults to ``"brute"``.
            optimizer_kwargs : keyword arguments for the ``optimizer``. For ``"brute"`` and
                ``"shgo"``, these kwargs are passed to the respective SciPy implementation.
                Has to be given as one dictionary, not variable length.
            full_output (bool): whether to return the intermediate minimized energy values from
                the univariate optimization steps.
            **kwargs : variable length keyword arguments for the objective function.

        Returns:
            tuple[list [array], float]: the new variable values :math:`x^{(t+1)}` and the objective
            function output prior to the step.
            If single arg is provided, list [array] is replaced by array.
        """
        x_new, H_0, *y_output = self.step_and_cost(
            objective_fn,
            *args,
            num_frequencies=num_frequencies,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            full_output=full_output,
            **kwargs,
        )
        if full_output:
            return x_new, y_output
        else:
            return x_new

    @staticmethod
    def _full_reconstruction_equ(fun, num_frequency, H_0):
        r"""Reconstruct a univariate trigonometric function using trigonometric interpolation.
        See `Gil Vidal and Theis (2018) <https://arxiv.org/abs/1812.06323>`_ or
        `Wierichs et al. (2021) <https://arxiv.org/abs/2107.12390>`_.

        Args:
            fun (callable): the function to reconstruct
            num_frequency (int): the number of (integer) frequencies present in ``fun``.

        Returns:
            callable: The reconstruction function with ``num_frequency`` frequencies,
            coinciding with ``fun`` on the same number of points.
        """
        if H_0 is None:
            mus = range(-num_frequency, num_frequency + 1)
            shifts = [2 * mu * np.pi / (2 * num_frequency + 1) for mu in mus]
            evals = [fun(shift) for shift in shifts]
        else:
            mus = range(1, num_frequency + 1)
            shifts_pos = [2 * mu * np.pi / (2 * num_frequency + 1) for mu in mus]
            shifts_neg = [-shift for shift in shifts_pos[::-1]]
            evals = list(map(fun, shifts_neg)) + [H_0] + list(map(fun, shifts_pos))
            shifts = shifts_neg + [0] + shifts_pos
        a, b = (num_frequency + 0.5) / np.pi, 0.5 / np.pi
        reconstruction = lambda x: np.sum(
            np.array(
                [
                    _eval * np.sinc(a * (x - shift)) / np.sinc(b * (x - shift))
                    for _eval, shift in zip(evals, shifts)
                ]
            )
        )
        return reconstruction

    @staticmethod
    def _rotosolve(objective_fn, num_frequency, optimizer, optimizer_kwargs, full_output, H_0=None):
        r"""The rotosolve step for a univariate (restriction of a) cost function.

        Updates the parameter of the ``objective_fn`` based on Equation 1 in
        `Ostaszewski et al. (2019) <https://arxiv.org/abs/1905.09692>`_ if ``num_frequency==1``,
        or based on a reconstruction and global minimization subroutine if ``num_frequency>1``.

        Args:
            objective_fn (function): the objective function for optimization. It should take a
            num_frequency (int): the number of frequencies in the ``objective_fn``.

        Returns:
            x_min (float): the minimizing input of ``objective_fn`` within :math:`(-\pi, \pi]`.
            y_min (float): the minimal value of ``objective_fn``.
        """
        # Use closed form expression from Ostaszewski et al., using notation of App. A
        if num_frequency == 1:
            H_0 = float(objective_fn(0.0)) if H_0 is None else H_0
            H_p = float(objective_fn(0.5 * np.pi))
            H_m = float(objective_fn(-0.5 * np.pi))
            C = 0.5 * (H_p + H_m)
            B = np.arctan2(2 * (H_0 - C), H_p - H_m)
            x_min = -np.pi / 2 - B
            if full_output:
                A = np.sqrt((H_0 - C) ** 2 + 0.25 * (H_p - H_m) ** 2)
                y_min = -A + C
            else:
                y_min = None
        else:
            reconstruction = RotosolveOptimizer._full_reconstruction_equ(
                objective_fn, num_frequency, H_0
            )
            x_min, y_min = optimizer(reconstruction, **optimizer_kwargs)
            if y_min is None and full_output:
                y_min = reconstruction(x_min)

        if x_min <= -np.pi:
            x_min += 2 * np.pi

        return x_min, y_min

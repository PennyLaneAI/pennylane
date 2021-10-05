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
# pylint: disable=too-many-branches,cell-var-from-loop

import numpy as np
from scipy.optimize import brute, shgo
from pennylane.utils import _flatten, unflatten


def _assert_integer(x):
    """Raises a ValueError if x is not of a sub-datatype of np.integer."""
    x_type = type(x)
    if not np.issubdtype(x_type, np.integer):
        raise ValueError(
            f"The numbers of frequencies are expected to be integers. Received {x_type}."
        )


def _validate_num_freqs(num_freqs, requires_grad):
    """Checks whether a number of frequencies input is valid
    Args:
        num_freqs (int or array[int]): Number of frequencies data
        requires_grad (list[bool]): Information whether an argument is to be optimized, per arg
    Returns:
        array[int]: Parsed input value to correct shape or simply return input if array
        array[int]: List of flattened inputs, with each input being unwrapped if it has length 1
    """
    if num_freqs is None:
        num_freqs = [1] * sum(requires_grad)
        num_freqs_flat = num_freqs
    elif np.isscalar(num_freqs):
        _assert_integer(num_freqs)
        num_freqs = [num_freqs] * sum(requires_grad)
        num_freqs_flat = num_freqs
    else:
        num_freqs_flat = []
        for num_frequency in num_freqs:
            if np.isscalar(num_frequency):
                _assert_integer(num_frequency)
                num_frequency_flat = num_frequency
            else:
                num_frequency_flat = list(_flatten(num_frequency))
                for _num_freq in num_frequency_flat:
                    _assert_integer(_num_freq)
                num_frequency_flat = np.array(num_frequency_flat, dtype=int)
            num_freqs_flat.append(num_frequency_flat)
        if len(num_freqs) != sum(requires_grad):
            raise ValueError(
                "The length of the provided numbers of frequencies "
                f"({len(num_freqs)}) does not match the number of function arguments "
                f"({sum(requires_grad)})."
            )
    return num_freqs, num_freqs_flat


def _brute_optimizer(fun, num_steps, **kwargs):
    r"""Brute force optimizer, wrapper of scipy.optimizer.brute that repeats it
    ``num_steps`` times. Signature is as expected by ``RotosolveOptimizer._rotosolve``
    below, providing a scalar minimal position and the function value at that position."""
    width = 2 * np.pi
    x_min = 0.0
    Ns = kwargs.pop("Ns")
    for _ in range(num_steps):
        _range = (x_min - width / 2, x_min + width / 2)
        x_min, y_min, *_ = brute(fun, ranges=(_range,), full_output=True, Ns=Ns, **kwargs)
        width /= Ns

    return x_min, y_min


def _shgo_optimizer(fun, **kwargs):
    r"""Wrapper for ``scipy.optimize.shgo`` (Simplicial Homology global optimizer).
    Signature is as expected by ``RotosolveOptimizer._rotosolve`` below, providing
    a scalar minimal position and the function value at that position."""
    opt_res = shgo(fun, **kwargs)
    return opt_res.x, opt_res.fun


class RotosolveOptimizer:
    r"""Rotosolve gradient-free optimizer.

    The Rotosolve optimizer minimizes an objective function with respect to the parameters of a
    quantum circuit without the need for calculating the gradient of the function. The algorithm
    updates the parameters :math:`\boldsymbol{\theta} = \theta_1, \dots, \theta_D` by
    separately reconstructing the cost function with respect to each circuit parameter,
    while keeping all other parameters fixed.

    For each parameter, a purely classical one-dimensional global optimization over the
    interval :math:`(-\pi,\pi]` is performed, which can be replaced by a closed-form expression for
    the optimal value if the :math:`d^{th}` parametrized gate has only two eigenvalues. In this
    case, the optimal value :math:`\theta^*_d` is given by

    .. math::

        \theta^*_d &= \underset{\theta_d}{\text{argmin}}\left<H\right>_{\theta_d}\\
              &= -\frac{\pi}{2} - \text{arctan2}\left(2\left<H\right>_{\theta_d=0}
              - \left<H\right>_{\theta_d=\pi/2} - \left<H\right>_{\theta_d=-\pi/2},
              \left<H\right>_{\theta_d=\pi/2} - \left<H\right>_{\theta_d=-\pi/2}\right),

    where :math:`\left<H\right>_{\theta_d}` is the expectation value of the objective function
    restricted to only depend on the parameter :math:`\theta_d`.

    The algorithm is described in further detail in
    `Vidal and Theis (2018) <https://arxiv.org/abs/1812.06323>`_ and
    `Ostaszewski et al. (2019) <https://arxiv.org/abs/1905.09692>`_, and the reconstruction
    method used for more general operations is described in
    `Wierichs et al. (2021) <https://arxiv.org/abs/2107.12390>`_.

    **Example:**

    Initialize the optimizer and set the number of steps to optimize over.

    >>> opt = qml.optimize.RotosolveOptimizer()
    >>> num_steps = 10

    Next, we create a QNode we wish to optimize:

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

    This QNode is defined simply by measuring the expectation value of the tensor
    product of ``PauliZ`` operators on all qubits.
    It takes three parameters:

    - ``rot_param`` controls three Pauli rotations with three parameters (one frequency each),
    - ``layer_par`` feeds into a layer of rotations with a single parameter (three frequencies), and
    - ``crot_param`` feeds three parameters into three controlled Pauli rotations (two frequencies
      each).

    We also initialize a set of parameters for all these operations, and summarize
    the numbers of frequencies in ``num_freqs``.

    .. code-block :: python

        init_param = [
            np.array([0.3, 0.2, 0.67], requires_grad=True),
            np.array(1.1, requires_grad=True),
            np.array([-0.2, 0.1, -2.5], requires_grad=True),
        ]

        num_freqs = [[1, 1, 1], 3, [2, 2, 2]]

    The keyword argument ``requires_grad`` can be used to determine whether the respective
    parameter should be optimized or not, following the behaviour of gradient computations and
    gradient-based optimizers when using Autograd.

    In addition, the optimization technique for the Rotosolve substeps can be chosen via the
    ``optimizer`` and ``optimizer_kwargs`` keyword arguments.
    As an extra feature, the minimized cost of the intermediate univariate reconstructions can
    be read out via ``full_output``, including the cost *after* the full Rotosolve step:

    .. code-block :: python

        param = init_param.copy()
        cost_rotosolve = []
        for step in range(num_steps):
            param, cost, sub_cost = opt.step_and_cost(
                cost_function,
                *param,
                num_freqs=num_freqs,
                full_output=True,
            )
            print(f"Cost before step: {cost}")
            print(f"Minimization substeps: {np.round(sub_cost, 6)}")
            cost_rotosolve.extend(sub_cost)

    The optimized values for ``x`` are now stored in ``param`` and (sub)steps-vs-cost can be
    assessed by plotting ``cost_rotosolve``.
    The ``full_output`` feature is available for both, ``step`` and ``step_and_cost``.

    The most general form ``RotosolveOptimizer`` is designed to tackle currently is any
    trigonometric cost function with integer frequencies up to the given value
    of ``num_freqs`` per parameter. Not all of the integers up to ``num_freqs`` have to
    be present in the frequency spectrum. In order to tackle equidistant but non-integer
    frequencies, we recommend rescaling the argument of the function of interest.
    """
    # pylint: disable=too-few-public-methods

    def step_and_cost(
        self,
        objective_fn,
        *args,
        num_freqs=None,
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
            num_freqs (int or array[int]): The number of frequencies in the ``objective_fn`` per
                parameter. If an ``int``, the same number is used for all parameters; if
                ``array[int]``, the shape of ``args`` and ``num_freqs`` has to coincide.
                Defaults to ``num_freqs=1``, corresponding to Pauli rotation gates.
            optimizer (callable or str): the optimization method used for the univariate
                minimization if there is more than one frequency with respect to the respective
                parameter. If a callable, should have the signature
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
            tuple(list [array] or array, float): the new variable values :math:`x^{(t+1)}` and
            the objective function output prior to the step.
            If a single arg is provided, list [array] is replaced by array.
            list [float]: the intermediate energy values, only returned if ``full_output=True``.
        """
        _requires_grad = [getattr(arg, "requires_grad", True) for arg in args]
        num_freqs, num_freqs_flat = _validate_num_freqs(num_freqs, _requires_grad)

        optimizer_kwargs = optimizer_kwargs or {}
        optimizer = optimizer or "brute"

        if optimizer == "brute":
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
        fun_at_zero = objective_fn(*args, **kwargs)

        train_arg_index = 0
        for arg_index, arg in enumerate(args):
            del after_args[0]

            if _requires_grad[arg_index]:
                num_frequency_flat = num_freqs_flat[train_arg_index]
                x_flat = np.fromiter(_flatten(arg), dtype=float)
                num_params = len(x_flat)
                if np.isscalar(num_frequency_flat):
                    num_frequency_flat = [num_frequency_flat] * num_params
                else:
                    if len(num_frequency_flat) != num_params:
                        raise ValueError(
                            "The number of the frequency counts "
                            f"({len(num_frequency_flat)}) for the {arg_index}th argument does "
                            f"not match the number of parameters in that argument ({num_params})."
                        )
                shift_vecs = np.eye(num_params)

                # Iterate over current arg:
                for par_index, _num_frequency in enumerate(num_frequency_flat):
                    shift_vec = shift_vecs[par_index]
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
                        fun_at_zero=(fun_at_zero if train_arg_index + par_index == 0 else None),
                    )
                    x_flat += shift_vec * x_min
                    if full_output:
                        y_output.append(y_min)

                args_new[arg_index] = unflatten(x_flat, arg)
                train_arg_index += 1

            # updating before_args for next loop
            before_args.append(args_new[arg_index])

        # unwrap arguments if only one, backward compatible and cleaner
        if len(args_new) == 1:
            args_new = args_new[0]

        if full_output:
            return args_new, fun_at_zero, y_output
        return args_new, fun_at_zero

    def step(
        self,
        objective_fn,
        *args,
        num_freqs=None,
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
            num_freqs (int or array[int]): The number of frequencies in the ``objective_fn`` per
                parameter. If an ``int``, the same number is used for all parameters; if
                ``array[int]``, the shape of ``args`` and ``num_freqs`` has to coincide.
                Defaults to ``num_freqs=1``, corresponding to Pauli rotation gates.
            optimizer (callable or str): the optimization method used for the univariate
                minimization if there is more than one frequency with respect to the respective
                parameter. If a callable, should have the signature
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
            If a single arg is provided, list [array] is replaced by array.
            list [float]: the intermediate energy values, only returned if ``full_output=True``.
        """
        x_new, _, *y_output = self.step_and_cost(
            objective_fn,
            *args,
            num_freqs=num_freqs,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            full_output=full_output,
            **kwargs,
        )
        if full_output:
            return x_new, y_output
        return x_new

    @staticmethod
    def full_reconstruction_equ(fun, num_frequency, fun_at_zero=None):
        r"""Reconstruct a univariate trigonometric function using trigonometric interpolation.
        See `Vidal and Theis (2018) <https://arxiv.org/abs/1812.06323>`_ or
        `Wierichs et al. (2021) <https://arxiv.org/abs/2107.12390>`_.

        Args:
            fun (callable): the function to reconstruct
            num_frequency (int): the number of (integer) frequencies present in ``fun``.
            fun_at_zero (float): The value of ``fun`` at 0. Computed if not provided.

        Returns:
            callable: The reconstruction function with ``num_frequency`` frequencies,
            coinciding with ``fun`` on the same number of points.
        """
        fun_at_zero = float(fun(0.0)) if fun_at_zero is None else fun_at_zero
        mus = np.arange(1, num_frequency + 1)
        shifts_pos = 2 * mus * np.pi / (2 * num_frequency + 1)
        shifts_neg = -shifts_pos[::-1]
        evals = list(map(fun, shifts_neg)) + [fun_at_zero] + list(map(fun, shifts_pos))
        shifts = np.concatenate([shifts_neg, [0.0], shifts_pos])
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
    def _rotosolve(
        objective_fn, num_frequency, optimizer, optimizer_kwargs, full_output, fun_at_zero=None
    ):
        r"""The rotosolve step for a univariate (restriction of a) cost function.

        Updates the parameter of the ``objective_fn`` based on Equation 1 in
        `Ostaszewski et al. (2019) <https://arxiv.org/abs/1905.09692>`_ if ``num_frequency==1``,
        or based on a reconstruction and global minimization subroutine if ``num_frequency>1``.

        Args:
            objective_fn (function): the objective function for optimization. It should take a
            num_frequency (int): the number of frequencies in the ``objective_fn``.
            optimizer (callable or str): the optimization method used if ``num_frequency>1``.
                If a callable, should have the signature
                ``(fun, **kwargs) -> x_min, y_min``, where ``y_min`` is tracked and returned
                if ``full_output==True`` but is not relevant to the optimization.
                If ``"brute"`` or ``"shgo"``, the corresponding global optimizer of SciPy is used.
            optimizer_kwargs : keyword arguments for the ``optimizer``. For ``"brute"`` and
                ``"shgo"``, these kwargs are passed to the respective SciPy implementation.
                Has to be given as one dictionary, *not variable length*.
            full_output (bool): Whether to track the intermediate minimized energy values.
            fun_at_zero (float): The value of ``fun`` at 0. Computed if not provided.

        Returns:
            x_min (float): the minimizing input of ``objective_fn`` within :math:`(-\pi, \pi]`.
            y_min (float): the minimal value of ``objective_fn``.
        """
        # pylint: disable=too-many-arguments
        fun_at_zero = float(objective_fn(0.0)) if fun_at_zero is None else fun_at_zero
        # Use closed form expression from Ostaszewski et al., using notation of App. A
        if num_frequency == 1:
            H_p = float(objective_fn(0.5 * np.pi))
            H_m = float(objective_fn(-0.5 * np.pi))
            C = 0.5 * (H_p + H_m)
            B = np.arctan2(2 * (fun_at_zero - C), H_p - H_m)
            x_min = -np.pi / 2 - B
            if full_output:
                A = np.sqrt((fun_at_zero - C) ** 2 + 0.25 * (H_p - H_m) ** 2)
                y_min = -A + C
            else:
                y_min = None
        else:
            reconstruction = RotosolveOptimizer.full_reconstruction_equ(
                objective_fn, num_frequency, fun_at_zero
            )
            x_min, y_min = optimizer(reconstruction, **optimizer_kwargs)
            if y_min is None and full_output:
                y_min = reconstruction(x_min)

        if x_min <= -np.pi:
            x_min += 2 * np.pi

        return x_min, y_min

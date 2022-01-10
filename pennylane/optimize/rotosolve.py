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

from inspect import signature
from functools import lru_cache
import numpy as np
from scipy.optimize import brute, shgo

import pennylane as qml


def _brute_optimizer(fun, num_steps, **kwargs):
    r"""Brute force optimizer, wrapper of scipy.optimizer.brute that repeats it
    ``num_steps`` times. Signature is as expected by ``RotosolveOptimizer._min_numeric``
    below, returning a scalar minimal position and the function value at that position."""
    width = 2 * np.pi
    x_min = 0.0
    Ns = kwargs.pop("Ns")
    for _ in range(num_steps):
        _range = (x_min - width / 2, x_min + width / 2)
        x_min, y_min, *_ = brute(fun, ranges=(_range,), full_output=True, Ns=Ns, **kwargs)
        x_min = x_min[0]
        width /= Ns

    return x_min, y_min


def _shgo_optimizer(fun, **kwargs):
    r"""Wrapper for ``scipy.optimize.shgo`` (Simplicial Homology global optimizer).
    Signature is as expected by ``RotosolveOptimizer._min_numeric`` below, providing
    a scalar minimal position and the function value at that position."""
    opt_res = shgo(fun, **kwargs)
    return opt_res.x[0], opt_res.fun


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

    def __init__(self, optimizer=None, optimizer_kwargs=None):
        self.optimizer, self.optimizer_kwargs = self._prepare_optimizer(optimizer, optimizer_kwargs)

    def _prepare_optimizer(self, optimizer, optimizer_kwargs):
        """Set default optimizer and optimizer keyword arguments
        for the one-dimensional optimization in each substep of Rotosolve."""
        optimizer_kwargs = optimizer_kwargs or {}
        optimizer = optimizer or "brute"

        if optimizer == "brute":
            optimizer = _brute_optimizer
            optimizer_kwargs.setdefault("num_steps", 4)
            optimizer_kwargs.setdefault("Ns", 100)
        elif optimizer == "shgo":
            optimizer = _shgo_optimizer
            optimizer_kwargs.setdefault("bounds", ((-np.pi, np.pi),))

        return optimizer, optimizer_kwargs

    def _validate_inputs(self, requires_grad, args, nums_frequency, spectra):
        """Checks that for each trainable argument either the number of
        frequencies or the frequency spectrum is given."""

        for arg, (arg_name, _requires_grad) in zip(args, requires_grad.items()):
            if _requires_grad:
                _nums_frequency = nums_frequency.get(arg_name, {})
                _spectra = spectra.get(arg_name, {})
                all_keys = set(_nums_frequency) | set(_spectra)

                shape = qml.math.shape(arg)
                indices = np.ndindex(shape) if len(shape) > 0 else [()]
                for par_idx in indices:
                    if par_idx not in all_keys:
                        raise ValueError(
                            "Neither the number of frequencies nor the frequency spectrum "
                            f"was provided for the entry {par_idx} of argument {arg_name}."
                        )

    def step_and_cost(
        self,
        objective_fn,
        *args,
        nums_frequency=None,
        spectra=None,
        shifts=None,
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
        # todo: does this signature call cover all cases?
        sign_fn = objective_fn.func if isinstance(objective_fn, qml.QNode) else objective_fn
        arg_names = list(signature(sign_fn).parameters.keys())
        requires_grad = {
            arg_name: qml.math.requires_grad(arg) for arg_name, arg in zip(arg_names, args)
        }
        nums_frequency = nums_frequency or {}
        spectra = spectra or {}
        self._validate_inputs(requires_grad, args, nums_frequency, spectra)

        # we will single out one arg to change at a time
        # the following hold the arguments not getting updated
        before_args = []
        after_args = list(args)
        # mutable version of args to get updated
        args_new = list(args)

        # Prepare intermediate minimization results cache
        if full_output:
            y_output = []
        # Compute the very first evaluation in order to be able to cache it
        fun_at_zero = objective_fn(*args, **kwargs)
        first_sub_update = True

        for arg_idx, (arg, arg_name) in enumerate(zip(args, arg_names)):
            del after_args[0]

            if not requires_grad[arg_name]:
                before_args.append(arg)
                continue
            shape = qml.math.shape(arg)
            indices = np.ndindex(shape) if len(shape) > 0 else [()]
            for par_idx in indices:
                # Set a single parameter in a single argument to be reconstructed
                num_freq = nums_frequency.get(arg_name, {}).get(par_idx, None)
                spectrum = spectra.get(arg_name, {}).get(par_idx, None)
                if spectrum is not None:
                    spectrum = np.array(spectrum)
                _fun_at_zero = fun_at_zero if first_sub_update else None

                if num_freq == 1 or (spectrum is not None and len(spectrum[spectrum > 0])) == 1:
                    _args = before_args + [arg] + after_args
                    univariate = self._restrict_to_univariate(
                        objective_fn, arg_idx, par_idx, _args, kwargs
                    )
                    freq = 1.0 if num_freq is not None else spectrum[0]
                    x_min, y_min = self._min_analytic(univariate, freq, _fun_at_zero)
                    arg = qml.math.scatter_element_add(arg, par_idx, x_min)

                else:
                    ids = {arg_name: (par_idx,)}
                    _nums_frequency = (
                        {arg_name: {par_idx: num_freq}} if num_freq is not None else None
                    )
                    _spectra = {arg_name: {par_idx: spectrum}} if spectrum is not None else None

                    # Set up the reconstruction function
                    recon_fn = qml.fourier.reconstruct(
                        objective_fn, ids, _nums_frequency, _spectra, shifts
                    )
                    # Perform the reconstruction
                    _args = before_args + [arg] + after_args
                    recon = recon_fn(*_args, f0=_fun_at_zero, **kwargs)[arg_name][par_idx]

                    __args = (
                        before_args + [qml.math.scatter_element_add(arg, par_idx, 0.3)] + after_args
                    )

                    x_min, y_min = self._min_numeric(recon)

                    # Update the currently treated argument
                    arg = qml.math.scatter_element_add(arg, par_idx, x_min - arg[par_idx])
                first_sub_update = False

                if full_output:
                    y_output.append(y_min)

            # updating before_args for next argument
            before_args.append(arg)

        # All arguments have been updated and/or passed to before_args
        args = before_args
        # unwrap arguments if only one, backward compatible and cleaner
        if len(args) == 1:
            args = args[0]

        if full_output:
            return args, fun_at_zero, y_output

        return args, fun_at_zero

    def step(
        self,
        objective_fn,
        *args,
        nums_frequency=None,
        spectra=None,
        shifts=None,
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
            nums_frequency=nums_frequency,
            spectra=spectra,
            shifts=shifts,
            full_output=full_output,
            **kwargs,
        )
        if full_output:
            return x_new, y_output
        return x_new

    def _rotosolve(
        objective_fn, single_frequency, optimizer, optimizer_kwargs, full_output, fun_at_zero=None
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

    def _restrict_to_univariate(self, fn, arg_idx, par_idx, args, kwargs):
        r"""Restrict a function to a univariate function for given argument
        and parameter indices.

        Args:
            fn (callable): Multivariate function
            arg_idx (int): Index of the argument that contains the parameter to restrict to
            par_idx (tuple[int]): Index of the parameter to restrict to within the argument
            args (tuple): Arguments at which to restrict the function.
            kwargs (dict): Keyword arguments at which to restrict the function.

        Returns:
            callable: Univariate restriction of ``fn``. That is, this callable takes
            a single float value as input and has the same return type as ``fn``.
            All arguments are set to the given ``args`` and the input value to this
            function is added to the marked parameter.
        """
        the_arg = args[arg_idx]
        if len(qml.math.shape(the_arg)) == 0:
            shift_vec = qml.math.ones_like(the_arg)
        else:
            shift_vec = qml.math.zeros_like(the_arg)
            shift_vec = qml.math.scatter_element_add(shift_vec, par_idx, 1.0)

        def _univariate_fn(x):
            return fn(*args[:arg_idx], the_arg + shift_vec * x, *args[arg_idx + 1 :], **kwargs)

        return _univariate_fn

    def _min_analytic(self, objective_fn, freq, f0):
        r"""Analytically minimize a trigonometric function that depends on a
        single parameter and has a single frequency. Uses two or
        three function evaluations.

        Args:
            objective_fn (callable): Trigonometric function to minimize
            freq (float): Frequency in the ``objective_fn``
            f0 (float): Value of the ``objective_fn`` at zero. Reduces the
                number of calls to the function from three to two if given.

        Returns:
            float: Position of the minimum of ``objective_fn``
            float: Value of the minimum of ``objective_fn``

        The closed form expression used here was derived in
        `Vidal & Theis (2018) <https://arxiv.org/abs/1812.06323>`__ ,
        `Parrish et al (2019) <https://arxiv.org/abs/1904.03206>`__ and
        `Ostaszewski et al (2019) <https://arxiv.org/abs/1905.09692>`__.
        We use the notation of Appendix A of the latter reference, allowing
        for an arbitrary frequency instead of restricting to ``freq=1``.
        The returned position is guaranteed to lie within :math:`(-\pi, \pi]`.
        """
        if f0 is None:
            f0 = objective_fn(0.0)
        fp = objective_fn(0.5 * np.pi / freq)
        fm = objective_fn(-0.5 * np.pi / freq)
        C = 0.5 * (fp + fm)
        B = np.arctan2(2 * f0 - fp - fm, fp - fm)
        x_min = (-np.pi / 2 - B) / freq
        A = np.sqrt((f0 - C) ** 2 + 0.25 * (fp - fm) ** 2)
        y_min = -A + C

        if x_min <= -np.pi:
            x_min = x_min + 2 * np.pi

        return x_min, y_min

    def _min_numeric(self, objective_fn):
        r"""Numerically minimize a trigonometric function that depends on a
        single parameter. Uses potentially large numbers of function evaluations.
        The optimization method and options are stored in
        ``RotosolveOptimizer.optimizer`` and ``RotosolveOptimizer.optimizer_kwargs``.

        Args:
            objective_fn (callable): Trigonometric function to minimize

        Returns:
            float: Position of the minimum of ``objective_fn``
            float: Value of the minimum of ``objective_fn``

        The returned position is guaranteed to lie within :math:`(-\pi, \pi]`.
        """

        x_min, y_min = self.optimizer(objective_fn, **self.optimizer_kwargs)
        if y_min is None:
            y_min = objective_fn(x_min)

        if x_min <= -np.pi:
            x_min += 2 * np.pi

        return x_min, y_min

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
import numpy as np
from scipy.optimize import brute, shgo

import pennylane as qml


def _brute_optimizer(fun, num_steps, bounds=None, **kwargs):
    r"""Brute force optimizer, wrapper of scipy.optimize.brute that repeats it
    ``num_steps`` times. Signature is as expected by ``RotosolveOptimizer._min_numeric``
    below, returning a scalar minimal position and the function value at that position."""
    Ns = kwargs.pop("Ns")
    width = bounds[0][1] - bounds[0][0]
    center = (bounds[0][1] + bounds[0][0]) / 2
    for _ in range(num_steps):
        range_ = (center - width / 2, center + width / 2)
        center, y_min, *_ = brute(fun, ranges=(range_,), full_output=True, Ns=Ns, **kwargs)
        # We only ever use this function for 1D optimization
        center = center[0]
        width /= Ns

    return center, y_min


def _shgo_optimizer(fun, **kwargs):
    r"""Wrapper for ``scipy.optimize.shgo`` (Simplicial Homology global optimizer).
    Signature is as expected by ``RotosolveOptimizer._min_numeric`` below, providing
    a scalar minimal position and the function value at that position."""
    opt_res = shgo(fun, **kwargs)
    return opt_res.x[0], opt_res.fun


def _validate_inputs(requires_grad, args, nums_frequency, spectra):
    """Checks that for each trainable argument either the number of
    frequencies or the frequency spectrum is given."""

    if not any(requires_grad.values()):
        raise ValueError(
            "Found no parameters to optimize. The parameters to optimize "
            "have to be marked as trainable."
        )
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


def _restrict_to_univariate(fn, arg_idx, par_idx, args, kwargs):
    r"""Restrict a function to a univariate function for given argument
    and parameter indices.

    Args:
        fn (callable): Multivariate function
        arg_idx (int): Index of the argument that contains the parameter to restrict
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


class RotosolveOptimizer:
    r"""Rotosolve gradient-free optimizer.

    The Rotosolve optimizer minimizes an objective function with respect to the parameters of a
    quantum circuit without the need for calculating the gradient of the function. The algorithm
    updates the parameters :math:`\boldsymbol{\theta} = \theta_1, \dots, \theta_D` by
    separately reconstructing the cost function with respect to each circuit parameter,
    while keeping all other parameters fixed.

    Args:
        substep_optimizer (str or callable): Optimizer to use for the substeps of Rotosolve
            that carries out a univariate (i.e., single-parameter) global optimization.
            *Only used if there are more than one frequency for a given parameter.*
            It must take as inputs:

            - A function ``fn`` that maps scalars to scalars,

            - the (keyword) argument ``bounds``, and

            - optional keyword arguments.

            It must return two scalars:

            - The input value ``x_min`` for which ``fn`` is minimal, and

            - the minimal value ``y_min=fn(x_min)`` or ``None``.

            Alternatively, the following optimizers are built-in and can be chosen by
            passing their name:

            - ``"brute"``: An iterative version of
              `SciPy's brute force optimizer <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brute.html>`_.
              It evaluates the function at ``Ns`` equidistant points across the range
              :math:`[-\pi, \pi]` and iteratively refines the range around the point
              with the smallest cost value for ``num_steps`` times.

            - ``"shgo"``: `SciPy's SHGO optimizer <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.shgo.html>`_.

        substep_kwargs (dict): Keyword arguments to be passed to the ``substep_optimizer``
            callable. For ``substep_optimizer="shgo"``, the original keyword arguments of
            the SciPy implementation are available, for ``substep_optimizer="brute"`` the
            keyword arguments ``ranges``, ``Ns`` and ``num_steps`` are useful.
            *Only used if there are more than one frequency for a given parameter.*

    For each parameter, a purely classical one-dimensional global optimization over the
    interval :math:`(-\pi,\pi]` is performed, which is replaced automatically by a
    closed-form expression for the optimal value if the :math:`d\text{th}` parametrized
    gate has only two eigenvalues. This means that ``substep_optimizer`` and
    ``substep_kwargs`` will not be used for these parameters.
    In this case, the optimal value :math:`\theta^*_d` is given analytically by

    .. math::

        \theta^*_d &= \underset{\theta_d}{\text{argmin}}\left<H\right>_{\theta_d}\\
              &= -\frac{\pi}{2} - \text{arctan2}\left(2\left<H\right>_{\theta_d=0}
              - \left<H\right>_{\theta_d=\pi/2} - \left<H\right>_{\theta_d=-\pi/2},
              \left<H\right>_{\theta_d=\pi/2} - \left<H\right>_{\theta_d=-\pi/2}\right),

    where :math:`\left<H\right>_{\theta_d}` is the expectation value of the objective function
    restricted to only depend on the parameter :math:`\theta_d`.

    .. warning::

        The built-in one-dimensional optimizers ``"brute"`` and ``"shgo"`` for the substeps
        of a Rotosolve optimization step use the interval :math:`(-\pi,\pi]`, rescaled with
        the inverse smallest frequency as default domain to optimize over. For complicated
        cost functions, this domain might not be suitable for the substep optimization and
        an appropriate range should be passed via ``bounds`` in ``substep_kwargs``.

    The algorithm is described in further detail in
    `Vidal and Theis (2018) <https://arxiv.org/abs/1812.06323>`_,
    `Nakanishi, Fujii and Todo (2019) <https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.043158>`_,
    `Parrish et al. (2019) <https://arxiv.org/abs/1904.03206>`_,
    and
    `Ostaszewski et al. (2019) <https://quantum-journal.org/papers/q-2021-01-28-391/>`_,
    and the reconstruction method used for more general operations is described in
    `Wierichs et al. (2022) <https://doi.org/10.22331/q-2022-03-30-677>`_.

    .. warning::

        ``RotosolveOptimizer`` will only update parameters that are *explicitly*
        marked as trainable. This can be done via ``requires_grad`` if using Autograd
        or PyTorch. ``RotosolveOptimizer`` is not yet implemented to work in a stable
        manner with TensorFlow or JAX.

    **Example:**

    Initialize the optimizer and set the number of steps to optimize over.
    Recall that the optimization with ``RotosolveOptimizer`` uses global optimization substeps
    of univariate functions. The optimization technique for these substeps can be chosen via the
    ``substep_optimizer`` and ``substep_kwargs`` keyword arguments.
    Here we use the built-in iterative version of
    `SciPy's brute force optimizer <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brute.html>`_
    with four iterations.
    We will run Rotosolve itself for three iterations.

    >>> opt_kwargs = {"num_steps": 4}
    >>> opt = qml.optimize.RotosolveOptimizer(substep_optimizer="brute", substep_kwargs=opt_kwargs)
    >>> num_steps = 3

    Next, we create a QNode we wish to optimize:

    .. code-block :: python

        dev = qml.device('default.qubit', wires=3, shots=None)

        @qml.qnode(dev)
        def cost_function(rot_param, layer_par, crot_param, rot_weights=None, crot_weights=None):
            for i, par in enumerate(rot_param * rot_weights):
                qml.RX(par, wires=i)
            for w in dev.wires:
                qml.RX(layer_par, wires=w)
            for i, par in enumerate(crot_param*crot_weights):
                qml.CRY(par, wires=[i, (i+1)%3])
            return qml.expval(qml.Z(0) @ qml.Z(1) @ qml.Z(2))

    This QNode is defined simply by measuring the expectation value of the tensor
    product of ``PauliZ`` operators on all qubits.
    It takes three parameters:

    - ``rot_param`` controls three Pauli rotations with three parameters, multiplied with ``rot_weights``,
    - ``layer_par`` feeds into a layer of rotations with a single parameter, and
    - ``crot_param`` feeds three parameters, multiplied with ``crot_weights``, into
      three controlled Pauli rotations.

    We also initialize a set of parameters for all these operations, and start with
    uniform weights, i.e., all ``rot_weights`` and ``crot_weights`` are set to one.
    This means that all frequencies with which the parameters in ``rot_param`` and
    ``crot_param`` enter the QNode are integer-valued.
    The number of frequencies per parameter are summarized in ``nums_frequency``.

    .. code-block :: python

        init_param = (
            np.array([0.3, 0.2, 0.67], requires_grad=True),
            np.array(1.1, requires_grad=True),
            np.array([-0.2, 0.1, -2.5], requires_grad=True),
        )
        rot_weights = np.ones(3)
        crot_weights = np.ones(3)

        nums_frequency = {
            "rot_param": {(0,): 1, (1,): 1, (2,): 1},
            "layer_par": {(): 3},
            "crot_param": {(0,): 2, (1,): 2, (2,): 2},
        }

    The keyword argument ``requires_grad`` can be used to determine whether the respective
    parameter should be optimized or not, following the behaviour of gradient computations and
    gradient-based optimizers when using Autograd or Torch.
    With TensorFlow, a ``tf.Variable`` inside a ``tf.GradientTape`` may be used to
    mark variables as trainable.

    Now we carry out the optimization.
    The minimized cost of the intermediate univariate reconstructions can
    be read out via ``full_output``, including the cost *after* the full Rotosolve step:

    >>> param = init_param
    >>> cost_rotosolve = []
    >>> for step in range(num_steps):
    ...     param, cost, sub_cost = opt.step_and_cost(
    ...         cost_function,
    ...         *param,
    ...         nums_frequency=nums_frequency,
    ...         full_output=True,
    ...         rot_weights=rot_weights,
    ...         crot_weights=crot_weights,
    ...     )
    ...     print(f"Cost before step: {cost}")
    ...     print(f"Minimization substeps: {np.round(sub_cost, 6)}")
    ...     cost_rotosolve.extend(sub_cost)
    Cost before step: 0.04200821039253547
    Minimization substeps: [-0.230905 -0.863336 -0.980072 -0.980072 -1.       -1.       -1.      ]
    Cost before step: -0.9999999990681161
    Minimization substeps: [-1. -1. -1. -1. -1. -1. -1.]
    Cost before step: -0.9999999999999996
    Minimization substeps: [-1. -1. -1. -1. -1. -1. -1.]

    The optimized values for the parameters are now stored in ``param``
    and the optimization behaviour can be assessed by plotting ``cost_rotosolve``,
    which include the substeps of the Rotosolve optimization.
    The ``full_output`` feature is available for both, ``step`` and ``step_and_cost``.

    In general, the frequencies in a QNode will not be integer-valued, requiring us
    to provide the ``RotosolveOptimizer`` not only with the number of frequencies
    but their concrete values. For the example QNode above, this happens if the
    weights are no longer one:

    >>> rot_weights = np.array([0.4, 0.8, 1.2], requires_grad=False)
    >>> crot_weights = np.array([0.5, 1.0, 1.5], requires_grad=False)
    >>> spectrum_fn = qml.fourier.qnode_spectrum(cost_function)
    >>> spectra = spectrum_fn(*param, rot_weights=rot_weights, crot_weights=crot_weights)
    >>> spectra["rot_param"]
    {(0,): [-0.4, 0.0, 0.4], (1,): [-0.8, 0.0, 0.8], (2,): [-1.2, 0.0, 1.2]}
    >>> spectra["crot_param"]
    {(0,): [-0.5, -0.25, 0.0, 0.25, 0.5], (1,): [-1.0, -0.5, 0.0, 0.5, 1.0], (2,): [-1.5, -0.75, 0.0, 0.75, 1.5]}

    We may provide these spectra instead of ``nums_frequency`` to Rotosolve to
    enable the optimization of the QNode at these weights:

    >>> param = init_param
    >>> for step in range(num_steps):
    ...     param, cost, sub_cost = opt.step_and_cost(
    ...         cost_function,
    ...         *param,
    ...         spectra=spectra,
    ...         full_output=True,
    ...         rot_weights = rot_weights,
    ...         crot_weights = crot_weights,
    ...     )
    ...     print(f"Cost before step: {cost}")
    ...     print(f"Minimization substeps: {np.round(sub_cost, 6)}")
    Cost before step: 0.09299359486191039
    Minimization substeps: [-0.268008 -0.713209 -0.24993  -0.871989 -0.907672 -0.907892 -0.940474]
    Cost before step: -0.9404742138557066
    Minimization substeps: [-0.940474 -1.       -1.       -1.       -1.       -1.       -1.      ]
    Cost before step: -1.0
    Minimization substeps: [-1. -1. -1. -1. -1. -1. -1.]

    As we can see, while the optimization got a bit harder and the optimizer takes a bit longer
    to converge than previously, Rotosolve was able to adapt to the more complicated
    dependence on the input arguments and still found the global minimum successfully.
    """

    # pylint: disable=too-few-public-methods

    def __init__(self, substep_optimizer="brute", substep_kwargs=None):
        self.substep_kwargs = {} if substep_kwargs is None else substep_kwargs
        if substep_optimizer == "brute":
            self.substep_optimizer = _brute_optimizer
            self.substep_kwargs.setdefault("num_steps", 4)
            self.substep_kwargs.setdefault("Ns", 100)
        elif substep_optimizer == "shgo":
            self.substep_optimizer = _shgo_optimizer
        else:
            self.substep_optimizer = substep_optimizer

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
        function value prior to the step. Each step includes multiple substeps, one per
        parameter.

        Args:
            objective_fn (function): the objective function for optimization. It should take a
                sequence of the values ``*args`` and a list of the gates ``generators`` as inputs,
                and return a single value.
            *args (Sequence): variable length sequence containing the initial values of the
                variables to be optimized over or a single float with the initial value.
            nums_frequency (dict[dict]): The number of frequencies in the ``objective_fn`` per
                parameter. The keys must correspond to argument names of the objective
                function, the values must be dictionaries that map parameter indices (``tuple``)
                in the argument to the number of frequencies with which it enters the objective
                function (``int``).
                The parameter index for a scalar QNode argument is ``()``, for
                one-dimensional array QNode arguments, it takes the form ``(i,)`` for the
                i-th parameter in the argument.
            spectra (dict[dict]): Frequency spectra in the ``objective_fn`` per parameter.
                The formatting is the same as for ``nums_frequency``, but the values
                of the inner dictionaries must be sequences of frequencies
                (``Sequence[float]``).
                For each parameter, ``num_frequency`` take precedence over ``spectra``.
            shifts (dict[dict]): Shift angles for the reconstruction per QNode parameter.
                The keys have to be argument names of ``qnode`` and the inner dictionaries have to
                be mappings from parameter indices to the respective shift angles to be used for
                that parameter. For :math:`R` non-zero frequencies, there must be :math:`2R+1`
                shifts given. Ignored if ``nums_frequency`` gives a number of frequencies
                for the respective parameter in the QNode argument.
            full_output (bool): whether to return the intermediate minimized energy values from
                the univariate optimization substeps.
            **kwargs : variable length keyword arguments for the objective function.

        Returns:
            list [array] or array: the new variable values :math:`x^{(t+1)}`.
            If a single arg is provided, list [array] is replaced by array.
            float: the objective function output prior to the step.
            list [float]: the intermediate objective values, only returned if
            ``full_output=True``.

        The optimization step consists of multiple substeps.

        For each substep,
        one of the parameters in one of the QNode arguments is singled out, and the
        objective function is considered as univariate function (i.e., function that
        depends on a single scalar) of that parameter.

        If ``nums_frequency`` states that there is only a single frequency, or ``spectra``
        only contains one positive frequency, for a parameter, an analytic formula is
        used to return the minimum of the univariate restriction.

        For multiple frequencies, :func:`.fourier.reconstruct` is used to reconstruct
        the univariate restriction and a numeric minimization is performed instead.
        The latter minimization is performed using the ``substep_optimizer`` passed to
        ``RotosolveOptimizer`` at initialization.

        .. note::

            One of ``nums_frequency`` and ``spectra`` must contain information
            about each parameter that is to be trained with ``RotosolveOptimizer``.
            For each univariate reconstruction, the data in ``nums_frequency`` takes
            precedence over the information in ``spectra``.

        """
        # todo: does this signature call cover all cases?
        sign_fn = objective_fn.func if isinstance(objective_fn, qml.QNode) else objective_fn
        arg_names = list(signature(sign_fn).parameters.keys())
        requires_grad = {
            arg_name: qml.math.requires_grad(arg) for arg_name, arg in zip(arg_names, args)
        }
        nums_frequency = nums_frequency or {}
        spectra = spectra or {}
        _validate_inputs(requires_grad, args, nums_frequency, spectra)

        # we will single out one arg to change at a time
        # the following hold the arguments not getting updated
        before_args = []
        after_args = list(args)

        # Prepare intermediate minimization results cache
        if full_output:
            y_output = []
        # Compute the very first evaluation in order to be able to cache it
        fun_at_zero = objective_fn(*args, **kwargs)
        first_substep_in_step = True

        for arg_idx, (arg, arg_name) in enumerate(zip(args, arg_names)):
            del after_args[0]

            if not requires_grad[arg_name]:
                before_args.append(arg)
                continue
            shape = qml.math.shape(arg)
            indices = np.ndindex(shape) if len(shape) > 0 else [()]
            for par_idx in indices:
                _fun_at_zero = fun_at_zero if first_substep_in_step else None
                # Set a single parameter in a single argument to be reconstructed
                num_freq = nums_frequency.get(arg_name, {}).get(par_idx, None)
                spectrum = spectra.get(arg_name, {}).get(par_idx, None)
                if spectrum is not None:
                    spectrum = np.array(spectrum)

                if num_freq == 1 or (spectrum is not None and len(spectrum[spectrum > 0])) == 1:
                    _args = before_args + [arg] + after_args
                    univariate = _restrict_to_univariate(
                        objective_fn, arg_idx, par_idx, _args, kwargs
                    )
                    freq = 1.0 if num_freq is not None else spectrum[spectrum > 0][0]
                    x_min, y_min = self.min_analytic(univariate, freq, _fun_at_zero)
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
                    recon = recon_fn(*before_args, arg, *after_args, f0=_fun_at_zero, **kwargs)[
                        arg_name
                    ][par_idx]
                    if spectrum is None:
                        spectrum = list(range(num_freq + 1))
                    x_min, y_min = self._min_numeric(recon, spectrum)

                    # Update the currently treated argument
                    arg = qml.math.scatter_element_add(arg, par_idx, x_min - arg[par_idx])
                first_substep_in_step = False

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
        r"""Update args with one step of the optimizer. Each step includes
        multiple substeps, one per parameter.

        Args:
            objective_fn (function): the objective function for optimization. It should take a
                sequence of the values ``*args`` and a list of the gates ``generators`` as inputs,
                and return a single value.
            *args (Sequence): variable length sequence containing the initial values of the
                variables to be optimized over or a single float with the initial value.
            nums_frequency (dict[dict]): The number of frequencies in the ``objective_fn`` per
                parameter. The keys must correspond to argument names of the objective
                function, the values must be dictionaries that map parameter indices (``tuple``)
                in the argument to the number of frequencies with which it enters the objective
                function (``int``).
                The parameter index for a scalar QNode argument is ``()``, for
                one-dimensional array QNode arguments, it takes the form ``(i,)`` for the
                i-th parameter in the argument.
            spectra (dict[dict]): Frequency spectra in the ``objective_fn`` per parameter.
                The formatting is the same as for ``nums_frequency``, but the values
                of the inner dictionaries must be sequences of frequencies
                (``Sequence[float]``).
                For each parameter, ``num_frequency`` take precedence over ``spectra``.
            shifts (dict[dict]): Shift angles for the reconstruction per QNode parameter.
                The keys have to be argument names of ``qnode`` and the inner dictionaries have to
                be mappings from parameter indices to the respective shift angles to be used for
                that parameter. For :math:`R` non-zero frequencies, there must be :math:`2R+1`
                shifts given. Ignored if ``nums_frequency`` gives a number of frequencies
                for the respective parameter in the QNode argument.
            full_output (bool): whether to return the intermediate minimized energy values from
                the univariate optimization substeps.
            **kwargs : variable length keyword arguments for the objective function.

        Returns:
            list [array] or array: the new variable values :math:`x^{(t+1)}`.
                If a single arg is provided, list [array] is replaced by array.
            list [float]: the intermediate objective values, only returned if
                ``full_output=True``.

        The optimization step consists of multiple substeps.

        For each substep,
        one of the parameters in one of the QNode arguments is singled out, and the
        objective function is considered as univariate function (i.e., function that
        depends on a single scalar) of that parameter.

        If ``nums_frequency`` states that there is only a single frequency, or ``spectra``
        only contains one positive frequency, for a parameter, an analytic formula is
        used to return the minimum of the univariate restriction.

        For multiple frequencies, :func:`.fourier.reconstruct` is used to reconstruct
        the univariate restriction and a numeric minimization is performed instead.
        The latter minimization is performed using the ``substep_optimizer`` passed to
        ``RotosolveOptimizer`` at initialization.

        .. note::

            One of ``nums_frequency`` and ``spectra`` must contain information
            about each parameter that is to be trained with ``RotosolveOptimizer``.
            For each univariate reconstruction, the data in ``nums_frequency`` takes
            precedence over the information in ``spectra``.

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
            # For full_output=True, y_output was wrapped in an outer list due
            # to the dynamic unpacking
            return x_new, y_output[0]

        return x_new

    def _min_numeric(self, objective_fn, spectrum):
        r"""Numerically minimize a trigonometric function that depends on a
        single parameter. Uses potentially large numbers of function evaluations,
        depending on the used substep_optimizer. The optimization method and
        options are stored in ``RotosolveOptimizer.substep_optimizer``
        and ``RotosolveOptimizer.substep_kwargs``.

        Args:
            objective_fn (callable): Trigonometric function to minimize

        Returns:
            float: Position of the minimum of ``objective_fn``
            float: Value of the minimum of ``objective_fn``

        The returned position is guaranteed to lie within :math:`(-\pi, \pi]`.
        """
        opt_kwargs = self.substep_kwargs.copy()
        if "bounds" not in self.substep_kwargs:
            spectrum = qml.math.array(spectrum)
            half_width = np.pi / qml.math.min(spectrum[spectrum > 0])
            opt_kwargs["bounds"] = ((-half_width, half_width),)

        x_min, y_min = self.substep_optimizer(objective_fn, **opt_kwargs)
        if y_min is None:
            y_min = objective_fn(x_min)

        return x_min, y_min

    @staticmethod
    def min_analytic(objective_fn, freq, f0):
        r"""Analytically minimize a trigonometric function that depends on a
        single parameter and has a single frequency. Uses two or
        three function evaluations.

        Args:
            objective_fn (callable): Trigonometric function to minimize
            freq (float): Frequency :math:`f` in the ``objective_fn``
            f0 (float): Value of the ``objective_fn`` at zero. Reduces the
                number of calls to the function from three to two if given.

        Returns:
            float: Position of the minimum of ``objective_fn``
            float: Value of the minimum of ``objective_fn``

        The closed form expression used here was derived in
        `Vidal & Theis (2018) <https://arxiv.org/abs/1812.06323>`__ ,
        `Parrish et al (2019) <https://arxiv.org/abs/1904.03206>`__ and
        `Ostaszewski et al (2021) <https://doi.org/10.22331/q-2021-01-28-391>`__.
        We use the notation of Appendix A of the last of these references,
        although we allow for an arbitrary frequency instead of restricting
        to :math:`f=1`.
        The returned position is guaranteed to lie within :math:`(-\pi/f, \pi/f]`.

        The used formula for the minimization of the :math:`d-\text{th}`
        parameter then reads

        .. math::

            \theta^*_d &= \underset{\theta_d}{\text{argmin}}\left<H\right>_{\theta_d}\\
                  &= -\frac{\pi}{2f} - \frac{1}{f}\text{arctan2}\left(2\left<H\right>_{\theta_d=0}
                  - \left<H\right>_{\theta_d=\pi/(2f)} - \left<H\right>_{\theta_d=-\pi/(2f)},
                  \left<H\right>_{\theta_d=\pi/(2f)} - \left<H\right>_{\theta_d=-\pi/(2f)}\right),

        """
        if f0 is None:
            f0 = objective_fn(0.0)
        shift = 0.5 * np.pi / freq
        fp = objective_fn(shift)
        fm = objective_fn(-shift)
        C = 0.5 * (fp + fm)
        B = np.arctan2(2 * f0 - fp - fm, fp - fm)
        x_min = -shift - B / freq
        A = np.sqrt((f0 - C) ** 2 + 0.25 * (fp - fm) ** 2)
        y_min = -A + C

        if x_min <= -2 * shift:
            x_min = x_min + 4 * shift

        return x_min, y_min

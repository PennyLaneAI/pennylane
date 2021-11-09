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
"""Contains a function that computes the fourier series of
a quantum expectation value."""
from functools import wraps
from inspect import signature
import warnings

import numpy as np
import pennylane as qml


def _reconstruct_equ(fun, num_frequency, x0=None, f0=None):
    r"""Reconstruct a univariate Fourier series with consecutive integer
    frequencies, using trigonometric interpolation and equidistant shifts.

    This technique is based on
    `Dirichlet kernels <https://en.wikipedia.org/wiki/Dirichlet_kernel>`_, see
    `Vidal and Theis (2018) <https://arxiv.org/abs/1812.06323>`_ or
    `Wierichs et al. (2021) <https://arxiv.org/abs/2107.12390>`_.

    Args:
        fun (callable): Univariate finite Fourier series to reconstruct.
            It must have signature ``float -> float`` .
        num_frequency (int): Number of integer frequencies in ``fun``
            All integer frequencies below ``num_frequency`` are assumed
            to be present in ``fun`` as well; if they are not, the output
            is correct put the reconstruction could have been performed
            with fewer evaluations of ``fun`` .
        x0 (float): Center to which to shift the reconstruction.
            The points at which ``fun`` is evaluated are *not* affected
            by ``x0`` .
        f0 (float): Value of ``fun`` at zero; Providing ``f0`` saves one
            evaluation of ``fun``.

    Returns:
        callable: Reconstructed Fourier series with ``num_frequency`` frequencies.
        This function is a purely classical function. Furthermore, it is fully
        differentiable.
    """
    if not abs(int(num_frequency)) == num_frequency:
        raise ValueError(f"num_frequency must be a non-negative integer, got {num_frequency}")

    a = (num_frequency + 0.5) / np.pi
    b = 0.5 / np.pi

    shifts_pos = qml.math.arange(1, num_frequency + 1) / a
    shifts_neg = -shifts_pos[::-1]
    f0 = fun(0.0) if f0 is None else f0
    evals = list(map(fun, shifts_neg)) + [f0] + list(map(fun, shifts_pos))
    shifts = qml.math.concatenate([shifts_neg, [0.0], shifts_pos])

    x0 = qml.math.array(0.0) if x0 is None else x0

    def _reconstruction(x):
        """Univariate reconstruction based on equidistant shifts and Dirichlet kernels."""
        _x = x - x0 - shifts
        return qml.math.tensordot(
            qml.math.sinc(a * _x) / qml.math.sinc(b * _x),
            evals,
            axes=[[0], [0]],
        )

    return _reconstruction


_warn_text_f0_ignored = (
    "The provided value of the function at zero will be ignored due to the "
    "provided shift values. This may lead to additional evaluations of the "
    "function to be reconstructed."
)


def _reconstruct_gen(fun, spectrum, shifts=None, x0=None, f0=None):
    r"""Reconstruct a univariate (real-valued) Fourier series with given spectrum.

    Args:
        fun (callable): Univariate finite Fourier series to reconstruct.
            It must have signature ``float -> float`` .
        spectrum (Collection): Frequency spectrum of the Fourier series;
            non-positive frequencies are ignored.
        shifts (Sequence): Shift angles at which to evaluate ``fun`` for the reconstruction
            Chosen equidistantly within the interval :math:`[0, 2\pi/f_\text{max}]`
            if ``shifts=None`` , where :math:`f_\text{max}` is the biggest
            frequency in ``spectrum``.
        x0 (float): Center to which to shift the reconstruction.
            The points at which ``fun`` is evaluated are *not* affected
            by ``x0`` .
        f0 (float): Value of ``fun`` at zero; If :math:`0` is among the ``shifts``
            and ``f0`` is provided, one evaluation of ``fun`` is saved.

    Returns:
        callable: Reconstructed Fourier series with :math:`R` frequencies in ``spectrum`` .
        This function is a purely classical function. Furthermore, it is fully differentiable.
    """
    # pylint: disable=unused-argument

    have_f0 = f0 is not None
    have_shifts = shifts is not None

    spectrum = qml.math.array(spectrum)
    spectrum = spectrum[spectrum > 0]
    f_max = qml.math.max(spectrum)

    # If no shifts are provided, choose equidistant ones
    if not have_shifts:
        R = len(spectrum)
        shifts = qml.math.arange(-R, R + 1) * 2 * np.pi / (f_max * (2 * R + 1)) * R
        zero_idx = R
        need_f0 = True
    elif have_f0:
        zero_idx = qml.math.argwhere(qml.math.isclose(shifts, 0.0)).T[0]
        zero_idx = zero_idx[0] if len(zero_idx) > 0 else None
        need_f0 = zero_idx is not None

    # Take care of shifts close to zero if f0 was provided
    if have_f0 and need_f0:
        # Only one shift may be zero at a time
        shifts = qml.math.concatenate(
            [[shifts[zero_idx]], shifts[:zero_idx], shifts[zero_idx + 1 :]]
        )
        evals = qml.math.array([f0] + list(map(fun, shifts[1:])))
    else:
        if have_f0 and not need_f0:
            warnings.warn(_warn_text_f0_ignored)
        evals = qml.math.array(list(map(fun, shifts)))

    L = len(shifts)
    # Construct the coefficient matrix case by case
    C1 = qml.math.ones((L, 1))
    C2 = qml.math.cos(qml.math.tensordot(shifts, spectrum, axes=0))
    C3 = qml.math.sin(qml.math.tensordot(shifts, spectrum, axes=0))
    C = qml.math.hstack([C1, C2, C3])

    # Solve the system of linear equations
    cond = qml.math.linalg.cond(C)
    if cond > 1e8:
        warnings.warn(
            f"The condition number of the Fourier transform matrix is very large: {cond}.",
            UserWarning,
        )
    W = qml.math.linalg.solve(C, evals)

    # Extract the Fourier coefficients
    R = (L - 1) // 2
    a0 = W[0]
    a = W[1 : R + 1]
    b = W[R + 1 :]

    x0 = qml.math.array(0.0) if x0 is None else x0
    # Construct the Fourier series
    def _reconstruction(x):
        """Univariate reconstruction based on arbitrary shifts."""
        x = x - x0
        return (
            a0
            + qml.math.dot(a, qml.math.cos(spectrum * x))
            + qml.math.dot(b, qml.math.sin(spectrum * x))
        )

    return _reconstruction


def _prepare_jobs(ids, nums_frequency, spectra, shifts, atol):
    r"""For inputs to reconstruct, determine how the given information yields
    function reconstruction tasks and collect them into a dictionary ``jobs``.
    Also determine whether the function at zero is needed.

    Args:
        ids (dict or Sequence or str): Indices for the QNode parameters with respect to which
            the QNode should be reconstructed as a univariate function, per QNode argument.
            Each key of the dict, entry of the list, or the single ``str`` has to be the name
            of an argument of ``qnode`` .
            If a ``dict`` , the values of ``ids`` have to contain the parameter indices
            for the respective array-valued QNode argument represented by the key.
            If a ``list`` , the parameter indices are inferred from ``nums_frequency`` if
            given or ``spectra`` else.
            If ``None``, all keys present in ``nums_frequency`` / ``spectra`` are considered.
        nums_frequency (dict[dict]): Numbers of integer frequencies -- and biggest
            frequency -- per QNode parameter. The keys have to be argument names of ``qnode``
            and the inner dictionaries have to be mappings from parameter indices to the
            respective integer number of frequencies. If the QNode frequencies are not contiguous
            integers, the argument ``spectra`` should be used to save evaluations of ``qnode`` .
            Takes precedence over ``spectra`` and leads to usage of equidistant shifts.
        spectra (dict[dict]): Frequency spectra per QNode parameter.
            The keys have to be argument names of ``qnode`` and the inner dictionaries have to
            be mappings from parameter indices to the respective frequency spectrum for that
            parameter. Ignored if ``nums_frequency!=None``.
        shifts (dict[dict]): Shift angles for the reconstruction per QNode parameter.
            The keys have to be argument names of ``qnode`` and the inner dictionaries have to
            be mappings from parameter indices to the respective shift angles to be used for that
            parameter. For :math:`R` non-zero frequencies, there must be :math:`2R+1` shifts
            given. Ignored if ``nums_frequency!=None``.
        atol (float): Absolute tolerance used to analyze shifts lying close to 0.

    Returns:
        dict[dict]: Indices for the QNode parameters with respect to which the QNode
            will be reconstructed. Cast to the dictionary structure explained above.
            If the input ``ids`` was a dictionary, it is returned unmodified.
        callable: The reconstruction method to use, one out of two internal methods.
        dict[dict[dict]]: Keyword arguments for the reconstruction method specifying
            how to carry out the reconstruction. The outer-most keys are QNode argument
            names, the middle keys are parameter indices like the inner keys of
            ``nums_frequency`` or ``spectra`` and the inner-most dictionary contains the
            keyword arguments, i.e. the keys are keyword argument names for the
            reconstruction method
        bool: Whether any of the reconstruction jobs will require the evaluation
            of the function at the position of reconstruction itself.
    """
    # pylint: disable=too-many-branches
    if nums_frequency is None:
        jobs = {}

        if spectra is None:
            raise ValueError("Either nums_frequency or spectra must be given.")
        if ids is None:
            ids = {outer_key: inner_dict.keys() for outer_key, inner_dict in spectra.items()}
        elif isinstance(ids, str):
            # ids only provides a single argument name but no parameter indices
            ids = {ids: spectra[ids].keys()}
        elif not isinstance(ids, dict):
            # ids only provides argument names but no parameter indices
            ids = {_id: spectra[_id].keys() for _id in ids}

        if shifts is None:
            shifts = {}

        need_f0 = False
        recon_fn = _reconstruct_gen

        # If no shifts are provided, compute them
        for arg_name, inner_dict in ids.items():
            _jobs = {}

            for par_idx in inner_dict:
                _shifts = shifts.get(arg_name)
                if _shifts is not None:
                    _shifts = _shifts.get(par_idx)

                # Determine spectrum and number of frequencies, discounting for 0
                _spectrum = spectra[arg_name][par_idx]
                _R = len(_spectrum) - 1
                # Determine whether f0 is needed and whether the shifts have the correct shape
                if _shifts is not None:
                    need_f0 = True
                    if len(_shifts) != 2 * _R + 1:
                        raise ValueError(
                            f"The number of provided shifts ({len(_shifts)}) does not fit to the "
                            f"number of frequencies (2R+1={2*_R+1}) for parameter {par_idx} in "
                            f"argument {arg_name}."
                        )
                if _shifts is None or any(qml.math.isclose(_shifts, 0.0, rtol=0, atol=atol)):
                    need_f0 = True

                # Store job
                if _R > 0:
                    _jobs[par_idx] = {"shifts": _shifts, "spectrum": _spectrum}
                else:
                    # _R=0 belongs to a constant function
                    _jobs[par_idx] = None

            jobs[arg_name] = _jobs

    else:
        jobs = {}
        need_f0 = True
        if ids is None:
            ids = {outer_key: inner_dict.keys() for outer_key, inner_dict in nums_frequency.items()}
        elif isinstance(ids, str):
            # ids only provides a single argument name but no parameter indices
            ids = {ids: nums_frequency[ids].keys()}
        elif not isinstance(ids, dict):
            # ids only provides argument names but no parameter indices
            ids = {_id: nums_frequency[_id].keys() for _id in ids}
        recon_fn = _reconstruct_equ

        for arg_name, inner_dict in ids.items():
            _jobs = {}

            for par_idx in inner_dict:
                _num_frequency = nums_frequency[arg_name][par_idx]
                # Store job; f0 missing
                _jobs[par_idx] = {"num_frequency": _num_frequency} if _num_frequency > 0 else None

            jobs[arg_name] = _jobs

    return ids, recon_fn, jobs, need_f0


def reconstruct(qnode, ids=None, nums_frequency=None, spectra=None, shifts=None):
    r"""Reconstruct univariate restrictions of an expectation value QNode.

    Args:
        qnode (pennylane.QNode): Quantum node to be reconstructed, representing a
            circuit that outputs an expectation value.
        ids (dict or Sequence or str): Indices for the QNode parameters with respect to which
            the QNode should be reconstructed as a univariate function, per QNode argument.
            Each key of the dict, entry of the list, or the single ``str`` has to be the name
            of an argument of ``qnode`` .
            If a ``dict`` , the values of ``ids`` have to contain the parameter indices
            for the respective array-valued QNode argument represented by the key.
            If a ``list`` , the parameter indices are inferred from ``nums_frequency`` if
            given or ``spectra`` else.
            If ``None``, all keys present in ``nums_frequency`` / ``spectra`` are considered.
        nums_frequency (dict[dict]): Numbers of integer frequencies -- and biggest
            frequency -- per QNode parameter. The keys have to be argument names of ``qnode``
            and the inner dictionaries have to be mappings from parameter indices to the
            respective integer number of frequencies. If the QNode frequencies are not contiguous
            integers, the argument ``spectra`` should be used to save evaluations of ``qnode`` .
            Takes precedence over ``spectra`` and leads to usage of equidistant shifts.
        spectra (dict[dict]): Frequency spectra per QNode parameter.
            The keys have to be argument names of ``qnode`` and the inner dictionaries have to
            be mappings from parameter indices to the respective frequency spectrum for that
            parameter. Ignored if ``nums_frequency!=None``.
        shifts (dict[dict]): Shift angles for the reconstruction per QNode parameter.
            The keys have to be argument names of ``qnode`` and the inner dictionaries have to
            be mappings from parameter indices to the respective shift angles to be used for that
            parameter. For :math:`R` non-zero frequencies, there must be :math:`2R+1` shifts
            given. Ignored if ``nums_frequency!=None``.

    Returns:
        function: Function which accepts the same arguments as the QNode.
            When called, this function will return a dictionary of dictionaries,
            formatted like ``nums_frequency`` or ``spectra`` ,
            that contains the univariate reconstructions per QNode parameter.

    For each provided ``id`` in ``ids``, the QNode is restricted to varying the single QNode
    parameter corresponding to the ``id`` . This univariate function is then reconstructed
    via a Fourier transform or Dirichlet kernels, depending on the provided input.
    Either the ``spectra`` of the QNode with respect to its input parameter or the numbers
    of frequencies, ``nums_frequency`` , per parameter must be provided.

    **Usage Details**

    *Input formatting*

    As described briefly above, the essential inputs to ``reconstruct`` that provide information
    about the QNode are given as dictionaries of dictionaries, where the outer keys reference
    the argument names of ``qnode`` and the inner keys reference the parameter indices within
    one array-valued QNode argument. For scalar-valued QNode parameters, the parameter index is
    set to ``0`` by convention (also see the examples below).
    This applies to ``nums_frequency`` , ``spectra`` , and ``shifts`` .

    Note that the information provided in ``nums_frequency`` / ``spectra`` is essential for
    the correctness of the reconstruction.

    On the other hand, the input format for ``ids`` is flexible and allows a collection of
    parameter indices for each QNode argument name (as a ``dict`` ), a collection of argument
    names (as a ``list``, ``set``, ``tuple`` or similar), or a single argument name
    (as a ``str`` ) to be defined. For ``ids=None`` , all argument names contained in
    ``nums_frequency`` -- or ``spectra`` if ``nums_frequency`` is not used -- are considered.
    For inputs that do not specify parameter indices per QNode argument name (all formats but
    ``dict`` ), these parameter indices are inferred from ``nums_frequency`` / ``spectra`` .

    *Reconstruction cost*

    The reconstruction cost -- in terms of calls to ``qnode`` -- depend on the number of
    frequencies given via ``nums_frequency`` or ``spectra`` . A univariate reconstruction
    for :math:`R` frequencies takes :math:`2R+1` evaluations. If multiple univariate
    reconstructions are performed at the same point with various numbers of frequencies
    :math:`R_k` , the cost are :math:`1+2\sum_k R_k` if the shift :math:`0` is used in all
    of them. This is in particular the case if ``nums_frequency`` or ``spectra`` with
    ``shifts=None`` is used.

    If the number of frequencies is too large or the given frequency spectrum contains
    more than the spectrum of ``qnode`` , the reconstruction is performed suboptimally
    but remains correct.
    For integer-valued spectra with gaps, the equidistant reconstruction is thus suboptimal
    and the non-equidistant version method be used.

    *Numerical stability*

    In general, the reconstruction with equidistant shifts for equidistant frequencies
    (used if ``nums_frequency`` is provided) is more stable numerically than the more
    general Fourier reconstruction (used if ``nums_frequency=None`` ).
    If the system of equations to be solved in the Fourier transform is
    ill-conditioned, a warning is raised as the output might become unstable.
    Examples for this are shift values or frequencies that lie very close to each other.

    **Examples**

    Consider the following QNode:

    .. code-block:: python

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x, Y, f=1.0):
            qml.RX(f*x, wires=0)
            qml.RY(Y[0], wires=0)
            qml.RY(Y[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(5*Y[1], wires=1)
            return qml.expval(qml.PauliZ(0)@qml.PauliZ(1))

        x = 0.4
        Y = np.array([1.9, -0.5])
        f = 2.3

        circuit_value = circuit(x, Y)

    It has three variational parameters ``x`` (a scalar) and two entries of ``Y``
    (an array-like) as well as a tunable frequency ``f`` for the ``qml.RX`` rotation.
    A first reconstruction job could be with respect to the first entry of ``Y``,
    which enters the circuit with a single integer frequency:

    >>> nums_frequency = {"Y": {0: 1, 1: 6}}
    >>> rec = qml.fourier.reconstruct(circuit, {"Y": [0]}, nums_frequency)(x, Y)
    >>> rec
    {'Y': {0: <function _reconstruct_equ.<locals>._reconstruction at 0x7f949627d790>}}
    >>> recon_Y0 = rec["Y"][0]
    >>> np.isclose(recon_Y0(Y[0]), circuit_value)
    True
    >>> np.isclose(recon_Y0(Y[0]+1.3), circuit(x, Y+np.eye(2)[0]*1.3)
    True

    We successfully reconstructed the dependence on ``Y[0]`` , keeping ``x`` and ``Y[1]``
    at the values initialized above. Note that even though information about ``Y[1]`` is
    contained in ``nums_frequency`` , ``ids`` determines which reconstructions are performed.
    We may do the same for ``Y[1]`` , which enters the circuit with maximal frequency
    :math:`1+5=4` . We will also track the number of executions needed:

    >>> dev._num_executions = 0
    >>> rec = qml.fourier.reconstruct(circuit, {"Y": [1]}, nums_frequency)(x, Y)
    >>> dev.num_executions
    13

    As expected, we required :math:`2R+1=2\cdot 6+1=13` circuit executions. However, not
    all frequencies below :math:`f_\text{max}=6` are present in the circuit, so that
    a reconstruction using knowledge of the spectrum will be cheaper:

    >>> dev._num_executions = 0
    >>> spectra = {"Y": {1: [0., 1., 4., 5., 6.]}}
    >>> rec = qml.fourier.reconstruct(circuit, {"Y": [1]}, None, spectra)(x, Y)
    >>> dev.num_executions
    9

    If we want to reconstruct the dependence of ``circuit`` on ``x`` , we can not use
    ``nums_frequency`` if ``f`` is not an integer. One could rescale ``x`` to obtain
    the frequency :math:`1` again, or directly use ``spectra`` . We will combine this
    with another reconstruction with respect to ``Y[0]`` :

    >>> dev._num_executions = 0
    >>> spectra = {"x": {0: [0., f]}, "Y": {0: [0., 1.]}}
    >>> rec = qml.fourier.reconstruct(circuit, None, None, spectra)(x, Y, f=f)
    >>> dev.num_executions
    5
    >>> recon_x = rec["x"][0]
    >>> np.isclose(recon_x(x+0.5), circuit(x+0.5, Y, f=f)
    True

    Note that by convention, the parameter index for a scalar variable is ``0`` and
    that the frequency :math:`0` always is included in the spectra. Furthermore, we
    here skipped the input ``ids`` so that the reconstruction was performed for all
    keys in ``spectra`` . While the reconstruction with a single non-zero frequency
    costs three evaluations of ``circuit`` for each, ``x`` and ``Y[0]`` , carrying them
    out at the same position allowed us to save one evaluation and reduce the number
    of calls to :math:`5`.
    """
    # pylint: disable=cell-var-from-loop, unused-argument

    atol = 1e-8
    ids, recon_fn, jobs, need_f0 = _prepare_jobs(ids, nums_frequency, spectra, shifts, atol)
    arg_names = list(signature(qnode).parameters.keys())

    @wraps(qnode)
    def wrapper(*args, **kwargs):
        nonlocal spectra
        if need_f0:
            f0 = qnode(*args, **kwargs)
        else:
            f0 = None

        def constant_fn(x):
            """Univariate reconstruction of a constant Fourier series."""
            return f0

        # Carry out the reconstruction jobs
        reconstructions = {}
        for arg_name, inner_dict in jobs.items():
            _reconstructions = {}
            arg_idx = arg_names.index(arg_name)

            for par_idx, job in inner_dict.items():
                if job is None:
                    _reconstructions[par_idx] = constant_fn
                else:
                    shift_vec = qml.math.zeros_like(args[arg_idx])
                    if len(qml.math.shape(shift_vec)) == 0:
                        shift_vec = 1.0
                        x0 = args[arg_idx]
                    else:
                        shift_vec[par_idx] = 1.0
                        x0 = args[arg_idx][par_idx]

                    def _univariate_fn(x):
                        new_arg = args[arg_idx] + shift_vec * x
                        new_args = args[:arg_idx] + (new_arg,) + args[arg_idx + 1 :]
                        return qnode(*new_args, **kwargs)

                    _reconstructions[par_idx] = recon_fn(_univariate_fn, **job, x0=x0, f0=f0)

            reconstructions[arg_name] = _reconstructions

        return reconstructions

    return wrapper

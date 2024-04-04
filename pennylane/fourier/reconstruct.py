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
from autoray import numpy as anp
import pennylane as qml


def _reconstruct_equ(fun, num_frequency, x0=None, f0=None, interface=None):
    r"""Reconstruct a univariate Fourier series with consecutive integer
    frequencies, using trigonometric interpolation and equidistant shifts.

    This technique is based on
    `Dirichlet kernels <https://en.wikipedia.org/wiki/Dirichlet_kernel>`_, see
    `Vidal and Theis (2018) <https://arxiv.org/abs/1812.06323>`_ or
    `Wierichs et al. (2022) <https://doi.org/10.22331/q-2022-03-30-677>`_.

    Args:
        fun (callable): Univariate finite Fourier series to reconstruct.
            It must have signature ``float -> float`` .
        num_frequency (int): Number of integer frequencies in ``fun``.
            All integer frequencies below ``num_frequency`` are assumed
            to be present in ``fun`` as well; if they are not, the output
            is correct put the reconstruction could have been performed
            with fewer evaluations of ``fun`` .
        x0 (float): Center to which to shift the reconstruction.
            The points at which ``fun`` is evaluated are *not* affected
            by ``x0`` .
        f0 (float): Value of ``fun`` at zero; Providing ``f0`` saves one
            evaluation of ``fun``.
        interface (str): Which auto-differentiation framework to use as
            interface. This determines in which interface the output
            reconstructed function is intended to be used.

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
    shifts = qml.math.concatenate([shifts_neg, [0.0], shifts_pos])
    shifts = anp.asarray(shifts, like=interface)
    f0 = fun(0.0) if f0 is None else f0
    evals = (
        list(map(fun, shifts[:num_frequency])) + [f0] + list(map(fun, shifts[num_frequency + 1 :]))
    )
    evals = anp.asarray(evals, like=interface)

    x0 = anp.asarray(np.float64(0.0), like=interface) if x0 is None else x0

    def _reconstruction(x):
        """Univariate reconstruction based on equidistant shifts and Dirichlet kernels.
        The derivative at of ``sinc`` are not well-implemented in TensorFlow and Autograd,
        use the Fourier transform reconstruction if this derivative is needed.
        """
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


def _reconstruct_gen(fun, spectrum, shifts=None, x0=None, f0=None, interface=None):
    r"""Reconstruct a univariate (real-valued) Fourier series with given spectrum.

    Args:
        fun (callable): Univariate finite Fourier series to reconstruct.
            It must have signature ``float -> float`` .
        spectrum (Collection): Frequency spectrum of the Fourier series;
            non-positive frequencies are ignored.
        shifts (Sequence): Shift angles at which to evaluate ``fun`` for the reconstruction.
            Chosen equidistantly within the interval :math:`[0, 2\pi/f_\text{max}]`
            if ``shifts=None`` , where :math:`f_\text{max}` is the biggest
            frequency in ``spectrum``.
        x0 (float): Center to which to shift the reconstruction.
            The points at which ``fun`` is evaluated are *not* affected
            by ``x0`` .
        f0 (float): Value of ``fun`` at zero; If :math:`0` is among the ``shifts``
            and ``f0`` is provided, one evaluation of ``fun`` is saved.
        interface (str): Which auto-differentiation framework to use as
            interface. This determines in which interface the output
            reconstructed function is intended to be used.

    Returns:
        callable: Reconstructed Fourier series with :math:`R` frequencies in ``spectrum`` .
        This function is a purely classical function. Furthermore, it is fully differentiable.
    """
    # pylint: disable=unused-argument, too-many-arguments

    have_f0 = f0 is not None
    have_shifts = shifts is not None

    spectrum = anp.asarray(spectrum, like=interface)
    spectrum = spectrum[spectrum > 0]
    f_max = qml.math.max(spectrum)

    # If no shifts are provided, choose equidistant ones
    if not have_shifts:
        R = qml.math.shape(spectrum)[0]
        shifts = qml.math.arange(-R, R + 1) * 2 * np.pi / (f_max * (2 * R + 1)) * R
        zero_idx = R
        need_f0 = True
    elif have_f0:
        zero_idx = qml.math.where(qml.math.isclose(shifts, qml.math.zeros_like(shifts[0])))
        zero_idx = zero_idx[0][0] if (len(zero_idx) > 0 and len(zero_idx[0]) > 0) else None
        need_f0 = zero_idx is not None

    # Take care of shifts close to zero if f0 was provided
    if have_f0 and need_f0:
        # Only one shift may be zero at a time
        shifts = qml.math.concatenate(
            [shifts[zero_idx : zero_idx + 1], shifts[:zero_idx], shifts[zero_idx + 1 :]]
        )
        shifts = anp.asarray(shifts, like=interface)
        evals = anp.asarray([f0] + list(map(fun, shifts[1:])), like=interface)
    else:
        shifts = anp.asarray(shifts, like=interface)
        if have_f0 and not need_f0:
            warnings.warn(_warn_text_f0_ignored)
        evals = anp.asarray(list(map(fun, shifts)), like=interface)

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
    a = anp.asarray(W[1 : R + 1], like=interface)
    b = anp.asarray(W[R + 1 :], like=interface)

    x0 = anp.asarray(np.float64(0.0), like=interface) if x0 is None else x0

    # Construct the Fourier series
    def _reconstruction(x):
        """Univariate reconstruction based on arbitrary shifts."""
        x = x - x0
        return (
            a0
            + qml.math.tensordot(qml.math.cos(spectrum * x), a, axes=[[0], [0]])
            + qml.math.tensordot(qml.math.sin(spectrum * x), b, axes=[[0], [0]])
        )

    return _reconstruction


def _parse_ids(ids, info_dict):
    """Parse different formats of ``ids`` into the right dictionary format,
    potentially using the information in ``info_dict`` to complete it.
    """
    if ids is None:
        # Infer all id information from info_dict
        return {outer_key: inner_dict.keys() for outer_key, inner_dict in info_dict.items()}
    if isinstance(ids, str):
        # ids only provides a single argument name but no parameter indices
        return {ids: info_dict[ids].keys()}
    if not isinstance(ids, dict):
        # ids only provides argument names but no parameter indices
        return {_id: info_dict[_id].keys() for _id in ids}

    return ids


def _parse_shifts(shifts, R, arg_name, par_idx, atol, need_f0):
    """Processes shifts for a single reconstruction and determines
    wheter the function at the reconstruction point, ``f0`` will be
    needed.
    """
    # pylint: disable=too-many-arguments
    _shifts = shifts.get(arg_name)
    if _shifts is not None:
        _shifts = _shifts.get(par_idx)
    if _shifts is not None:
        # Check whether the _shifts have the correct size
        if len(_shifts) != 2 * R + 1:
            raise ValueError(
                f"The number of provided shifts ({len(_shifts)}) does not fit to the "
                f"number of frequencies (2R+1={2*R+1}) for parameter {par_idx} in "
                f"argument {arg_name}."
            )
        if any(qml.math.isclose(_shifts, qml.math.zeros_like(_shifts), rtol=0, atol=atol)):
            # If 0 is among the shifts, f0 is needed
            return _shifts, True
        # If 0 is not among the shifts, f0 is not needed
        return _shifts, (False or need_f0)
    # If no shifts are given, f0 is needed always
    return _shifts, True


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
            These indices always are tuples, i.e. ``()`` for scalar and ``(i,)`` for
            one-dimensional arguments.
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
    if nums_frequency is None:
        if spectra is None:
            raise ValueError("Either nums_frequency or spectra must be given.")

        ids = _parse_ids(ids, spectra)

        if shifts is None:
            shifts = {}

        need_f0 = False
        recon_fn = _reconstruct_gen

        jobs = {}

        # If no shifts are provided, compute them
        for arg_name, inner_dict in ids.items():
            _jobs = {}

            for par_idx in inner_dict:
                # Determine spectrum and number of frequencies, discounting for 0
                _spectrum = spectra[arg_name][par_idx]
                R = len(_spectrum) - 1
                _shifts, need_f0 = _parse_shifts(shifts, R, arg_name, par_idx, atol, need_f0)

                # Store job
                if R > 0:
                    _jobs[par_idx] = {"shifts": _shifts, "spectrum": _spectrum}
                else:
                    # R=0 belongs to a constant function
                    _jobs[par_idx] = None

            jobs[arg_name] = _jobs

    else:
        jobs = {}
        need_f0 = True

        ids = _parse_ids(ids, nums_frequency)

        recon_fn = _reconstruct_equ

        for arg_name, inner_dict in ids.items():
            _jobs = {}

            for par_idx in inner_dict:
                _num_frequency = nums_frequency[arg_name][par_idx]
                _jobs[par_idx] = {"num_frequency": _num_frequency} if _num_frequency > 0 else None

            jobs[arg_name] = _jobs

    return ids, recon_fn, jobs, need_f0


def reconstruct(qnode, ids=None, nums_frequency=None, spectra=None, shifts=None):
    r"""Reconstruct an expectation value QNode along a single parameter direction.
    This means we restrict the QNode to vary only one parameter, a univariate restriction.
    For common quantum gates, such restrictions are finite Fourier series with known
    frequency spectra. Thus they may be reconstructed using Dirichlet kernels or
    a non-uniform Fourier transform.

    Args:
        qnode (pennylane.QNode): Quantum node to be reconstructed, representing a
            circuit that outputs an expectation value.
        ids (dict or Sequence or str): Indices for the QNode parameters with respect to which
            the QNode should be reconstructed as a univariate function, per QNode argument.
            Each key of the dict, entry of the list, or the single ``str`` has to be the name
            of an argument of ``qnode`` .
            If a ``dict`` , the values of ``ids`` have to contain the parameter indices
            for the respective array-valued QNode argument represented by the key.
            These indices always are tuples, i.e., ``()`` for scalar and ``(i,)`` for
            one-dimensional arguments.
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
        function: Function which accepts the same arguments as the QNode and one additional
        keyword argument ``f0`` to provide the QNode value at the given arguments.
        When called, this function will return a dictionary of dictionaries,
        formatted like ``nums_frequency`` or ``spectra`` ,
        that contains the univariate reconstructions per QNode parameter.

    For each provided ``id`` in ``ids``, the QNode is restricted to varying the single QNode
    parameter corresponding to the ``id`` . This univariate function is then reconstructed
    via a Fourier transform or Dirichlet kernels, depending on the provided input.
    Either the frequency ``spectra`` of the QNode with respect to its input parameters or
    the numbers of frequencies, ``nums_frequency`` , per parameter must be provided.

    For quantum-circuit specific details, we refer the reader to
    `Vidal and Theis (2018) <https://arxiv.org/abs/1812.06323>`__ ,
    `Vidal and Theis (2020) <https://www.frontiersin.org/articles/10.3389/fphy.2020.00297/full>`__ ,
    `Schuld, Sweke and Meyer (2021) <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.103.032430>`__ ,
    and
    `Wierichs, Izaac, Wang and Lin (2022) <https://doi.org/10.22331/q-2022-03-30-677>`__ .
    An introduction to the concept of quantum circuits as Fourier series can also be found in
    the
    `Quantum models as Fourier series <https://pennylane.ai/qml/demos/tutorial_expressivity_fourier_series.html>`__
    and
    `General parameter-shift rules <https://pennylane.ai/qml/demos/tutorial_general_parshift.html>`__
    demos as well as the
    :mod:`qml.fourier <pennylane.fourier>` module docstring.

    **Example**

    Consider the following QNode:

    .. code-block:: python

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x, Y):
            qml.RX(x, wires=0)
            qml.RY(Y[0], wires=0)
            qml.RY(Y[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(5*  Y[1], wires=1)
            return qml.expval(qml.Z(0) @ qml.Z(1))

        x = 0.4
        Y = np.array([1.9, -0.5])
        f = 2.3

        circuit_value = circuit(x, Y)

    It has three variational parameters ``x`` (a scalar) and two entries of ``Y``
    (an array-like).
    A reconstruction job could then be with respect to the two entries of ``Y``,
    which enter the circuit with one and six integer frequencies, respectively
    (see the additional examples below for details on how to obtain the frequency
    spectrum if it is not known):

    >>> nums_frequency = {"Y": {(0,): 1, (1,): 6}}
    >>> with qml.Tracker(circuit.device) as tracker:
    ...     rec = qml.fourier.reconstruct(circuit, {"Y": [(0,), (1,)]}, nums_frequency)(x, Y)
    >>> rec.keys()
    dict_keys(['Y'])
    >>> print(*rec["Y"].items(), sep="\n")
    ((0,), <function _reconstruct_equ.<locals>._reconstruction at 0x7fbd685aee50>)
    ((1,), <function _reconstruct_equ.<locals>._reconstruction at 0x7fbd6866eee0>)
    >>> recon_Y0 = rec["Y"][(0,)]
    >>> recon_Y1 = rec["Y"][(1,)]
    >>> np.isclose(recon_Y0(Y[0]), circuit_value)
    True
    >>> np.isclose(recon_Y1(Y[1]+1.3), circuit(x, Y+np.eye(2)[1]*1.3))
    True

    We successfully reconstructed the dependence on the two entries of ``Y`` ,
    keeping ``x`` and the respective other entry in ``Y`` at their initial values.
    Let us also see how many executions of the device were used to obtain the
    reconstructions:

    >>> tracker.totals
    {'batches': 15, 'simulations': 15, 'executions': 15}

    The example above used that we already knew the frequency spectra of the
    QNode of interest. However, this is in general not the case and we may need
    to compute the spectrum first. This can be done with
    :func:`.fourier.qnode_spectrum` :

    >>> spectra = qml.fourier.qnode_spectrum(circuit)(x, Y)
    >>> spectra.keys()
    dict_keys(['x', 'Y'])
    >>> spectra["x"]
    {(): [-1.0, 0.0, 1.0]}
    >>> print(*spectra["Y"].items(), sep="\n")
    ((0,), [-1.0, 0.0, 1.0])
    ((1,), [-6.0, -5.0, -4.0, -1.0, 0.0, 1.0, 4.0, 5.0, 6.0])

    For more detailed explanations, usage details and additional examples, see
    the usage details section below.

    .. details::
        :title: Usage Details

        **Input formatting**

        As described briefly above, the essential inputs to ``reconstruct`` that provide information
        about the QNode are given as dictionaries of dictionaries, where the outer keys reference
        the argument names of ``qnode`` and the inner keys reference the parameter indices within
        each array-valued QNode argument. These parameter indices always are tuples, so that
        for scalar-valued QNode parameters, the parameter index is ``()`` by convention and the
        ``i`` -th parameter of a one-dimensional array can be accessed via ``(i,)`` .
        For example, providing ``nums_frequency``

        - for a scalar argument: ``nums_frequency = {"x": {(): 4}}``
        - for a one-dimensional argument: ``nums_frequency = {"Y": {(0,): 2, (1,): 9, (4,): 1}}``
        - for a three-dimensional argument: ``nums_frequency = {"Z": {(0, 2, 5): 2, (1, 1, 4): 1}}``

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

        **Reconstruction cost**

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
        and the non-equidistant version method be used (also see the examples below).

        **Numerical stability**

        In general, the reconstruction with equidistant shifts for equidistant frequencies
        (used if ``nums_frequency`` is provided) is more stable numerically than the more
        general Fourier reconstruction (used if ``nums_frequency=None`` ).
        If the system of equations to be solved in the Fourier transform is
        ill-conditioned, a warning is raised as the output might become unstable.
        Examples for this are shift values or frequencies that lie very close to each other.

        **Differentiability**

        The returned scalar functions are differentiable in all interfaces with respect
        to their scalar input variable. They expect these inputs to be in the same
        interface as the one used by the QNode. More advanced differentiability, for example
        of the reconstructions with respect to QNode properties, is not supported
        reliably yet.

        .. warning::

            When using ``TensorFlow`` or ``Autograd`` *and* ``nums_frequency`` ,
            the reconstructed functions are not differentiable at the point of
            reconstruction. One workaround for this is to use ``spectra`` as
            input instead and to thereby use the Fourier transform instead of
            Dirichlet kernels. Alternatively, the original QNode evaluation can
            be used.

        **More examples**

        Consider the QNode from the example above, now with an additional, tunable frequency
        ``f`` for the Pauli-X rotation that is controlled by ``x`` :

        .. code-block:: python

            @qml.qnode(dev)
            def circuit(x, Y, f=1.0):
                qml.RX(f * x, wires=0)
                qml.RY(Y[0], wires=0)
                qml.RY(Y[1], wires=1)
                qml.CNOT(wires=[0, 1])
                qml.RY(5*  Y[1], wires=1)
                return qml.expval(qml.Z(0) @ qml.Z(1))

            f = 2.3

            circuit_value = circuit(x, Y)


        We repeat the reconstruction job for the dependence on ``Y[1]`` .
        Note that even though information about ``Y[0]`` is contained in ``nums_frequency`` ,
        ``ids`` determines which reconstructions are performed.

        >>> with qml.Tracker(circuit.device) as tracker:
        ...     rec = qml.fourier.reconstruct(circuit, {"Y": [(1,)]}, nums_frequency)(x, Y)
        >>> tracker.totals
        {'executions': 13}

        As expected, we required :math:`2R+1=2\cdot 6+1=13` circuit executions. However, not
        all frequencies below :math:`f_\text{max}=6` are present in the circuit, so that
        a reconstruction using knowledge of the full frequency spectrum will be cheaper:

        >>> spectra = {"Y": {(1,): [0., 1., 4., 5., 6.]}}
        >>> with tracker:
        ...     rec = qml.fourier.reconstruct(circuit, {"Y": [(1,)]}, None, spectra)(x, Y)
        >>> tracker.totals
        {'executions': 9}

        We again obtain the full univariate dependence on ``Y[1]`` but with considerably
        fewer executions on the quantum device.
        Once we obtained the classical function that describes the dependence, no
        additional circuit evaluations are performed:

        >>> with tracker:
        ...     for Y1 in np.arange(-np.pi, np.pi, 20):
        ...         rec["Y"][(1,)](-2.1)
        >>> tracker.totals
        {}

        If we want to reconstruct the dependence of ``circuit`` on ``x`` , we cannot use
        ``nums_frequency`` if ``f`` is not an integer. One could rescale ``x`` to obtain
        the frequency :math:`1` again, or directly use ``spectra`` . We will combine the
        latter with another reconstruction with respect to ``Y[0]`` :

        >>> spectra = {"x": {(): [0., f]}, "Y": {(0,): [0., 1.]}}
        >>> with tracker:
        ...     rec = qml.fourier.reconstruct(circuit, None, None, spectra)(x, Y, f=f)
        >>> tracker.totals
        {'executions': 5}
        >>> recon_x = rec["x"][()]
        >>> np.isclose(recon_x(x+0.5), circuit(x+0.5, Y, f=f)
        True

        Note that by convention, the parameter index for a scalar variable is ``()`` and
        that the frequency :math:`0` always needs to be included in the spectra.
        Furthermore, we here skipped the input ``ids`` so that the reconstruction
        was performed for all keys in ``spectra`` .
        The reconstruction with a single non-zero frequency
        costs three evaluations of ``circuit`` for each, ``x`` and ``Y[0]`` . Performing
        both reconstructions at the same position allowed us to save one of the
        evaluations and reduce the number of calls to :math:`5`.

    """
    # pylint: disable=cell-var-from-loop, unused-argument

    atol = 1e-8
    ids, recon_fn, jobs, need_f0 = _prepare_jobs(ids, nums_frequency, spectra, shifts, atol)
    sign_fn = qnode.func if isinstance(qnode, qml.QNode) else qnode
    arg_names = list(signature(sign_fn).parameters.keys())
    arg_idx_from_names = {arg_name: i for i, arg_name in enumerate(arg_names)}

    @wraps(qnode)
    def wrapper(*args, f0=None, **kwargs):
        if f0 is None and need_f0:
            f0 = qnode(*args, **kwargs)

        interface = qml.math.get_interface(args[0])

        def constant_fn(x):
            """Univariate reconstruction of a constant Fourier series."""
            return f0

        # Carry out the reconstruction jobs
        reconstructions = {}
        for arg_name, inner_dict in jobs.items():
            _reconstructions = {}
            arg_idx = arg_idx_from_names[arg_name]

            for par_idx, job in inner_dict.items():
                if job is None:
                    _reconstructions[par_idx] = constant_fn
                else:
                    if len(qml.math.shape(args[arg_idx])) == 0:
                        shift_vec = qml.math.ones_like(args[arg_idx])
                        x0 = args[arg_idx]
                    else:
                        shift_vec = qml.math.zeros_like(args[arg_idx])
                        shift_vec = qml.math.scatter_element_add(shift_vec, par_idx, 1.0)
                        x0 = args[arg_idx][par_idx]

                    def _univariate_fn(x):
                        new_arg = args[arg_idx] + shift_vec * x
                        new_args = args[:arg_idx] + (new_arg,) + args[arg_idx + 1 :]
                        return qnode(*new_args, **kwargs)

                    _reconstructions[par_idx] = recon_fn(
                        _univariate_fn, **job, x0=x0, f0=f0, interface=interface
                    )

            reconstructions[arg_name] = _reconstructions

        return reconstructions

    return wrapper

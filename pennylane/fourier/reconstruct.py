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
from collections.abc import Collection
from functools import wraps
from inspect import signature
import warnings

import pennylane as qml
from pennylane import numpy as np


def _reconstruct_equ(fun, num_frequency, x0=None, f0=None):
    r"""Reconstruct a univariate Fourier series with consecutive integer
    frequencies, using trigonometric interpolation and equidistant shifts.

    This technique is based on
    `Dirichlet kernels <https://en.wikipedia.org/wiki/Dirichlet_kernel>`_, see
    `Vidal and Theis (2018) <https://arxiv.org/abs/1812.06323>`_ or
    `Wierichs et al. (2021) <https://arxiv.org/abs/2107.12390>`_.

    Args:
        fun (callable): Univariate finite Fourier series to reconstruct.
        num_frequency (int): Number of integer frequencies in ``fun``
        x0 (float): Center with respect to which to reconstruct.
        f0 (float): Value of ``fun`` at ``x0`` ; Providing ``f0`` saves one
            evaluation of ``fun``

    Returns:
        callable: Reconstructed Fourier series with ``num_frequency`` frequencies,
        as ``qml.numpy`` based function.
    """
    if not abs(int(num_frequency)) == num_frequency:
        raise ValueError(f"num_frequency must be a non-negative integer, got {num_frequency}")

    a = (num_frequency + 0.5) / np.pi
    b = 0.5 / np.pi

    shifts_pos = qml.math.arange(1, num_frequency + 1) / a
    shifts_neg = -shifts_pos[::-1]
    x0 = qml.math.array(0.0) if x0 is None else x0
    f0 = fun(0.0) if f0 is None else f0
    evals = list(map(fun, shifts_neg)) + [f0] + list(map(fun, shifts_pos))
    shifts = qml.math.concatenate([shifts_neg, [0.0], shifts_pos])

    def _reconstruction(x):
        """Univariate reconstruction based on equidistant shifts and Dirichlet kernels."""
        x = x - x0
        return qml.math.tensordot(
            qml.math.sinc(a * (x - shifts)) / qml.math.sinc(b * (x - shifts)),
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
        fun (callable): Fourier series to reconstruct with signature ``float -> float``
        spectrum (Collection): Frequency spectrum of the Fourier series; non-positive
            frequencies are ignored
        shifts (Sequence): Shift angles at which to evaluate ``fun`` for the reconstruction
            Chosen equidistantly within the interval :math:`[0, 2\pi/f_\text{min}]` if ``shifts=None``
            where :math:`f_\text{min}` is the smallest frequency in ``spectrum``.
        x0 (float): Center with respect to which to reconstruct.
        f0 (float): Value of ``fun`` at ``x0`` . If :math:`0` is among the ``shifts``
            and ``f0`` is provided, one evaluation of ``fun`` is saved.

    Returns:
        callable: Reconstructed Fourier series with :math:`R` frequencies in ``spectrum``,
        as ``qml.numpy`` based function and coinciding with ``fun`` on :math:`2R+1` points.
    """
    # pylint: disable=unused-argument

    have_f0 = f0 is not None
    have_shifts = shifts is not None

    spectrum = qml.math.array(spectrum)
    spectrum = spectrum[spectrum > 0]
    f_max = max(spectrum)

    # If no shifts are provided, choose equidistant ones
    if not have_shifts:
        R = len(spectrum)
        shifts = np.arange(-R, R + 1) * 2 * np.pi / (f_max * (2 * R + 1)) * R
        zero_idx = R
        need_f0 = True
    elif have_f0:
        zero_idx = np.where(np.isclose(shifts, 0.0))[0]
        zero_idx = zero_idx[0] if len(zero_idx) > 0 else None
        need_f0 = zero_idx is not None

    # Take care of shifts close to zero if f0 was provided
    if have_f0 and need_f0:
        # Only one shift may be zero at a time
        shifts = np.concatenate([[shifts[zero_idx]], shifts[:zero_idx], shifts[zero_idx + 1 :]])
        evals = np.array([f0] + list(map(fun, shifts[1:])))
    else:
        if have_f0 and not need_f0:
            warnings.warn(_warn_text_f0_ignored)
        evals = np.array(list(map(fun, shifts)))

    L = len(shifts)
    # Construct the coefficient matrix case by case
    C1 = np.ones((L, 1))
    C2 = np.cos(np.outer(shifts, spectrum))
    C3 = np.sin(np.outer(shifts, spectrum))
    C = np.hstack([C1, C2, C3])

    # Solve the system of linear equations
    cond = np.linalg.cond(C)
    if cond > 1e8:
        warnings.warn(
            f"The condition number of the Fourier transform matrix is very large: {cond}.",
            UserWarning,
        )
    W = np.linalg.solve(C, evals)

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
        return a0 + np.dot(a, np.cos(spectrum * x)) + np.dot(b, np.sin(spectrum * x))

    return _reconstruction


def _prepare_jobs(ids, spectra, shifts, nums_frequency, atol):
    r"""For inputs to reconstruct, determine how the given information yields
    function reconstruction tasks and collect them into a dictionary ``jobs``.
    Also determine whether the function at zero is needed.

    Args:
        ids (dict or Collection or key): Indices for the QNode parameters with respect to which
            the QNode should be reconstructed as a univariate function.
            If a dictionary, ``ids`` contain the QNode argument names as keys and a ``Collection``
            for each value, indicating the parameter index in the argument.
            If a ``Collection``, all parameter indices of the given arguments are considered.
            If a single key, ``ids`` is interpreted as a ``Collection`` with that key as only entry.
            If ``None``, all keys of ``nums_frequency``/``spectra`` are considered.
        spectra (dict[dict]): Frequency spectra per QNode parameter.
            Ignored if ``nums_frequency!=None``.
        shifts (dict[dict]): Shift angles for computing the reconstruction per
            QNode parameter. Ignored if ``nums_frequency!=None``.
        nums_frequency (dict[dict]): Numbers of integer frequencies -- and biggest
            frequency -- per QNode parameter. If the frequencies are not contiguous integers,
            the argument ``spectra`` should be used. Takes precedence over ``spectra`` and
            leads to usage of equidistant shifts.
        atol (float): Absolute tolerance used to analyze shifts lying close to 0.

    Returns:

    """
    # pylint: disable=too-many-branches
    if nums_frequency is None:
        jobs = {}

        if spectra is None:
            raise ValueError("Either nums_frequency or spectra must be given.")
        if ids is None:
            ids = {outer_key: inner_dict.keys() for outer_key, inner_dict in spectra.items()}
        elif not isinstance(ids, dict):
            if isinstance(ids, Collection):
                # ids only provides argument names but no parameter indices
                ids = {_id: spectra[_id].keys() for _id in ids}
            else:
                # ids only provides a single argument name but no parameter indices
                ids = {ids: spectra[ids].keys()}

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

                _spectrum = spectra[arg_name][par_idx]
                # Determine whether f0 is needed
                if _shifts is None or any(np.isclose(_shifts, 0.0, rtol=0, atol=atol)):
                    need_f0 = True

                # Store job; f0 missing
                if len(_spectrum) > 1:
                    _jobs[par_idx] = {"shifts": _shifts, "spectrum": _spectrum}
                else:
                    # As we assume 0.0 to be in the spectrum, any spectrum with length 1
                    # belongs to a constant function
                    _jobs[par_idx] = None

            jobs[arg_name] = _jobs

    else:
        jobs = {}
        need_f0 = True
        if ids is None:
            ids = {outer_key: inner_dict.keys() for outer_key, inner_dict in nums_frequency.items()}
        elif not isinstance(ids, dict):
            if isinstance(ids, Collection):
                # ids only provides argument names but no parameter indices
                ids = {_id: nums_frequency[_id].keys() for _id in ids}
            else:
                # ids only provides a single argument name but no parameter indices
                ids = {ids: nums_frequency[ids].keys()}
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
        ids (dict or list or key): Indices for the QNode parameters with respect to which
            the QNode should be reconstructed as a univariate function, per QNode argument.
            Each key of the dict or entry of the list should be a valid key for
            ``nums_frequency`` if it is provided or ``spectra`` otherwise.
            Alternatively, a single valid key may be provided.
            If ``None``, all keys of ``nums_frequency``/``spectra`` are considered.
        nums_frequency (dict[dict]): Numbers of integer frequencies -- and biggest
            frequency -- per QNode parameter. If the frequencies are not contiguous integers,
            the argument ``spectra`` should be used. Takes precedence over ``spectra`` and
            leads to usage of equidistant shifts.
        spectra (dict[dict]): Frequency spectra per QNode parameter.
            Ignored if ``nums_frequency!=None``.
        shifts (dict[dict]): Shift angles for computing the reconstruction per
            QNode parameter. Ignored if ``nums_frequency!=None``.
    Returns:
        function: Function which accepts the same arguments as the QNode.
            When called, this function will return a dictionary of dictionaries,
            formatted like ``nums_frequency`` or ``spectra``,
            that contains the univariate reconstructions per QNode parameter.

    For each provided ``id`` in ``ids``, the QNode is restricted to varying the single QNode
    parameter corresponding to the ``id``. This univariate function is then restricted
    via a Fourier transform or Dirichlet kernels depending on the provided input.
    Either the ``spectra`` of the QNode with respect to its input parameter or the numbers
    of frequencies, ``nums_frequency``, per parameter must be provided.
    """
    # pylint: disable=cell-var-from-loop, unused-argument

    atol = 1e-8
    ids, recon_fn, jobs, need_f0 = _prepare_jobs(ids, spectra, shifts, nums_frequency, atol)
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
                    if len(np.shape(shift_vec)) == 0:
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

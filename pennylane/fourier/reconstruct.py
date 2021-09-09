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
"""Contains a function that computes the fourier series equivalent to
a quantum expectation value."""
# from itertools import chain, combinations
from functools import wraps
import numpy as np
import pennylane as qml


def _reconstruct_gen(fun, spectrum, shifts=None, fun_at_zero=None):
    r"""Reconstruct a univariate (real-valued) Fourier series with given spectrum.
    Args:
        fun (callable): Fourier series to reconstruct with signature ``float -> float``
        spectrum (list): Frequency spectrum of the Fourier series; negative frequencies are ignored
        shifts (list): Shift angles at which to evaluate ``fun`` for the reconstruction
            Chosen equidistantly within the interval :math:`[0, 2\pi/f_\text{min}]` where
            :math:`f_\text{min}` is the smallest frequency in ``spectrum``.
        fun_at_zero (float): Value of ``fun`` at zero. If :math:`0` is among the ``shifts``
            and ``fun_at_zero`` is provided, one evaluation of ``fun`` is saved.

    Returns:
        callable: Reconstructed Fourier series with :math:`R` frequencies in ``spectrum``,
        as ``qml.numpy`` based function and coinciding with ``fun`` on :math:`2R+1` points.
    """
    spectrum = [f for f in spectrum if f>0.]
    f_min = min(spectrum)
    # If no shifts are provided, choose equidistant ones
    if shifts is None:
        R = len(spectrum) 
        shifts = np.arange(-R, R + 1) * 2 * np.pi / (f_min * (2 * R + 1))
        close_to_zero = np.eye(2*R+1)[R]
    else:
        close_to_zero = np.isclose(shifts, 0.0)
    # Take care of shifts close to zero if fun_at_zero was provided
    if fun_at_zero is not None and any(close_to_zero):
        # Only one shift may be zero at a time
        zero_idx = int(np.where(close_to_zero)[0])
        shifts = np.concatenate([[shifts[zero_idx]], shifts[:zero_idx], shifts[zero_idx + 1 :]])
        evals = np.array([fun_at_zero] + list(map(fun, shifts[1:])))
    else:
        evals = np.array(list(map(fun, shifts)))

    L = len(shifts)
    # Construct the coefficient matrix case by case
    C1 = np.ones((L, 1))
    C2 = np.cos(np.outer(shifts, spectrum))
    C3 = np.sin(np.outer(shifts, spectrum))
    C = np.hstack([C1, C2, C3])

    # Solve the system of linear equations
    W = np.linalg.solve(C, evals)

    # Extract the Fourier coefficients
    R = (L - 1) // 2
    a0 = W[0]
    a = W[1 : R + 1]
    b = W[R + 1 :]

    # Construct the Fourier series
    def _reconstruction(x):
        """Univariate reconstruction based on arbitrary shifts."""
        return a0 + np.dot(a, np.cos(spectrum * x)) + np.dot(b, np.sin(spectrum * x))

    return _reconstruction


def _reconstruct_equ(fun, num_frequency, fun_at_zero=None):
    r"""Reconstruct a univariate Fourier series with consecutive integer
    frequencies, using trigonometric interpolation and equidistant shifts.

    This technique is based on
    `Dirichlet kernels <https://en.wikipedia.org/wiki/Dirichlet_kernel>`_, see
    `Vidal and Theis (2018) <https://arxiv.org/abs/1812.06323>`_ or
    `Wierichs et al. (2021) <https://arxiv.org/abs/2107.12390>`_.

    Args:
        fun (callable): Function to reconstruct
        num_frequency (int): Number of integer frequencies in ``fun``
        fun_at_zero (float): Value of ``fun`` at zero; Providing ``fun_at_zero`` saves one
            evaluation of ``fun``

    Returns:
        callable: Reconstructed Fourier series with ``num_frequency`` frequencies,
        as ``qml.numpy`` based function.
    """
    a, b = (num_frequency + 0.5) / np.pi, 0.5 / np.pi
    shifts_pos = np.arange(1, num_frequency + 1) / a
    shifts_neg = -shifts_pos[::-1]
    fun_at_zero = float(fun(0.0)) if fun_at_zero is None else fun_at_zero
    evals = list(map(fun, shifts_neg)) + [fun_at_zero] + list(map(fun, shifts_pos))
    shifts = np.concatenate([shifts_neg, [0.0], shifts_pos])

    def _reconstruction(x):
        """Univariate reconstruction based on equidistant shifts and Dirichlet kernels."""
        return np.dot(evals, np.sinc(a * (x - shifts)) / np.sinc(b * (x - shifts)))

    return _reconstruction

def _interpret_id(_id):
    r"""Interpret how a given id is meant to access a container of spectrum-type.
    Args:
        _id (tuple[int, tuple] or tuple[int] or int): Id to be interpreted

    Returns:
        int, tuple[int] or int: argument index and parameter index belonging to ``_id``.
        To extract the corresponding value, call ``_getitem(data, arg_id, param_id, False)``,
        where ``data`` is the spectrum-type container.
    """
    if isinstance(_id, tuple):
        if len(_id)>1 and isinstance(_id[1], tuple):
            # Argument index and parameter index for dictionary of dictionaries
            return _id[0], _id[1]
        # Single tuple index: Outer dictionary expected to be unpacked
        return None, _id
    elif np.issubdtype(type(_id), np.int):
        # Single integer index: Inner dictionary was unpacked
        return _id, None
    elif id is None:
        # A simple spectrum/number of frequencies
        return None, None
    else:
        raise ValueError(f"Could not interpret id {_id} .")


def _getitem(_dict, arg_id, param_id, default):
    r"""Access an entry of a spectrum-type container via two indices.
    """
    try:
        sub_dict = _dict[arg_id] if arg_id is not None else _dict
        sub_dict = sub_dict[param_id] if param_id is not None else sub_dict
    except:
        if default is False:
            raise ValueError(
                f"Could not get item with arg_id={arg_id} and param_id={param_id} "
                f"from the object and default was set to False.\nobject={_dict}"
            )
        sub_dict = default

    return sub_dict


def _setitem(_dict, arg_id, param_id, value):
    r"""Set an entry of a spectrum-type container via two indices.
    """
    if arg_id is not None:
        if param_id is None:
            _dict[arg_id] = value
        else:
            if arg_id in _dict.keys():
                _dict[arg_id][param_id] = value
            else:
                _dict[arg_id] = {param_id: value}
    else:
        if param_id is None:
            _dict = value
        else:
            _dict[param_id] = value

    return _dict


def _get_ids_from_data(data):
    r"""Extract the keys of a spectrum-type container as ids.
    Args:
        data (container): Container from which to get the ids

    Returns:
        list[tuple]: Ids belonging to the keys of ``data``; each entry is one of 
        ``tuple[int, tuple[int]]`` or ``tuple[int]``.
    """
    if not isinstance(data, dict):
        return [(None, None)]
    ids = []
    for outer_key, outer_val in data.items():
        if not isinstance(outer_val, dict):
            ids.append((outer_key, None))
        else:
            ids.extend([(outer_key, inner_key) for inner_key in outer_val])
    return ids


def _prepare_jobs(ids, spectra, shifts, nums_frequency):
    r"""For inputs to reconstruct, determine how the given information yields
    function reconstruction tasks.
    """
    if ids is None:
        if nums_frequency is None:
            ids = _get_ids_from_data(spectra)
        else:
            ids = _get_ids_from_data(nums_frequency)
    # Obtain argument and parameter ids from ids
    ids = [_interpret_id(_id) for _id in ids]

    jobs = []
    if nums_frequency is None:
        if shifts is None:
            shifts = {}
        need_fun_at_zero = True
        recon_fn = _reconstruct_gen
        # If no shifts are provided, compute them
        for (arg_id, param_id) in ids:
            _shifts = _getitem(shifts, arg_id, param_id, None)
            # Determine whether fun_at_zero is needed
            if _shifts is None or any(np.isclose(_shifts, 0.0, rtol=0, atol=atol)):
                need_fun_at_zero = True
            # Store job, spectrum and fun_at_zero missing in _kwargs
            _kwargs = {"shifts": _shifts}
            jobs.append( (arg_id, param_id, _kwargs) )
    else:
        recon_fn = _reconstruct_equ
        for (arg_id, param_id) in ids:
            _num_frequency = _getitem(nums_frequency, arg_id, param_id, False)
            _kwargs = {"num_frequency": _num_frequency} if _num_frequency>0 else None
            # Store job, fun_at_zero missing in _kwargs
            jobs.append( (arg_id, param_id, _kwargs) )

    return ids, recon_fn, jobs, need_fun_at_zero

def reconstruct(
    qnode, ids=None, spectra=None, shifts=None, nums_frequency=None, decimals=5
):
    r"""Reconstructs univariate restrictions of an expectation value QNode.
    """

    ids, recon_fn, jobs, need_fun_at_zero = _prepare_jobs(
        ids, spectra, shifts, nums_frequency
    )

    atol = 1e-8
    @wraps(qnode)
    def wrapper(*args, **kwargs):
        nonlocal spectra
        if need_fun_at_zero or nums_frequency is not None:
            fun_at_zero = qnode(*args, **kwargs)

        if nums_frequency is None:
            # If no spectra are provided, compute them
            if spectra is None:
                # Obtain arguments that are treated at all to get the spectrum for
                arg_ids, _ = zip(*interpreted_ids)
                argnums = [_arg_id if _arg_id is not None else 0 for _arg_id in arg_ids]
                spectra = qml.fourier.spectrum(qnode, encoding_args=argnums, decimals=decimals)(
                    *args, **kwargs
                )
            # Update reconstruction jobs with the spectra and fun_at_zero
            for i, job in enumerate(jobs):
                _spectrum = _getitem(spectra, job[0], job[1], False)
                if len(_spectrum)>1:
                    job[-1].update({"spectrum": _spectrum, "fun_at_zero": fun_at_zero})
                else:
                    jobs[i] = job[:-1]+(None,)
        else:
            # Update reconstruction jobs with fun_at_zero
            for job in jobs:
                if job[-1] is not None:
                    job[-1].update({"fun_at_zero": fun_at_zero})

        def constant_fn(x):
            return fun_at_zero

        # Carry out the reconstruction jobs
        reconstructions = {}
        for arg_id, param_id, _kwargs in jobs:
            if _kwargs is None:
                _recon = constant_fn
            else:
                shift_vec = np.zeros_like(args[arg_id])
                shift_vec[param_id] = 1.

                def _univariate_fn(x):
                    new_arg = args[arg_id] + shift_vec * x
                    new_args = ((arg if j != arg_id else new_arg) for j, arg in enumerate(args))
                    return qnode(*new_args, **kwargs)

                _recon = recon_fn(_univariate_fn, **_kwargs)

            reconstructions = _setitem(reconstructions, arg_id, param_id, _recon)

        return reconstructions

    return wrapper

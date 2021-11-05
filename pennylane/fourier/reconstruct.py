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
from functools import wraps
from inspect import signature

# import numpy as np
import pennylane as qml
from pennylane import numpy as np


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
    if not abs(int(num_frequency)) == num_frequency:
        raise ValueError(f"num_frequency must be a non-negative integer, got {num_frequency}")

    a, b = (num_frequency + 0.5) / np.pi, 0.5 / np.pi
    shifts_pos = qml.math.arange(1, num_frequency + 1) / a
    shifts_neg = -shifts_pos[::-1]
    fun_at_zero = fun(qml.math.array(0.0)) if fun_at_zero is None else fun_at_zero
    evals = list(map(fun, shifts_neg)) + [fun_at_zero] + list(map(fun, shifts_pos))
    shifts = qml.math.concatenate([shifts_neg, [0.0], shifts_pos])

    def _reconstruction(x):
        """Univariate reconstruction based on equidistant shifts and Dirichlet kernels."""
        return qml.math.tensordot(
            qml.math.sinc(a * (x - shifts)) / qml.math.sinc(b * (x - shifts)),
            evals,
            axes=[[0], [0]],
        )

    return _reconstruction


def _reconstruct_gen(fun, spectrum, shifts=None, fun_at_zero=None):
    r"""Reconstruct a univariate (real-valued) Fourier series with given spectrum.
    Args:
        fun (callable): Fourier series to reconstruct with signature ``float -> float``
        spectrum (Sequence): Frequency spectrum of the Fourier series; non-positive
            frequencies are ignored
        shifts (list): Shift angles at which to evaluate ``fun`` for the reconstruction
            Chosen equidistantly within the interval :math:`[0, 2\pi/f_\text{min}]` if ``shift=None``
            where :math:`f_\text{min}` is the smallest frequency in ``spectrum``.
        fun_at_zero (float): Value of ``fun`` at zero. If :math:`0` is among the ``shifts``
            and ``fun_at_zero`` is provided, one evaluation of ``fun`` is saved.

    Returns:
        callable: Reconstructed Fourier series with :math:`R` frequencies in ``spectrum``,
        as ``qml.numpy`` based function and coinciding with ``fun`` on :math:`2R+1` points.
    """
    if spectrum in ([], [0.0]):
        if fun_at_zero is None:
            fun_at_zero = fun(0.0)
        def _reconstruction(x):
            """Univariate reconstruction of a constant Fourier series."""
            return fun_at_zero
        return _reconstruction

    spectrum = np.array([f for f in spectrum if f > 0.0])
    f_max = max(spectrum)
    # If no shifts are provided, choose equidistant ones
    if shifts is None:
        R = len(spectrum)
        shifts = np.arange(-R, R + 1) * 2 * np.pi / (f_max * (2 * R + 1))*R
        zero_idx = R
    elif fun_at_zero is not None:
        zero_idx = np.where(np.isclose(shifts, 0.0))[0]
        if len(zero_idx) > 0:
            zero_idx = int(np.where(zero_idx))
        else:
            zero_idx = None
    # Take care of shifts close to zero if fun_at_zero was provided
    if fun_at_zero is not None and zero_idx is not None:
        # Only one shift may be zero at a time
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
    print(np.linalg.cond(C))
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


def _prepare_jobs(ids, spectra, shifts, nums_frequency):
    r"""For inputs to reconstruct, determine how the given information yields
    function reconstruction tasks and collect them into an ArgMap ``jobs``.
    Also determine whether the function at zero is needed."""
    if nums_frequency is None:
        jobs = {}
        if spectra is None:
            raise ValueError("Either nums_frequency or spectra must be given.")
        if ids is None:
            ids = {outer_key: inner_dict.keys() for outer_key, inner_dict in spectra.items()}
        if shifts is None:
            shifts = {}
        need_fun_at_zero = False
        recon_fn = _reconstruct_gen
        # If no shifts are provided, compute them
        for arg_name, inner_dict in ids.items():
            _jobs = {}
            for par_idx in inner_dict:
                _shifts = shifts.get(arg_name)
                if _shifts is not None:
                    _shifts = _shifts.get(par_idx)
                _spectrum = spectra[arg_name][par_idx]
                # Determine whether fun_at_zero is needed
                if _shifts is None or any(np.isclose(_shifts, 0.0, rtol=0, atol=atol)):
                    need_fun_at_zero = True
                # Store job; fun_at_zero missing
                if len(_spectrum) > 1:
                    _jobs[par_idx] = {"shifts": _shifts, "spectrum": _spectrum}
                else:
                    _jobs[par_idx] = None
            jobs[arg_name] = _jobs
    else:
        jobs = {}
        need_fun_at_zero = True
        if ids is None:
            # ids = nums_frequency.keys()
            ids = {outer_key: inner_dict.keys() for outer_key, inner_dict in nums_frequency.items()}
        recon_fn = _reconstruct_equ
        for arg_name, inner_dict in ids.items():
            _jobs = {}
            for par_idx in inner_dict:
                _num_frequency = nums_frequency[arg_name][par_idx]
                # Store job; fun_at_zero missing
                jobs[idx] = {"num_frequency": _num_frequency} if _num_frequency > 0 else None
            jobs[arg_name] = _jobs

    return ids, recon_fn, jobs, need_fun_at_zero


def reconstruct(qnode, ids=None, nums_frequency=None, spectra=None, shifts=None):
    r"""Reconstruct univariate restrictions of an expectation value QNode.

    Args:
        qnode (pennylane.QNode): Quantum node to be reconstructed, representing a
            circuit that outputs an expectation value.
        ids (list): Indices for the QNode parameters with respect to which
            the QNode should be reconstructed as a univariate function.
            Each entry of the list should be a valid key for ``nums_frequency`` if
            it is provided or ``spectra``. Alternatively, a single valid key may be provided.
            If ``None``, all keys of ``nums_frequency``/``spectra`` are considered.
        nums_frequency (pennylane.ArgMap[int]): Numbers of integer frequencies -- and biggest
            frequency -- per QNode parameter. If the frequencies are not contiguous integers,
            the argument ``spectra`` should be used. Takes precedence over ``spectra`` and
            uses equidistant shifts.
        spectra (pennylane.ArgMap[list[float]]): Frequency spectra per QNode parameter.
            Ignored if ``nums_frequency!=None``.
        shifts (pennylane.ArgMap[list[float]]): Shift angles for computing the reconstruction per
            QNode parameter. Ignored if ``nums_frequency!=None``.
    Returns:
        function: Function which accepts the same arguments as the QNode.
            When called, this function will return a :class:`~.pennylane.argmap.ArgMap`
            that contains the univariate reconstructions per QNode parameter.

    For each provided ``id`` in ``ids``, the QNode is restricted to varying the single QNode
    parameter corresponding to the ``id``. This univariate function is then restricted
    via a Fourier transform or Dirichlet kernels depending on the provided input.
    Either the ``spectra`` of the QNode with respect to its input parameter or the numbers
    of frequencies, ``nums_frequency``, per parameter must be provided.
    """

    ids, recon_fn, jobs, need_fun_at_zero = _prepare_jobs(ids, spectra, shifts, nums_frequency)

    atol = 1e-8

    arg_names = list(signature(qnode).parameters.keys())

    @wraps(qnode)
    def wrapper(*args, **kwargs):
        nonlocal spectra
        if need_fun_at_zero:
            fun_at_zero = qnode(*args, **kwargs)

        def constant_fn(x):
            return fun_at_zero

        # Carry out the reconstruction jobs
        # reconstructions = ArgMap(like=jobs)
        reconstructions = {}
        for arg_name, inner_dict in jobs.items():
            _reconstructions = {}
            arg_idx = arg_names.index(arg_name)
            for par_idx, job in inner_dict.items():
                # arg_idx = idx[0] if not (jobs.single_arg or jobs.single_object) else 0
                # param_idx = idx[1] if not jobs.single_object else 0
                if job is None:
                    _reconstructions[par_idx] = constant_fn
                else:
                    shift_vec = np.zeros_like(args[arg_idx])
                    shift_vec[par_idx] = 1.0

                    def _univariate_fn(x):
                        new_arg = args[arg_idx] + shift_vec * x
                        new_args = args[:arg_idx] + (new_arg,) + args[arg_idx + 1 :]
                        return qnode(*new_args, **kwargs)

                    _reconstructions[par_idx] = recon_fn(_univariate_fn, **job)
            reconstructions[arg_name] = _reconstructions

        return reconstructions

    return wrapper

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
"""
This module contains functions for computing the finite-difference gradient
of a quantum tape.
"""
# pylint: disable=protected-access,too-many-arguments
import numpy as np
from scipy.special import factorial

import pennylane as qml


def finite_diff_coeffs(n, approx, strategy):
    r"""Generate the finite difference shift values and corresponding
    term coefficients for a given derivative order, approximation accuracy,
    and strategy.

    Args:
        n (int): Positive integer specifying the order of the derivative. For example, ``n=1``
            corresponds to the first derivative, ``n=2`` the second derivative, etc.
        approx (int): Positive integer referring to the approximation order of the
            returned coefficients, e.g., ``approx=1`` corresponds to the
            first-order approximation to the derivative.
        strategy (str): One of ``"forward"``, ``"center"``, or ``"backward"``.
            For the ``"forward"`` strategy, the finite-difference shifts occur at the points
            :math:`x_0, x_0+h, x_0+2h,\dots`, where :math:`h` is some small
            step size. The ``"backwards"`` strategy is similar, but in
            reverse: :math:`x_0, x_0-h, x_0-2h, \dots`. Finally, the
            ``"center"`` strategy results in shifts symmetric around the
            unshifted point: :math:`\dots, x_0-2h, x_0-h, x_0, x_0+h, x_0+2h,\dots`.

    Returns:
        array[float]: A ``(2, N)`` array. The first row corresponds to the
        coefficients, and the second row corresponds to the shifts.

    **Example**

    >>> finite_diff_coeffs(n=1, approx=1, strategy="forward")
    array([[-1.,  1.],
           [ 0.,  1.]])

    The first row corresponds to the coefficients, and the second corresponds
    to the shifts. For example, this results in the linear combination:

    .. math:: \frac{-y(x_0) + y(x_0 + h)}{h}

    where :math:`h` is the finite-difference step size.

    More examples:

    >>> finite_diff_coeffs(n=1, approx=2, strategy="center")
    array([[-0.5,  0.5],
           [-1. ,  1. ]])
    >>> finite_diff_coeffs(n=2, approx=2, strategy="center")
    array([[-2.,  1.,  1.],
           [ 0., -1.,  1.]])
    """
    if n < 1 or not isinstance(n, int):
        raise ValueError("Derivative order n must be a positive integer.")

    if approx < 1 or not isinstance(approx, int):
        raise ValueError("Approximation order must be a positive integer.")

    num_points = approx + 2 * np.floor((n + 1) / 2) - 1
    N = num_points + 1 if n % 2 == 0 else num_points

    if strategy == "forward":
        shifts = np.arange(N, dtype=np.float64)

    elif strategy == "backward":
        shifts = np.arange(-N + 1, 1, dtype=np.float64)

    elif strategy == "center":
        if approx % 2 != 0:
            raise ValueError("Centered finite-difference requires an even order approximation.")

        N = num_points // 2
        shifts = np.arange(-N, N + 1, dtype=np.float64)

    else:
        raise ValueError(
            f"Unknown strategy {strategy}. Must be one of 'forward', 'backward', 'center'."
        )

    # solve for the coefficients
    A = shifts ** np.arange(len(shifts)).reshape(-1, 1)
    b = np.zeros_like(shifts)
    b[n] = factorial(n)
    coeffs = np.linalg.solve(A, b)

    coeffs_and_shifts = np.stack([coeffs, shifts])

    # remove all small coefficients and shifts
    coeffs_and_shifts[np.abs(coeffs_and_shifts) < 1e-10] = 0

    # remove columns where the coefficients are 0
    coeffs_and_shifts = coeffs_and_shifts[:, ~np.all(coeffs_and_shifts == 0, axis=0)]

    # sort columns in ascending order according to abs(shift)
    coeffs_and_shifts = coeffs_and_shifts[:, np.argsort(np.abs(coeffs_and_shifts)[1])]
    return coeffs_and_shifts


def generate_shifted_tapes(tape, idx, shifts, multipliers=None):
    r"""Generate a list of tapes where the corresponding trainable parameter
    index has been shifted by the values given.

    Args:
        tape (.QuantumTape): input quantum tape
        idx (int): trainable parameter index to shift the parameter of
        shifts (Sequence[float or int]): sequence of shift values
        multipliers (Sequence[float or int]): Sequence of multiplier values to
            scale the parameter by. If not provided, the parameter will
            not be scaled.

    Returns:
        list[QuantumTape]: List of quantum tapes. Each tape has parameter
        ``idx`` shifted by consecutive values of ``shift``. The length
        of the returned list of tapes will match the length of ``shifts``.
    """
    params = tape.get_parameters()
    tapes = []

    for i, s in enumerate(shifts):
        new_params = params.copy()
        shifted_tape = tape.copy(copy_operations=True)

        if multipliers is not None:
            m = multipliers[i]
            new_params[idx] = new_params[idx] * qml.math.convert_like(m, new_params[idx])

        new_params[idx] = new_params[idx] + qml.math.convert_like(s, new_params[idx])
        shifted_tape.set_parameters(new_params)
        tapes.append(shifted_tape)

    return tapes


def finite_diff(tape, argnum=None, h=1e-7, approx=1, n=1, strategy="forward", f0=None):
    r"""Generate the finite-difference tapes and postprocessing methods required
    to compute the gradient of a gate parameter with respect to its outputs.

    Args:
        tape (.QuantumTape): quantum tape to differentiate
        argnum (int or list[int] or None): Trainable parameter indices to differentiate
            with respect to. If not provided, the derivatives with respect to all
            trainable indices are returned.
        h (float): finite difference method step size
        approx (int): The approximation order of the finite-difference method to use.
        n (int): compute the :math:`n`-th derivative
        strategy (str): The strategy of the finite difference method. Must be one of
            ``"forward"``, ``"center"``, or ``"backward"``.
            For the ``"forward"`` strategy, the finite-difference shifts occur at the points
            :math:`x_0, x_0+h, x_0+2h,\dots`, where :math:`h` is some small
            stepsize. The ``"backwards"`` strategy is similar, but in
            reverse: :math:`x_0, x_0-h, x_0-2h, \dots`. Finally, the
            ``"center"`` strategy results in shifts symmetric around the
            unshifted point: :math:`\dots, x_0-2h, x_0-h, x_0, x_0+h, x_0+2h,\dots`.
        f0 (tensor_like[float] or None): Output of the evaluated input tape. If provided,
            and the gradient recipe contains an unshifted term, this value is used,
            saving a quantum evaluation.

    Returns:
        tuple[list[QuantumTape], function]: A tuple containing a
        list of generated tapes, in addition to a post-processing
        function to be applied to the evaluated tapes.

    **Example**

    >>> with qml.tape.QuantumTape() as tape:
    ...     qml.RX(params[0], wires=0)
    ...     qml.RY(params[1], wires=0)
    ...     qml.RX(params[2], wires=0)
    ...     qml.expval(qml.PauliZ(0))
    ...     qml.var(qml.PauliZ(0))
    >>> tape.trainable_params = {0, 1, 2}
    >>> gradient_tapes, fn = qml.gradients.finite_diff.grad(tape)
    >>> res = dev.batch_execute(gradient_tapes)
    >>> fn(res)
    [[-0.38751721 -0.18884787 -0.38355704]
     [ 0.69916862  0.34072424  0.69202359]]

    The output Jacobian matrix is of size ``(number_outputs, number_parameters)``.
    """
    # TODO: replace the JacobianTape._grad_method_validation
    # functionality before deprecation.
    diff_methods = tape._grad_method_validation("numeric")

    if not tape.trainable_params or all(g == "0" for g in diff_methods):
        # Either all parameters have grad method 0, or there are no trainable
        # parameters.
        return [], lambda x: np.zeros([tape.output_dim, len(tape.trainable_params)])

    gradient_tapes = []
    shapes = []
    c0 = None

    coeffs, shifts = finite_diff_coeffs(n=n, approx=approx, strategy=strategy)

    if 0 in shifts:
        # Stencil includes a term with zero shift.

        if f0 is None:
            # Ensure that the unshifted tape is appended
            # to the gradient tapes, if not already.
            gradient_tapes.append(tape)

        # Store the unshifted coefficient. We know that
        # it will always be the first coefficient due to processing.
        c0 = coeffs[0]
        shifts = shifts[1:]
        coeffs = coeffs[1:]

    # TODO: replace the JacobianTape._choose_params_with_methods
    # functionality before deprecation.
    method_map = dict(tape._choose_params_with_methods(diff_methods, argnum))

    for i, _ in enumerate(tape.trainable_params):
        if i not in method_map or method_map[i] == "0":
            # parameter has zero gradient
            shapes.append(0)
            continue

        g_tapes = generate_shifted_tapes(tape, i, shifts * h)
        gradient_tapes.extend(g_tapes)
        shapes.append(len(g_tapes))

    def processing_fn(results):
        grads = []
        start = 1 if c0 is not None and f0 is None else 0
        r0 = f0 or results[0]

        for s in shapes:

            if s == 0:
                # parameter has zero gradient
                g = qml.math.zeros_like(results[0])
                grads.append(g)
                continue

            res = results[start : start + s]
            start = start + s

            # compute the linear combination of results and coefficients
            res = qml.math.stack(res)
            g = sum([c * r for c, r in zip(coeffs, res)])

            if c0 is not None:
                # add on the unshifted term
                g = g + c0 * r0

            grads.append(g / (h ** n))

        # The following is for backwards compatibility; currently,
        # the device stacks multiple measurement arrays, even if not the same
        # size, resulting in a ragged array.
        # In the future, we might want to change this so that only tuples
        # of arrays are returned.
        for i, g in enumerate(grads):
            g = qml.math.convert_like(g, res[0])
            if hasattr(g, "dtype") and g.dtype is np.dtype("object"):
                grads[i] = qml.math.hstack(g)

        return qml.math.T(qml.math.stack(grads))

    return gradient_tapes, processing_fn

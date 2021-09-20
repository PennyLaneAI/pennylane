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
import functools

import numpy as np
from scipy.special import factorial

import pennylane as qml

from .gradient_transform import gradient_transform


@functools.lru_cache(maxsize=None)
def finite_diff_coeffs(n, approx_order, strategy):
    r"""Generate the finite difference shift values and corresponding
    term coefficients for a given derivative order, approximation accuracy,
    and strategy.

    Args:
        n (int): Positive integer specifying the order of the derivative. For example, ``n=1``
            corresponds to the first derivative, ``n=2`` the second derivative, etc.
        approx_order (int): Positive integer referring to the approximation order of the
            returned coefficients, e.g., ``approx_order=1`` corresponds to the
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

    >>> finite_diff_coeffs(n=1, approx_order=1, strategy="forward")
    array([[-1.,  1.],
           [ 0.,  1.]])

    For example, this results in the linear combination:

    .. math:: \frac{-y(x_0) + y(x_0 + h)}{h}

    where :math:`h` is the finite-difference step size.

    More examples:

    >>> finite_diff_coeffs(n=1, approx_order=2, strategy="center")
    array([[-0.5,  0.5],
           [-1. ,  1. ]])
    >>> finite_diff_coeffs(n=2, approx_order=2, strategy="center")
    array([[-2.,  1.,  1.],
           [ 0., -1.,  1.]])

    **Details**

    Consider a function :math:`y(x)`. We wish to approximate the :math:`n`-th
    derivative at point :math:`x_0`, :math:`y^{(n)}(x_0)`, by sampling the function
    at :math:`N<n` distinct points:

    .. math:: y^{(n)}(x_0) \approx \sum_{i=1}^N c_i y(x_i)

    where :math:`c_i` are coefficients, and :math:`x_i=x_0 + s_i` are the points we sample
    the function at.

    Consider the Taylor expansion of :math:`y(x_i)` around the point :math:`x_0`:

    .. math::

        y^{(n)}(x_0) \approx \sum_{i=1}^N c_i y(x_i)
            &= \sum_{i=1}^N c_i \left[ y(x_0) + y'(x_0)(x_i-x_0) + \frac{1}{2} y''(x_0)(x_i-x_0)^2 + \cdots \right]\\
            & = \sum_{j=0}^m y^{(j)}(x_0) \left[\sum_{i=1}^N \frac{c_i s_i^j}{j!} + \mathcal{O}(s_i^m) \right],

    where :math:`s_i = x_i-x_0`. For this approximation to be satisfied, we must therefore have

    .. math::

        \sum_{i=1}^N s_i^j c_i = \begin{cases} j!, &j=n\\ 0, & j\neq n\end{cases}.

    Thus, to determine the coefficients :math:`c_i \in \{c_1, \dots, c_N\}` for particular
    shift values :math:`s_i \in \{s_1, \dots, s_N\}` and derivative order :math:`n`,
    we must solve this linear system of equations.
    """
    if n < 1 or not isinstance(n, int):
        raise ValueError("Derivative order n must be a positive integer.")

    if approx_order < 1 or not isinstance(approx_order, int):
        raise ValueError("Approximation order must be a positive integer.")

    num_points = approx_order + 2 * np.floor((n + 1) / 2) - 1
    N = num_points + 1 if n % 2 == 0 else num_points

    if strategy == "forward":
        shifts = np.arange(N, dtype=np.float64)

    elif strategy == "backward":
        shifts = np.arange(-N + 1, 1, dtype=np.float64)

    elif strategy == "center":
        if approx_order % 2 != 0:
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
    params = list(tape.get_parameters())
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


@gradient_transform
def finite_diff(tape, argnum=None, h=1e-7, approx_order=1, n=1, strategy="forward", f0=None):
    r"""Transform a QNode to compute the finite-difference gradient of all gate
    parameters with respect to its inputs.

    Args:
        qnode (.QNode or .QuantumTape): quantum tape or QNode to differentiate
        argnum (int or list[int] or None): Trainable parameter indices to differentiate
            with respect to. If not provided, the derivatives with respect to all
            trainable parameters are returned.
        h (float): finite difference method step size
        approx_order (int): The approximation order of the finite-difference method to use.
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
        tensor_like or tuple[list[QuantumTape], function]:

        - If the input is a QNode, a tensor
          representing the output Jacobian matrix of size ``(number_outputs, number_gate_parameters)``
          is returned.

        - If the input is a tape, a tuple containing a list of generated tapes,
          in addition to a post-processing function to be applied to the
          evaluated tapes.

    **Example**

    This transform can be registered directly as the quantum gradient transform
    to use during autodifferentiation:

    >>> dev = qml.device("default.qubit", wires=2)
    >>> @qml.qnode(dev, gradient_fn=qml.gradients.finite_diff)
    ... def circuit(params):
    ...     qml.RX(params[0], wires=0)
    ...     qml.RY(params[1], wires=0)
    ...     qml.RX(params[2], wires=0)
    ...     return qml.expval(qml.PauliZ(0)), qml.var(qml.PauliZ(0))
    >>> params = np.array([0.1, 0.2, 0.3], requires_grad=True)
    >>> qml.jacobian(circuit)(params)
    tensor([[-0.38751725, -0.18884792, -0.38355708],
            [ 0.69916868,  0.34072432,  0.69202365]], requires_grad=True)


    .. UsageDetails::

        This gradient transform can also be applied directly to :class:`~.QNode` objects:

        >>> @qml.qnode(dev)
        ... def circuit(params):
        ...     qml.RX(params[0], wires=0)
        ...     qml.RY(params[1], wires=0)
        ...     qml.RX(params[2], wires=0)
        ...     return qml.expval(qml.PauliZ(0)), qml.var(qml.PauliZ(0))
        >>> qml.gradients.finite_diff(circuit)(params)
        tensor([[-0.38751725, -0.18884792, -0.38355708],
                [ 0.69916868,  0.34072432,  0.69202365]], requires_grad=True)

        This quantum gradient transform can also be applied to low-level
        :class:`~.QuantumTape` objects. This will result in no implicit quantum
        device evaluation. Instead, the processed tapes, and post-processing
        function, which together define the gradient are directly returned:

        >>> with qml.tape.JacobianTape() as tape:
        ...     qml.RX(params[0], wires=0)
        ...     qml.RY(params[1], wires=0)
        ...     qml.RX(params[2], wires=0)
        ...     qml.expval(qml.PauliZ(0))
        ...     qml.var(qml.PauliZ(0))
        >>> gradient_tapes, fn = qml.gradients.finite_diff(tape)
        >>> gradient_tapes
        [<JacobianTape: wires=[0, 1], params=3>,
         <JacobianTape: wires=[0, 1], params=3>,
         <JacobianTape: wires=[0, 1], params=3>,
         <JacobianTape: wires=[0, 1], params=3>]

        This can be useful if the underlying circuits representing the gradient
        computation need to be analyzed.

        The output tapes can then be evaluated and post-processed to retrieve
        the gradient:

        >>> dev = qml.device("default.qubit", wires=2)
        >>> fn(qml.execute(gradient_tapes, dev, None))
        [[-0.38751721 -0.18884787 -0.38355704]
         [ 0.69916862  0.34072424  0.69202359]]
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

    coeffs, shifts = finite_diff_coeffs(n=n, approx_order=approx_order, strategy=strategy)

    if 0 in shifts:
        # Finite difference formula includes a term with zero shift.

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
            g = qml.math.convert_like(g, results[0])
            if hasattr(g, "dtype") and g.dtype is np.dtype("object"):
                grads[i] = qml.math.hstack(g)

        return qml.math.T(qml.math.stack(grads))

    return gradient_tapes, processing_fn

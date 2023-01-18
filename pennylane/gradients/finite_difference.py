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
# pylint: disable=protected-access,too-many-arguments,too-many-branches,too-many-statements
import functools
import warnings
from collections.abc import Sequence

import numpy as np
from scipy.special import factorial

import pennylane as qml
from pennylane._device import _get_num_copies
from pennylane.measurements import ProbabilityMP

from .general_shift_rules import generate_shifted_tapes
from .gradient_transform import (
    choose_grad_methods,
    grad_method_validation,
    gradient_analysis,
    gradient_transform,
)


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


def _no_trainable_grad_new(tape, shots=None):
    warnings.warn(
        "Attempted to compute the gradient of a tape with no trainable parameters. "
        "If this is unintended, please mark trainable parameters in accordance with the "
        "chosen auto differentiation framework, or via the 'tape.trainable_params' property."
    )
    if isinstance(shots, Sequence):
        len_shot_vec = _get_num_copies(shots)
        if len(tape.measurements) == 1:
            return [], lambda _: tuple(qml.math.zeros([0]) for _ in range(len_shot_vec))
        return [], lambda _: tuple(
            tuple(qml.math.zeros([0]) for _ in range(len(tape.measurements)))
            for _ in range(len_shot_vec)
        )

    if len(tape.measurements) == 1:
        return [], lambda _: qml.math.zeros([0])
    return [], lambda _: tuple(qml.math.zeros([0]) for _ in range(len(tape.measurements)))


def _all_zero_grad_new(tape, shots=None):
    """Auxiliary function to return zeros for the all-zero gradient case."""
    list_zeros = []

    for m in tape.measurements:
        # TODO: Update shape for CV variables
        if isinstance(m, ProbabilityMP):
            shape = 2 ** len(m.wires)
        else:
            shape = ()

        if len(tape.trainable_params) == 1:
            sub_list_zeros = qml.math.zeros(shape)
        else:
            sub_list_zeros = [qml.math.zeros(shape) for _ in range(len(tape.trainable_params))]
            sub_list_zeros = tuple(sub_list_zeros)

        list_zeros.append(sub_list_zeros)

    if isinstance(shots, Sequence):
        len_shot_vec = _get_num_copies(shots)
        if len(tape.measurements) == 1:
            return [], lambda _: tuple(list_zeros[0] for _ in range(len_shot_vec))
        return [], lambda _: tuple(tuple(list_zeros) for _ in range(len_shot_vec))

    if len(tape.measurements) == 1:
        return [], lambda _: list_zeros[0]

    return [], lambda _: tuple(list_zeros)


@gradient_transform
def _finite_diff_new(
    tape,
    argnum=None,
    h=1e-7,
    approx_order=1,
    n=1,
    strategy="forward",
    f0=None,
    validate_params=True,
    shots=None,
):
    r"""Transform a QNode to compute the finite-difference gradient of all gate
    parameters with respect to its inputs. This function is adapted to the new return system.

    Args:
        tape (pennylane.QNode or .QuantumTape): quantum tape or QNode to differentiate
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
        validate_params (bool): Whether to validate the tape parameters or not. If ``True``,
            the ``Operation.grad_method`` attribute and the circuit structure will be analyzed
            to determine if the trainable parameters support the finite-difference method.
            If ``False``, the finite-difference method will be applied to all parameters.
        shots (None, int, list[int], list[ShotTuple]): The device shots that will be used to execute the tapes outputted by this
            transform. Note that this argument doesn't influence the shots used for tape execution, but provides information
            to the transform about the device shots and helps in determining if a shot sequence was used to define the
            device shots for the new return types output system.

    Returns:
        function or tuple[list[QuantumTape], function]:

        - If the input is a QNode, an object representing the Jacobian (function) of the QNode
          that can be executed to obtain the Jacobian matrix.
          The type of the matrix returned is either a tensor, a tuple or a
          nested tuple depending on the nesting structure of the original QNode output.

        - If the input is a tape, a tuple containing a
          list of generated tapes, together with a post-processing
          function to be applied to the results of the evaluated tapes
          in order to obtain the Jacobian matrix.

    **Example**

    This transform can be registered directly as the quantum gradient transform
    to use during autodifferentiation:

    >>> dev = qml.device("default.qubit", wires=2)
    >>> @qml.qnode(dev, interface="autograd", diff_method="finite-diff")
    ... def circuit(params):
    ...     qml.RX(params[0], wires=0)
    ...     qml.RY(params[1], wires=0)
    ...     qml.RX(params[2], wires=0)
    ...     return qml.expval(qml.PauliZ(0))
    >>> params = np.array([0.1, 0.2, 0.3], requires_grad=True)
    >>> qml.jacobian(circuit)(params)
    array([-0.38751725, -0.18884792, -0.38355708])

    When differentiating QNodes with multiple measurements using Autograd or TensorFlow, the outputs of the QNode first
    need to be stacked. The reason is that those two frameworks only allow differentiating functions with array or
    tensor outputs, instead of functions that output sequences. In contrast, Jax and Torch require no additional
    post-processing.

    >>> import jax
    >>> dev = qml.device("default.qubit", wires=2)
    >>> @qml.qnode(dev, interface="jax", diff_method="finite-diff")
    ... def circuit(params):
    ...     qml.RX(params[0], wires=0)
    ...     qml.RY(params[1], wires=0)
    ...     qml.RX(params[2], wires=0)
    ...     return qml.expval(qml.PauliZ(0)), qml.var(qml.PauliZ(0))
    >>> params = jax.numpy.array([0.1, 0.2, 0.3])
    >>> jax.jacobian(circuit)(params)
    (Array([-0.38751727, -0.18884793, -0.3835571 ], dtype=float32),
    Array([0.6991687 , 0.34072432, 0.6920237 ], dtype=float32))


    .. details::
        :title: Usage Details

        This gradient transform can also be applied directly to :class:`QNode <pennylane.QNode>` objects:

        >>> @qml.qnode(dev)
        ... def circuit(params):
        ...     qml.RX(params[0], wires=0)
        ...     qml.RY(params[1], wires=0)
        ...     qml.RX(params[2], wires=0)
        ...     return qml.expval(qml.PauliZ(0)), qml.var(qml.PauliZ(0))
        >>> params = np.array([0.1, 0.2, 0.3], requires_grad=True)
        >>> qml.gradients.finite_diff(circuit)(params)
        ((tensor(-0.38751724, requires_grad=True),
          tensor(-0.18884792, requires_grad=True),
          tensor(-0.38355709, requires_grad=True)),
         (tensor(0.69916868, requires_grad=True),
          tensor(0.34072432, requires_grad=True),
          tensor(0.69202366, requires_grad=True)))

        This quantum gradient transform can also be applied to low-level
        :class:`~.QuantumTape` objects. This will result in no implicit quantum
        device evaluation. Instead, the processed tapes, and post-processing
        function, which together define the gradient are directly returned:

        >>> with qml.tape.QuantumTape() as tape:
        ...     qml.RX(params[0], wires=0)
        ...     qml.RY(params[1], wires=0)
        ...     qml.RX(params[2], wires=0)
        ...     qml.expval(qml.PauliZ(0))
        ...     qml.var(qml.PauliZ(0))
        >>> gradient_tapes, fn = qml.gradients.finite_diff(tape)
        >>> gradient_tapes
        [<QuantumTape: wires=[0], params=3>,
         <QuantumTape: wires=[0], params=3>,
         <QuantumTape: wires=[0], params=3>,
         <QuantumTape: wires=[0], params=3>]

        This can be useful if the underlying circuits representing the gradient
        computation need to be analyzed.

        The output tapes can then be evaluated and post-processed to retrieve
        the gradient:

        >>> dev = qml.device("default.qubit", wires=2)
        >>> fn(qml.execute(gradient_tapes, dev, None))
        ((array(-0.38751724), array(-0.18884792), array(-0.38355709)),
         (array(0.69916868), array(0.34072432), array(0.69202366)))

        Devices that have a shot vector defined can also be used for execution, provided
        the ``shots`` argument was passed to the transform:

        >>> shots = (10, 100, 1000)
        >>> dev = qml.device("default.qubit", wires=2, shots=shots)
        >>> @qml.qnode(dev)
        ... def circuit(params):
        ...     qml.RX(params[0], wires=0)
        ...     qml.RY(params[1], wires=0)
        ...     qml.RX(params[2], wires=0)
        ...     return qml.expval(qml.PauliZ(0)), qml.var(qml.PauliZ(0))
        >>> params = np.array([0.1, 0.2, 0.3], requires_grad=True)
        >>> qml.gradients.finite_diff(circuit, shots=shots, h=10e-2)(params)
        (((array(-2.), array(-2.), array(0.)), (array(3.6), array(3.6), array(0.))),
         ((array(1.), array(0.4), array(1.)),
          (array(-1.62), array(-0.624), array(-1.62))),
         ((array(-0.48), array(-0.34), array(-0.46)),
          (array(0.84288), array(0.6018), array(0.80868))))

        The outermost tuple contains results corresponding to each element of the shot vector.
    """
    if argnum is None and not tape.trainable_params:
        return _no_trainable_grad_new(tape, shots)

    if validate_params:
        if "grad_method" not in tape._par_info[0]:
            gradient_analysis(tape, grad_fn=_finite_diff_new)
        diff_methods = grad_method_validation("numeric", tape)
    else:
        diff_methods = ["F" for i in tape.trainable_params]

    if all(g == "0" for g in diff_methods):
        return _all_zero_grad_new(tape, shots)

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

    method_map = choose_grad_methods(diff_methods, argnum)

    for i, _ in enumerate(tape.trainable_params):
        if i not in method_map or method_map[i] == "0":
            # parameter has zero gradient
            shapes.append(0)
            continue

        g_tapes = generate_shifted_tapes(tape, i, shifts * h)
        gradient_tapes.extend(g_tapes)
        shapes.append(len(g_tapes))

    def _single_shot_batch_result(results):
        """Auxiliary function for post-processing one batch of results corresponding to finite shots or a single
        component of a shot vector"""

        grads = []
        start = 1 if c0 is not None and f0 is None else 0
        r0 = f0 or results[0]

        output_dims = []
        # TODO: Update shape for CV variables
        for m in tape.measurements:
            if isinstance(m, ProbabilityMP):
                output_dims.append(2 ** len(m.wires))
            else:
                output_dims.append(1)

        for s in shapes:

            if s == 0:
                # parameter has zero gradient
                if not isinstance(results[0], tuple):
                    g = qml.math.zeros_like(results[0])
                else:
                    g = []
                    for i in output_dims:
                        zero = qml.math.squeeze(qml.math.zeros(i))
                        g.append(zero)

                grads.append(g)
                continue

            res = results[start : start + s]
            start = start + s

            # compute the linear combination of results
            # and coefficients

            pre_grads = []

            if len(tape.measurements) == 1:
                res = qml.math.stack(res)
                c = qml.math.convert_like(coeffs, res)
                lin_comb = qml.math.tensordot(res, c, [[0], [0]])
                pre_grads.append(lin_comb)
            else:
                for i in range(len(tape.measurements)):
                    r = qml.math.stack([r[i] for r in res])
                    c = qml.math.convert_like(coeffs, r)
                    lin_comb = qml.math.tensordot(r, c, [[0], [0]])
                    pre_grads.append(lin_comb)

            # Add on the unshifted term
            if c0 is not None:
                if len(tape.measurements) == 1:
                    c = qml.math.convert_like(c0, r0)
                    pre_grads = [pre_grads[0] + r0 * c]
                else:
                    for i in range(len(tape.measurements)):
                        r_i = r0[i]
                        c = qml.math.convert_like(c0, r_i)
                        pre_grads[i] = pre_grads[i] + r_i * c

            coeff_div = qml.math.cast_like(
                qml.math.convert_like(1 / h**n, pre_grads[0]), pre_grads[0]
            )

            if len(tape.measurements) > 1:
                pre_grads = tuple(
                    qml.math.convert_like(i * coeff_div, coeff_div) for i in pre_grads
                )
            else:
                pre_grads = qml.math.convert_like(pre_grads[0] * coeff_div, coeff_div)

            grads.append(pre_grads)
        # Single measurement
        if len(tape.measurements) == 1:
            if len(tape.trainable_params) == 1:
                return grads[0]
            return tuple(grads)

        # Reordering to match the right shape for multiple measurements
        grads_reorder = [[0] * len(tape.trainable_params) for _ in range(len(tape.measurements))]
        for i in range(len(tape.measurements)):
            for j in range(len(tape.trainable_params)):
                grads_reorder[i][j] = grads[j][i]

        # To tuple
        if len(tape.trainable_params) == 1:
            grads_tuple = tuple(elem[0] for elem in grads_reorder)
        else:
            grads_tuple = tuple(tuple(elem) for elem in grads_reorder)
        return grads_tuple

    def processing_fn(results):
        shot_vector = isinstance(shots, Sequence)

        if not shot_vector:
            grads_tuple = _single_shot_batch_result(results)
        else:
            grads_tuple = []
            len_shot_vec = _get_num_copies(shots)
            for idx in range(len_shot_vec):
                res = [tape_res[idx] for tape_res in results]
                g_tuple = _single_shot_batch_result(res)
                grads_tuple.append(g_tuple)
            grads_tuple = tuple(grads_tuple)

        return grads_tuple

    return gradient_tapes, processing_fn


@gradient_transform
def finite_diff(
    tape,
    argnum=None,
    h=1e-7,
    approx_order=1,
    n=1,
    strategy="forward",
    f0=None,
    validate_params=True,
    shots=None,
):
    r"""Transform a QNode to compute the finite-difference gradient of all gate
    parameters with respect to its inputs.

    Args:
        qnode (pennylane.QNode or .QuantumTape): quantum tape or QNode to differentiate
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
        validate_params (bool): Whether to validate the tape parameters or not. If ``True``,
            the ``Operation.grad_method`` attribute and the circuit structure will be analyzed
            to determine if the trainable parameters support the finite-difference method.
            If ``False``, the finite-difference method will be applied to all parameters.
        shots (None, int, list[int]): The device shots that will be used to execute the tapes outputted by this
            transform. Note that this argument doesn't influence the shots used for tape execution, but provides information
            to the transform about the device shots and helps in determining if a shot sequence was used to define the
            device shots for the new return types output system.

    Returns:
        function or tuple[list[QuantumTape], function]:

        - If the input is a QNode, an object representing the Jacobian (function) of the QNode
          that can be executed to obtain the Jacobian matrix.
          The returned matrix is a tensor of size ``(number_outputs, number_gate_parameters)``

        - If the input is a tape, a tuple containing a
          list of generated tapes, together with a post-processing
          function to be applied to the results of the evaluated tapes
          in order to obtain the Jacobian matrix.

    **Example**

    This transform can be registered directly as the quantum gradient transform
    to use during autodifferentiation:

    >>> dev = qml.device("default.qubit", wires=2)
    >>> @qml.qnode(dev, diff_method=qml.gradients.finite_diff)
    ... def circuit(params):
    ...     qml.RX(params[0], wires=0)
    ...     qml.RY(params[1], wires=0)
    ...     qml.RX(params[2], wires=0)
    ...     return qml.expval(qml.PauliZ(0)), qml.var(qml.PauliZ(0))
    >>> params = np.array([0.1, 0.2, 0.3], requires_grad=True)
    >>> qml.jacobian(circuit)(params)
    tensor([[-0.38751725, -0.18884792, -0.38355708],
            [ 0.69916868,  0.34072432,  0.69202365]], requires_grad=True)


    .. details::
        :title: Usage Details

        This gradient transform can also be applied directly to :class:`QNode <pennylane.QNode>` objects:

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

        >>> with qml.tape.QuantumTape() as tape:
        ...     qml.RX(params[0], wires=0)
        ...     qml.RY(params[1], wires=0)
        ...     qml.RX(params[2], wires=0)
        ...     qml.expval(qml.PauliZ(0))
        ...     qml.var(qml.PauliZ(0))
        >>> gradient_tapes, fn = qml.gradients.finite_diff(tape)
        >>> gradient_tapes
        [<QuantumTape: wires=[0], params=3>,
         <QuantumTape: wires=[0], params=3>,
         <QuantumTape: wires=[0], params=3>,
         <QuantumTape: wires=[0], params=3>]

        This can be useful if the underlying circuits representing the gradient
        computation need to be analyzed.

        The output tapes can then be evaluated and post-processed to retrieve
        the gradient:

        >>> dev = qml.device("default.qubit", wires=2)
        >>> fn(qml.execute(gradient_tapes, dev, None))
        [[-0.38751721 -0.18884787 -0.38355704]
         [ 0.69916862  0.34072424  0.69202359]]
    """
    if qml.active_return():
        return _finite_diff_new(
            tape,
            argnum=argnum,
            h=h,
            approx_order=approx_order,
            n=n,
            strategy=strategy,
            f0=f0,
            validate_params=validate_params,
            shots=shots,
        )

    if argnum is None and not tape.trainable_params:
        warnings.warn(
            "Attempted to compute the gradient of a tape with no trainable parameters. "
            "If this is unintended, please mark trainable parameters in accordance with the "
            "chosen auto differentiation framework, or via the 'tape.trainable_params' property."
        )
        return [], lambda _: qml.math.zeros([tape.output_dim, 0])

    if validate_params:
        if "grad_method" not in tape._par_info[0]:
            gradient_analysis(tape, grad_fn=finite_diff)
        diff_methods = grad_method_validation("numeric", tape)
    else:
        diff_methods = ["F" for i in tape.trainable_params]

    if all(g == "0" for g in diff_methods):
        return [], lambda _: np.zeros([tape.output_dim, len(tape.trainable_params)])

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

    method_map = choose_grad_methods(diff_methods, argnum)

    for i, _ in enumerate(tape.trainable_params):
        if i not in method_map or method_map[i] == "0":
            # parameter has zero gradient
            shapes.append(0)
            continue

        g_tapes = generate_shifted_tapes(tape, i, shifts * h)
        gradient_tapes.extend(g_tapes)
        shapes.append(len(g_tapes))

    def processing_fn(results):
        # HOTFIX: Apply the same squeezing as in qml.QNode to make the transform output consistent.
        # pylint: disable=protected-access
        if tape._qfunc_output is not None and not isinstance(tape._qfunc_output, Sequence):
            results = [qml.math.squeeze(res) for res in results]

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
            g = sum(r * c for c, r in zip(coeffs, res))

            if c0 is not None:
                # add on the unshifted term
                g = g + r0 * c0

            grads.append(g * (h ** (-n)))

        # The following is for backwards compatibility; currently,
        # the device stacks multiple measurement arrays, even if not the same
        # size, resulting in a ragged array.
        # In the future, we might want to change this so that only tuples
        # of arrays are returned.
        for i, g in enumerate(grads):
            if hasattr(g, "dtype") and g.dtype is np.dtype("object"):
                if qml.math.ndim(g) > 0:
                    grads[i] = qml.math.hstack(g)

        return qml.math.T(qml.math.stack(grads))

    return gradient_tapes, processing_fn

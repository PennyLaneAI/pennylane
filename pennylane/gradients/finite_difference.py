# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

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
import functools
from collections.abc import Callable
from functools import partial
from typing import Literal

# pylint: disable=too-many-arguments,too-many-branches,too-many-statements,unused-argument
from warnings import warn

import numpy as np
from scipy.linalg import solve as linalg_solve
from scipy.special import factorial

from pennylane import math, pytrees
from pennylane.devices.preprocess import decompose
from pennylane.exceptions import DecompositionUndefinedError
from pennylane.gradients.gradient_transform import contract_qjac_with_cjac
from pennylane.measurements import ProbabilityMP
from pennylane.operation import Operator
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms.core import transform
from pennylane.typing import PostprocessingFn

from .general_shift_rules import generate_shifted_tapes
from .gradient_transform import (
    _all_zero_grad,
    _no_trainable_grad,
    assert_no_trainable_tape_batching,
    choose_trainable_param_indices,
    find_and_validate_gradient_methods,
)


@functools.cache
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

    # Note: using np.linalg.solve instead of scipy.linalg.solve can cause a bus error when this
    # is inside a tf.py_function inside a tf.function, as occurs with the tensorflow-autograph interface
    # Bus errors were potentially specific to the M1 Mac. Change with caution.
    coeffs = linalg_solve(A, b)

    coeffs_and_shifts = np.stack([coeffs, shifts])

    # remove all small coefficients and shifts
    coeffs_and_shifts[np.abs(coeffs_and_shifts) < 1e-10] = 0

    # remove columns where the coefficients are 0
    coeffs_and_shifts = coeffs_and_shifts[:, ~np.all(coeffs_and_shifts == 0, axis=0)]

    # sort columns in ascending order according to abs(shift)
    coeffs_and_shifts = coeffs_and_shifts[:, np.argsort(np.abs(coeffs_and_shifts)[1])]
    return coeffs_and_shifts


def finite_diff_jvp(
    f: Callable,
    args: tuple,
    tangents: tuple,
    *,
    h: float = 1e-7,
    approx_order: int = 1,
    strategy: Literal["forward", "backward", "center"] = "forward",
) -> tuple:
    r"""Compute the jvp of a generic function using finite differences.

    Args:
        f (Callable): a generic function that returns a pytree of tensors. Note that this

            function should not have keyword arguments.
        args (tuple[TensorLike]): the tuple of arguments to the function ``f``

        tangents (tuple[TensorLike]): the tuple of tangents for the arguments ``args``


    Keyword Args:
        h=1e-7 (float): finite difference method step size
        approx_order=1 (int): The approximation order of the finite-difference method to use.
        strategy="forward" (str): The strategy of the finite difference method. Must be one of
            ``"forward"``, ``"center"``, or ``"backward"``.
            For the ``"forward"`` strategy, the finite-difference shifts occur at the points
            :math:`x_0, x_0+h, x_0+2h,\dots`, where :math:`h` is some small
            stepsize. The ``"backwards"`` strategy is similar, but in
            reverse: :math:`x_0, x_0-h, x_0-2h, \dots`. Finally, the
            ``"center"`` strategy results in shifts symmetric around the
            unshifted point: :math:`\dots, x_0-2h, x_0-h, x_0, x_0+h, x_0+2h,\dots`.

    Returns:
        tuple(TensorLike, TensorLike): the results and their cotangents


    >>> def f(x, y):
    ...     return 2 * x * y, x**2
    >>> args = (0.5, 1.2)
    >>> tangents = (1.0, 1.0)
    >>> results, dresults = qml.gradients.finite_diff_jvp(f, args, tangents)
    >>> results
    (1.2, 0.25)
    >>> dresults
    [np.float64(3.399999999986747), np.float64(1.000001000006634)]

    """
    coeffs, shifts = finite_diff_coeffs(n=1, approx_order=approx_order, strategy=strategy)

    initial_res = f(*args)
    flat_initial_res, pytree_structure = pytrees.flatten(initial_res)

    jvps = [0 for _ in flat_initial_res]
    for i, t in enumerate(tangents):
        if type(t).__name__ == "Zero":  # Zero = jax.interpreters.ad.Zero
            continue
        t = np.array(t) if isinstance(t, (int, float, complex)) else t

        if math.get_dtype_name(args[i]) in ("float32", "complex64"):

            warn(
                "Detected 32 bits precision parameter with finite differences. It is recommended to use 64 bit precision when computing finite differences.",
                UserWarning,
            )

        shifted_args = list(args)
        for index in np.ndindex(math.shape(args[i])):
            ti = t[index]

            if not math.is_abstract(ti) and math.allclose(ti, 0):
                continue

            ti_over_h = ti / h
            for coeff, shift in zip(coeffs, shifts):
                if shift == 0:
                    res = flat_initial_res
                else:
                    shifted_args[i] = math.scatter_element_add(args[i], index, h * shift)
                    res, _ = pytrees.flatten(f(*shifted_args))

                for result_idx, r in enumerate(res):
                    jvps[result_idx] += ti_over_h * coeff * r

    return initial_res, pytrees.unflatten(jvps, pytree_structure)


def _processing_fn(results, shots, single_shot_batch_fn):
    if not shots.has_partitioned_shots:
        return single_shot_batch_fn(results)
    grads_tuple = []
    for idx in range(shots.num_copies):
        res = [tape_res[idx] for tape_res in results]
        g_tuple = single_shot_batch_fn(res)
        grads_tuple.append(g_tuple)
    return tuple(grads_tuple)


def _finite_diff_stopping_condition(op) -> bool:
    if not op.has_decomposition:
        # let things without decompositions through without error
        # error will happen when calculating shifted tapes for finite diff

        return True
    if isinstance(op, Operator) and any(math.requires_grad(p) for p in op.data):
        return op.grad_method is not None
    return True


# pylint: disable=too-many-positional-arguments
def _expand_transform_finite_diff(
    tape: QuantumScript,
    argnum=None,
    h: float = 1e-7,
    approx_order: int = 1,
    n: int = 1,
    strategy: Literal["forward", "backward", "center"] = "forward",
    f0=None,
    validate_params: bool = True,
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Expand function to be applied before finite difference."""
    [new_tape], postprocessing = decompose(
        tape,
        stopping_condition=_finite_diff_stopping_condition,
        skip_initial_state_prep=False,
        name="finite_diff",
        error=DecompositionUndefinedError,
    )
    if new_tape is tape:
        return [tape], postprocessing
    params = new_tape.get_parameters(trainable_only=False)
    new_tape.trainable_params = math.get_trainable_indices(params)
    return [new_tape], postprocessing


# pylint: disable=too-many-positional-arguments
@partial(
    transform,
    expand_transform=_expand_transform_finite_diff,
    classical_cotransform=contract_qjac_with_cjac,
    final_transform=True,
)
def finite_diff(
    tape: QuantumScript,
    argnum=None,
    h: float = 1e-7,
    approx_order: int = 1,
    n: int = 1,
    strategy: Literal["forward", "backward", "center"] = "forward",
    f0=None,
    validate_params: bool = True,
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    r"""Transform a circuit to compute the finite-difference gradient of all gate parameters with respect to its inputs.

    Args:
        tape (QNode or QuantumTape): quantum circuit to differentiate
        argnum (int or list[int] or None): Trainable parameter indices to differentiate
            with respect to. If not provided, the derivatives with respect to all
            trainable parameters are returned. Note that the indices are with respect to
            the list of trainable parameters.
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

    Returns:
        qnode (QNode) or tuple[List[QuantumTape], function]:

        The transformed circuit as described in :func:`qml.transform <pennylane.transform>`. Executing this circuit
        will provide the Jacobian in the form of a tensor, a tuple, or a nested tuple depending upon the nesting
        structure of measurements in the original circuit.

    **Example**

    This transform can be registered directly as the quantum gradient transform
    to use during autodifferentiation:

    >>> dev = qml.device("default.qubit")
    >>> @qml.qnode(dev, interface="autograd", diff_method="finite-diff")
    ... def circuit(params):
    ...     qml.RX(params[0], wires=0)
    ...     qml.RY(params[1], wires=0)
    ...     qml.RX(params[2], wires=0)
    ...     return qml.expval(qml.Z(0))
    >>> params = np.array([0.1, 0.2, 0.3], requires_grad=True)
    >>> qml.jacobian(circuit)(params)
    array([-0.38751725, -0.18884792, -0.38355708])

    When differentiating QNodes with multiple measurements using Autograd or TensorFlow, the outputs of the QNode first
    need to be stacked. The reason is that those two frameworks only allow differentiating functions with array or
    tensor outputs, instead of functions that output sequences. In contrast, Jax and Torch require no additional
    post-processing.

    >>> import jax
    >>> dev = qml.device("default.qubit")
    >>> @qml.qnode(dev, interface="jax", diff_method="finite-diff")
    ... def circuit(params):
    ...     qml.RX(params[0], wires=0)
    ...     qml.RY(params[1], wires=0)
    ...     qml.RX(params[2], wires=0)
    ...     return qml.expval(qml.Z(0)), qml.var(qml.Z(0))
    >>> params = jax.numpy.array([0.1, 0.2, 0.3])
    >>> jax.jacobian(circuit)(params)
    (Array([-0.38751727, -0.18884793, -0.3835571 ], dtype=float32),
     Array([0.6991687 , 0.34072432, 0.6920237 ], dtype=float32))


    .. details::
        :title: Usage Details

        This gradient transform can be applied directly to :class:`QNode <pennylane.QNode>`
        objects. However, for performance reasons, we recommend providing the gradient transform
        as the ``diff_method`` argument of the QNode decorator, and differentiating with your
        preferred machine learning framework.

        >>> @qml.qnode(dev)
        ... def circuit(params):
        ...     qml.RX(params[0], wires=0)
        ...     qml.RY(params[1], wires=0)
        ...     qml.RX(params[2], wires=0)
        ...     return qml.expval(qml.Z(0)), qml.var(qml.Z(0))
        >>> params = np.array([0.1, 0.2, 0.3], requires_grad=True)
        >>> qml.gradients.finite_diff(circuit)(params)
        (tensor([-0.38751724, -0.18884792, -0.38355708], requires_grad=True),
         tensor([0.69916868, 0.34072432, 0.69202365], requires_grad=True))

        This quantum gradient transform can also be applied to low-level
        :class:`~.QuantumTape` objects. This will result in no implicit quantum
        device evaluation. Instead, the processed tapes, and post-processing
        function, which together define the gradient are directly returned:

        >>> ops = [qml.RX(p, wires=0) for p in params]
        >>> measurements = [qml.expval(qml.Z(0)), qml.var(qml.Z(0))]
        >>> tape = qml.tape.QuantumTape(ops, measurements)
        >>> gradient_tapes, fn = qml.gradients.finite_diff(tape)
        >>> gradient_tapes
        [<QuantumTape: wires=[0], params=3>,
         <QuantumScript: wires=[0], params=3>,
         <QuantumScript: wires=[0], params=3>,
         <QuantumScript: wires=[0], params=3>]

        This can be useful if the underlying circuits representing the gradient
        computation need to be analyzed.

        Note that ``argnum`` refers to the index of a parameter within the list of trainable
        parameters. For example, if we have:

        >>> tape = qml.tape.QuantumScript(
        ...     [qml.RX(1.2, wires=0), qml.RY(2.3, wires=0), qml.RZ(3.4, wires=0)],
        ...     [qml.expval(qml.Z(0))],
        ...     trainable_params = [1, 2]
        ... )
        >>> qml.gradients.finite_diff(tape, argnum=1)

        The code above will differentiate the third parameter rather than the second.

        The output tapes can then be evaluated and post-processed to retrieve the gradient:

        >>> dev = qml.device("default.qubit")
        >>> fn(qml.execute(gradient_tapes, dev, None))
        ((tensor(-0.56464251, requires_grad=True),
          tensor(-0.56464251, requires_grad=True),
          tensor(-0.56464251, requires_grad=True)),
         (tensor(0.93203912, requires_grad=True),
          tensor(0.93203912, requires_grad=True),
          tensor(0.93203912, requires_grad=True)))

        This gradient transform is compatible with devices that use shot vectors for execution.

        >>> from functools import partial
        >>> shots = (10, 100, 1000)
        >>> dev = qml.device("default.qubit")
        >>> @partial(qml.set_shots, shots=shots)
        ... @qml.qnode(dev)
        ... def circuit(params):
        ...     qml.RX(params[0], wires=0)
        ...     qml.RY(params[1], wires=0)
        ...     qml.RX(params[2], wires=0)
        ...     return qml.expval(qml.Z(0)), qml.var(qml.Z(0))
        >>> params = np.array([0.1, 0.2, 0.3], requires_grad=True)
        >>> qml.gradients.finite_diff(circuit, h=0.1)(params)
        ((array([-2., -2.,  0.]), array([3.6, 3.6, 0. ])),
         (array([1. , 0.2, 0.4]), array([-1.78 , -0.34 , -0.688])),
         (array([-0.9 , -0.22, -0.48]), array([1.5498 , 0.3938 , 0.84672])))

        The outermost tuple contains results corresponding to each element of the shot vector.
    """

    transform_name = "finite difference"
    assert_no_trainable_tape_batching(tape, transform_name)

    if any(math.get_dtype_name(p) == "float32" for p in tape.get_parameters()):
        warn(
            "Finite differences with float32 detected. Answers may be inaccurate. float64 is recommended.",
            UserWarning,
        )
    number_parameters = len(tape.trainable_params)
    number_measurements = len(tape.measurements)
    if argnum is None and not tape.trainable_params:
        return _no_trainable_grad(tape)

    trainable_params_indices = choose_trainable_param_indices(tape, argnum)
    diff_methods = (
        find_and_validate_gradient_methods(tape, "numeric", trainable_params_indices)
        if validate_params
        else {idx: "F" for idx in trainable_params_indices}
    )

    if all(g == "0" for g in diff_methods.values()):
        return _all_zero_grad(tape)

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

    for i, _ in enumerate(tape.trainable_params):
        if i not in diff_methods or diff_methods[i] == "0":
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
                    g = math.zeros_like(results[0])
                else:
                    g = []
                    for i in output_dims:
                        zero = math.squeeze(math.zeros(i))
                        g.append(zero)

                grads.append(g)
                continue

            res = results[start : start + s]
            start = start + s

            # compute the linear combination of results
            # and coefficients

            pre_grads = []

            if number_measurements == 1:
                res = math.stack(res)
                c = math.convert_like(coeffs, res)
                lin_comb = math.tensordot(res, c, [[0], [0]])
                pre_grads.append(lin_comb)
            else:
                for i in range(number_measurements):
                    r = math.stack([r[i] for r in res])
                    c = math.convert_like(coeffs, r)
                    lin_comb = math.tensordot(r, c, [[0], [0]])
                    pre_grads.append(lin_comb)

            # Add on the unshifted term
            if c0 is not None:
                if number_measurements == 1:
                    c = math.convert_like(c0, r0)
                    pre_grads = [pre_grads[0] + r0 * c]
                else:
                    for i in range(number_measurements):
                        r_i = r0[i]
                        c = math.convert_like(c0, r_i)
                        pre_grads[i] = pre_grads[i] + r_i * c

            coeff_div = math.cast_like(math.convert_like(1 / h**n, pre_grads[0]), pre_grads[0])

            if len(tape.measurements) > 1:
                pre_grads = tuple(math.convert_like(i * coeff_div, coeff_div) for i in pre_grads)
            else:
                pre_grads = math.convert_like(pre_grads[0] * coeff_div, coeff_div)

            grads.append(pre_grads)
        # Single measurement
        if number_measurements == 1:
            if number_parameters == 1:
                return grads[0]
            return tuple(grads)

        # Reordering to match the right shape for multiple measurements
        grads_reorder = [[0] * number_parameters for _ in range(len(tape.measurements))]
        for i in range(number_measurements):
            for j in range(number_parameters):
                grads_reorder[i][j] = grads[j][i]

        # To tuple
        if number_parameters == 1:
            return tuple(elem[0] for elem in grads_reorder)
        return tuple(tuple(elem) for elem in grads_reorder)

    processing_fn = functools.partial(
        _processing_fn, shots=tape.shots, single_shot_batch_fn=_single_shot_batch_result
    )

    return gradient_tapes, processing_fn

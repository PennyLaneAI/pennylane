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
# pylint: disable=protected-access
import functools

import findiff
import numpy as np

import pennylane as qml


def arbitrary(tape, idx, h=1e-7, order=1, n=1, form="forward"):
    r"""Generate the first-order forward finite-difference tapes and postprocessing
    methods required to compute the gradient of a gate parameter.

    Args:
        tape (.QuantumTape): quantum tape to differentiate
        idx (int): trainable parameter index to differentiate with respect to
        h=1e-7 (float): finite difference method step size

    Returns:
        tuple[list[QuantumTape], function]: A tuple containing the list of generated tapes,
        in addition to a post-processing function to be applied to the evaluated
        tapes.
    """
    params = qml.math.stack(tape.get_parameters())
    tapes = []

    stencil = findiff.coefficients(deriv=n, acc=order)[form]
    coeffs = stencil["coefficients"]
    shifts = stencil["offsets"]

    if 0 in shifts:
        i = np.where(shifts == 0)[0][0]
        shifts = np.delete(shifts, i)
        shifts = np.concatenate([0, shifts])

        c0 = coeffs[i]
        coeffs = np.delete(coeffs, i)
        coeffs = np.concatenate([c0, coeffs])

    for s in shifts:
        shifted_tape = tape.copy(copy_operations=True)

        shift = np.zeros(qml.math.shape(params), dtype=np.float64)
        shift[idx] = s * h

        shifted_params = params + qml.math.convert_like(shift, params)
        shifted_tape.set_parameters(qml.math.unstack(shifted_params))

        tapes.append(shifted_tape)

    def processing_fn(results):
        """Computes the gradient of the parameter at index idx via first-order
        forward finite differences.

        Args:
            results (list[real]): evaluated quantum tapes

        Returns:
            array[float]: 1-dimensional array of length determined by the tape output
            measurement statistics
        """
        results = qml.math.squeeze(qml.math.stack(results))
        return sum([c * r for c, r in zip(coeffs, results)])

    return tapes, processing_fn


def first_order_forward(tape, idx, h=1e-7):
    r"""Generate the first-order forward finite-difference tapes and postprocessing
    methods required to compute the gradient of a gate parameter.

    Args:
        tape (.QuantumTape): quantum tape to differentiate
        idx (int): trainable parameter index to differentiate with respect to
        h=1e-7 (float): finite difference method step size

    Returns:
        tuple[list[QuantumTape], function]: A tuple containing the list of generated tapes,
        in addition to a post-processing function to be applied to the evaluated
        tapes.
    """
    params = qml.math.stack(tape.get_parameters())
    shift = np.zeros(qml.math.shape(params), dtype=np.float64)
    coeffs = []
    tapes = []

    shift[idx] = h
    shifted_tape = tape.copy(copy_operations=True)

    shifted_params = params + qml.math.convert_like(shift, params)
    shifted_tape.set_parameters(qml.math.unstack(shifted_params))

    tapes.append(shifted_tape)
    tapes.append(tape)

    def processing_fn(results):
        """Computes the gradient of the parameter at index idx via first-order
        forward finite differences.

        Args:
            results (list[real]): evaluated quantum tapes

        Returns:
            array[float]: 1-dimensional array of length determined by the tape output
            measurement statistics
        """
        shifted = results[0]
        y0 = results[1]

        return qml.math.squeeze((shifted - y0) / h)

    return tapes, processing_fn


def second_order_centered(tape, idx, h=1e-7):
    r"""Generate the second-order centered finite-difference tapes and postprocessing
    methods required to compute the gradient of a gate parameter.

    Args:
        tape (.QuantumTape): quantum tape to differentiate
        idx (int): trainable parameter index to differentiate with respect to
        h=1e-7 (float): finite difference method step size

    Returns:
        tuple[list[QuantumTape], function]: A tuple containing the list of generated tapes,
        in addition to a post-processing function to be applied to the evaluated
        tapes.
    """
    params = qml.math.stack(tape.get_parameters())
    shift = np.zeros(qml.math.shape(params), dtype=np.float64)
    shift[idx] = h
    shift = qml.math.convert_like(shift, params)

    shifted_forward = tape.copy(copy_operations=True)
    shifted_backward = tape.copy(copy_operations=True)

    shifted_params = params + shift / 2.0
    shifted_forward.set_parameters(qml.math.unstack(shifted_params))

    shifted_params = params - shift / 2.0
    shifted_backward.set_parameters(qml.math.unstack(shifted_params))

    def processing_fn(results):
        """Computes the gradient of the parameter at index idx via first-order
        forward finite differences.

        Args:
            results (list[real]): evaluated quantum tapes

        Returns:
            array[float]: 1-dimensional array of length determined by the tape output
            measurement statistics
        """
        return qml.math.squeeze((results[0] - results[1]) / h)

    return [shifted_forward, shifted_backward], processing_fn


def grad(tape, argnum=None, h=1e-7, order=1, n=1, form="forward"):
    r"""Generate the parameter-shift tapes and postprocessing methods required
    to compute the gradient of an gate parameter with respect to an
    expectation value.

    Args:
        tape (.QuantumTape): quantum tape to differentiate
        argnum (int or list[int] or None): Trainable parameter indices to differentiate
            with respect to. If not provided, the derivative with respect to all
            trainable indices are returned.
        h=1e-7 (float): finite difference method step size
        order=1 (int): The order of the finite difference method to use. ``1`` corresponds
            to forward finite differences, ``2`` to centered finite differences.

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
    >>> gradient_tapes, fn = gradients.finite_difference.grad(tape, order=2)
    >>> res = dev.batch_execute(gradient_tapes)
    >>> fn(res)
    [[-0.38751721 -0.18884787 -0.38355704]
     [ 0.69916862  0.34072424  0.69202359]]
    """
    if any(m.return_type is qml.operation.State for m in tape.measurements):
        raise ValueError("Does not support circuits that return the state")

    # TODO: replace the JacobianTape._grad_method_validation
    # functionality before deprecation.
    diff_methods = tape._grad_method_validation("numeric")

    if not tape.trainable_params or all(g == "0" for g in diff_methods):
        # Either all parameters have grad method 0, or there are no trainable
        # parameters.
        return [[]], []

    gradient_tapes = []
    processing_fns = []
    shapes = []

    if order == 1:
        gradient_tapes.append(tape)
        gradient_fn = first_order_forward

    elif order == 2:
        gradient_fn = second_order_centered

    # TODO: replace the JacobianTape._choose_params_with_methods
    # functionality before deprecation.
    for idx, (t_idx, dm) in enumerate(tape._choose_params_with_methods(diff_methods, argnum)):
        if dm == "0":
            shapes.append(0)
            processing_fns.append([])
            continue

        g_tapes, fn = gradient_fn(tape, t_idx, h=h)

        if order == 1:
            g_tapes = g_tapes[:1]

        gradient_tapes.extend(g_tapes)
        processing_fns.append(fn)
        shapes.append(len(g_tapes))

    def processing_fn(results):
        grads = []
        start = 1 if order == 1 else 0
        y0 = results[0]

        for s, f in zip(shapes, processing_fns):

            if s == 0:
                g = qml.math.convert_like(np.zeros([tape.output_dim]), results)
                grads.append(g)
                continue

            r = results[start : start + s]
            g = f(qml.math.stack([r[0], y0])) if order == 1 else f(r)

            grads.append(g)
            start = start + s

        return qml.math.stack(grads).T

    return gradient_tapes, processing_fn

# Copyright 2022 Xanadu Quantum Technologies Inc.

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
This module contains functions for computing the Jacobian-vector product
of tapes.
"""
import numpy as np

from pennylane import math


def compute_jvp(dy, jac, num=None):
    """Convenience function to compute the Jacobian-vector product for a given
    vector of gradient outputs and a Jacobian.

    Args:
        dy (tensor_like): vector of gradient outputs
        jac (tensor_like): Jacobian matrix. For an n-dimensional ``dy``
            vector, the first n-dimensions of ``jac`` should match
            the shape of ``dy``.
        num (int): The length of the flattened ``dy`` argument. This is an
            optional argument, but can be useful to provide if ``dy`` potentially
            has no shape (for example, due to tracing or just-in-time compilation).

    Returns:
        tensor_like: the Jacobian-vector product
    """
    if jac is None:
        return None

    dy_row = math.reshape(dy, [-1])

    if num is None:
        num = math.shape(dy_row)[0]
    if not isinstance(dy_row, np.ndarray):
        jac = math.convert_like(jac, dy_row)
        jac = math.cast(jac, dy_row.dtype)

    jac = math.reshape(jac, [-1, num])

    try:
        if math.allclose(dy, 0):
            # If the dy vector is zero, then the
            # corresponding element of the VJP will be zero.
            num_params = jac.shape[0]
            res = math.convert_like(np.zeros([num_params]), dy)
            return math.cast(res, dy.dtype)
    except (AttributeError, TypeError):
        pass

    return math.tensordot(jac, dy_row, axes=[[1], [0]])


def jvp(tape, dy, gradient_fn, gradient_kwargs=None):
    r"""Generate the gradient tapes and processing function required to compute
    the Jacobian-vector products of a tape.

    Args:
        tape (.QuantumTape): quantum tape to differentiate
        dy (tensor_like): Gradient-output vector. Must have shape
            matching the output shape of the corresponding tape.
        gradient_fn (callable): the gradient transform to use to differentiate
            the tape
        gradient_kwargs (dict): dictionary of keyword arguments to pass when
            determining the gradients of tapes

    Returns:
        tensor_like or None: Jacobian-vector product. Returns None if the tape
        has no trainable parameters.

    **Example**

    """
    gradient_kwargs = gradient_kwargs or {}
    num_params = len(tape.trainable_params)

    if num_params == 0:
        # The tape has no trainable parameters; the JVP
        # is simply none.
        return [], lambda _, num=None: None

    try:
        if math.allclose(dy, 0):
            # If the dy vector is zero, then the
            # corresponding element of the JVP will be zero,
            # and we can avoid a quantum computation.

            def func(_, num=None):  # pylint: disable=unused-argument
                res = math.convert_like(np.zeros([num_params]), dy)
                return math.cast(res, dy.dtype)

            return [], func
    except (AttributeError, TypeError):
        pass

    gradient_tapes, fn = gradient_fn(tape, **gradient_kwargs)

    def processing_fn(results, num=None):
        # Postprocess results to compute the Jacobian-vector product
        jac = fn(results)
        return compute_jvp(dy, jac, num=num)

    return gradient_tapes, processing_fn


def batch_jvp(tapes, dys, gradient_fn, reduction="append", gradient_kwargs=None):
    r"""Generate the gradient tapes and processing function required to compute
    the Jacobian-vector products of a batch of tapes.

    Args:
        tapes (Sequence[.QuantumTape]): sequence of quantum tapes to differentiate
        dys (Sequence[tensor_like]): Sequence of gradient-output vectors ``dy``. Must be the
            same length as ``tapes``. Each ``dy`` tensor should have shape
            matching the output shape of the corresponding tape.
        gradient_fn (callable): the gradient transform to use to differentiate
            the tapes
        reduction (str): Determines how the Jacobian-vector products are returned.
            If ``append``, then the output of the function will be of the form
            ``List[tensor_like]``, with each element corresponding to the JVP of each
            input tape. If ``extend``, then the output JVPs will be concatenated.
        gradient_kwargs (dict): dictionary of keyword arguments to pass when
            determining the gradients of tapes

    Returns:
        List[tensor_like or None]: list of Jacobian-vector products. ``None`` elements corresponds
        to tapes with no trainable parameters.

    **Example**

    """
    gradient_kwargs = gradient_kwargs or {}
    reshape_info = []
    gradient_tapes = []
    processing_fns = []

    # Loop through the tapes and dys vector
    for tape, dy in zip(tapes, dys):
        g_tapes, fn = jvp(tape, dy, gradient_fn, gradient_kwargs)

        reshape_info.append(len(g_tapes))
        processing_fns.append(fn)
        gradient_tapes.extend(g_tapes)

    def processing_fn(results, nums=None):
        jvps = []
        start = 0

        if nums is None:
            nums = [None] * len(tapes)

        for t_idx in range(len(tapes)):
            # extract the correct results from the flat list
            res_len = reshape_info[t_idx]
            res_t = results[start : start + res_len]
            start += res_len

            # postprocess results to compute the JVP
            jvp_ = processing_fns[t_idx](res_t, num=nums[t_idx])

            if jvp_ is None:
                if reduction == "append":
                    jvps.append(None)
                continue

            if isinstance(reduction, str):
                getattr(jvps, reduction)(jvp_)
            elif callable(reduction):
                reduction(jvps, jvp_)

        return jvps

    return gradient_tapes, processing_fn

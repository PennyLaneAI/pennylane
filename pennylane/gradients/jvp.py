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
This module contains functions for computing the Jacobian vector product
of tapes.
"""
import numpy as np

import pennylane as qml
from pennylane import math


def _convert(jac, tangent):
    """Utility to convert and cast the jacobian as tangent."""
    if isinstance(jac, tuple):
        jac_new = []
        for j in jac:
            j_ = math.convert_like(j, tangent)
            j_ = math.cast(j_, tangent.dtype)
            jac_new.append(j_)
        jac = tuple(jac_new)
    else:
        jac = math.convert_like(jac, tangent)
        jac = math.cast(jac, tangent.dtype)
    return jac


def compute_jvp_single(tangent, jac):
    """Convenience function to compute the Jacobian vector product for a given
    vector of gradient outputs and a Jacobian for a single measurement tape.
    Args:
        dy (tensor_like): vector of gradient outputs
        jac (tensor_like, tuple): Jacobian matrix
    Returns:
        tensor_like: the Jacobian vector product

    **Examples**

    """
    if jac is None:
        return None

    tangent = qml.math.stack(tangent)
    jac = _convert(jac, tangent)

    # Single measurement with a single param
    if not isinstance(jac, tuple):
        # Single measurement with no dimension e.g. expval
        if jac.shape == ():
            jac = math.reshape(jac, (1,))
            tangent = math.reshape(tangent, (1,))
            res = math.tensordot(jac, tangent, [[0], [0]])
        # Single measurement with dimension e.g. probs
        else:
            tangent = math.reshape(tangent, (1,))
            jac = math.reshape(jac, (1, len(jac)))
            res = math.tensordot(jac, tangent, [[0], [0]])
    # Single measurement with multiple params
    else:
        # Single measurement with no dimension e.g. expval
        if jac[0].shape == ():
            jac = qml.math.stack(jac)
            res = qml.math.tensordot(jac, tangent, 1)
        # Single measurement with dimension e.g. probs
        else:
            jac = qml.math.stack(jac)
            res = qml.math.tensordot(jac, tangent, [[0], [0]])
    return res


def compute_jvp_multi(tangent, jac):
    """Convenience function to compute the vector-Jacobian product for a given
    vector of gradient outputs and a Jacobian for a multiple measurements tape.
    Args:
        dy (tensor_like): vector of gradient outputs
        jac (tensor_like, tuple): Jacobian matrix
    Returns:
        tensor_like: the vector-Jacobian product
    **Examples**

    """
    if jac is None:
        return None
    res = []

    for j in jac:
        res.append(compute_jvp_single(tangent, j))

    return tuple(res)


def jvp(tape, tangent, gradient_fn, gradient_kwargs=None):
    r"""Generate the gradient tapes and processing function required to compute
    the Jacobian vector products of a tape.

    Args:
        tape (.QuantumTape): quantum tape to differentiate
        tangent (tensor_like): Gradient-output vector. Must have shape
            matching the output shape of the corresponding tape.
        gradient_fn (callable): the gradient transform to use to differentiate
            the tape
        gradient_kwargs (dict): dictionary of keyword arguments to pass when
            determining the gradients of tapes

    Returns:
        tensor_like or None: Vector-Jacobian product. Returns None if the tape
        has no trainable parameters.

    **Example**

    """
    gradient_kwargs = gradient_kwargs or {}
    num_params = len(tape.trainable_params)
    num_measurements = len(tape.measurements)

    if num_params == 0:
        # The tape has no trainable parameters; the JVP
        # is simply none.
        return [], lambda _, num=None: None

    multi_m = num_measurements > 1
    multi_p = num_params > 1

    try:
        if math.allclose(qml.math.stack(tangent), 0):
            # If the tangent vector is zero, then the
            # corresponding element of the JVP will be zero,
            # and we can avoid a quantum computation.

            def func(_):  # pylint: disable=unused-argument
                if not multi_m:
                    res = math.convert_like(np.zeros([num_params]), tangent)
                    res = math.cast(res, tangent.dtype)
                    if multi_p:
                        res = tuple(res)
                else:
                    res = []
                    for m in tape.measurements:
                        # TODO: Update shape for CV variables
                        if m.return_type is qml.measurements.Probability:
                            dim = 2 ** len(m.wires)
                        else:
                            dim = ()
                        sub_res = math.convert_like(np.zeros(dim), tangent)
                        sub_res = math.cast(sub_res, tangent.dtype)
                        res.append(sub_res)
                    res = tuple(res)

                return res

            return [], func
    except (AttributeError, TypeError):
        pass

    gradient_tapes, fn = gradient_fn(tape, **gradient_kwargs)

    def processing_fn(results):
        # postprocess results to compute the Jacobian
        jac = fn(results)
        if multi_m:
            return compute_jvp_multi(tangent, jac)
        return compute_jvp_single(tangent, jac)

    return gradient_tapes, processing_fn


def batch_jvp(tapes, tangents, gradient_fn, reduction="append", gradient_kwargs=None):
    r"""Generate the gradient tapes and processing function required to compute
    the Jacobian vector products of a batch of tapes.

    Args:
        tapes (Sequence[.QuantumTape]): sequence of quantum tapes to differentiate
        dys (Sequence[tensor_like]): Sequence of gradient-output vectors ``dy``. Must be the
            same length as ``tapes``. Each ``dy`` tensor should have shape
            matching the output shape of the corresponding tape.
        gradient_fn (callable): the gradient transform to use to differentiate
            the tapes
        reduction (str): Determines how the vector-Jacobian products are returned.
            If ``append``, then the output of the function will be of the form
            ``List[tensor_like]``, with each element corresponding to the VJP of each
            input tape. If ``extend``, then the output VJPs will be concatenated.
        gradient_kwargs (dict): dictionary of keyword arguments to pass when
            determining the gradients of tapes

    Returns:
        List[tensor_like or None]: list of vector-Jacobian products. ``None`` elements corresponds
        to tapes with no trainable parameters.

    **Example**

    """
    gradient_kwargs = gradient_kwargs or {}
    reshape_info = []
    gradient_tapes = []
    processing_fns = []

    # Loop through the tapes and dys vector
    for tape, tangent in zip(tapes, tangents):
        g_tapes, fn = jvp(tape, tangent, gradient_fn, gradient_kwargs)

        reshape_info.append(len(g_tapes))
        processing_fns.append(fn)
        gradient_tapes.extend(g_tapes)

    def processing_fn(results):
        jvps = []
        start = 0

        for t_idx in range(len(tapes)):
            # extract the correct results from the flat list
            res_len = reshape_info[t_idx]
            res_t = results[start : start + res_len]
            start += res_len

            # postprocess results to compute the VJP
            jvp_ = processing_fns[t_idx](res_t)

            if jvp_ is None:
                if reduction == "append":
                    jvps.append(None)
                continue

            if isinstance(reduction, str):
                if not isinstance(jvp_, tuple) and jvp_.shape == ():
                    jvp_ = math.reshape(jvp_, (1,))
                getattr(jvps, reduction)(jvp_)
            elif callable(reduction):
                reduction(jvps, jvp_)

        return jvps

    return gradient_tapes, processing_fn

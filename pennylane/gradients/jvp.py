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


def _convert(jac, tangent):
    """Utility to convert and cast the jacobian as tangent."""
    if isinstance(jac, tuple):
        jac_new = []
        for j in jac:
            j_ = qml.math.convert_like(j, tangent)
            j_ = qml.math.cast_like(j_, tangent)
            jac_new.append(j_)
        jac = tuple(jac_new)
    else:
        jac = qml.math.convert_like(jac, tangent)
        jac = qml.math.cast_like(jac, tangent)
    return jac


def compute_jvp_single(tangent, jac):
    """Convenience function to compute the Jacobian vector product for a given
    tangent vector and a Jacobian for a single measurement tape.

    Args:
        tangent (list, tensor_like): tangent vector
        jac (tensor_like, tuple): Jacobian matrix

    Returns:
        tensor_like: the Jacobian vector product

    **Examples**

    1. For a single parameter and a single measurement without shape (e.g. expval, var):

    .. code-block:: pycon

        >>> tangent = np.array([1.0])
        >>> jac = np.array(0.2)
        >>> qml.gradients.compute_jvp_single(tangent, jac)
        np.array(0.2)

    2. For a single parameter and a single measurment with shape (e.g. probs):

    .. code-block:: pycon

        >>> tangent = np.array([2.0])
        >>> jac = np.array([0.3, 0.4])
        >>> qml.gradients.compute_jvp_single(tangent, jac)
        np.array([0.6, 0.8])

    3. For multiple parameters (in this case 2 parameters) and a single measurement without shape (e.g. expval, var):

    .. code-block:: pycon

        >>> tangent = np.array([1.0, 2.0])
        >>> jac = tuple([np.array(0.1), np.array(0.2)])
        >>> qml.gradients.compute_jvp_single(tangent, jac)
        np.array(0.5)

    4. For multiple parameters (in this case 2 parameters) and a single measurement with shape (e.g. probs):

    .. code-block:: pycon

        >>> tangent = np.array([1.0, 0.5])
        >>> jac = tuple([np.array([0.1, 0.3]), np.array([0.2, 0.4])])
        >>> qml.gradients.compute_jvp_single(tangent, jac)
        np.array([0.2, 0.5])

    """
    if jac is None:
        return None

    tangent = qml.math.stack(tangent)
    jac = _convert(jac, tangent)

    # Single param
    if not isinstance(jac, tuple):
        # No trainable parameters
        if jac.shape == (0,):
            res = qml.math.zeros((1, 0))
            return res

        tangent = qml.math.reshape(tangent, (1,))

        # No dimension e.g. expval
        if jac.shape == ():
            jac = qml.math.reshape(jac, (1,))

        # With dimension e.g. probs
        else:
            jac = qml.math.reshape(jac, (1, -1))
        res = qml.math.tensordot(jac, tangent, [[0], [0]])
    # Multiple params
    else:
        # No trainable parameters (adjoint)
        if len(jac) == 0:
            res = qml.math.zeros((1, 0))
            return res

        jac = qml.math.stack(jac)

        # No dimension e.g. expval
        if jac[0].shape == ():
            res = qml.math.tensordot(jac, tangent, 1)

        # With dimension e.g. probs
        else:
            res = qml.math.tensordot(jac, tangent, [[0], [0]])
    return res


def compute_jvp_multi(tangent, jac):
    """Convenience function to compute the Jacobian-vector product for a given
    vector of gradient outputs and a Jacobian for a tape with multiple measurements.

    Args:
        tangent (tensor_like, list): tangent vector
        jac (tensor_like, tuple): Jacobian matrix

    Returns:
        tensor_like: the Jacobian-vector product

    **Examples**

    1. For a single parameter and multiple measurements (one without shape and one with shape, e.g. expval and probs):

    .. code-block:: pycon

        >>> tangent = np.array([2.0])
        >>> jac = tuple([np.array([0.3]), np.array([0.2, 0.5])])
        >>> qml.gradients.compute_jvp_multi(tangent, jac)
        (np.array([0.6]), np.array([0.4, 1. ]))

    2. For multiple parameters (in this case 2 parameters) and multiple measurements (one without shape and one with
    shape, e.g. expval and probs):

    .. code-block:: pycon

        >>> tangent = np.array([1.0, 2.0])
        >>> jac = tuple([tuple([np.array([0.3]), np.array([0.4])]), tuple([np.array([0.2, 0.5]), np.array([0.3, 0.8])]),])
        >>> qml.gradients.compute_jvp_multi(tangent, jac)
        (np.array([1.1]), np.array([0.8, 2.1]))
    """
    if jac is None:
        return None
    res = tuple(compute_jvp_single(tangent, j) for j in jac)
    return res


def jvp(tape, tangent, gradient_fn, gradient_kwargs=None):
    r"""Generate the gradient tapes and processing function required to compute
    the Jacobian vector products of a tape.

    Args:
        tape (.QuantumTape): quantum tape to differentiate
        tangent (tensor_like, list): Gradient-output vector. Must have shape
            matching the number of trainable parameters.
        gradient_fn (callable): the gradient transform to use to differentiate
            the tape
        gradient_kwargs (dict): dictionary of keyword arguments to pass when
            determining the gradients of tapes

    Returns:
        tensor_like or tuple or None: Jacobian vector product. Returns None if the tape
        has no trainable parameters.

    **Example**
    #TODO: add examples
    """
    gradient_kwargs = gradient_kwargs or {}
    num_params = len(tape.trainable_params)
    num_measurements = len(tape.measurements)

    if num_params == 0:
        # The tape has no trainable parameters; the JVP
        # is simply none.
        return [], lambda _, num=None: None

    multi_m = num_measurements > 1

    try:
        if qml.math.allclose(qml.math.stack(tangent), 0):
            # If the tangent vector is zero, then the
            # corresponding element of the JVP will be zero,
            # and we can avoid a quantum computation.

            def func(_):  # pylint: disable=unused-argument
                if not multi_m:
                    # TODO: Update shape for CV variables and for qutrit simulations
                    res = _single_measurement_zero(tape.measurements[0], tangent)
                else:
                    # TODO: Update shape for CV variables and for qutrit simulations
                    res = [_single_measurement_zero(m, tangent) for m in tape.measurements]
                    res = tuple(res)
                return res

            return [], func
        # TODO remove the pragma in PR #3170
    except (AttributeError, TypeError):  # pragma: no cover
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
        tangentss (Sequence[tensor_like]): Sequence of gradient-output vectors ``dy``. Must be the
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
        List[tensor_like or None]: list of Jacobian vector products. ``None`` elements corresponds
        to tapes with no trainable parameters.

    **Example**
    # TODO: add examples
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

            # postprocess results to compute the JVP
            jvp_ = processing_fns[t_idx](res_t)

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


def _single_measurement_zero(m, tangent):
    """Aux function to create a zero tensor from a measurement."""
    if m.return_type is qml.measurements.Probability:
        dim = 2 ** len(m.wires)
    else:
        dim = ()
    res = qml.math.convert_like(np.zeros(dim), tangent)
    res = qml.math.cast_like(res, tangent)
    return res

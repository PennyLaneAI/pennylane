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
from collections.abc import Sequence

import numpy as np

import pennylane as qml
from pennylane._device import _get_num_copies
from pennylane.measurements import ProbabilityMP


def _convert(jac, tangent):
    """Utility to convert and cast the jacobian as tangent."""
    if isinstance(jac, tuple):
        jac_new = []
        for j in jac:
            j_ = qml.math.convert_like(j, tangent)
            j_ = qml.math.cast(j_, tangent.dtype)
            jac_new.append(j_)
        jac = tuple(jac_new)
    else:
        jac = qml.math.convert_like(jac, tangent)
        jac = qml.math.cast(jac, tangent.dtype)
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


def jvp(tape, tangent, gradient_fn, shots=None, gradient_kwargs=None):
    r"""Generate the gradient tapes and processing function required to compute
    the Jacobian vector product of a tape. This function only works with the new return type system on.

    Args:
        tape (.QuantumTape): quantum tape to differentiate
        tangent (tensor_like, list): Gradient-output vector. Must have shape
            matching the number of trainable parameters.
        gradient_fn (callable): the gradient transform to use to differentiate
            the tape
        shots (None, int, list[int]): The device shots that will be used to
            execute the tapes outputted by this
        gradient_kwargs (dict): dictionary of keyword arguments to pass when
            determining the gradients of tapes

    Returns:
        tensor_like or tuple or None: Jacobian vector product. Returns None if the tape
        has no trainable parameters.

    **Example**

    Consider the following quantum tape with Jax parameters:

    .. code-block:: python

        import jax

        qml.enable_return()

        x = jax.numpy.array([[0.1, 0.2, 0.3],
                             [0.4, 0.5, 0.6]])

        with qml.tape.QuantumTape() as tape:
            qml.RX(x[0, 0], wires=0)
            qml.RY(x[0, 1], wires=1)
            qml.RZ(x[0, 2], wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RX(x[1, 0], wires=1)
            qml.RY(x[1, 1], wires=0)
            qml.RZ(x[1, 2], wires=1)
            qml.expval(qml.PauliZ(0))
            qml.probs(wires=1)

    We can use the ``jvp`` function to compute the Jacobian vector product,
    given a tangent vector ``tangent``:

    >>> tangent = [jax.numpy.array(1.0), jax.numpy.array(1.0), jax.numpy.array(1.0), jax.numpy.array(1.0), jax.numpy.array(1.0), jax.numpy.array(1.0)]
    >>> jvp_tapes, fn = qml.gradients.jvp(tape, tangent, qml.gradients.param_shift)

    Note that ``tangent`` has six elements, matching the parameter dimension of the tape.

    Executing the JVP tapes, and applying the processing function:

    >>> dev = qml.device("default.qubit", wires=2)
    >>> jvp = fn(dev.batch_execute(jvp_tapes))
    >>> jvp
    (Array(-0.62073976, dtype=float32), Array([-0.3259707 ,  0.32597077], dtype=float32))
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
    except (AttributeError, TypeError):
        pass

    gradient_tapes, fn = gradient_fn(tape, shots=shots, **gradient_kwargs)

    def processing_fn(results):
        # postprocess results to compute the Jacobian
        jac = fn(results)
        shot_vector = isinstance(shots, Sequence)

        # Jacobian without shot vectors
        if not shot_vector:
            if multi_m:
                return compute_jvp_multi(tangent, jac)
            return compute_jvp_single(tangent, jac)

        # The jacobian is calculated for shot vectors
        len_shot_vec = _get_num_copies(shots)
        jvps = []
        if multi_m:
            for i in range(len_shot_vec):
                jvps.append(compute_jvp_multi(tangent, jac[i]))
        else:
            for i in range(len_shot_vec):
                jvps.append(compute_jvp_single(tangent, jac[i]))

        return tuple(jvps)

    return gradient_tapes, processing_fn


def batch_jvp(tapes, tangents, gradient_fn, shots=None, reduction="append", gradient_kwargs=None):
    r"""Generate the gradient tapes and processing function required to compute
    the Jacobian vector products of a batch of tapes.

    Args:
        tapes (Sequence[.QuantumTape]): sequence of quantum tapes to differentiate
        tangents (Sequence[tensor_like]): Sequence of gradient-output vectors ``dy``. Must be the
            same length as ``tapes``. Each ``dy`` tensor should have shape
            matching the output shape of the corresponding tape.
        gradient_fn (callable): the gradient transform to use to differentiate
            the tapes
        shots (None, int, list[int]): The device shots that will be used to
            execute the tapes outputted by this
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

    .. code-block:: python

        import jax
        qml.enable_return()
        x = jax.numpy.array([[0.1, 0.2, 0.3],
                             [0.4, 0.5, 0.6]])

        def ansatz(x):
            qml.RX(x[0, 0], wires=0)
            qml.RY(x[0, 1], wires=1)
            qml.RZ(x[0, 2], wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RX(x[1, 0], wires=1)
            qml.RY(x[1, 1], wires=0)
            qml.RZ(x[1, 2], wires=1)

        with qml.tape.QuantumTape() as tape1:
            ansatz(x)
            qml.expval(qml.PauliZ(0))
            qml.probs(wires=1)

        with qml.tape.QuantumTape() as tape2:
            ansatz(x)
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        tapes = [tape1, tape2]

    Both tapes share the same circuit ansatz, but have different measurement outputs.

    We can use the ``batch_jvp`` function to compute the Jacobian vector product,
    given a list of tangents ``tangent``:

    >>> tangent_0 = [jax.numpy.array(1.0), jax.numpy.array(1.0), jax.numpy.array(1.0), jax.numpy.array(1.0), jax.numpy.array(1.0), jax.numpy.array(1.0)]
    >>> tangent_1 = [jax.numpy.array(1.0), jax.numpy.array(1.0), jax.numpy.array(1.0), jax.numpy.array(1.0), jax.numpy.array(1.0), jax.numpy.array(1.0)]
    >>> tangents = [tangent_0, tangent_1]

    Note that each ``tangents`` has shape matching the parameter dimension of the tape.

    Executing the JVP tapes, and applying the processing function:

    >>> jvp_tapes, fn = qml.gradients.batch_jvp(tapes, tangents, qml.gradients.param_shift)

    >>> dev = qml.device("default.qubit", wires=2)
    >>> jvps = fn(dev.batch_execute(jvp_tapes))
    >>> jvps
    [(Array(-0.62073976, dtype=float32), Array([-0.3259707 ,  0.32597077], dtype=float32)), Array(-0.6900841, dtype=float32)]

    We have two JVPs; one per tape. Each one corresponds to the shape of the output of their respective tape.
    """
    # pylint: disable=too-many-arguments
    gradient_kwargs = gradient_kwargs or {}
    reshape_info = []
    gradient_tapes = []
    processing_fns = []

    # Loop through the tapes and dys vector
    for tape, tangent in zip(tapes, tangents):
        g_tapes, fn = jvp(tape, tangent, gradient_fn, shots, gradient_kwargs)

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
    dim = 2 ** len(m.wires) if isinstance(m, ProbabilityMP) else ()
    res = qml.math.convert_like(np.zeros(dim), tangent)
    res = qml.math.cast_like(res, tangent)
    return res

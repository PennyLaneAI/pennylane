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
from pennylane.measurements import ProbabilityMP


def compute_jvp_single(tangent, jac):
    r"""Convenience function to compute the Jacobian vector product for a given
    tangent vector and a Jacobian for a single measurement tape.

    Args:
        tangent (list, tensor_like): tangent vector
        jac (tensor_like, tuple): Jacobian matrix

    Returns:
        tensor_like: the Jacobian vector product

    **Examples**

    We start with a number of examples. A more complete, technical description is given
    further below.

    1. For a single parameter and a single measurement without shape (e.g. ``expval``, ``var``):

    .. code-block:: pycon

        >>> tangent = np.array([1.0])
        >>> jac = np.array(0.2)
        >>> qml.gradients.compute_jvp_single(tangent, jac)
        array(0.2)

    2. For a single parameter and a single measurement with shape (e.g. ``probs``):

    .. code-block:: pycon

        >>> tangent = np.array([2.0])
        >>> jac = np.array([0.3, 0.4])
        >>> qml.gradients.compute_jvp_single(tangent, jac)
        array([0.6, 0.8])

    3. For multiple parameters (in this case 2 parameters) and a single measurement
       without shape (e.g. ``expval``, ``var``):

    .. code-block:: pycon

        >>> tangent = np.array([1.0, 2.0])
        >>> jac = tuple([np.array(0.1), np.array(0.2)])
        >>> qml.gradients.compute_jvp_single(tangent, jac)
        array(0.5)

    4. For multiple parameters (in this case 2 parameters) and a single measurement with
       shape (e.g. ``probs``):

    .. code-block:: pycon

        >>> tangent = np.array([1.0, 0.5])
        >>> jac = tuple([np.array([0.1, 0.3]), np.array([0.2, 0.4])])
        >>> qml.gradients.compute_jvp_single(tangent, jac)
        array([0.2, 0.5])

    .. details::
        :title: Technical description
        :href: technical-description

        There are multiple case distinctions in this function, for particular examples see above.

        - The JVP may be for one **(A)** or multiple **(B)** parameters. We call the number of
          parameters ``k``

        - The number ``R`` of tape return type dimensions may be between 0 and 3.
          We call the return type dimensions ``r_j``

        - Each parameter may have an arbitrary number ``L_i>=0`` of dimensions

        In the following, ``(a, b)`` denotes a tensor_like of shape ``(a, b)`` and ``[(a,), (b,)]``
        / ``((a,), (b,))`` denotes a ``list`` / ``tuple`` of tensors with the indicated shapes,
        respectively. Ignore the case of no trainable parameters, as it is filtered out in advance.

        For scenario **(A)**, the input shapes can be in

        .. list-table::
           :widths: 30 40 30
           :header-rows: 1

           * - ``tangent`` shape
             - ``jac`` shape
             - Comment
           * - ``(1,)`` or ``[()]`` or ``(())``
             - ``()``
             - scalar return, scalar parameter
           * - ``(1,)`` or ``[()]`` or ``(())``
             - ``(r_1,..,r_R)``
             - tensor return, scalar parameter
           * - ``[(l_1,..,l_{L_1})]`` [1]
             - ``(l_1,..,l_{L_1})``
             - scalar return, tensor parameter
           * - ``[(l_1,..,l_{L_1})]`` [1]
             - ``(r_1,..,r_R, l_1,..,l_{L_1})``
             - tensor return, tensor parameter

        [1] Note that intuitively, ``tangent`` could be allowed to be a tensor of shape
        ``(l_1,..,l_{L_1})`` without an outer list. However, this is excluded in order
        to allow for the distinction from scenario **(B)**. Internally, this input shape for
        ``tangent`` never occurs for scenario **(A)**.

        In this scenario, the tangent is reshaped into a one-dimensional tensor with shape
        ``(tangent_size,)`` and the Jacobian is reshaped to have the dimensions
        ``(r_1, ... r_R, tangent_size)``. This is followed by a ``tensordot`` contraction over the
        ``tangent_size`` axis of both tensors.

        For scenario **(B)**, the input shapes can be in

        .. list-table::
           :widths: 30 40 30
           :header-rows: 1

           * - ``tangent`` shape
             - ``jac`` shape
             - Comment
           * - ``(k,)`` or ``[(),..,()]`` or ``((),..,())``
             - ``((),..,())`` (length ``k``)
             - scalar return, ``k`` scalar parameters
           * - ``(k,)`` or ``[(),..,()]`` or ``((),..,())``
             - ``((r_1,..,r_R),..,(r_1,..,r_R))`` [1]
             - tensor return, ``k`` scalar parameters
           * - ``[(l_1,..,l_{L_1}),..,(l_1,..,l_{L_k})]``
             - ``((l_1,..,l_{L_1}),..,(l_1,..,l_{L_k}))``
             - scalar return, ``k`` tensor parameters
           * - ``[(l_1,..,l_{L_1}),..,(l_1,..,l_{L_k})]``
             - ``((r_1,..,r_R, l_1,..,l_{L_1}),..,(r_1,..,r_R, l_1,..,l_{L_k}))`` [1]
             - tensor return, ``k`` tensor parameters

        [1] Note that the return type dimensions ``(r_1,..,r_R)`` are the same for all entries
        of ``jac``, whereas the dimensions of the entries in ``tanget``, and the according
        dimensions ``(l_1,..,l_{L_k})`` of the ``jac`` entries may differ.

        In this scenario, another case separation is used: If any of the parameters is a
        tensor (i.e. not a scalar), all tangent entries are reshaped into one-dimensional
        tensors with shapes ``(tangent_size_i,)`` and then stacked into one one-dimensional tensor.
        If there are no tensor parameters, the tangent is just stacked and reshaped.
        The Jacobians are reshaped to have the dimensions ``(r_1, ... r_R, tangent_size_i)``
        and then are concatenated along their last (potentially mismatching) axis.
        This is followed by a tensordot contraction over the axes of size
        :math:`\sum_i` ``tangent_size_i``.

    """
    if jac is None:
        return None
    single_param = not isinstance(jac, tuple)
    if (single_param and jac.shape == (0,)) or (not single_param and len(jac) == 0):
        # No trainable parameters
        return qml.math.zeros((1, 0))

    if single_param:
        tangent = qml.math.stack(tangent)
        first_tangent_ndim = len(tangent.shape[1:])
        tangent = qml.math.flatten(tangent)
        tangent_size = tangent.shape[0]
        shape = jac.shape
        new_shape = shape[: len(shape) - first_tangent_ndim] + (tangent_size,)
        jac = qml.math.cast(qml.math.convert_like(jac, tangent), tangent.dtype)
        jac = qml.math.reshape(jac, new_shape)
        return qml.math.tensordot(jac, tangent, [[-1], [0]])

    tangent_ndims = [getattr(t, "ndim", 0) for t in tangent]
    if isinstance(tangent, (tuple, list)) and any(ndim > 0 for ndim in tangent_ndims):
        # At least one tangent entry is not a scalar, requiring us to flatten them and hstack
        tangent = [qml.math.flatten(t) for t in tangent]
        tangent_sizes = [t.shape[0] for t in tangent]
        tangent = qml.math.hstack(tangent)
    else:
        # Only scalar tangent entries, no flattening required and we may use stack
        tangent_sizes = [1] * len(tangent)
        tangent = qml.math.stack(tangent)
    jac_shapes = [j.shape for j in jac]
    new_shapes = [
        shape[: len(shape) - t_ndim] + (tsize,)
        for shape, t_ndim, tsize in zip(jac_shapes, tangent_ndims, tangent_sizes)
    ]
    jac = qml.math.concatenate([qml.math.reshape(j, s) for j, s in zip(jac, new_shapes)], axis=-1)
    jac = qml.math.cast(qml.math.convert_like(jac, tangent), tangent.dtype)
    return qml.math.tensordot(jac, tangent, [[-1], [0]])


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
        (array([0.6]), array([0.4, 1. ]))

    2. For multiple parameters (in this case 2 parameters) and multiple measurements (one without shape and one with
    shape, e.g. expval and probs):

    .. code-block:: pycon

        >>> tangent = np.array([1.0, 2.0])
        >>> jac = tuple([tuple([np.array([0.3]), np.array([0.4])]), tuple([np.array([0.2, 0.5]), np.array([0.3, 0.8])]),])
        >>> qml.gradients.compute_jvp_multi(tangent, jac)
        (array([1.1]), array([0.8, 2.1]))
    """
    if jac is None:
        return None
    return tuple(compute_jvp_single(tangent, j) for j in jac)


def jvp(tape, tangent, gradient_fn, gradient_kwargs=None):
    r"""Generate the gradient tapes and processing function required to compute
    the Jacobian vector product of a tape. This function only works with the new return type system on.

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

    Consider the following quantum tape with Jax parameters:

    .. code-block:: python

        import jax

        x = jax.numpy.array([[0.1, 0.2, 0.3],
                             [0.4, 0.5, 0.6]])

        ops = [
            qml.RX(x[0, 0], wires=0),
            qml.RY(x[0, 1], wires=1),
            qml.RZ(x[0, 2], wires=0),
            qml.CNOT(wires=[0, 1]),
            qml.RX(x[1, 0], wires=1),
            qml.RY(x[1, 1], wires=0),
            qml.RZ(x[1, 2], wires=1)
        ]
        measurements = [qml.expval(qml.Z(0)), qml.probs(wires=1)]
        tape = qml.tape.QuantumTape(ops, measurements)

    We can use the ``jvp`` function to compute the Jacobian vector product,
    given a tangent vector ``tangent``:

    >>> tangent = [jax.numpy.array(1.0), jax.numpy.array(1.0), jax.numpy.array(1.0), jax.numpy.array(1.0), jax.numpy.array(1.0), jax.numpy.array(1.0)]
    >>> jvp_tapes, fn = qml.gradients.jvp(tape, tangent, qml.gradients.param_shift)

    Note that ``tangent`` has six elements, matching the parameter dimension of the tape.

    Executing the JVP tapes, and applying the processing function:

    >>> dev = qml.device("default.qubit")
    >>> jvp = fn(dev.execute(jvp_tapes))
    >>> jvp
    (Array(-0.62073968, dtype=float64),
     Array([-0.32597067,  0.32597067], dtype=float64))
    """
    if len(tape.trainable_params) == 0:
        # The tape has no trainable parameters; the JVP
        # is simply none.
        def zero_vjp(_):
            res = tuple(np.zeros(mp.shape(None, tape.shots)) for mp in tape.measurements)
            return res[0] if len(tape.measurements) == 1 else res

        return tuple(), zero_vjp

    multi_m = len(tape.measurements) > 1

    try:
        # if qml.math.allclose(qml.math.stack(tangent), 0):
        if qml.math.allclose(tangent, 0):
            # If the tangent vector is zero, then the
            # corresponding element of the JVP will be zero,
            # and we can avoid a quantum computation.

            def func(_):  # pylint: disable=unused-argument
                # TODO: Update shape for CV variables and for qutrit simulations
                res = tuple(_single_measurement_zero(m, tangent) for m in tape.measurements)
                if not multi_m:
                    res = res[0]
                return res

            return [], func
    except (AttributeError, TypeError):
        pass

    gradient_kwargs = gradient_kwargs or {}
    gradient_tapes, fn = gradient_fn(tape, **gradient_kwargs)

    def processing_fn(results):
        # postprocess results to compute the Jacobian
        jac = fn(results)
        _jvp_fn = compute_jvp_multi if multi_m else compute_jvp_single

        # Jacobian without shot vectors
        if not tape.shots.has_partitioned_shots:
            return _jvp_fn(tangent, jac)

        # The jacobian is calculated for shot vectors
        return tuple(_jvp_fn(tangent, jac[i]) for i in range(tape.shots.num_copies))

    return gradient_tapes, processing_fn


def batch_jvp(tapes, tangents, gradient_fn, reduction="append", gradient_kwargs=None):
    r"""Generate the gradient tapes and processing function required to compute
    the Jacobian vector products of a batch of tapes.

    Args:
        tapes (Sequence[.QuantumTape]): sequence of quantum tapes to differentiate
        tangents (Sequence[tensor_like]): Sequence of gradient-output vectors ``dy``. Must be the
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

    .. code-block:: python

        import jax
        x = jax.numpy.array([[0.1, 0.2, 0.3],
                             [0.4, 0.5, 0.6]])

        ops = [
            qml.RX(x[0, 0], wires=0),
            qml.RY(x[0, 1], wires=1),
            qml.RZ(x[0, 2], wires=0),
            qml.CNOT(wires=[0, 1]),
            qml.RX(x[1, 0], wires=1),
            qml.RY(x[1, 1], wires=0),
            qml.RZ(x[1, 2], wires=1)
        ]
        measurements1 = [qml.expval(qml.Z(0)), qml.probs(wires=1)]
        tape1 = qml.tape.QuantumTape(ops, measurements1)

        measurements2 = [qml.expval(qml.Z(0) @ qml.Z(1))]
        tape2 = qml.tape.QuantumTape(ops, measurements2)

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

    >>> dev = qml.device("default.qubit")
    >>> jvps = fn(dev.execute(jvp_tapes))
    >>> jvps
    ((Array(-0.62073968, dtype=float64),
      Array([-0.32597067,  0.32597067], dtype=float64)),
     Array(-0.690084, dtype=float64))

    We have two JVPs; one per tape. Each one corresponds to the shape of the output of their respective tape.
    """
    # pylint: disable=too-many-arguments
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

        return tuple(jvps)

    return gradient_tapes, processing_fn


def _single_measurement_zero(m, tangent):
    """Aux function to create a zero tensor from a measurement."""
    dim = 2 ** len(m.wires) if isinstance(m, ProbabilityMP) else ()
    res = qml.math.convert_like(np.zeros(dim), tangent)
    res = qml.math.cast_like(res, tangent)
    return res

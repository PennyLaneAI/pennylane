# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
This module contains functions for adding the JAX interface
to a PennyLane Device class.
"""

# pylint: disable=too-many-arguments, no-member
import jax
import jax.numpy as jnp
import numpy as np

import pennylane as qml
from pennylane.interfaces import InterfaceUnsupportedError
from pennylane.interfaces.jax import _raise_vector_valued_fwd
from pennylane.measurements import ProbabilityMP

from .jax import set_parameters_on_copy_and_unwrap

dtype = jnp.float64


def execute_legacy(
    tapes, device, execute_fn, gradient_fn, gradient_kwargs, _n=1, max_diff=1, mode=None
):
    """Execute a batch of tapes with JAX parameters on a device.

    Args:
        tapes (Sequence[.QuantumTape]): batch of tapes to execute
        device (pennylane.Device): Device to use to execute the batch of tapes.
            If the device does not provide a ``batch_execute`` method,
            by default the tapes will be executed in serial.
        execute_fn (callable): The execution function used to execute the tapes
            during the forward pass. This function must return a tuple ``(results, jacobians)``.
            If ``jacobians`` is an empty list, then ``gradient_fn`` is used to
            compute the gradients during the backwards pass.
        gradient_kwargs (dict): dictionary of keyword arguments to pass when
            determining the gradients of tapes
        gradient_fn (callable): the gradient function to use to compute quantum gradients
        _n (int): a positive integer used to track nesting of derivatives, for example
            if the nth-order derivative is requested.
        max_diff (int): If ``gradient_fn`` is a gradient transform, this option specifies
            the maximum order of derivatives to support. Increasing this value allows
            for higher order derivatives to be extracted, at the cost of additional
            (classical) computational overhead during the backwards pass.
        mode (str): Whether the gradients should be computed on the forward
            pass (``forward``) or the backward pass (``backward``).

    Returns:
        list[list[float]]: A nested list of tape results. Each element in
        the returned list corresponds in order to the provided tapes.
    """
    # pylint: disable=unused-argument
    if max_diff > 1:
        raise InterfaceUnsupportedError("The JAX interface only supports first order derivatives.")

    if _n == 1:
        for tape in tapes:
            # set the jitted parameters
            params = tape.get_parameters(trainable_only=False)
            trainable_params = set()
            for idx, p in enumerate(params):
                if isinstance(p, jax.core.Tracer) or qml.math.requires_grad(p):
                    trainable_params.add(idx)
            tape.trainable_params = trainable_params

    parameters = tuple(list(t.get_parameters()) for t in tapes)

    if gradient_fn is None:
        return _execute_with_fwd_legacy(
            parameters,
            tapes=tapes,
            device=device,
            execute_fn=execute_fn,
            gradient_kwargs=gradient_kwargs,
            _n=_n,
        )

    return _execute_legacy(
        parameters,
        tapes=tapes,
        device=device,
        execute_fn=execute_fn,
        gradient_fn=gradient_fn,
        gradient_kwargs=gradient_kwargs,
        _n=_n,
    )


def _numeric_type_to_dtype(numeric_type, device):
    """Auxiliary function for converting from Python numeric types to JAX
    dtypes based on the precision defined for the interface."""

    single_precision = dtype is jnp.float32
    if numeric_type is bool:
        if isinstance(device, qml.Device):
            numeric_type = int
        else:
            return jnp.bool_

    if numeric_type is int:
        return jnp.int32 if single_precision else jnp.int64

    if numeric_type is float:
        return jnp.float32 if single_precision else jnp.float64

    # numeric_type is complex
    return jnp.complex64 if single_precision else jnp.complex128


def _extract_shape_dtype_structs(tapes, device):
    """Auxiliary function for defining the jax.ShapeDtypeStruct objects given
    the tapes and the device.

    The host_callback.call function expects jax.ShapeDtypeStruct objects to
    describe the output of the function call.
    """
    shape_dtypes = []

    for t in tapes:
        shape = t.shape(device)

        tape_dtype = _numeric_type_to_dtype(t.numeric_type, device)
        shape_and_dtype = jax.ShapeDtypeStruct(tuple(shape), tape_dtype)

        shape_dtypes.append(shape_and_dtype)

    return shape_dtypes


def _execute_legacy(
    params,
    tapes=None,
    device=None,
    execute_fn=None,
    gradient_fn=None,
    gradient_kwargs=None,
    _n=1,
):  # pylint: disable=dangerous-default-value,unused-argument
    total_params = np.sum([len(p) for p in params])

    @jax.custom_vjp
    def wrapped_exec(params):
        result_shapes_dtypes = _extract_shape_dtype_structs(tapes, device)

        def wrapper(p):
            """Compute the forward pass."""
            new_tapes = set_parameters_on_copy_and_unwrap(tapes, p)
            res, _ = execute_fn(new_tapes, **gradient_kwargs)

            # When executed under `jax.vmap` the `result_shapes_dtypes` will contain
            # the shape without the vmap dimensions, while the function here will be
            # executed with objects containing the vmap dimensions. So res[i].ndim
            # will have an extra dimension for every `jax.vmap` wrapping this execution.
            #
            # The execute_fn will return an object with shape `(n_observables, batches)`
            # but the default behaviour for `jax.pure_callback` is to add the extra
            # dimension at the beginning, so `(batches, n_observables)`. So in here
            # we detect with the heuristic above if we are executing under vmap and we
            # swap the order in that case.

            # pylint: disable=consider-using-enumerate
            for i in range(len(res)):
                if res[i].ndim > result_shapes_dtypes[i].ndim:
                    res[i] = res[i].T
            return res

        res = jax.pure_callback(wrapper, result_shapes_dtypes, params, vectorized=True)
        return res

    def wrapped_exec_fwd(params):
        return wrapped_exec(params), params

    def wrapped_exec_bwd(params, g):
        if isinstance(gradient_fn, qml.gradients.gradient_transform):
            for t in tapes:
                multi_probs = (
                    any(isinstance(m, ProbabilityMP) for m in t.measurements)
                    and len(t.measurements) > 1
                )

                if multi_probs:
                    raise InterfaceUnsupportedError(
                        "The JAX-JIT interface doesn't support differentiating QNodes that "
                        "return multiple probabilities."
                    )

            def non_diff_wrapper(args):
                """Compute the VJP in a non-differentiable manner."""
                p = args[:-1]
                dy = args[-1]

                new_tapes = set_parameters_on_copy_and_unwrap(tapes, p, unwrap=False)
                vjp_tapes, processing_fn = qml.gradients.batch_vjp(
                    new_tapes,
                    dy,
                    gradient_fn,
                    reduction="append",
                    gradient_kwargs=gradient_kwargs,
                )

                partial_res = execute_fn(vjp_tapes)[0]
                res = processing_fn(partial_res)
                return np.concatenate(res)

            args = tuple(params) + (g,)
            vjps = jax.pure_callback(
                non_diff_wrapper,
                jax.ShapeDtypeStruct((total_params,), dtype),
                args,
            )

            param_idx = 0
            res = []

            # Group the vjps based on the parameters of the tapes
            for p in params:
                param_vjp = vjps[param_idx : param_idx + len(p)]
                res.append(param_vjp)
                param_idx += len(p)

            # Unwrap partial results into ndim=0 arrays to allow
            # differentiability with JAX
            # E.g.,
            # [Array([-0.9553365], dtype=float32), Array([0., 0.],
            # dtype=float32)]
            # is mapped to
            # [[Array(-0.9553365, dtype=float32)], [Array(0.,
            # dtype=float32), Array(0., dtype=float32)]].
            need_unstacking = any(r.ndim != 0 for r in res)
            if need_unstacking:
                res = [qml.math.unstack(x) for x in res]

            return (tuple(res),)

        def jacs_wrapper(p):
            """Compute the jacs"""
            new_tapes = set_parameters_on_copy_and_unwrap(tapes, p)
            jacs = gradient_fn(new_tapes, **gradient_kwargs)
            return jacs

        shapes = [
            jax.ShapeDtypeStruct((len(t.measurements), len(p)), dtype)
            for t, p in zip(tapes, params)
        ]
        jacs = jax.pure_callback(jacs_wrapper, shapes, params)
        vjps = [qml.gradients.compute_vjp(d, jac) for d, jac in zip(g, jacs)]
        res = [[jnp.array(p) for p in v] for v in vjps]
        return (tuple(res),)

    wrapped_exec.defvjp(wrapped_exec_fwd, wrapped_exec_bwd)
    return wrapped_exec(params)


# The execute function in forward mode
def _execute_with_fwd_legacy(
    params,
    tapes=None,
    device=None,
    execute_fn=None,
    gradient_kwargs=None,
    _n=1,
):  # pylint: disable=dangerous-default-value,unused-argument
    @jax.custom_vjp
    def wrapped_exec(params):
        def wrapper(p):
            """Compute the forward pass by returning the jacobian too."""
            new_tapes = []

            for t, a in zip(tapes, p):
                new_tapes.append(t.copy(copy_operations=True))
                new_tapes[-1].set_parameters(a)

            res, jacs = execute_fn(new_tapes, **gradient_kwargs)

            # On the forward execution return the jacobian too
            return res, jacs

        fwd_shape_dtype_struct = _extract_shape_dtype_structs(tapes, device)

        # params should be the parameters for each tape queried prior, assert
        # to double-check
        assert len(tapes) == len(params)

        jacobian_shape = []
        for t, p in zip(tapes, params):
            shape = t.shape(device) + (len(p),)
            _dtype = _numeric_type_to_dtype(t.numeric_type, device)
            shape = [shape] if isinstance(shape, int) else shape
            o = jax.ShapeDtypeStruct(tuple(shape), _dtype)
            jacobian_shape.append(o)

        res, jacs = jax.pure_callback(
            wrapper, tuple([fwd_shape_dtype_struct, jacobian_shape]), params
        )
        return res, jacs

    def wrapped_exec_fwd(params):
        res, jacs = wrapped_exec(params)
        return res, tuple([jacs, params])

    def wrapped_exec_bwd(params, g):
        _raise_vector_valued_fwd(tapes)

        # Use the jacobian that was computed on the forward pass
        jacs, params = params

        # Adjust the structure of how the jacobian is returned to match the
        # non-forward mode cases
        # E.g.,
        # [Array([[ 0.06695931,  0.01383095, -0.46500877]], dtype=float32)]
        # is mapped to
        # [[Array(0.06695931, dtype=float32), Array(0.01383095,
        # dtype=float32), Array(-0.46500877, dtype=float32)]]
        res_jacs = []
        for j in jacs:
            this_j = []
            for i in range(j.shape[1]):
                this_j.append(j[0, i])
            res_jacs.append(this_j)
        return tuple([tuple(res_jacs)])

    wrapped_exec.defvjp(wrapped_exec_fwd, wrapped_exec_bwd)
    res = wrapped_exec(params)

    tracing = any(isinstance(r, jax.interpreters.ad.JVPTracer) for r in res)

    # When there are no tracers (not differentiating), we have the result of
    # the forward pass and the jacobian, but only need the result of the
    # forward pass
    if len(res) == 2 and not tracing:
        res = res[0]

    return res

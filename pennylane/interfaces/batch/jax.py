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
# pylint: disable=too-many-arguments
import jax
import jax.numpy as jnp

import pennylane as qml
from pennylane.operation import Sample, Probability
from pennylane.interfaces.batch import InterfaceUnsupportedError

dtype = jnp.float64


def get_jax_interface_name(tapes):
    """Check all parameters in each tape and output the name of the suitable
    JAX interface.

    This function checks each tape and determines if any of the gate parameters
    was transformed by a JAX transform such as ``jax.jit``. If so, it outputs
    the name of the JAX interface with jit support.

    Note that determining if jit support should be turned on is done by
    checking if parameters are abstract. Parameters can be abstract not just
    for ``jax.jit``, but for other JAX transforms (vmap, pmap, etc.) too. The
    reason is that JAX doesn't have a public API for checking whether or not
    the execution is within the jit transform.

    Args:
        tapes (Sequence[.QuantumTape]): batch of tapes to execute

    Returns:
        str: name of JAX interface that fits the tape parameters, "jax" or
        "jax-jit"
    """
    for t in tapes:
        for op in t.operations:
            for param in op.data:
                if qml.math.is_abstract(param):
                    return "jax-jit"

    return "jax"


def execute(tapes, device, execute_fn, gradient_fn, gradient_kwargs, _n=1, max_diff=1, mode=None):
    """Execute a batch of tapes with JAX parameters on a device.

    Args:
        tapes (Sequence[.QuantumTape]): batch of tapes to execute
        device (.Device): Device to use to execute the batch of tapes.
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

    _validate_tapes(tapes)

    for tape in tapes:
        # set the trainable parameters
        params = tape.get_parameters(trainable_only=False)
        tape.trainable_params = qml.math.get_trainable_indices(params)

    parameters = tuple(list(t.get_parameters()) for t in tapes)

    if gradient_fn is None:
        return _execute_with_fwd(
            parameters,
            tapes=tapes,
            device=device,
            execute_fn=execute_fn,
            gradient_kwargs=gradient_kwargs,
            _n=_n,
        )

    return _execute(
        parameters,
        tapes=tapes,
        device=device,
        execute_fn=execute_fn,
        gradient_fn=gradient_fn,
        gradient_kwargs=gradient_kwargs,
        _n=_n,
    )


def _validate_tapes(tapes):
    """Validates that the input tapes are compatible with JAX support.

    Note: the goal of this validation is to filter out cases where ragged
    outputs for QNodes may arise. Such QNodes involve creating arrays from
    ragged nested sequences that can not be handled by JAX.

    Raises:
        InterfaceUnsupportedError: if tapes that produce ragged outputs were provided
    """
    for t in tapes:

        return_types = [o.return_type for o in t.observables]
        set_of_return_types = set(return_types)
        probs_or_sample_measure = Sample in return_types or Probability in return_types
        if probs_or_sample_measure and len(set_of_return_types) > 1:
            raise InterfaceUnsupportedError(
                "Using the JAX interface, sample and probability measurements cannot be mixed with other measurement types."
            )

        if Probability in return_types:
            set_len_wires = set(len(o.wires) for o in t.observables)
            if len(set_len_wires) > 1:
                raise InterfaceUnsupportedError(
                    "Using the JAX interface, multiple probability measurements need to have the same number of wires specified."
                )


def _execute(
    params,
    tapes=None,
    device=None,
    execute_fn=None,
    gradient_fn=None,
    gradient_kwargs=None,
    _n=1,
):  # pylint: disable=dangerous-default-value,unused-argument
    """The main interface execution function where jacobians of the execute
    function are computed by the registered backward function."""

    @jax.custom_vjp
    def wrapped_exec(params):
        new_tapes = []

        for t, a in zip(tapes, params):
            new_tapes.append(t.copy(copy_operations=True))
            new_tapes[-1].set_parameters(a)

        with qml.tape.Unwrap(*new_tapes):
            res, _ = execute_fn(new_tapes, **gradient_kwargs)

        if len(tapes) > 1:
            res = [jnp.array(r) for r in res]
        else:
            res = jnp.array(res)

        return res

    def wrapped_exec_fwd(params):
        return wrapped_exec(params), params

    def wrapped_exec_bwd(params, g):

        if isinstance(gradient_fn, qml.gradients.gradient_transform):

            args = tuple(params) + (g,)
            new_tapes = []
            p = args[:-1]
            dy = args[-1]

            for t, a in zip(tapes, p):
                new_tapes.append(t.copy(copy_operations=True))
                new_tapes[-1].set_parameters(a)
                new_tapes[-1].trainable_params = t.trainable_params

            vjp_tapes, processing_fn = qml.gradients.batch_vjp(
                new_tapes,
                dy,
                gradient_fn,
                reduction="append",
                gradient_kwargs=gradient_kwargs,
            )

            partial_res = execute_fn(vjp_tapes)[0]

            for t in tapes:
                multi_probs = (
                    any(o.return_type is Probability for o in t.observables)
                    and len(t.observables) > 1
                )

            if multi_probs:
                # For multiple probability measurements, adjust the
                # rows/columns in the result to match other interfaces
                new_partial_res = []
                for r in partial_res:
                    if r.ndim > 1:
                        new_partial_res.append(r.swapaxes(0, 1))
                    else:
                        new_partial_res.append(r)
                partial_res = new_partial_res

            res = processing_fn(partial_res)
            vjps = jnp.concatenate(res)

            param_idx = 0
            res = []

            # Group the vjps based on the parameters of the tapes
            for p in params:
                param_vjp = vjps[param_idx : param_idx + len(p)]
                res.append(param_vjp)
                param_idx += len(p)

            # Unstack partial results into ndim=0 arrays to allow
            # differentiability with JAX
            # E.g.,
            # [DeviceArray([-0.9553365], dtype=float32), DeviceArray([0., 0.],
            # dtype=float32)]
            # is mapped to
            # [[DeviceArray(-0.9553365, dtype=float32)], [DeviceArray(0.,
            # dtype=float32), DeviceArray(0., dtype=float32)]].
            need_unstacking = any(r.ndim != 0 for r in res)
            if need_unstacking:
                res = [qml.math.unstack(x) for x in res]

            return (tuple(res),)

        # Gradient function is a device method.
        with qml.tape.Unwrap(*tapes):
            jacs = gradient_fn(tapes, **gradient_kwargs)

        vjps = [qml.gradients.compute_vjp(d, jac) for d, jac in zip(g, jacs)]
        res = [[jnp.array(p) for p in v] for v in vjps]
        return (tuple(res),)

    wrapped_exec.defvjp(wrapped_exec_fwd, wrapped_exec_bwd)
    return wrapped_exec(params)


def _execute_with_fwd(
    params,
    tapes=None,
    device=None,
    execute_fn=None,
    gradient_kwargs=None,
    _n=1,
):  # pylint: disable=dangerous-default-value,unused-argument
    """The auxiliary execute function for cases when the user requested
    jacobians to be computed in forward mode or when no gradient function was
    provided."""

    @jax.custom_vjp
    def wrapped_exec(params):
        new_tapes = []

        for t, a in zip(tapes, params):
            new_tapes.append(t.copy(copy_operations=True))
            new_tapes[-1].set_parameters(a)

        with qml.tape.Unwrap(*new_tapes):
            res, jacs = execute_fn(new_tapes, **gradient_kwargs)

        if len(tapes) > 1:
            res, jacs = [jnp.array(r) for r in res], [jnp.array(j) for j in jacs]
        else:
            res, jacs = jnp.array(res), jnp.array(jacs)
        return res, jacs

    def wrapped_exec_fwd(params):
        res, jacs = wrapped_exec(params)
        return res, tuple([jacs, params])

    def wrapped_exec_bwd(params, g):

        # Use the jacobian that was computed on the forward pass
        jacs, params = params

        # Note: there is an issue when jax.jacobian is being used, either due
        # to issues with tensor updating (TypeError: Updates tensor must be of
        # rank 0; got 1) or because jax.vmap introduces a redundant
        # dimensionality in the result by duplicating entries
        # Example to the latter:
        #
        # 1. Output when using jax.jacobian:
        # DeviceArray([[-0.09983342,  0.01983384],\n
        #              [-0.09983342, 0.01983384]], dtype=float64),
        # DeviceArray([[ 0.        , -0.97517033],\n
        #              [ 0.        , -0.97517033]], dtype=float64)),
        #
        # 2. Expected output:
        # DeviceArray([[-0.09983342, 0.01983384],\n
        #             [ 0.        , -0.97517033]]
        #
        # The output produced by this function matches 2.
        scalar_outputs = all(t.output_dim == 1 for t in tapes)
        if not scalar_outputs:
            raise InterfaceUnsupportedError(
                "Computing the jacobian of vector-valued tapes is not supported currently in forward mode."
            )

        # Adjust the structure of how the jacobian is returned to match the
        # non-forward mode cases
        # E.g.,
        # [DeviceArray([[ 0.06695931,  0.01383095, -0.46500877]], dtype=float32)]
        # is mapped to
        # [[DeviceArray(0.06695931, dtype=float32), DeviceArray(0.01383095,
        # dtype=float32), DeviceArray(-0.46500877, dtype=float32)]]
        res_jacs = []
        for j in jacs:
            this_j = []
            for i in range(j.shape[1]):
                arr = (
                    j[0, i] if j.shape[0] == 1 else jnp.array([j[k, i] for k in range(j.shape[0])])
                )
                this_j.append(arr)
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

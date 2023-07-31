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
This module contains functions for adding the PyTorch interface
to a PennyLane Device class.
"""
# pylint: disable=too-many-arguments,protected-access,abstract-method
import numpy as np
import torch
import torch.utils._pytree as pytree

import pennylane as qml
from pennylane.measurements import CountsMP
from pennylane.transforms import convert_to_numpy_parameters


def _compute_vjp(dy, jacs, device=None):
    vjps = []

    for d, jac in zip(dy, jacs):
        if isinstance(jac, np.ndarray):
            jac = torch.from_numpy(jac)

        jac = torch.as_tensor(jac, device=device)
        vjp = qml.gradients.compute_vjp(d, jac)
        vjps.extend(vjp)

    return vjps


class ExecuteTapesLegacy(torch.autograd.Function):
    """The signature of this ``torch.autograd.Function`` is designed to
    work around Torch restrictions.

    In particular, ``torch.autograd.Function``:

    - Cannot accept keyword arguments. As a result, we pass a dictionary
      as the first argument ``kwargs``. This dictionary **must** contain:

      * ``"tapes"``: the quantum tapes to batch evaluate
      * ``"device"``: the quantum device to use to evaluate the tapes
      * ``"execute_fn"``: the execution function to use on forward passes
      * ``"gradient_fn"``: the gradient transform function to use
        for backward passes
      * ``"gradient_kwargs"``: gradient keyword arguments to pass to the
        gradient function
      * ``"max_diff``: the maximum order of derivatives to support

    Further, note that the ``parameters`` argument is dependent on the
    ``tapes``; this function should always be called
    with the parameters extracted directly from the tapes as follows:

    >>> parameters = []
    >>> [parameters.extend(t.get_parameters()) for t in tapes]
    >>> kwargs = {"tapes": tapes, "device": device, "gradient_fn": gradient_fn, ...}
    >>> ExecuteTapes.apply(kwargs, *parameters)

    The private argument ``_n`` is used to track nesting of derivatives, for example
    if the nth-order derivative is requested. Do not set this argument unless you
    understand the consequences!
    """

    @staticmethod
    def forward(ctx, kwargs, *parameters):  # pylint: disable=arguments-differ
        """Implements the forward pass batch tape evaluation."""
        ctx.tapes = kwargs["tapes"]
        ctx.device = kwargs["device"]

        ctx.execute_fn = kwargs["execute_fn"]
        ctx.gradient_fn = kwargs["gradient_fn"]

        ctx.gradient_kwargs = kwargs["gradient_kwargs"]
        ctx.max_diff = kwargs["max_diff"]
        ctx._n = kwargs.get("_n", 1)

        unwrapped_tapes = tuple(convert_to_numpy_parameters(t) for t in ctx.tapes)
        res, ctx.jacs = ctx.execute_fn(unwrapped_tapes, **ctx.gradient_kwargs)

        # if any input tensor uses the GPU, the output should as well
        ctx.torch_device = None

        for p in parameters:
            if isinstance(p, torch.Tensor) and p.is_cuda:  # pragma: no cover
                ctx.torch_device = p.get_device()
                break

        for i, r in enumerate(res):
            if isinstance(r, np.ndarray) and r.dtype is np.dtype("object"):
                # For backwards compatibility, we flatten ragged tape outputs
                r = np.hstack(r)

            if any(isinstance(m, CountsMP) for m in ctx.tapes[i].measurements):
                continue

            if isinstance(r, (list, tuple)):
                res[i] = [torch.as_tensor(t) for t in r]

                if isinstance(r, tuple):
                    res[i] = tuple(res[i])
            else:
                res[i] = torch.as_tensor(r, device=ctx.torch_device)

            if ctx.jacs:
                ctx.jacs[i] = torch.as_tensor(ctx.jacs[i], device=ctx.torch_device)

        return tuple(res)

    @staticmethod
    def backward(ctx, *dy):
        """Returns the vector-Jacobian product with given
        parameter values p and output gradient dy"""

        if ctx.jacs:
            # Jacobians were computed on the forward pass (mode="forward")
            # No additional quantum evaluations needed; simply compute the VJPs directly.
            vjps = _compute_vjp(dy, ctx.jacs)

        else:
            # Need to compute the Jacobians on the backward pass (accumulation="backward")

            if isinstance(ctx.gradient_fn, qml.gradients.gradient_transform):
                # Gradient function is a gradient transform.

                # Generate and execute the required gradient tapes
                if ctx._n < ctx.max_diff:
                    # The derivative order is less than the max derivative order.
                    # Compute the VJP recursively by using the gradient transform
                    # and calling ``execute`` to compute the results.
                    # This will allow higher-order derivatives to be computed
                    # if requested.

                    vjp_tapes, processing_fn = qml.gradients.batch_vjp(
                        ctx.tapes,
                        dy,
                        ctx.gradient_fn,
                        reduction="extend",
                        gradient_kwargs=ctx.gradient_kwargs,
                    )

                    # This is where the magic happens. Note that we call ``execute``.
                    # This recursion, coupled with the fact that the gradient transforms
                    # are differentiable, allows for arbitrary order differentiation.
                    vjps = processing_fn(
                        execute(
                            vjp_tapes,
                            ctx.device,
                            ctx.execute_fn,
                            ctx.gradient_fn,
                            ctx.gradient_kwargs,
                            _n=ctx._n + 1,
                            max_diff=ctx.max_diff,
                        )
                    )
                else:
                    # The derivative order is at the maximum. Compute the VJP
                    # in a non-differentiable manner to reduce overhead.

                    unwrapped_tapes = tuple(convert_to_numpy_parameters(t) for t in ctx.tapes)
                    vjp_tapes, processing_fn = qml.gradients.batch_vjp(
                        unwrapped_tapes,
                        dy,
                        ctx.gradient_fn,
                        reduction="extend",
                        gradient_kwargs=ctx.gradient_kwargs,
                    )

                    vjps = processing_fn(ctx.execute_fn(vjp_tapes)[0])

            else:
                # Gradient function is not a gradient transform
                # (e.g., it might be a device method).
                # Note that unlike the previous branch:
                #
                # - there is no recursion here
                # - gradient_fn is not differentiable
                #
                # so we cannot support higher-order derivatives.

                unwrapped_tapes = tuple(convert_to_numpy_parameters(t) for t in ctx.tapes)
                jacs = ctx.gradient_fn(unwrapped_tapes, **ctx.gradient_kwargs)

                vjps = _compute_vjp(dy, jacs, device=ctx.torch_device)

        # The output of backward must match the input of forward.
        # Therefore, we return `None` for the gradient of `kwargs`.
        return (None,) + tuple(vjps)


def _execute_legacy(tapes, device, execute_fn, gradient_fn, gradient_kwargs, _n=1, max_diff=2):
    """Execute a batch of tapes with Torch parameters on a device.

    This function may be called recursively, if ``gradient_fn`` is a differentiable
    transform, and ``_n < max_diff``.

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

    Returns:
        list[list[torch.Tensor]]: A nested list of tape results. Each element in
        the returned list corresponds in order to the provided tapes.
    """

    parameters = []
    for tape in tapes:
        # set the trainable parameters
        params = tape.get_parameters(trainable_only=False)
        tape.trainable_params = qml.math.get_trainable_indices(params)
        parameters.extend(tape.get_parameters())

    kwargs = {
        "tapes": tapes,
        "device": device,
        "execute_fn": execute_fn,
        "gradient_fn": gradient_fn,
        "gradient_kwargs": gradient_kwargs,
        "_n": _n,
        "max_diff": max_diff,
    }
    return ExecuteTapesLegacy.apply(kwargs, *parameters)


def pytreeify(cls):
    """Pytrees refer to a tree-like structure built out of container-like Python objects. The pytreeify class is used
    to bypass some PyTorch limitation of `autograd.Function`. The forward pass can only return tuple of tensors but
    not any other nested structure. This class apply flatten to the forward pass and unflatten the results in the
    apply function. In this way, it is possible to treat multiple tapes with multiple measurements.
    """
    orig_fw = cls.forward
    orig_bw = cls.backward
    orig_apply = cls.apply

    def new_apply(*inp):
        # Inputs already flat
        out_struct_holder = []
        flat_out = orig_apply(out_struct_holder, *inp)
        return pytree.tree_unflatten(flat_out, out_struct_holder[0])

    def new_forward(ctx, out_struct_holder, *inp):
        out = orig_fw(ctx, *inp)
        flat_out, out_struct = pytree.tree_flatten(out)
        ctx._out_struct = out_struct
        out_struct_holder.append(out_struct)
        return tuple(flat_out)

    def new_backward(ctx, *flat_grad_outputs):
        grad_outputs = pytree.tree_unflatten(flat_grad_outputs, ctx._out_struct)
        grad_inputs = orig_bw(ctx, *grad_outputs)
        # None corresponds to the diff of out_struct_holder
        return (None,) + tuple(grad_inputs)

    cls.apply = new_apply
    cls.forward = new_forward
    cls.backward = new_backward
    return cls


def _compute_vjps(dys, jacs, multi_measurements):
    """Compute the vjps of multiple tapes, directly for a Jacobian and tangents."""
    vjps = []

    for i, multi in enumerate(multi_measurements):
        compute_func = (
            qml.gradients.compute_vjp_multi if multi else qml.gradients.compute_vjp_single
        )
        vjps.extend(compute_func(dys[i], jacs[i]))
    return vjps


@pytreeify
class ExecuteTapes(torch.autograd.Function):
    """The signature of this ``torch.autograd.Function`` is designed to
    work around Torch restrictions.

    In particular, ``torch.autograd.Function``:

    - Cannot accept keyword arguments. As a result, we pass a dictionary
      as the first argument ``kwargs``. This dictionary **must** contain:

      * ``"tapes"``: the quantum tapes to batch evaluate
      * ``"device"``: the quantum device to use to evaluate the tapes
      * ``"execute_fn"``: the execution function to use on forward passes
      * ``"gradient_fn"``: the gradient transform function to use
        for backward passes
      * ``"gradient_kwargs"``: gradient keyword arguments to pass to the
        gradient function
      * ``"max_diff``: the maximum order of derivatives to support

    Further, note that the ``parameters`` argument is dependent on the
    ``tapes``; this function should always be called
    with the parameters extracted directly from the tapes as follows:

    >>> parameters = []
    >>> [parameters.extend(t.get_parameters()) for t in tapes]
    >>> kwargs = {"tapes": tapes, "device": device, "gradient_fn": gradient_fn, ...}
    >>> ExecuteTapes.apply(kwargs, *parameters)

    The private argument ``_n`` is used to track nesting of derivatives, for example
    if the nth-order derivative is requested. Do not set this argument unless you
    understand the consequences!
    """

    @staticmethod
    def forward(ctx, kwargs, *parameters):  # pylint: disable=arguments-differ
        """Implements the forward pass batch tape evaluation."""
        ctx.tapes = kwargs["tapes"]
        ctx.device = kwargs["device"]

        ctx.execute_fn = kwargs["execute_fn"]
        ctx.gradient_fn = kwargs["gradient_fn"]

        ctx.gradient_kwargs = kwargs["gradient_kwargs"]
        ctx.max_diff = kwargs["max_diff"]
        ctx._n = kwargs.get("_n", 1)

        res, ctx.jacs = ctx.execute_fn(ctx.tapes, **ctx.gradient_kwargs)

        # if any input tensor uses the GPU, the output should as well
        ctx.torch_device = None

        for p in parameters:
            if isinstance(p, torch.Tensor) and p.is_cuda:  # pragma: no cover
                ctx.torch_device = p.get_device()
                break

        res = tuple(_res_to_torch(r, ctx) for r in res)
        for i, _ in enumerate(res):
            # In place change of the numpy array Jacobians to Torch objects
            _jac_to_torch(i, ctx)

        return res

    @staticmethod
    def backward(ctx, *dy):
        """Returns the vector-Jacobian product with given
        parameter values p and output gradient dy"""
        multi_measurements = [len(tape.measurements) > 1 for tape in ctx.tapes]

        if ctx.jacs:
            # Jacobians were computed on the forward pass (mode="forward")
            # No additional quantum evaluations needed; simply compute the VJPs directly.
            vjps = _compute_vjps(dy, ctx.jacs, multi_measurements)

        else:
            # Need to compute the Jacobians on the backward pass (accumulation="backward")

            if isinstance(ctx.gradient_fn, qml.gradients.gradient_transform):
                # Gradient function is a gradient transform.

                # Generate and execute the required gradient tapes
                if ctx._n < ctx.max_diff:
                    # The derivative order is less than the max derivative order.
                    # Compute the VJP recursively by using the gradient transform
                    # and calling ``execute`` to compute the results.
                    # This will allow higher-order derivatives to be computed
                    # if requested.

                    vjp_tapes, processing_fn = qml.gradients.batch_vjp(
                        ctx.tapes,
                        dy,
                        ctx.gradient_fn,
                        reduction="extend",
                        gradient_kwargs=ctx.gradient_kwargs,
                    )
                    # This is where the magic happens. Note that we call ``execute``.
                    # This recursion, coupled with the fact that the gradient transforms
                    # are differentiable, allows for arbitrary order differentiation.
                    res = execute(
                        vjp_tapes,
                        ctx.device,
                        ctx.execute_fn,
                        ctx.gradient_fn,
                        ctx.gradient_kwargs,
                        _n=ctx._n + 1,
                        max_diff=ctx.max_diff,
                    )
                    vjps = processing_fn(res)

                else:
                    # The derivative order is at the maximum. Compute the VJP
                    # in a non-differentiable manner to reduce overhead.
                    vjp_tapes, processing_fn = qml.gradients.batch_vjp(
                        ctx.tapes,
                        dy,
                        ctx.gradient_fn,
                        reduction="extend",
                        gradient_kwargs=ctx.gradient_kwargs,
                    )

                    vjps = processing_fn(ctx.execute_fn(vjp_tapes)[0])

            else:
                # Gradient function is not a gradient transform
                # (e.g., it might be a device method).
                # Note that unlike the previous branch:
                #
                # - there is no recursion here
                # - gradient_fn is not differentiable
                #
                # so we cannot support higher-order derivatives.

                jacs = ctx.gradient_fn(ctx.tapes, **ctx.gradient_kwargs)

                vjps = _compute_vjps(dy, jacs, multi_measurements)

        # Remove empty vjps (from tape with non trainable params)
        vjps = [vjp for vjp in vjps if list(vjp.shape) != [0]]
        # The output of backward must match the input of forward.
        # Therefore, we return `None` for the gradient of `kwargs`.
        return (None,) + tuple(vjps)


def execute(tapes, device, execute_fn, gradient_fn, gradient_kwargs, _n=1, max_diff=1):
    """Execute a batch of tapes with Torch parameters on a device.
    This function may be called recursively, if ``gradient_fn`` is a differentiable
    transform, and ``_n < max_diff``.

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
    Returns:
        list[list[torch.Tensor]]: A nested list of tape results. Each element in
        the returned list corresponds in order to the provided tapes.
    """
    # pylint: disable=unused-argument
    parameters = []
    for tape in tapes:
        # set the trainable parameters
        params = tape.get_parameters(trainable_only=False)
        tape.trainable_params = qml.math.get_trainable_indices(params)
        parameters.extend(tape.get_parameters())

    kwargs = {
        "tapes": tapes,
        "device": device,
        "execute_fn": execute_fn,
        "gradient_fn": gradient_fn,
        "gradient_kwargs": gradient_kwargs,
        "_n": _n,
        "max_diff": max_diff,
    }

    return ExecuteTapes.apply(kwargs, *parameters)


def _res_to_torch(r, ctx):
    """Convert results from unwrapped execution to torch."""
    if isinstance(r, (list, tuple)):
        res = []
        for t in r:
            if isinstance(t, dict) or isinstance(t, list) and all(isinstance(i, dict) for i in t):
                # count result, single or broadcasted
                res.append(t)
            else:
                if isinstance(t, tuple):
                    res.append(tuple(torch.as_tensor(el, device=ctx.torch_device) for el in t))
                else:
                    res.append(torch.as_tensor(t, device=ctx.torch_device))
        if isinstance(r, tuple):
            res = tuple(res)
    elif isinstance(r, dict):
        res = r
    else:
        res = torch.as_tensor(r, device=ctx.torch_device)

    return res


def _jac_to_torch(i, ctx):
    """Convert Jacobian from unwrapped execution to torch in the given ctx."""
    if ctx.jacs:
        ctx_jacs = list(ctx.jacs)
        multi_m = len(ctx.tapes[i].measurements) > 1
        multi_p = len(ctx.tapes[i].trainable_params) > 1

        # Multiple measurements and parameters: Jacobian is a tuple of tuple
        if multi_p and multi_m:
            jacobians = []
            for jacobian in ctx_jacs[i]:
                inside_nested_jacobian = [
                    torch.as_tensor(j, device=ctx.torch_device) for j in jacobian
                ]
                inside_nested_jacobian_tuple = tuple(inside_nested_jacobian)
                jacobians.append(inside_nested_jacobian_tuple)
            ctx_jacs[i] = tuple(jacobians)
        # Single measurement and single parameter: Jacobian is a tensor
        elif not multi_p and not multi_m:
            ctx_jacs[i] = torch.as_tensor(np.array(ctx_jacs[i]), device=ctx.torch_device)
        # Multiple measurements or multiple parameters: Jacobian is a tuple
        else:
            jacobian = [torch.as_tensor(jac, device=ctx.torch_device) for jac in ctx_jacs[i]]
            ctx_jacs[i] = tuple(jacobian)
        ctx.jacs = tuple(ctx_jacs)

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
This module contains functions for adding the PyTorch interface
to a PennyLane Device class.
"""
# pylint: disable=too-many-arguments,protected-access,abstract-method
import numpy as np
import torch

import pennylane as qml


def _compute_vjp(dy, jacs, device=None):
    vjps = []

    for d, jac in zip(dy, jacs):
        if isinstance(jac, np.ndarray):
            jac = torch.from_numpy(jac)

        jac = torch.as_tensor(jac, device=device)
        vjp = qml.gradients.compute_vjp(d, jac)
        vjps.extend(vjp)

    return vjps


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

        with qml.tape.Unwrap(*ctx.tapes):
            res, ctx.jacs = ctx.execute_fn(ctx.tapes, **ctx.gradient_kwargs)

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

            if any(m.return_type is qml.measurements.Counts for m in ctx.tapes[i].measurements):
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

                    with qml.tape.Unwrap(*ctx.tapes):
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

                with qml.tape.Unwrap(*ctx.tapes):
                    jacs = ctx.gradient_fn(ctx.tapes, **ctx.gradient_kwargs)

                vjps = _compute_vjp(dy, jacs, device=ctx.torch_device)

        # The output of backward must match the input of forward.
        # Therefore, we return `None` for the gradient of `kwargs`.
        return (None,) + tuple(vjps)


def execute(tapes, device, execute_fn, gradient_fn, gradient_kwargs, _n=1, max_diff=2, mode=None):
    """Execute a batch of tapes with Torch parameters on a device.

    This function may be called recursively, if ``gradient_fn`` is a differentiable
    transform, and ``_n < max_diff``.

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

    kwargs = dict(
        tapes=tapes,
        device=device,
        execute_fn=execute_fn,
        gradient_fn=gradient_fn,
        gradient_kwargs=gradient_kwargs,
        _n=_n,
        max_diff=max_diff,
    )
    return ExecuteTapes.apply(kwargs, *parameters)

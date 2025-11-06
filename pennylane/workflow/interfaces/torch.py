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

**How to bind a custom derivative with Torch.**

See `the Torch documentation <https://pytorch.org/docs/stable/notes/extending.html>`_ for more complete
information.

Suppose I have a function ``f`` that I want to define a custom vjp for.

We need to inherit from ``torch.autograd.Function`` and define ``forward`` and ``backward`` static
methods.

.. code-block:: python

    class CustomFunction(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x, exponent=2):
            ctx.saved_info = {'x': x, 'exponent': exponent}
            return x ** exponent

        @staticmethod
        def backward(ctx, dy):
            x = ctx.saved_info['x']
            exponent = ctx.saved_info['exponent']
            print(f"Calculating the gradient with x={x}, dy={dy}, exponent={exponent}")
            return dy * exponent * x ** (exponent-1), None

To use the ``CustomFunction`` class, we call it with the static ``apply`` method.

>>> val = torch.tensor(2.0, requires_grad=True)
>>> res = CustomFunction.apply(val)
>>> res
tensor(4., grad_fn=<CustomFunctionBackward>)
>>> res.backward()
>>> val.grad
Calculating the gradient with x=2.0, dy=1.0, exponent=2
tensor(4.)

Note that for custom functions, the output of ``forward`` and the output of ``backward`` are flattened iterables of
Torch arrays.  While autograd and jax can handle nested result objects like ``((np.array(1), np.array(2)), np.array(3))``,
torch requires that it be flattened like ``(np.array(1), np.array(2), np.array(3))``.  The ``pytreeify`` class decorator
modifies the output of ``forward`` and the input to ``backward`` to unpack and repack the nested structure of the PennyLane
result object.

"""
# pylint: disable=protected-access
import inspect
import logging

import numpy as np
import torch
import torch.utils._pytree as pytree

import pennylane as qml

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


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


def _recursive_conj(dy):
    if isinstance(dy, torch.Tensor):
        return torch.conj(dy)
    return tuple(_recursive_conj(d) for d in dy)


@pytreeify
class ExecuteTapes(torch.autograd.Function):
    """The signature of this ``torch.autograd.Function`` is designed to
    work around Torch restrictions.

    In particular, ``torch.autograd.Function``:

    - Cannot accept keyword arguments. As a result, we pass a dictionary
      as the first argument ``kwargs``. This dictionary **must** contain:

      * ``"tapes"``: the quantum tapes to batch evaluate
      * ``"execute_fn"``: a function that calculates the results of the tapes
      * ``"jpc"``: a :class:`~.JacobianProductCalculator` that can compute the vjp.

    Further, note that the ``parameters`` argument is dependent on the
    ``tapes``; this function should always be called
    with the parameters extracted directly from the tapes as follows:

    >>> parameters = [p for t in tapes for p in t.get_parameters()]
    >>> kwargs = {"tapes": tapes, "execute_fn": execute_fn, "jpc": jpc}
    >>> ExecuteTapes.apply(kwargs, *parameters)

    """

    @staticmethod
    def forward(ctx, kwargs, *parameters):
        """Implements the forward pass batch tape evaluation."""
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Entry with args=(ctx=%s, kwargs=%s, parameters=%s) called by=%s",
                ctx,
                kwargs,
                parameters,
                "::L".join(
                    str(i) for i in inspect.getouterframes(inspect.currentframe(), 2)[1][1:3]
                ),
            )

        ctx.tapes = kwargs["tapes"]
        ctx.jpc = kwargs["jpc"]

        res = tuple(kwargs["execute_fn"](ctx.tapes))

        # if any input tensor uses the GPU, the output should as well
        ctx.torch_device = None

        for p in parameters:
            if isinstance(p, torch.Tensor) and p.is_cuda:  # pragma: no cover
                ctx.torch_device = p.get_device()
                break
        res = tuple(_res_to_torch(r, ctx) for r in res)
        return res

    @staticmethod
    def backward(ctx, *dy):
        """Returns the vector-Jacobian product with given
        parameter values p and output gradient dy"""
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Entry with args=(ctx=%s, dy=%s) called by=%s",
                ctx,
                dy,
                "::L".join(
                    str(i) for i in inspect.getouterframes(inspect.currentframe(), 2)[1][1:3]
                ),
            )
        # Torch obeys the dL/dz_conj convention instead of the
        # dL/dz convention of PennyLane, autograd and jax. This converts between the formats
        dy = _recursive_conj(dy)
        vjps = ctx.jpc.compute_vjp(ctx.tapes, dy)
        # split tensor into separate entries
        unpacked_vjps = []
        for vjp_slice in vjps:
            if vjp_slice is not None and np.squeeze(vjp_slice).shape != (0,):
                unpacked_vjps.extend(_res_to_torch(vjp_slice, ctx))
        vjps = tuple(unpacked_vjps)
        # The output of backward must match the input of forward.
        # Therefore, we return `None` for the gradient of `kwargs`.
        return (None,) + vjps


# pylint: disable=unused-argument
def execute(tapes, execute_fn, jpc, device=None):
    """Execute a batch of tapes with Torch parameters on a device.

    Args:
        tapes (Sequence[.QuantumTape]): batch of tapes to execute
        execute_fn (Callable[[Sequence[.QuantumTape]], ResultBatch]): a function that turns a batch of circuits into results
        jpc (JacobianProductCalculator): a class that can compute the vector jacobian product for the input tapes.

    Returns:
        TensorLike: A nested tuple of tape results. Each element in
        the returned tuple corresponds in order to the provided tapes.
    """
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Entry with args=(tapes=%s, execute_fn=%s, jpc=%s",
            tapes,
            (
                f"\n{inspect.getsource(execute_fn)}\n"
                if logger.isEnabledFor(qml.logging.TRACE)
                else execute_fn
            ),
            jpc,
        )

    parameters = []
    for tape in tapes:
        # set the trainable parameters
        params = tape.get_parameters(trainable_only=False)
        tape.trainable_params = qml.math.get_trainable_indices(params)
        parameters.extend(tape.get_parameters())

    kwargs = {
        "tapes": tuple(tapes),
        "execute_fn": execute_fn,
        "jpc": jpc,
    }

    return ExecuteTapes.apply(kwargs, *parameters)


def _res_to_torch(r, ctx):
    """Convert results from unwrapped execution to torch."""
    if isinstance(r, dict):
        return r
    if isinstance(r, (list, tuple)):
        return type(r)(_res_to_torch(t, ctx) for t in r)
    return torch.as_tensor(r, device=ctx.torch_device)

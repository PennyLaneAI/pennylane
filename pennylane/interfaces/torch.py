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

    >>> parameters = []
    >>> [parameters.extend(t.get_parameters()) for t in tapes]
    >>> kwargs = {"tapes": tapes, "execute_fn": execute_fn, "jpc": jpc}
    >>> ExecuteTapes.apply(kwargs, *parameters)

    """

    @staticmethod
    def forward(ctx, kwargs, *parameters):  # pylint: disable=arguments-differ
        """Implements the forward pass batch tape evaluation."""
        ctx.tapes = kwargs["tapes"]
        ctx.execute_fn = kwargs["execute_fn"]
        ctx.jpc = kwargs["jpc"]

        res = tuple(ctx.execute_fn(ctx.tapes))

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

        vjps = ctx.jpc.compute_vjp(ctx.tapes, dy)

        # split tensor into separate entries
        unpacked_vjps = []
        for vjp_slice in vjps:
            if vjp_slice is not None and np.squeeze(vjp_slice).shape != (0,):
                unpacked_vjps.extend(_res_to_torch(vjp_slice, ctx))
        vjps = tuple(unpacked_vjps)
        # Remove empty vjps (from tape with non trainable params)
        # vjps = tuple(vjp for vjp in vjps if list(vjp.shape) != [0])
        # The output of backward must match the input of forward.
        # Therefore, we return `None` for the gradient of `kwargs`.
        return (None,) + vjps


def execute(tapes, execute_fn, jpc):
    """Execute a batch of tapes with Torch parameters on a device.

    Args:
        tapes (Sequence[.QuantumTape]): batch of tapes to execute
        execute_fn (Callable[[Sequence[.QuantumTape]], ResultBatch]): a function that turns a batch of circuits into results
        jpc (JacobianProductCalculator): a class that can compute the vector jacobian product for the input tapes.

    Returns:
        TensorLike: A nested tuple of tape results. Each element in
        the returned tuple corresponds in order to the provided tapes.
    """

    # pylint: disable=unused-argument
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
        res = []
        for t in r:
            if isinstance(t, dict) or isinstance(t, list) and all(isinstance(i, dict) for i in t):
                # count result, single or broadcasted
                res.append(t)
            elif isinstance(t, tuple):
                res.append(tuple(torch.as_tensor(el, device=ctx.torch_device) for el in t))
            else:
                res.append(torch.as_tensor(t, device=ctx.torch_device))
        if isinstance(r, tuple):
            res = tuple(res)
        return res
    return torch.as_tensor(r, device=ctx.torch_device)

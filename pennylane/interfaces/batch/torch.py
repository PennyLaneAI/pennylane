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
This module contains functions for adding the Torch interface
to a PennyLane Device class.
"""
# pylint: disable=protected-access
import contextlib

import numpy as np
import torch

import pennylane as qml


from .unwrap import UnwrapTape


def get_trainable_params(tape):
    """Gets the trainable Torch parameters of a tape.

    Trainable Torch parameters are any tensor that have the ``requires_grad``
    attribute. If not provided, parameters are assumed to be non-trainable
    by default.

    Args:
        tape (.QuantumTape): a quantum tape

    Returns:
        set[int]: a set containing integers corresponding to tape
        parameters that are differentiable Torch tensors

    **Example**

    >>> with qml.tape.QuantumTape() as tape:
    ...     qml.RX(torch.tensor(0.1, requires_grad=True), wires=0)
    ...     qml.RY(0.2, wires=0)
    ...     qml.RZ(torch.tensor(0.3, requires_grad=True), wires=0)
    >>> trainable_params, params = get_trainable_params(tape)
    >>> trainable_params
    {0, 2}
    >>> params
    [tensor(0.1000, requires_grad=True), 0.2, tensor(0.3000, requires_grad=True)]
    """
    params = tape.get_parameters(trainable_only=False)
    trainable_params = set()

    for idx, p in enumerate(params):
        if getattr(p, "requires_grad", False):
            trainable_params.add(idx)

    return trainable_params, params


def convert_to_numpy(tensors):
    """Converts any Torch tensor in a sequence to NumPy arrays."""
    res = []

    for i in tensors:
        if isinstance(i, torch.Tensor):
            if i.is_cuda:  # pragma: no cover
                res.append(i.cpu().detach().numpy())
            else:
                res.append(i.detach().numpy())
        else:
            res.append(i)

    # if NumPy array is scalar, convert to a Python float
    res = [i.tolist() if (isinstance(i, np.ndarray) and not i.shape) else i for i in res]

    return res


class BatchExecute(torch.autograd.Function):
    """The signature of this ``torch.autograd.Function`` is designed to
    workaround Torch restrictions.

    In particular, ``torch.autograd.Function``:

    - Cannot accept keyword arguments. As a result, we pass a dictionary
      as the first argument ``kwargs``. This dictionary **must** contain:

      * ``"tapes"``: the quantum tapes to batch evaluate
      * ``"device"``: the device to use to evaluate the tapes
      * ``"gradient_fn"``: The gradient transform function to use
        for backward passes.
      * ``"cache"``: the cache list

    Further, note that the ``parameters`` argument is dependent on the
    ``tapes``; this Function should always be called
    with the parameters extracted directly from the tapes as follows:

    >>> parameters = []
    >>> [parameters.extend(t.get_parameters()) for t in tapes])
    >>> kwargs = {"tapes": tapes, "device": device, "gradient_fn": gradient_fn}
    >>> BatchExecute.apply(kwargs, *parameters)

    The private argument ``_n`` is used to track nesting of derivatives, for example
    if the nth-order derivative is requested. Do not set this argument unless you
    understand the consequences!
    """

    @staticmethod
    def forward(ctx, kwargs, *parameters):  # pylint: disable=unused-argument
        """Implements the forward pass batch tape evaluation."""
        ctx.tapes = kwargs["tapes"]
        ctx.device = kwargs["device"]
        ctx.gradient_fn = kwargs["gradient_fn"]
        ctx.cache = kwargs.get("cache", [])
        ctx._n = kwargs.get("_n", 1)

        with contextlib.ExitStack() as stack:
            unwrapped_tapes = [
                stack.enter_context(UnwrapTape(t, convert_to_numpy, get_trainable_params))
                for t in ctx.tapes
            ]
            res = ctx.device.batch_execute(unwrapped_tapes)

        return tuple(torch.as_tensor(torch.from_numpy(r)) for r in res)

    @staticmethod
    def backward(ctx, *dy):
        """Returns the vector-Jacobian product with given
        parameter values p and output gradient dy"""
        reshape_info = []
        gradient_tapes = []
        processing_fns = []

        for t in ctx.tapes:
            processing_fns.append([])

            for idx, _ in enumerate(t.trainable_params):
                g_tapes, fn = ctx.gradient_fn(t, idx)

                reshape_info.append(len(g_tapes))
                gradient_tapes.extend(g_tapes)
                processing_fns[-1].append(fn)

        results = batch_execute(gradient_tapes, ctx.device, gradient_fn=None, cache=ctx.cache)
        vjp = []
        start = 0

        for t, d in zip(range(len(ctx.tapes)), dy):
            num_params = len(ctx.tapes[t].trainable_params)
            jac = []

            if num_params == 0:
                vjp.extend([None])
                continue

            for fn, res_len in zip(processing_fns[t], reshape_info):
                # extract the correct results from the flat list
                res = results[start : start + res_len]
                start += res_len

                # postprocess results to compute the gradient
                jac.append(fn(res))

            dy_row = d.view(-1)
            jac = qml.math.T(torch.stack(jac))
            jac = torch.reshape(jac, [-1, num_params])
            vjp.extend(torch.tensordot(dy_row, jac, dims=[[0], [0]]))

        return (None,) + tuple(vjp)


def batch_execute(
    tapes, device, gradient_fn=None, cache=[], _n=1
):  # pylint: disable=dangerous-default-value
    """Execute a batch of tapes with Torch tensors on a device.

    Args:
        tapes (Sequence[.QuantumTape]): batch of tapes to execute
        device (.Device): Device to use to execute the batch of tapes.
            If the device does not provide a ``batch_execute`` method,
            by default the tapes will be executed in serial.
        gradient_fn (None or callable): The gradient transform function to use
            for backward passes. The provided gradient transform should have
            the signature

            .. code-block:: python

                gradient_fn(tape, idx)

            where ``tape`` is the quantum function to differentiate, and
            ``idx`` is the trainable parameter to return the partial
            derivative of. The function should return a tuple
            ``(gradient_tape, fn)`` containing the list of generated tapes, in
            addition to a post-processing function to be applied to the
            evaluated tapes.

            If not provided, the 'best' gradient function will be determined.

        cache (list[dict[str, float]]): cache of tape parameter-shifts

    Returns:
        list[list[float]]: A nested list of tape results. Each element in
        the returned list corresponds in order to the provided tapes.

    **Example**

    Consider the following cost function:

    .. code-block:: python

        def cost_fn(params, x, dev):
            with qml.tape.QuantumTape() as tape1:
                qml.RX(params[0], wires=0)
                qml.RY(params[1], wires=0)
                qml.expval(qml.PauliZ(0))

            with qml.tape.QuantumTape() as tape2:
                qml.RX(params[2], wires=0)
                qml.RY(x[0], wires=0)
                qml.CNOT(wires=[0, 1])
                qml.probs(wires=0)

            tapes = [tape1, tape2]

            # execute both tapes in a batch on the given device
            res = batch_execute(tapes, dev)

            return res[0][0] + res[1][0, 0] - res[1][0, 1]

    In this cost function, two **independent** quantum tapes are being
    constructed; one returning an expectation value, the other probabilities.
    We then batch execute the two tapes, and reduce the results to obtain
    a scalar.

    Let's execute this cost function while tracking the gradient:

    >>> dev = qml.device("lightning.qubit", wires=2)
    >>> params = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
    >>> x = torch.tensor([0.5], requires_grad=True)
    >>> res = cost_fn(params, x, dev)
    >>> res
    tensor(1.8136, dtype=torch.float64, grad_fn=<SubBackward0>)

    Since the ``batch_execute`` function is differentiable, we can
    also compute the gradient:

    >>> res.backward()
    >>> params.grad
    tensor([-0.0978, -0.1977, -0.2593])
    >>> x.grad
    tensor([-0.4580])

    Finally, we can also compute any nth-order derivative. Let's compute the Hessian:

    >>> cost = lambda params: cost_fn(params, x, dev)
    >>> hess = torch.autograd.functional.hessian(cost, params)
    tensor([[-0.9752,  0.0198,  0.0000],
           [ 0.0198, -0.9752,  0.0000],
           [ 0.0000,  0.0000, -0.8384]])
    """
    if gradient_fn is None:
        gradient_fn = qml.transforms.gradients.qubit_parameter_shift.expval_grad

    parameters = []
    for t in tapes:
        parameters.extend(t.get_parameters())

    kwargs = dict(tapes=tapes, device=device, gradient_fn=gradient_fn, cache=cache, _n=_n)
    return BatchExecute.apply(kwargs, *parameters)

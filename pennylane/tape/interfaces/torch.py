# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
This module contains the mixin interface class for creating differentiable quantum tapes with
PyTorch.
"""
# pylint: disable=protected-access, attribute-defined-outside-init, arguments-differ, no-member, import-self
import numpy as np
import semantic_version
import torch

from pennylane import QuantumFunctionError
from pennylane.interfaces.torch import args_to_numpy

from pennylane.tape.queuing import AnnotatedQueue

COMPLEX_SUPPORT = semantic_version.match(">=1.6.0", torch.__version__)


class _TorchInterface(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_kwargs, *input_):
        """Implements the forward pass QNode evaluation"""
        # detach all input tensors, convert to NumPy array
        ctx.args = args_to_numpy(input_)
        ctx.kwargs = input_kwargs
        ctx.save_for_backward(*input_)

        tape = ctx.kwargs["tape"]
        device = ctx.kwargs["device"]

        # unwrap constant parameters
        ctx.all_params = tape.get_parameters(trainable_only=False)
        ctx.all_params_unwrapped = args_to_numpy(ctx.all_params)

        # evaluate the tape
        tape.set_parameters(ctx.all_params_unwrapped, trainable_only=False)
        res = tape.execute_device(ctx.args, device)
        tape.set_parameters(ctx.all_params, trainable_only=False)

        if hasattr(res, "numpy"):
            res = res.numpy()

        # if any input tensor uses the GPU, the output should as well
        for i in input_:
            if isinstance(i, torch.Tensor):
                if i.is_cuda:  # pragma: no cover
                    cuda_device = i.get_device()
                    return torch.as_tensor(
                        torch.from_numpy(res), device=cuda_device, dtype=tape.dtype
                    )

        if tape.is_sampled and not tape.all_sampled:
            return tuple([torch.as_tensor(t, dtype=tape.dtype) for t in res])

        if res.dtype == np.dtype("object"):
            res = np.hstack(res)

        return torch.as_tensor(torch.from_numpy(res), dtype=tape.dtype)

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        """Implements the backwards pass QNode vector-Jacobian product"""
        tape = ctx.kwargs["tape"]
        device = ctx.kwargs["device"]

        tape.set_parameters(ctx.all_params_unwrapped, trainable_only=False)
        jacobian = tape.jacobian(device, params=ctx.args, **tape.jacobian_options)
        tape.set_parameters(ctx.all_params, trainable_only=False)

        jacobian = torch.as_tensor(jacobian, dtype=grad_output.dtype).to(grad_output)

        vjp = grad_output.view(1, -1) @ jacobian
        grad_input_list = torch.unbind(vjp.flatten())
        grad_input = []

        # match the type and device of the input tensors
        for i, j in zip(grad_input_list, ctx.saved_tensors):
            res = torch.as_tensor(i, dtype=tape.dtype)
            if j.is_cuda:  # pragma: no cover
                cuda_device = j.get_device()
                res = torch.as_tensor(res, device=cuda_device)
            grad_input.append(res)

        return (None,) + tuple(grad_input)


class TorchInterface(AnnotatedQueue):
    """Mixin class for applying an Torch interface to a :class:`~.JacobianTape`.

    Torch-compatible quantum tape classes can be created via subclassing:

    .. code-block:: python

        class MyTorchQuantumTape(TorchInterface, JacobianTape):

    Alternatively, the Torch interface can be dynamically applied to existing
    quantum tapes via the :meth:`~.apply` class method. This modifies the
    tape **in place**.

    Once created, the Torch interface can be used to perform quantum-classical
    differentiable programming.

    **Example**

    Once a Torch quantum tape has been created, it can be evaluated and differentiated:

    .. code-block:: python

        dev = qml.device("default.qubit", wires=1)
        p = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)

        with TorchInterface.apply(JacobianTape()) as qtape:
            qml.Rot(p[0], p[1] ** 2 + p[0] * p[2], p[1] * torch.sin(p[2]), wires=0)
            expval(qml.PauliX(0))

        result = qtape.execute(dev)

    >>> print(result)
    tensor([0.0698], dtype=torch.float64, grad_fn=<_TorchInterfaceBackward>)
    >>> result.backward()
    >>> print(p.grad)
    tensor([0.2987, 0.3971, 0.0988])

    The Torch interface defaults to ``torch.float64`` output. This can be modified by
    providing the ``dtype`` argument when applying the interface:

    >>> p = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
    >>> with TorchInterface.apply(JacobianTape()) as qtape:
    ...     qml.Rot(p[0], p[1] ** 2 + p[0] * p[2], p[1] * torch.sin(p[2]), wires=0)
    ...     expval(qml.PauliX(0))
    >>> result = qtape.execute(dev)
    >>> print(result)
    tensor([0.0698], grad_fn=<_TorchInterfaceBackward>)
    >>> print(result.dtype)
    torch.float32
    >>> result.backward()
    >>> print(p.grad)
    tensor([0.2987, 0.3971, 0.0988])
    >>> print(p.grad.dtype)
    torch.float32
    """

    dtype = torch.float64

    @property
    def interface(self):  # pylint: disable=missing-function-docstring
        return "torch"

    def _update_trainable_params(self):
        params = self.get_parameters(trainable_only=False)

        trainable_params = set()

        for idx, p in enumerate(params):
            if getattr(p, "requires_grad", False):
                trainable_params.add(idx)

        self.trainable_params = trainable_params
        return params

    def _execute(self, params, **kwargs):
        kwargs["tape"] = self
        res = _TorchInterface.apply(kwargs, *params)
        return res

    @classmethod
    def apply(cls, tape, dtype=torch.float64):
        """Apply the Torch interface to an existing tape in-place.

        Args:
            tape (.JacobianTape): a quantum tape to apply the Torch interface to
            dtype (torch.dtype): the dtype that the returned quantum tape should
                output

        **Example**

        >>> with JacobianTape() as tape:
        ...     qml.RX(0.5, wires=0)
        ...     expval(qml.PauliZ(0))
        >>> TorchInterface.apply(tape)
        >>> tape
        <TorchQuantumTape: wires=<Wires = [0]>, params=1>
        """
        if (dtype is torch.complex64 or dtype is torch.complex128) and not COMPLEX_SUPPORT:
            raise QuantumFunctionError(
                "Version 1.6.0 or above of PyTorch must be installed for complex support, "
                "which is required for quantum functions that return the state."
            )

        tape_class = getattr(tape, "__bare__", tape.__class__)
        tape.__bare__ = tape_class
        tape.__class__ = type("TorchQuantumTape", (cls, tape_class), {"dtype": dtype})
        tape._update_trainable_params()
        return tape

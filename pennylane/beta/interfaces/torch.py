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
# pylint: disable=protected-access, attribute-defined-outside-init
import numpy as np
import torch

from pennylane.interfaces.torch import args_to_numpy
from pennylane.beta.queuing import AnnotatedQueue


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

        # # unwrap constant parameters
        ctx.all_params = tape.get_parameters(free_only=False)
        ctx.all_params_unwrapped = args_to_numpy(ctx.all_params)

        # evaluate the tape
        tape.set_parameters(ctx.all_params_unwrapped, free_only=False)
        res = tape.execute_device(ctx.args, device)
        tape.set_parameters(ctx.all_params, free_only=False)

        # if any input tensor uses the GPU, the output should as well
        for i in input_:
            if isinstance(i, torch.Tensor):
                if i.is_cuda:  # pragma: no cover
                    cuda_device = i.get_device()
                    return torch.as_tensor(torch.from_numpy(res), device=cuda_device, dtype=tape.dtype)

        return torch.as_tensor(torch.from_numpy(res), dtype=tape.dtype)

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        """Implements the backwards pass QNode vector-Jacobian product"""
        tape = ctx.kwargs["tape"]
        device = ctx.kwargs["device"]

        tape.set_parameters(ctx.all_params_unwrapped, free_only=False)
        jacobian = tape.jacobian(device, params=ctx.args)
        tape.set_parameters(ctx.all_params, free_only=False)

        jacobian = torch.as_tensor(jacobian, dtype=grad_output.dtype)

        vjp = torch.transpose(grad_output.view(-1, 1), 0, 1) @ jacobian
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

    @property
    def interface(self):  # pylint: disable=missing-function-docstring
        return "torch"

    def _update_trainable_params(self):
        params = [o.data for o in self.operations + self.observables]
        params = [item for sublist in params for item in sublist]

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
        tape_class = getattr(tape, "__bare__", tape.__class__)
        tape.__bare__ = tape_class
        tape.__class__ = type("TorchQuantumTape", (cls, tape_class), {"dtype": dtype})
        tape._update_trainable_params()
        return tape

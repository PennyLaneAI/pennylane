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
This module contains the :func:`to_torch` function to convert Numpy-interfacing quantum nodes to PyTorch
compatible quantum nodes.
"""
# pylint: disable=redefined-outer-name,arguments-differ
import inspect
from functools import partial

import numpy as np
import torch

from pennylane.utils import unflatten


def _get_default_args(func):
    """Get the default arguments of a function.

    Args:
        func (function): a valid Python function

    Returns:
        dict: dictionary containing the argument name and tuple
        (positional idx, default value)
    """
    signature = inspect.signature(func)
    return {
        k: (idx, v.default)
        for idx, (k, v) in enumerate(signature.parameters.items())
        if v.default is not inspect.Parameter.empty
    }


def args_to_numpy(args):
    """Converts all Torch tensors in a list to NumPy arrays

    Args:
        args (list): list containing QNode arguments, including Torch tensors

    Returns:
        list: returns the same list, with all Torch tensors converted to NumPy arrays
    """
    res = []

    for i in args:
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


def kwargs_to_numpy(kwargs):
    """Converts all Torch tensors in a dictionary to NumPy arrays

    Args:
        args (dict): dictionary containing QNode keyword arguments, including Torch tensors

    Returns:
        dict: returns the same dictionary, with all Torch tensors converted to NumPy arrays
    """
    res = {}

    for key, val in kwargs.items():
        if isinstance(val, torch.Tensor):
            if val.is_cuda:  # pragma: no cover
                res[key] = val.cpu().detach().numpy()
            else:
                res[key] = val.detach().numpy()
        else:
            res[key] = val

    # if NumPy array is scalar, convert to a Python float
    res = {
        k: v.tolist() if (isinstance(v, np.ndarray) and not v.shape) else v for k, v in res.items()
    }

    return res


def to_torch(qnode):
    """Function that accepts a :class:`~.QNode`, and returns a PyTorch-compatible QNode.

    Args:
        qnode (~pennylane.qnode.QNode): a PennyLane QNode

    Returns:
        torch.autograd.Function: the QNode as a PyTorch autograd function
    """

    class _TorchQNode(torch.autograd.Function):
        """The TorchQNode"""

        @staticmethod
        def forward(ctx, input_kwargs, *input_):
            """Implements the forward pass QNode evaluation"""
            # detach all input tensors, convert to NumPy array
            ctx.args = args_to_numpy(input_)
            ctx.kwargs = kwargs_to_numpy(input_kwargs)
            ctx.save_for_backward(*input_)

            # evaluate the QNode
            res = qnode(*ctx.args, **ctx.kwargs)

            if not isinstance(res, np.ndarray):
                # scalar result, cast to NumPy scalar
                res = np.array(res)

            # if any input tensor uses the GPU, the output should as well
            for i in input_:
                if isinstance(i, torch.Tensor):
                    if i.is_cuda:  # pragma: no cover
                        cuda_device = i.get_device()
                        return torch.as_tensor(torch.from_numpy(res), device=cuda_device)

            return torch.from_numpy(res)

        @staticmethod
        def backward(ctx, grad_output):  # pragma: no cover
            """Implements the backwards pass QNode vector-Jacobian product"""
            # NOTE: This method is definitely tested by the `test_torch.py` test suite,
            # however does not show up in the coverage. This is likely due to
            # subtleties in the torch.autograd.FunctionMeta metaclass, specifically
            # the way in which the backward class is created on the fly

            # evaluate the Jacobian matrix of the QNode
            jacobian = qnode.jacobian(ctx.args, ctx.kwargs)

            if grad_output.is_cuda:  # pragma: no cover
                grad_output_np = grad_output.cpu().detach().numpy()
            else:
                grad_output_np = grad_output.detach().numpy()

            # perform the vector-Jacobian product
            if not grad_output_np.shape:
                temp = grad_output_np * jacobian
            else:
                temp = grad_output_np.T @ jacobian

            # restore the nested structure of the input args
            temp = [
                np.array(i) if not isinstance(i, np.ndarray) else i
                for i in unflatten(temp.flat, ctx.args)
            ]

            # convert the result to torch tensors, matching
            # the type of the input tensors
            grad_input = []
            for i, j in zip(temp, ctx.saved_tensors):
                res = torch.as_tensor(torch.from_numpy(i), dtype=j.dtype)
                if j.is_cuda:  # pragma: no cover
                    cuda_device = j.get_device()
                    res = torch.as_tensor(res, device=cuda_device)
                grad_input.append(res)

            return (None,) + tuple(grad_input)

    class qnode_str(partial):
        """Torch QNode"""

        # pylint: disable=too-few-public-methods

        @property
        def interface(self):
            """String representing the QNode interface"""
            return "torch"

        def __str__(self):
            """String representation"""
            detail = "<QNode: device='{}', func={}, wires={}, interface={}>"
            return detail.format(
                qnode.device.short_name, qnode.func.__name__, qnode.num_wires, self.interface
            )

        def __repr__(self):
            """REPL representation"""
            return self.__str__()

        print_applied = qnode.print_applied
        jacobian = qnode.jacobian
        metric_tensor = qnode.metric_tensor
        draw = qnode.draw
        func = qnode.func

    @qnode_str
    def custom_apply(*args, **kwargs):
        """Custom apply wrapper, to allow passing kwargs to the TorchQNode"""

        # get default kwargs that weren't passed
        keyword_sig = _get_default_args(qnode.func)
        keyword_defaults = {k: v[1] for k, v in keyword_sig.items()}
        # keyword_positions = {v[0]: k for k, v in keyword_sig.items()}

        # create a keyword_values dict, that contains defaults
        # and any user-passed kwargs
        keyword_values = {}
        keyword_values.update(keyword_defaults)
        keyword_values.update(kwargs)

        # sort keyword values into a list of args, using their position
        # [keyword_values[k] for k in sorted(keyword_positions, key=keyword_positions.get)]

        return _TorchQNode.apply(keyword_values, *args)

    return custom_apply

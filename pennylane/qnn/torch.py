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
"""This module contains the classes and functions for integrating QNodes with the Torch Module
API."""
import functools
import inspect
import math
from collections.abc import Iterable
from typing import Callable, Optional

try:
    import torch
    from torch.nn import Module
    from pennylane.interfaces.torch import to_torch

    TORCH_IMPORTED = True
except ImportError:
    # The following allows this module to be imported even if PyTorch is not installed. Users
    # will instead see an ImportError when instantiating the TorchLayer.
    from abc import ABC

    Module = ABC
    TORCH_IMPORTED = False


class TorchLayer(Module):
    """TODO"""

    def __init__(self, qnode, weight_shapes: dict, output_dim, init_method: Optional[Callable] =
    None):
        if not TORCH_IMPORTED:
            raise ImportError("TorchLayer requires PyTorch")
        super().__init__()

        self.sig = qnode.func.sig

        if self.input_arg not in self.sig:
            raise TypeError(
                "QNode must include an argument with name {} for inputting data".format(
                    self.input_arg
                )
            )

        if self.input_arg in set(weight_shapes.keys()):
            raise ValueError(
                "{} argument should not have its dimension specified in "
                "weight_shapes".format(self.input_arg)
            )

        if set(weight_shapes.keys()) | {self.input_arg} != set(self.sig.keys()):
            raise ValueError("Must specify a shape for every non-input parameter in the QNode")

        if qnode.func.var_pos:
            raise TypeError("Cannot have a variable number of positional arguments")

        if qnode.func.var_keyword:
            raise TypeError("Cannot have a variable number of keyword arguments")

        self.qnode = to_torch(qnode)
        weight_shapes = {
            weight: (tuple(size) if isinstance(size, Iterable) else (size,) if size > 1 else ())
            for weight, size in weight_shapes.items()
        }

        # Allows output_dim to be specified as an int, e.g., 5, or as a length-1 tuple, e.g., (5,)
        self.output_dim = output_dim[0] if isinstance(output_dim, Iterable) else output_dim

        defaults = {
            name for name, sig in self.sig.items() if sig.par.default != inspect.Parameter.empty
        }
        self.input_is_default = self.input_arg in defaults
        if defaults - {self.input_arg} != set():
            raise TypeError(
                "Only the argument {} is permitted to have a default".format(self.input_arg)
            )

        if not init_method:
            init_method = functools.partial(torch.nn.init.uniform_, b=2 * math.pi)

        self.qnode_weights = {}
        for name, size in weight_shapes.items():
            if len(size) == 0:
                self.qnode_weights[name] = torch.nn.Parameter(init_method(torch.Tensor(1))[0])
            else:
                self.qnode_weights[name] = torch.nn.Parameter(init_method(torch.Tensor(*size)))
            self.register_parameter(name, self.qnode_weights[name])

    def forward(self, inputs):  # pylint: disable=arguments-differ
        if len(inputs.shape) == 1:
            return self._evaluate_qnode(inputs)

        return torch.stack([self._evaluate_qnode(x) for x in inputs])

    def _evaluate_qnode(self, x):
        qnode = self.qnode

        for arg in self.sig:
            if arg is not self.input_arg:  # Non-input arguments must always be positional
                w = self.qnode_weights[arg]

                qnode = functools.partial(qnode, w)
            else:
                if self.input_is_default:  # The input argument can be positional or keyword
                    qnode = functools.partial(qnode, **{self.input_arg: x})
                else:
                    qnode = functools.partial(qnode, x)
        return qnode()

    def extra_repr(self):
        detail = "Quantum Torch Layer: func={}"
        return detail.format(self.qnode.func.__name__)

    _input_arg = "inputs"

    @property
    def input_arg(self):
        """TODO"""
        return self._input_arg

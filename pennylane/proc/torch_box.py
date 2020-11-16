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
"""This module contains the TorchBox implementation of the TensorBox API.
"""
import numpy as np
import torch

import pennylane as qml


class TorchBox(qml.proc.TensorBox):
    """Implements the :class:`~.TensorBox` API for Torch tensors.

    For more details, please refer to the :class:`~.TensorBox` documentation.
    """

    @qml.proc.tensorbox.wrap_output
    def abs(self):
        return torch.abs(self.data)

    def angle(self):
        return TorchBox(torch.angle(self.data))

    arcsin = qml.proc.tensorbox.wrap_output(lambda self: torch.arcsin(self.data))

    @staticmethod
    def astensor(tensor):
        return torch.as_tensor(tensor)

    def cast(self, dtype):
        if isinstance(dtype, torch.dtype):
            return TorchBox(self.data.to(dtype))

        dtype_name = np.dtype(dtype).name
        torch_dtype = getattr(torch, dtype_name, None)

        if torch_dtype is None:
            raise ValueError(f"Unable to convert {dtype} to a Torch dtype")

        return TorchBox(self.data.to(torch_dtype))

    @staticmethod
    def concatenate(values, axis=0):
        if axis is None:
            # flatten and then concatenate zero'th dimension
            # to reproduce numpy's behaviour
            tensors = [TorchBox.astensor(t).flatten() for t in TorchBox.unbox_list(values)]
            res = torch.cat(tensors, dim=0)

        else:
            tensors = [TorchBox.astensor(t) for t in TorchBox.unbox_list(values)]
            res = torch.cat(tensors, dim=axis)

        return TorchBox(res)

    def expand_dims(self, axis):
        return TorchBox(torch.unsqueeze(self.data, dim=axis))

    def dot(self, other):
        other = self.astensor(other)

        dtype1 = self.data.dtype
        dtype2 = other.dtype

        if dtype1 is not dtype2:
            complex_type = {dtype1, dtype2}.intersection({torch.complex64, torch.complex128})
            float_type = {dtype1, dtype2}.intersection(
                {torch.float16, torch.float32, torch.float64}
            )
            int_type = {dtype1, dtype2}.intersection(
                {torch.int8, torch.int16, torch.int32, torch.int64}
            )

            cast_type = complex_type or float_type or int_type
            cast_type = list(cast_type)[-1]

            other = other.to(cast_type)
            self.data = self.data.to(cast_type)

        if other.ndim == 2 and self.data.ndim == 2:
            return TorchBox(self.data @ other)

        if other.ndim == 0 and self.data.ndim == 0:
            return TorchBox(self.data * other)

        return TorchBox(torch.tensordot(self.data, other, dims=[[-1], [-2]]))

    @property
    def interface(self):
        return "torch"

    def numpy(self):
        return self.data.detach().cpu().numpy()

    def ones_like(self):
        return TorchBox(torch.ones_like(self.data))

    @property
    def requires_grad(self):
        return self.data.requires_grad

    @property
    def shape(self):
        return tuple(self.data.shape)

    def sqrt(self):
        return TorchBox(torch.sqrt(self.data))

    @staticmethod
    def stack(values, axis=0):
        tensors = [TorchBox.astensor(t) for t in TorchBox.unbox_list(values)]
        res = torch.stack(tensors, axis=axis)
        return TorchBox(res)

    def sum(self, axis=None, keepdims=False):
        if axis is None:
            return TorchBox(torch.sum(self.data))

        return TorchBox(torch.sum(self.data, dim=axis, keepdim=keepdims))

    def take(self, indices, axis=None):
        if isinstance(indices, qml.proc.TensorBox):
            indices = indices.numpy()

        if not isinstance(indices, torch.Tensor):
            indices = self.astensor(indices)

        if axis is None:
            return TorchBox(self.data.flatten()[indices])

        if indices.ndim == 1:
            return TorchBox(torch.index_select(self.data, dim=axis, index=indices))

        fancy_indices = [slice(None)] * axis + [indices]
        return TorchBox(self.data[fancy_indices])

    @property
    def T(self):
        return TorchBox(self.data.T)

    @staticmethod
    def where(condition, x, y):
        return TorchBox(torch.where(TorchBox.astensor(condition), *TorchBox.unbox_list([x, y])))

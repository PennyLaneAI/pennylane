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
"""This module contains the TorchBox implementation of the TensorBox API.
"""
import numpy as np
import torch

import pennylane as qml


wrap_output = qml.math.wrap_output


class TorchBox(qml.math.TensorBox):
    """Implements the :class:`~.TensorBox` API for Torch tensors.

    For more details, please refer to the :class:`~.TensorBox` documentation.
    """

    abs = wrap_output(lambda self: torch.abs(self.data))
    angle = wrap_output(lambda self: torch.angle(self.data))
    arcsin = wrap_output(lambda self: torch.asin(self.data))
    conj = wrap_output(lambda self: torch.conj(self.data))
    expand_dims = wrap_output(lambda self, axis: torch.unsqueeze(self.data, dim=axis))
    gather = wrap_output(lambda self, indices: self.data[indices])
    ones_like = wrap_output(lambda self: torch.ones_like(self.data))
    reshape = wrap_output(lambda self, shape: torch.reshape(self.data, shape))
    sqrt = wrap_output(
        lambda self: torch.sqrt(
            self.data.to(torch.float64)
            if self.data.dtype in (torch.int64, torch.int32)
            else self.data
        )
    )
    T = wrap_output(lambda self: self.data.T)
    squeeze = wrap_output(lambda self: self.data.squeeze())

    @staticmethod
    def astensor(tensor):
        return torch.as_tensor(tensor)

    @staticmethod
    @wrap_output
    def block_diag(values):
        tensors = [TorchBox.astensor(t) for t in TorchBox.unbox_list(values)]
        tensors = TorchBox._coerce_types(tensors)

        sizes = np.array([t.shape for t in tensors])
        res = torch.zeros(np.sum(sizes, axis=0).tolist(), dtype=tensors[0].dtype)

        p = np.cumsum(sizes, axis=0)
        ridx, cidx = np.stack([p - sizes, p]).T

        for t, r, c in zip(tensors, ridx, cidx):
            row = np.arange(*r).reshape(-1, 1)
            col = np.arange(*c).reshape(1, -1)
            res[row, col] = t

        return res

    @wrap_output
    def cast(self, dtype):
        if isinstance(dtype, torch.dtype):
            return self.data.to(dtype)

        dtype_name = np.dtype(dtype).name
        torch_dtype = getattr(torch, dtype_name, None)

        if torch_dtype is None:
            raise ValueError(f"Unable to convert {dtype} to a Torch dtype")

        return self.data.to(torch_dtype)

    @staticmethod
    def _coerce_types(tensors):
        dtypes = {i.dtype for i in tensors}

        if len(dtypes) == 1:
            return tensors

        complex_priority = [torch.complex64, torch.complex128]
        float_priority = [torch.float16, torch.float32, torch.float64]
        int_priority = [torch.int8, torch.int16, torch.int32, torch.int64]

        complex_type = [i for i in complex_priority if i in dtypes]
        float_type = [i for i in float_priority if i in dtypes]
        int_type = [i for i in int_priority if i in dtypes]

        cast_type = complex_type or float_type or int_type
        cast_type = list(cast_type)[-1]

        return [t.to(cast_type) for t in tensors]

    @staticmethod
    @wrap_output
    def concatenate(values, axis=0):
        if axis is None:
            # flatten and then concatenate zero'th dimension
            # to reproduce numpy's behaviour
            tensors = [TorchBox.astensor(t).flatten() for t in TorchBox.unbox_list(values)]
            return torch.cat(tensors, dim=0)

        tensors = [TorchBox.astensor(t) for t in TorchBox.unbox_list(values)]
        return torch.cat(tensors, dim=axis)

    @staticmethod
    @wrap_output
    def diag(values, k=0):
        if isinstance(values, torch.Tensor):
            return torch.diag(values, diagonal=k)

        return torch.diag(TorchBox.stack(values).data, diagonal=k)

    @staticmethod
    @wrap_output
    def dot(x, y):
        x, y = [TorchBox.astensor(t) for t in TorchBox.unbox_list([x, y])]
        x, y = TorchBox._coerce_types([x, y])

        if x.ndim == 0 and y.ndim == 0:
            return x * y

        if x.ndim <= 2 and y.ndim <= 2:
            return x @ y

        return torch.tensordot(x, y, dims=[[-1], [-2]])

    @property
    def interface(self):
        return "torch"

    def numpy(self):
        return self.data.detach().cpu().numpy()

    @property
    def requires_grad(self):
        return self.data.requires_grad

    @wrap_output
    def scatter_element_add(self, index, value):
        if self.data.is_leaf:
            self.data = self.data.clone()
        self.data[tuple(index)] += value
        return self.data

    @property
    def shape(self):
        return tuple(self.data.shape)

    @staticmethod
    @wrap_output
    def stack(values, axis=0):
        tensors = [TorchBox.astensor(t) for t in TorchBox.unbox_list(values)]
        res = torch.stack(tensors, axis=axis)
        return res

    @wrap_output
    def sum(self, axis=None, keepdims=False):
        if axis is None:
            return torch.sum(self.data)

        return torch.sum(self.data, dim=axis, keepdim=keepdims)

    @wrap_output
    def take(self, indices, axis=None):
        if not isinstance(indices, torch.Tensor):
            indices = self.astensor(indices)

        if axis is None:
            return self.data.flatten()[indices]

        if indices.ndim == 1:
            if (indices < 0).any():
                # index_select doesn't allow negative indices
                dim_length = self.data.size()[0] if axis is None else self.shape[axis]

                indices = qml.math.where(indices >= 0, indices, indices + dim_length)

            return torch.index_select(self.data, dim=axis, index=indices)

        fancy_indices = [slice(None)] * axis + [indices]
        return self.data[fancy_indices]

    @staticmethod
    @wrap_output
    def where(condition, x, y):
        return torch.where(TorchBox.astensor(condition), *TorchBox.unbox_list([x, y]))

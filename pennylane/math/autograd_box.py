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
"""This module contains the AutogradBox implementation of the TensorBox API.
"""
# pylint: disable=no-member,protected-access
import pennylane as qml
from pennylane import numpy as np


wrap_output = qml.math.wrap_output


class AutogradBox(qml.math.TensorBox):
    """Implements the :class:`~.TensorBox` API for ``pennylane.numpy`` tensors.

    For more details, please refer to the :class:`~.TensorBox` documentation.
    """

    abs = wrap_output(lambda self: np.abs(self.data))
    angle = wrap_output(lambda self: np.angle(self.data))
    arcsin = wrap_output(lambda self: np.arcsin(self.data))
    cast = wrap_output(lambda self, dtype: np.tensor(self.data, dtype=dtype))
    expand_dims = wrap_output(lambda self, axis: np.expand_dims(self.data, axis=axis))
    ones_like = wrap_output(lambda self: np.ones_like(self.data))
    sqrt = wrap_output(lambda self: np.sqrt(self.data))
    sum = wrap_output(
        lambda self, axis=None, keepdims=False: np.sum(self.data, axis=axis, keepdims=keepdims)
    )
    T = wrap_output(lambda self: self.data.T)

    @staticmethod
    def astensor(tensor):
        return np.tensor(tensor)

    @staticmethod
    @wrap_output
    def concatenate(values, axis=0):
        return np.concatenate(AutogradBox.unbox_list(values), axis=axis)

    @staticmethod
    @wrap_output
    def dot(x, y):
        x, y = AutogradBox.unbox_list([x, y])

        if x.ndim == 0 and y.ndim == 0:
            return x * y

        if x.ndim == 2 and y.ndim == 2:
            return x @ y

        return np.dot(x, y)

    @property
    def interface(self):
        return "autograd"

    def numpy(self):
        if hasattr(self.data, "_value"):
            # Catches the edge case where the data is an Autograd arraybox,
            # which only occurs during backpropagation.
            return self.data._value

        return self.data.numpy()

    @property
    def requires_grad(self):
        return self.data.requires_grad

    @property
    def shape(self):
        return self.data.shape

    @staticmethod
    @wrap_output
    def stack(values, axis=0):
        return np.stack(AutogradBox.unbox_list(values), axis=axis)

    @wrap_output
    def take(self, indices, axis=None):
        indices = self.astensor(indices)

        if axis is None:
            return self.data.flatten()[indices]

        fancy_indices = [slice(None)] * axis + [indices]
        return self.data[tuple(fancy_indices)]

    @staticmethod
    @wrap_output
    def where(condition, x, y):
        return np.where(condition, *AutogradBox.unbox_list([x, y]))

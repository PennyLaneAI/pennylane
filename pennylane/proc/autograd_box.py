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
# pylint: disable=no-member
import pennylane as qml
from pennylane import numpy as np


class AutogradBox(qml.proc.TensorBox):
    """Implements the :class:`~.TensorBox` API for ``pennylane.numpy`` tensors.

    For more details, please refer to the :class:`~.TensorBox` documentation.
    """

    @staticmethod
    def astensor(tensor):
        return np.tensor(tensor)

    def cast(self, dtype):
        return AutogradBox(np.tensor(self.data, dtype=dtype))

    def expand_dims(self, axis):
        return AutogradBox(np.expand_dims(self.data, axis=axis))

    @property
    def interface(self):
        return "autograd"

    def numpy(self):
        return self.data.numpy()

    def ones_like(self):
        return AutogradBox(np.ones_like(self.data))

    @property
    def requires_grad(self):
        return self.data.requires_grad

    @property
    def shape(self):
        return self.data.shape

    @staticmethod
    def stack(values, axis=0):
        return AutogradBox(np.stack(AutogradBox.unbox_list(values), axis=axis))

    @property
    def T(self):
        return AutogradBox(self.data.T)

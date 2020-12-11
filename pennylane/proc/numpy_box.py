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
"""This module contains the NumpyBox implementation of the TensorBox API.
"""
import numpy as np
import pennylane as qml


class NumpyBox(qml.proc.TensorBox):
    """Implements the :class:`~.TensorBox` API for ``numpy.ndarray``.

    For more details, please refer to the :class:`~.TensorBox` documentation.
    """

    def __init__(self, tensor):
        if not isinstance(tensor, np.ndarray):
            tensor = np.asarray(tensor)

        super().__init__(tensor)

    @staticmethod
    def astensor(tensor):
        return np.asarray(tensor)

    def cast(self, dtype):
        return NumpyBox(np.asarray(self.data, dtype=dtype))

    def expand_dims(self, axis):
        return NumpyBox(np.expand_dims(self.data, axis=axis))

    @property
    def interface(self):
        return "numpy"

    def numpy(self):
        return self.data

    def ones_like(self):
        return NumpyBox(np.ones_like(self.data))

    @property
    def requires_grad(self):
        return False

    @property
    def shape(self):
        return self.data.shape

    @staticmethod
    def stack(values, axis=0):
        return NumpyBox(np.stack(NumpyBox.unbox_list(values), axis=axis))

    @property
    def T(self):
        return NumpyBox(self.data.T)

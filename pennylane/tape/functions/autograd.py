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
"""This module contains the AutogradTensor implementation of the UnifiedTensor API.
"""
from pennylane import numpy as np

from .unified import UnifiedTensor


class AutogradTensor(UnifiedTensor):
    """Implements the :class:`~.UnifiedTensor` API for ``pennylane.numpy`` tensors."""

    def expand_dims(self, axis):
        return AutogradTensor(np.expand_dims(self.data, axis=axis))

    def numpy(self):
        if hasattr(self.data, "numpy"):
            return self.data.numpy()

        return self.data

    def ones_like(self):
        return AutogradTensor(np.ones_like(self.data))

    @staticmethod
    def stack(values, axis=0):
        return AutogradTensor(np.stack(AutogradTensor.unwrap(values), axis=axis))

    @property
    def shape(self):
        return self.data.shape

    @property
    def T(self):
        return AutogradTensor(self.data.T)

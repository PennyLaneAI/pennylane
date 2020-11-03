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
"""This module contains the TorchTensor implementation of the UnifiedTensor API.
"""
import numpy as np
import torch

from .unified import UnifiedTensor


class TorchTensor(UnifiedTensor):
    """Implements the :class:`~.UnifiedTensor` API for Torch tensors."""

    @staticmethod
    def stack(values, axis=0):
        res = torch.stack(TorchTensor.unwrap(values), axis=axis)
        return TorchTensor(res)

    @property
    def shape(self):
        return tuple(self.data.shape)

    def expand_dims(self, axis):
        return TorchTensor(torch.unsqueeze(self.data, dim=axis))

    def numpy(self):
        return self.data.detach().cpu().numpy()

    def ones_like(self):
        return TorchTensor(torch.ones_like(self.data))

    @property
    def T(self):
        return TorchTensor(torch.transpose(self.data))

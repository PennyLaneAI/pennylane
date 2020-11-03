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
import torch

import pennylane as qml


class TorchBox(qml.TensorBox):
    """Implements the :class:`~.TensorBox` API for Torch tensors.

    For more details, please refer to the :class:`~.TensorBox` documentation.
    """

    @property
    def interface(self):
        return "torch"

    @staticmethod
    def stack(values, axis=0):
        res = torch.stack(TorchBox.unbox_list(values), axis=axis)
        return TorchBox(res)

    @property
    def shape(self):
        return tuple(self.unbox().shape)

    def expand_dims(self, axis):
        return TorchBox(torch.unsqueeze(self.unbox(), dim=axis))

    def numpy(self):
        return self.unbox().detach().cpu().numpy()

    def ones_like(self):
        return TorchBox(torch.ones_like(self.unbox()))

    @property
    def T(self):
        return TorchBox(torch.transpose(self.unbox()))

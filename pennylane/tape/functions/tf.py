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
"""This module contains the TensorFlowTensor implementation of the UnifiedTensor API.
"""
import numpy as np
import tensorflow as tf

from .unified import UnifiedTensor


class TensorFlowTensor(UnifiedTensor):
    """Implements the :class:`~.UnifiedTensor` API for TensorFlow tensors."""

    @staticmethod
    def stack(values, axis=0):
        res = tf.stack(TensorFlowTensor.unwrap(values), axis=axis)
        return TensorFlowTensor(res)

    @property
    def shape(self):
        return tuple(self.data.shape)

    def expand_dims(self, axis):
        return TensorFlowTensor(tf.expand_dims(self.data, axis=axis))

    def numpy(self):
        return self.data.numpy()

    def ones_like(self):
        return TensorFlowTensor(tf.ones_like(self.data))

    @property
    def T(self):
        return TensorFlowTensor(tf.transpose(self.data))

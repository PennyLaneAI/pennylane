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
"""This module contains the TensorFlowBox implementation of the TensorBox API.
"""
import numpy as np
import tensorflow as tf

from .tensorbox import TensorBox


class TensorFlowBox(TensorBox):
    """Implements the :class:`~.TensorBox` API for TensorFlow tensors.

    For more details, please refer to the :class:`~.TensorBox` documentation.
    """

    @staticmethod
    def stack(values, axis=0):
        res = tf.stack(TensorFlowBox.unbox_list(values), axis=axis)
        return TensorFlowBox(res)

    @property
    def shape(self):
        return tuple(self.unbox().shape)

    def expand_dims(self, axis):
        return TensorFlowBox(tf.expand_dims(self.unbox(), axis=axis))

    def numpy(self):
        return self.unbox().numpy()

    def ones_like(self):
        return TensorFlowBox(tf.ones_like(self.unbox()))

    @property
    def T(self):
        return TensorFlowBox(tf.transpose(self.unbox()))

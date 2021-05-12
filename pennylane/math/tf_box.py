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
"""This module contains the TensorFlowBox implementation of the TensorBox API.
"""
import numbers
import tensorflow as tf


try:
    from tensorflow.python.eager.tape import should_record_backprop
except ImportError:  # pragma: no cover
    from tensorflow.python.eager.tape import should_record as should_record_backprop


import pennylane as qml


wrap_output = qml.math.wrap_output


class TensorFlowBox(qml.math.TensorBox):
    """Implements the :class:`~.TensorBox` API for TensorFlow tensors.

    For more details, please refer to the :class:`~.TensorBox` documentation.
    """

    abs = wrap_output(lambda self: tf.abs(self.data))
    angle = wrap_output(lambda self: tf.math.angle(self.data))
    arcsin = wrap_output(lambda self: tf.math.asin(self.data))
    cast = wrap_output(lambda self, dtype: tf.cast(self.data, dtype))
    conj = wrap_output(lambda self: tf.math.conj(self.data))
    diag = staticmethod(wrap_output(lambda values, k=0: tf.linalg.diag(values, k=k)))
    expand_dims = wrap_output(lambda self, axis: tf.expand_dims(self.data, axis=axis))
    gather = wrap_output(lambda self, indices: tf.gather(self.data, indices))
    ones_like = wrap_output(lambda self: tf.ones_like(self.data))
    reshape = wrap_output(lambda self, shape: tf.reshape(self.data, shape))
    sqrt = wrap_output(
        lambda self: tf.sqrt(
            tf.cast(self.data, dtype=tf.float64)
            if self.data.dtype in (tf.dtypes.int64, tf.dtypes.int32)
            else self.data
        )
    )
    sum = wrap_output(
        lambda self, axis=None, keepdims=False: tf.reduce_sum(
            self.data, axis=axis, keepdims=keepdims
        )
    )
    T = wrap_output(lambda self: tf.transpose(self.data))
    squeeze = wrap_output(lambda self: tf.squeeze(self.data))

    def __len__(self):
        if isinstance(self.data, tf.Variable):
            return len(tf.convert_to_tensor(self.data))

        return super().__len__()

    @staticmethod
    def astensor(tensor):
        return tf.convert_to_tensor(tensor)

    @staticmethod
    @wrap_output
    def block_diag(values):
        tensors = TensorFlowBox.unbox_list(values)
        tensors = TensorFlowBox._coerce_types(tensors)
        int_dtype = None

        if tensors[0].dtype in (tf.int32, tf.int64):
            int_dtype = tensors[0].dtype
            tensors = [tf.cast(t, tf.float32) for t in tensors]

        linop_blocks = [tf.linalg.LinearOperatorFullMatrix(block) for block in tensors]
        linop_block_diagonal = tf.linalg.LinearOperatorBlockDiag(linop_blocks)
        res = linop_block_diagonal.to_dense()

        if int_dtype is None:
            return res

        return tf.cast(res, int_dtype)

    @staticmethod
    def _coerce_types(tensors):
        tensors = [TensorFlowBox.astensor(t) for t in tensors]
        dtypes = {i.dtype for i in tensors}

        if len(dtypes) == 1:
            return tensors

        complex_type = dtypes.intersection({tf.complex64, tf.complex128})
        float_type = dtypes.intersection({tf.float16, tf.float32, tf.float64})
        int_type = dtypes.intersection({tf.int8, tf.int16, tf.int32, tf.int64})

        cast_type = complex_type or float_type or int_type
        cast_type = list(cast_type)[-1]

        return [tf.cast(t, cast_type) for t in tensors]

    @staticmethod
    @wrap_output
    def concatenate(values, axis=0):
        if axis is None:
            # flatten and then concatenate zero'th dimension
            # to reproduce numpy's behaviour
            tensors = [
                tf.reshape(TensorFlowBox.astensor(t), [-1])
                for t in TensorFlowBox.unbox_list(values)
            ]
            tensors = TensorFlowBox._coerce_types(tensors)
            return tf.concat(tensors, axis=0)

        return tf.concat(TensorFlowBox.unbox_list(values), axis=axis)

    @staticmethod
    @wrap_output
    def dot(x, y):
        x, y = [TensorFlowBox.astensor(t) for t in TensorFlowBox.unbox_list([x, y])]
        x, y = TensorFlowBox._coerce_types([x, y])

        if x.ndim == 0 and y.ndim == 0:
            return x * y

        if y.ndim == 1:
            return tf.tensordot(x, y, axes=[[-1], [0]])

        if x.ndim == 2 and y.ndim == 2:
            return x @ y

        return tf.tensordot(x, y, axes=[[-1], [-2]])

    @property
    def interface(self):
        return "tf"

    def numpy(self):
        return self.data.numpy()

    @property
    def requires_grad(self):
        return should_record_backprop([self.astensor(self.data)])

    @wrap_output
    def scatter_element_add(self, index, value):
        indices = tf.expand_dims(index, 0)
        tensor = tf.cast(tf.expand_dims(value, 0), self.data.dtype)
        self.data = tf.tensor_scatter_nd_add(self.data, indices, tensor)
        return self.data

    @property
    def shape(self):
        return tuple(self.data.shape)

    @staticmethod
    @wrap_output
    def stack(values, axis=0):
        values = TensorFlowBox._coerce_types(TensorFlowBox.unbox_list(values))
        res = tf.stack(values, axis=axis)
        return res

    @wrap_output
    def take(self, indices, axis=None):
        if isinstance(indices, numbers.Number):
            indices = [indices]

        indices = self.astensor(indices)

        if qml.numpy.any(indices < 0):
            # Unlike NumPy, TensorFlow doesn't support negative indices.
            dim_length = tf.size(self.data).numpy() if axis is None else self.shape[axis]

            indices = qml.math.where(indices >= 0, indices, indices + dim_length)

        if axis is None:
            # Unlike NumPy, if axis=None TensorFlow defaults to the first
            # dimension rather than flattening the array.
            data = tf.reshape(self.data, [-1])
            return tf.gather(data, indices)

        return tf.gather(self.data, indices, axis=axis)

    @staticmethod
    @wrap_output
    def where(condition, x, y):
        return tf.where(TensorFlowBox.astensor(condition), *TensorFlowBox.unbox_list([x, y]))

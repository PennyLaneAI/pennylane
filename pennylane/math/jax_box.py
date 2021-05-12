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
"""This module contains the JaxBox implementation of the TensorBox API.
"""
import jax
from jax.scipy.linalg import block_diag
import jax.numpy as jnp
import pennylane as qml


wrap_output = qml.math.wrap_output


class JaxBox(qml.math.TensorBox):
    """Implements the :class:`~.TensorBox` API for ``numpy.ndarray``.

    For more details, please refer to the :class:`~.TensorBox` documentation.
    """

    abs = wrap_output(lambda self: jnp.abs(self.data))
    angle = wrap_output(lambda self: jnp.angle(self.data))
    arcsin = wrap_output(lambda self: jnp.arcsin(self.data))
    cast = wrap_output(lambda self, dtype: jnp.array(self.data, dtype=dtype))
    conj = wrap_output(lambda self: jnp.conj(self.data))
    diag = staticmethod(wrap_output(lambda values, k=0: jnp.diag(jnp.array(values), k=k)))
    expand_dims = wrap_output(lambda self, axis: jnp.expand_dims(self.data, axis=axis))
    gather = wrap_output(lambda self, indices: self.data[indices])
    reshape = wrap_output(lambda self, shape: jnp.reshape(self.data, shape))
    ones_like = wrap_output(lambda self: jnp.ones_like(self.data))
    sqrt = wrap_output(lambda self: jnp.sqrt(self.data))
    sum = wrap_output(
        lambda self, axis=None, keepdims=False: jnp.sum(self.data, axis=axis, keepdims=keepdims)
    )
    T = wrap_output(lambda self: self.data.T)
    take = wrap_output(
        lambda self, indices, axis=None: jnp.take(self.data, indices, axis=axis, mode="wrap")
    )
    squeeze = wrap_output(lambda self: self.data.squeeze())

    def __init__(self, tensor):
        tensor = jnp.asarray(tensor)

        super().__init__(tensor)

    @staticmethod
    def astensor(tensor):
        return jnp.asarray(tensor)

    @staticmethod
    @wrap_output
    def block_diag(values):
        return block_diag(*JaxBox.unbox_list(values))

    @staticmethod
    @wrap_output
    def concatenate(values, axis=0):
        return jnp.concatenate(JaxBox.unbox_list(values), axis=axis)

    @staticmethod
    @wrap_output
    def dot(x, y):
        x, y = JaxBox.unbox_list([x, y])
        x = jnp.asarray(x)
        y = jnp.asarray(y)

        if x.ndim == 0 and y.ndim == 0:
            return x * y

        if x.ndim == 2 and y.ndim == 2:
            return x @ y

        return jnp.dot(x, y)

    @property
    def interface(self):
        return "jax"

    def numpy(self):
        return self.data

    @property
    def requires_grad(self):
        return True

    @wrap_output
    def scatter_element_add(self, index, value):
        self.data = jax.ops.index_add(self.data, tuple(index), value)
        return self.data

    @property
    def shape(self):
        return self.data.shape

    @staticmethod
    @wrap_output
    def stack(values, axis=0):
        return jnp.stack(JaxBox.unbox_list(values), axis=axis)

    @staticmethod
    @wrap_output
    def where(condition, x, y):
        return jnp.where(condition, *JaxBox.unbox_list([x, y]))

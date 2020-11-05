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
"""Function wrappers for the TensorBox API"""
from functools import wraps
import warnings

import numpy as np

from .tensorbox import TensorBox


def _get_multi_tensorbox(values):
    """Determines the correct framework to dispatch to given a
    sequence of tensor like objects.

    Args:
        values (Sequence[tensor_like]): a sequence of tensor like objects

    Returns:
        .TensorBox: A TensorBox that will dispatch to the correct framework
        given the rules of precedence. This TensorBox will contain the *first*
        tensor like object in ``values`` that corresponds to the highest-priority
        framework.

    To determine the framework to dispatch to, the following rules
    are followed:

    * Tensors that are incompatible (such as Torch and TensorFlow tensors)
      cannot both be present.

    * Autograd tensors *may* be present alongside Torch and TensorFlow tensors,
      but Torch and TensorFlow take precendence; the autograd arrays will
      be treated as non-differentiable NumPy arrays. A warning will be raised
      suggesting that vanilla NumPy be used instead.

    * Vanilla NumPy arrays can be used alongside other tensor objects; they will
      always be treated as non-differentiable constants.
    """
    interfaces = [get_interface(v) for v in values]

    if "tf" in interfaces and "torch" in interfaces:
        raise ValueError("Tensors contain mixed types; cannot determine dispatch library")

    non_numpy_interfaces = set(interfaces) - numpy

    if len(non_numpy_interfaces) == 2:
        warnings.warn(
            f"Tensors contain types {non_numpy_interfaces}; dispatch will prioritize "
            "TensorFlow and PyTorch over autograd. Consider replacing Autograd with vanilla NumPy.",
            UserWarning,
        )

    if "tf" in interfaces:
        return TensorBox(values[interfaces.index("tf")])

    if "torch" in interfaces:
        return TensorBox(values[interfaces.index("torch")])

    if "autograd" in interfaces:
        return TensorBox(values[interfaces.index("autograd")])

    return TensorBox(values[interfaces.index("numpy")])


def allequal(tensor1, tensor2, **kwargs):
    """Returns True if two tensors are element-wise equal along a given axis.

    This function is equivalent to calling ``np.all(tensor1 == tensor2, **kwargs)``,
    but allows for ``tensor1`` and ``tensor2`` to differ in type.

    Args:
        tensor1 (tensor_like): tensor to compare
        tensor2 (tensor_like): tensor to compare
        **kwargs: Accepts any keyword argument that is accepted by ``np.all``,
            such as ``axis``, ``out``, and ``keepdims``. See the `NumPy documentation
            <https://numpy.org/doc/stable/reference/generated/numpy.all.html>`__ for
            more details.

    Returns:
        ndarray, bool: If ``axis=None``, a logical AND reduction is applied to all elements
        and a boolean will be returned, indicating if all elements evaluate to True. Otherwise,
        a boolean NumPy array will be returned.

    **Example**

    >>> a = torch.tensor([1, 2])
    >>> b = np.array([1, 2])
    >>> allequal(a, b)
    True
    """
    t1 = TensorBox(tensor1).numpy()
    t2 = TensorBox(tensor2).numpy()
    return np.all(t1 == t2, **kwargs)


@wraps(np.allclose)
def allclose(a, b, rtol=1e-05, atol=1e-08, **kwargs):
    t1 = TensorBox(a).numpy()
    t2 = TensorBox(b).numpy()
    return np.allclose(t1, t2, rtol=rtol, atol=atol, **kwargs)


def cast(tensor, dtype):
    """Casts the given tensor to a new type.

    Args:
        tensor (tensor_like): tensor to cast
        dtype (str, np.dtype): Any supported NumPy dtype specified; this can be
            a string (``"float64"``), a ``np.dtype`` object (``np.dtype("float64")``), or
            a dtype class (``np.float64``). If ``tensor`` is not a NumPy array, the
            **equivalent** dtype in the dispatched framework is used.

    Returns:
        tensor_like: a tensor with the same shape and values as ``tensor`` and the
        same dtype as ``dtype``.

    **Example**

    We can use NumPy dtype specifiers:

    >>> x = torch.tensor([1, 2])
    >>> cast(x, np.float64)
    tensor([1., 2.], dtype=torch.float64)

    We can also use strings:

    >>> x = tf.tensor(x, "float64")
    <tf.Tensor: shape=(2,), dtype=complex128, numpy=array([1.+0.j, 2.+0.j])>
    """
    return TensorBox(tensor).cast(dtype).data


def cast_like(tensor1, tensor2):
    """Casts a tensor to the same dtype as another.

    Args:
        tensor1 (tensor_like): tensor to cast
        tensor2 (tensor_like): tensor with corresponding dtype to cast to

    Returns:
        tensor_like: a tensor with the same shape and values as ``tensor`` and the
        same dtype as ``tensor2``.

    **Example**

    >>> x = torch.tensor([1, 2])
    >>> y = torch.tensor([3., 4.])
    >>> cast(x, y)
    tensor([1., 2.])
    """
    dtype = TensorBox(tensor2).numpy().dtype.type
    return TensorBox(tensor1).cast(dtype).data


def convert_like(tensor1, tensor2):
    return TensorBox(tensor1).astensor(tensor2).data


def expand_dims(tensor, axis):
    return TensorBox(tensor, axis).data


def get_interface(tensor):
    return TensorBox(tensor).interface


def toarray(tensor):
    return TensorBox(tensor).numpy()


def ones_like(tensor, dtype=None):
    return TensorBox(tensor).ones_like(dtype=dtype).data


def requires_grad(tensor):
    return TensorBox(tensor).requires_grad


def shape(tensor):
    return TensorBox(tensor).shape


def stack(values, axis=0):
    return _get_multi_tensorbox(values).stack(values, axis=0)


def T(tensor):
    return TensorBox(tensor).T.data

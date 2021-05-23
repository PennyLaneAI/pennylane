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
"""Multiple dispatch functions"""
# pylint: disable=import-outside-toplevel,too-many-return-statements
import warnings

from autoray import numpy as np

from . import single_dispatch  # pylint:disable=unused-import
from .utils import cast, get_interface


def _multi_dispatch(values):
    """Determines the correct framework to dispatch to given a
    sequence of tensor-like objects.

    Args:
        values (Sequence[tensor_like]): a sequence of tensor like objects

    Returns:
        str: the name of the interface

    To determine the framework to dispatch to, the following rules
    are applied:

    * Tensors that are incompatible (such as Torch and TensorFlow tensors)
      cannot both be present.

    * Autograd tensors *may* be present alongside Torch and TensorFlow tensors,
      but Torch and TensorFlow take precendence; the autograd arrays will
      be treated as non-differentiable NumPy arrays. A warning will be raised
      suggesting that vanilla NumPy be used instead.

    * Vanilla NumPy arrays can be used alongside other tensor objects; they will
      always be treated as non-differentiable constants.
    """
    if "resource_variable" in getattr(values, "__module__", tuple()):
        values = np.asarray(values)

    interfaces = {get_interface(v) for v in values}

    if len(set(interfaces) - {"numpy", "autograd"}) > 1:
        # contains multiple non-autograd interfaces
        raise ValueError("Tensors contain mixed types; cannot determine dispatch library")

    non_numpy_interfaces = set(interfaces) - {"numpy"}

    if len(non_numpy_interfaces) > 1:
        # contains autograd and another interface
        warnings.warn(
            f"Contains tensors of types {non_numpy_interfaces}; dispatch will prioritize "
            "TensorFlow and PyTorch over autograd. Consider replacing Autograd with vanilla NumPy.",
            UserWarning,
        )

    if "tensorflow" in interfaces:
        return "tensorflow"

    if "torch" in interfaces:
        return "torch"

    if "autograd" in interfaces:
        return "autograd"

    if "jax" in interfaces:
        return "jax"

    return "numpy"


def block_diag(values):
    """Combine a sequence of 2D tensors to form a block diagonal tensor.

    Args:
        values (Sequence[tensor_like]): Sequence of 2D arrays/tensors to form
            the block diagonal tensor.

    Returns:
        tensor_like: the block diagonal tensor

    **Example**

    >>> t = [
    ...     np.array([[1, 2], [3, 4]]),
    ...     torch.tensor([[1, 2, 3], [-1, -6, -3]]),
    ...     torch.tensor(5)
    ... ]
    >>> qml.math.block_diag(t)
    tensor([[ 1,  2,  0,  0,  0,  0],
            [ 3,  4,  0,  0,  0,  0],
            [ 0,  0,  1,  2,  3,  0],
            [ 0,  0, -1, -6, -3,  0],
            [ 0,  0,  0,  0,  0,  5]])
    """
    interface = _multi_dispatch(values)
    values = np.coerce(values, like=interface)
    return np.block_diag(values, like=interface)


def concatenate(values, axis=0):
    """Concatenate a sequence of tensors along the specified axis.

    .. warning::

        Tensors that are incompatible (such as Torch and TensorFlow tensors)
        cannot both be present.

    Args:
        values (Sequence[tensor_like]): Sequence of tensor-like objects to
            concatenate. The objects must have the same shape, except in the dimension corresponding
            to axis (the first, by default).
        axis (int): The axis along which the input tensors are concatenated. If axis is None,
            tensors are flattened before use. Default is 0.

    Returns:
        tensor_like: The concatenated tensor.

    **Example**

    >>> x = tf.constant([0.6, 0.1, 0.6])
    >>> y = tf.Variable([0.1, 0.2, 0.3])
    >>> z = np.array([5., 8., 101.])
    >>> concatenate([x, y, z])
    <tf.Tensor: shape=(3, 3), dtype=float32, numpy=
    array([6.00e-01, 1.00e-01, 6.00e-01, 1.00e-01, 2.00e-01, 3.00e-01, 5.00e+00, 8.00e+00, 1.01e+02], dtype=float32)>
    """
    interface = _multi_dispatch(values)

    if interface == "torch":
        import torch

        if axis is None:
            # flatten and then concatenate zero'th dimension
            # to reproduce numpy's behaviour
            values = [np.flatten(torch.as_tensor(t)) for t in values]
            axis = 0
        else:
            values = [torch.as_tensor(t) for t in values]

    if interface == "tensorflow" and axis is None:
        # flatten and then concatenate zero'th dimension
        # to reproduce numpy's behaviour
        values = [np.flatten(np.array(t)) for t in values]
        axis = 0

    return np.concatenate(values, axis=axis, like=interface)


def diag(values, k=0):
    """Construct a diagonal tensor from a list of scalars.

    Args:
        values (tensor_like or Sequence[scalar]): sequence of numeric values that
            make up the diagonal
        k (int): The diagonal in question. ``k=0`` corresponds to the main diagonal.
            Use ``k>0`` for diagonals above the main diagonal, and ``k<0`` for
            diagonals below the main diagonal.

    Returns:
        tensor_like: the 2D diagonal tensor

    **Example**

    >>> x = [1., 2., tf.Variable(3.)]
    >>> diag(x)
    <tf.Tensor: shape=(3, 3), dtype=float32, numpy=
    array([[1., 0., 0.],
           [0., 2., 0.],
           [0., 0., 3.]], dtype=float32)>
    >>> y = tf.Variable([0.65, 0.2, 0.1])
    >>> diag(y, k=-1)
    <tf.Tensor: shape=(4, 4), dtype=float32, numpy=
    array([[0.  , 0.  , 0.  , 0.  ],
           [0.65, 0.  , 0.  , 0.  ],
           [0.  , 0.2 , 0.  , 0.  ],
           [0.  , 0.  , 0.1 , 0.  ]], dtype=float32)>
    >>> z = torch.tensor([0.1, 0.2])
    >>> qml.math.diag(z, k=1)
    tensor([[0.0000, 0.1000, 0.0000],
            [0.0000, 0.0000, 0.2000],
            [0.0000, 0.0000, 0.0000]])
    """
    interface = _multi_dispatch(values)

    if isinstance(values, (list, tuple)):
        values = np.stack(np.coerce(values, like=interface), like=interface)

    return np.diag(values, k=k, like=interface)


def dot(tensor1, tensor2):
    """Returns the matrix or dot product of two tensors.

    * If both tensors are 0-dimensional, elementwise multiplication
      is performed and a 0-dimensional scalar returned.

    * If both tensors are 1-dimensional, the dot product is returned.

    * If the first array is 2-dimensional and the second array 1-dimensional,
      the matrix-vector product is returned.

    * If both tensors are 2-dimensional, the matrix product is returned.

    * Finally, if the the first array is N-dimensional and the second array
      M-dimensional, a sum product over the last dimension of the first array,
      and the second-to-last dimension of the second array is returned.

    Args:
        tensor1 (tensor_like): input tensor
        tensor2 (tensor_like): input tensor
    """
    interface = _multi_dispatch([tensor1, tensor2])
    x, y = np.coerce([tensor1, tensor2], like=interface)

    if interface == "torch":
        if x.ndim == 0 and y.ndim == 0:
            return x * y

        if x.ndim <= 2 and y.ndim <= 2:
            return x @ y

        return np.tensordot(x, y, dims=[[-1], [-2]], like=interface)

    if interface == "tensorflow":
        if x.ndim == 0 and y.ndim == 0:
            return x * y

        if y.ndim == 1:
            return np.tensordot(x, y, axes=[[-1], [0]], like=interface)

        if x.ndim == 2 and y.ndim == 2:
            return x @ y

        return np.tensordot(x, y, axes=[[-1], [-2]], like=interface)

    return np.dot(x, y, like=interface)


def ones_like(tensor, dtype=None):
    """Returns a tensor of all ones with the same shape and dtype
    as the input tensor.

    Args:
        tensor (tensor_like): input tensor
        dtype (str, np.dtype, None): The desired output datatype of the array. If not provided, the dtype of
            ``tensor`` is used. This argument can be any supported NumPy dtype representation, including
            a string (``"float64"``), a ``np.dtype`` object (``np.dtype("float64")``), or
            a dtype class (``np.float64``). If ``tensor`` is not a NumPy array, the
            **equivalent** dtype in the dispatched framework is used.

    Returns:
        tensor_like: an all-ones tensor with the same shape and
        size as ``tensor``

    **Example**

    >>> x = torch.tensor([1., 2.])
    >>> ones_like(x)
    tensor([1, 1])
    >>> y = tf.Variable([[0], [5]])
    >>> ones_like(y, dtype=np.complex128)
    <tf.Tensor: shape=(2, 1), dtype=complex128, numpy=
    array([[1.+0.j],
           [1.+0.j]])>
    """
    if dtype is not None:
        return cast(np.ones_like(tensor), dtype)

    return np.ones_like(tensor)


def stack(values, axis=0):
    """Stack a sequence of tensors along the specified axis.

    .. warning::

        Tensors that are incompatible (such as Torch and TensorFlow tensors)
        cannot both be present.

    Args:
        values (Sequence[tensor_like]): Sequence of tensor-like objects to
            stack. Each object in the sequence must have the same size in the given axis.
        axis (int): The axis along which the input tensors are stacked. ``axis=0`` corresponds
            to vertical stacking.

    Returns:
        tensor_like: The stacked array. The stacked array will have one additional dimension
        compared to the unstacked tensors.

    **Example**

    >>> x = tf.constant([0.6, 0.1, 0.6])
    >>> y = tf.Variable([0.1, 0.2, 0.3])
    >>> z = np.array([5., 8., 101.])
    >>> stack([x, y, z])
    <tf.Tensor: shape=(3, 3), dtype=float32, numpy=
    array([[6.00e-01, 1.00e-01, 6.00e-01],
           [1.00e-01, 2.00e-01, 3.00e-01],
           [5.00e+00, 8.00e+00, 1.01e+02]], dtype=float32)>
    """
    interface = _multi_dispatch(values)
    values = np.coerce(values, like=interface)
    return np.stack(values, axis=axis, like=interface)


def where(condition, x, y):
    """Returns elements chosen from x or y depending on a boolean tensor condition.

    The input tensors ``condition``, ``x``, and ``y`` must all be broadcastable to the same shape.

    Args:
        condition (tensor_like[bool]): A boolean tensor. Where True, elements from
            ``x`` will be chosen, otherwise ``y``.
        x (tensor_like): values from which to choose if the condition evaluates to ``True``
        y (tensor_like): values from which to choose if the condition evaluates to ``False``

    Returns:
        tensor_like: A tensor with elements from ``x`` where the condition is ``True``, and
        ``y`` otherwise. The output tensor has the same shape as the input tensors.

    **Example**

    >>> a = torch.tensor([0.6, 0.23, 0.7, 1.5, 1.7], requires_grad=True)
    >>> b = torch.tensor([-1., -2., -3., -4., -5.], requires_grad=True)
    >>> math.where(a < 1, a, b)
    tensor([ 0.6000,  0.2300,  0.7000, -4.0000, -5.0000], grad_fn=<SWhereBackward>)
    """
    return np.where(condition, x, y, like=_multi_dispatch([condition, x, y]))

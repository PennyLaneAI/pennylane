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
"""This module contains the TensorBox abstract base class."""
# pylint: disable=import-outside-toplevel,too-many-public-methods
import abc
import functools
import numbers
import sys


def wrap_output(func):
    """Decorator to automate the wrapping of TensorBox method outputs.

    When applied to a TensorBox method, it inserts an additional argument
    into the signature, ``wrap_output``. By default, this is True, causing
    the output of the method to be wrapped as a TensorBox; specifying
    ``wrap_output=False`` when calling the method results in the
    underlying tensor itself being returned.
    """

    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        wrap = kwargs.pop("wrap_output", True)

        if wrap:
            cls = vars(sys.modules[func.__module__])[func.__qualname__.split(".")[0]]
            return cls(func(*args, **kwargs))

        return func(*args, **kwargs)

    return _wrapper


class TensorBox(abc.ABC):
    """A container for array-like objects that allows array manipulation to be performed in a
    unified manner for supported tensor/array manipulation frameworks.

    Args:
        tensor (tensor_like): instantiate the ``TensorBox`` container with an array-like object

    .. warning::

        The :class:`TensorBox` class is designed for internal use **only**, to ensure that
        PennyLane templates, cost functions, and optimizers retain differentiability
        across all supported interfaces.

        Consider instead using the function wrappers provided in :mod:`~.tensorbox`.

    By wrapping array-like objects in a ``TensorBox`` class, array manipulations are
    performed by simply chaining method calls. Under the hood, the method call is dispatched
    to the corresponding tensor/array manipulation library based on the wrapped array type, without
    the need to import any external libraries manually. As a result, autodifferentiation is
    preserved where needed.

    **Example**

    While this is an abstract base class, this class may be 'instantiated' directly;
    by overloading ``__new__``, the tensor argument is inspected, and the correct subclass
    is returned:

    >>> x = tf.Variable([0.4, 0.1, 0.5])
    >>> y = TensorBox(x)
    >>> print(y)
    TensorBox: <tf.Variable 'Variable:0' shape=(3,) dtype=float32, numpy=array([0.4, 0.1, 0.5], dtype=float32)>

    The original tensor is available via the :meth:`~.unbox` method or the :attr:`data` attribute:

    >>> y.unbox()
    <tf.Variable 'Variable:0' shape=(3,) dtype=float32, numpy=array([0.4, 0.1, 0.5], dtype=float32)>

    In addition, this class defines various abstract methods that all subclasses
    must define. These methods allow for common manipulations and
    linear algebra transformations without the need for importing.

    >>> y.ones_like()
    tf.Tensor([1. 1. 1.], shape=(3,), dtype=float32)

    Unless specified, the returned tensors are also ``TensorBox`` instances, allowing
    for method chaining:

    >>> y.ones_like().expand_dims(0)
    tf.Tensor([[1. 1. 1.]], shape=(1, 3), dtype=float32)
    """

    __array_priority__ = 100
    _initialized = False

    def __new__(cls, tensor):
        if isinstance(tensor, TensorBox):
            return tensor

        if cls is not TensorBox:
            return super(TensorBox, cls).__new__(cls)

        namespace = tensor.__class__.__module__.split(".")[0]
        box = None
        if namespace in ("jax", "jaxlib"):
            from .jax_box import JaxBox

            box = JaxBox.__new__(JaxBox, tensor)

        if isinstance(tensor, (numbers.Number, list, tuple)) or namespace == "numpy":
            from .numpy_box import NumpyBox

            box = NumpyBox.__new__(NumpyBox, tensor)

        if namespace in ("pennylane", "autograd"):
            from .autograd_box import AutogradBox

            box = AutogradBox.__new__(AutogradBox, tensor)

        if namespace == "tensorflow":
            from .tf_box import TensorFlowBox

            box = TensorFlowBox.__new__(TensorFlowBox, tensor)

        if namespace == "torch":
            from .torch_box import TorchBox

            box = TorchBox.__new__(TorchBox, tensor)

        if box is None:
            raise ValueError(f"Unknown tensor type {type(tensor)}")
        return box

    def __init__(self, tensor):
        if self._initialized:
            return

        self.data = tensor
        self._initialized = True

    def __repr__(self):
        return f"TensorBox: {self.data.__repr__()}"

    def __len__(self):
        return len(self.data)

    def __eq__(self, other):
        if other is NotImplemented:
            return NotImplemented

        other = TensorBox(other).numpy()
        return self.numpy() == other

    def __add__(self, other):
        if isinstance(other, TensorBox):
            other = other.data

        return self.__class__(self.data + other)

    def __sub__(self, other):
        if isinstance(other, TensorBox):
            other = other.data

        return self.__class__(self.data - other)

    def __rsub__(self, other):
        if isinstance(other, TensorBox):
            other = other.data

        return self.__class__(other - self.data)

    def __mul__(self, other):
        if isinstance(other, TensorBox):
            other = other.data

        return self.__class__(self.data * other)

    def __truediv__(self, other):
        if isinstance(other, TensorBox):
            other = other.data

        return self.__class__(self.data / other)

    def __rtruediv__(self, other):
        return self.__class__(other / self.data)

    def __pow__(self, other):
        if isinstance(other, TensorBox):
            other = other.data

        return self.__class__(self.data ** other)

    def __rpow__(self, other):
        return self.__class__(other ** self.data)

    __radd__ = __add__
    __rmul__ = __mul__

    @staticmethod
    def unbox_list(tensors):
        """Unboxes or unwraps a list of tensor-like objects, converting any :class:`TensorBox`

        objects in the list into raw interface tensors.

        Args:
            tensors (list[tensor_like]): list of arrays, tensors, or :class:`~.TensorBox` objects

        Returns
            list[tensor_like]: the input list with all :class:`TensorBox` objects
            unwrapped

        **Example**

        >>> x = tf.Variable([0.4, 0.1, 0.5])
        >>> y = TensorBox(x)
        >>> z = tf.constant([0.1, 0.2])

        Note that this is a static method, so we must pass the tensor represented by the ``TensorBox``
        if we would like it to be included.

        >>> res = y.unwrap([y, z])
        >>> res
        [<tf.Variable 'Variable:0' shape=(3,) dtype=float32, numpy=array([0.4, 0.1, 0.5], dtype=float32)>,
         <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.1, 0.2], dtype=float32)>]
        >>> print([type(v) for v in res])
        [<class 'tensorflow.python.ops.resource_variable_ops.ResourceVariable'>,
         <class 'tensorflow.python.framework.ops.EagerTensor'>]
        """
        return [v.data if isinstance(v, TensorBox) else v for v in tensors]

    def unbox(self):
        """Unboxes the ``TensorBox`` container, returning the raw interface tensor."""
        return self.data

    ###############################################################################
    # Abstract methods and properties
    ###############################################################################

    @abc.abstractmethod
    def abs(self):
        """TensorBox: Returns the element-wise absolute value."""

    @abc.abstractmethod
    def angle(self):
        """TensorBox: Returns the elementwise complex angle."""

    @abc.abstractmethod
    def arcsin(self):
        """Returns the element-wise inverse sine of the tensor"""

    @staticmethod
    @abc.abstractmethod
    def astensor(tensor):
        """Converts the input to the native tensor type of the TensorBox.

        Args:
            tensor (tensor_like): array to convert
        """

    @staticmethod
    @abc.abstractmethod
    def block_diag(values):
        """Combine a sequence of 2D tensors to form a block diagonal tensor.

        Args:
            values (Sequence[tensor_like]): Sequence of 2D arrays/tensors to form
                the block diagonal tensor.

        Returns:
            tensor_like: the block diagonal tensor
        """

    @abc.abstractmethod
    def cast(self, dtype):
        """Cast the dtype of the TensorBox.

        Args:
            dtype (np.dtype, str): the NumPy datatype to cast to
                If the boxed tensor is not a NumPy array, the equivalent
                datatype in the target framework is chosen.
        """

    @abc.abstractmethod
    def conj(self):
        """TensorBox: Returns the elementwise conjugation."""

    @staticmethod
    @abc.abstractmethod
    def concatenate(values, axis=0):
        """Join a sequence of tensors along an existing axis.

        Args:
            values (Sequence[tensor_like]): sequence of arrays/tensors to concatenate
            axis (int): axis on which to concatenate

        **Example**

        >>> x = tf.Variable([[1, 2], [3, 4]])
        >>> a = tf.constant([[5, 6]])
        >>> y = TensorBox(x)
        >>> y.concatenate([a, y], axis=0)
        <tf.Tensor: shape=(2, 3), dtype=float32, numpy=
        array([[1, 2],
               [3, 4],
               [5, 6]]), dtype=float32)>

        >>> y.concatenate([a, y], axis=1)
        <tf.Tensor: shape=(2, 3), dtype=float32, numpy=
        array([[1, 2, 5],
               [3, 4, 6]]), dtype=float32)>

        >>> y.concatenate([a, y], axis=None)
        <tf.Tensor: shape=(2, 3), dtype=float32, numpy=
        array([1, 2, 3, 4, 5, 6]), dtype=float32)>

        Note that this is a static method, so we must pass the unified tensor itself
        if we would like it to be included.
        """

    @staticmethod
    @abc.abstractmethod
    def diag(values, k=0):
        """Construct a diagonal tensor from a list of scalars.

        Args:
            values (Sequence[int or float or complex]): sequence of numeric values that
                make up the diagonal
            k (int): The diagonal in question. ``k=0`` corresponds to the main diagonal.
                Use ``k>0`` for diagonals above the main diagonal, and ``k<0`` for
                diagonals below the main diagonal.

        Returns:
            TensorBox: TensorBox containing the 2D diagonal tensor
        """

    @staticmethod
    @abc.abstractmethod
    def dot(x, y):
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
            other (tensor_like): the tensor-like object to right-multiply the TensorBox by
        """

    @abc.abstractmethod
    def expand_dims(self, axis):
        """Expand the shape of the tensor.

        Args:
            axis (int or tuple[int]): the axis or axes where the additional
                dimensions should be inserted
        """

    @abc.abstractmethod
    def gather(self, indices):
        """Gather tensor values given a tuple of indices.

        This is equivalent to the following NumPy fancy indexing:

        ..code-block:: python

            tensor[indices]

        Args:
            indices (Sequence[int]): the indices of the values to extract
        """

    @property
    @abc.abstractmethod
    def interface(self):
        """str, None: The package that the :class:`.TensorBox` class
        will dispatch to. The returned strings correspond to those used for PennyLane
        :doc:`interfaces </introduction/interfaces>`."""

    @abc.abstractmethod
    def numpy(self):
        """Converts the tensor to a standard, non-differentiable NumPy ndarray, or to a Python scalar if
        the tensor is 0-dimensional.

        Returns:
            array, float, int: NumPy ndarray, or Python scalar if the input is 0-dimensional

        **Example**

        >>> x = tf.Variable([0.4, 0.1, 0.5])
        >>> y = TensorBox(x)
        >>> y.numpy()
        array([0.4, 0.1, 0.5], dtype=float32)
        """

    @abc.abstractmethod
    def ones_like(self):
        """Returns a unified tensor of all ones, with the shape and dtype
        of the unified tensor.

        Returns:
            TensorBox: all ones array

        **Example**

        >>> x = tf.Variable([[0.4, 0.1], [0.1, 0.5]])
        >>> y = TensorBox(x)
        >>> y.ones_like()
        tf.Tensor(
        [[1. 1.]
         [1. 1.]], shape=(2, 2), dtype=float32)
        """

    @abc.abstractmethod
    def reshape(self, shape):
        """Gives a new shape to a tensor without changing its data.

        Args:
            shape (tuple[int]): The new shape. The special value of -1 indicates
                that the size of that dimension is computed so that the total size
                remains constant. A dimension of -1 can only be specified once.

        Returns:
            TensorBox: TensorBox containing a new view into the tensor with
            shape ``shape``
        """

    @property
    @abc.abstractmethod
    def requires_grad(self):
        """bool: Whether the TensorBox is considered trainable.


        Note that the implemetation depends on the contained tensor type, and
        may be context dependent.

        For example, Torch tensors and PennyLane tensors track trainability
        as a property of the tensor itself. TensorFlow, on the other hand,

        only tracks trainability if being watched by a gradient tape.
        """

    @abc.abstractmethod
    def scatter_element_add(self, index, value):
        """Add a scalar value to an element of the tensor.

        Args:
            index (tuple[int]): the index of the tensor to update
            value (int or float or complex): Scalar value to add to the
                tensor element, in place.
        """

    @property
    @abc.abstractmethod
    def shape(self):
        """tuple[int]: returns the shape of the tensor as a tuple of integers"""

    @abc.abstractmethod
    def sqrt(self):
        """Returns the square root of the tensor"""

    @staticmethod
    @abc.abstractmethod
    def stack(values, axis=0):
        """Stacks a list of tensors along the specified index.

        Args:
            values (Sequence[tensor_like]): sequence of arrays/tensors to stack
            axis (int): axis on which to stack

        Returns:
            TensorBox: TensorBox containing the stacked array

        **Example**

        >>> x = tf.Variable([0.4, 0.1, 0.5])
        >>> a = tf.constant([1., 2., 3.])
        >>> y = TensorBox(x)
        >>> y.stack([a, y])
        <tf.Tensor: shape=(2, 3), dtype=float32, numpy=
        array([[1. , 2. , 3. ],
               [0.4, 0.1, 0.5]], dtype=float32)>

        Note that this is a static method, so we must pass the unified tensor itself
        if we would like it to be included.
        """

    @abc.abstractmethod
    def sum(self, axis=None, keepdims=False):
        """TensorBox: Returns the sum of the tensor elements across the specified dimensions.

        Args:
            axis (int or tuple[int]): The axis or axes along which to perform the sum.
                If not specified, all elements of the tensor across all dimensions
                will be summed, returning a tensor.
            keepdims (bool): If True, retains all summed dimensions.

        **Example**

        Summing over all dimensions:

        >>> x = tf.Variable([[1., 2.], [3., 4.]])
        >>> y = TensorBox(x)
        >>> y.sum()
        TensorBox: <tf.Tensor: shape=(), dtype=float32, numpy=10.0>

        Summing over specified dimensions:

        >>> x = np.array([[[1, 1], [5, 3]], [[1, 4], [-6, -1]]])
        >>> y = TensorBox(x)
        >>> y.shape
        (2, 2, 2)
        >>> y.sum(axis=(0, 2))
        TensorBox: tensor([7, 1], requires_grad=True)
        >>> y.sum(axis=(0, 2), keepdims=True)
        TensorBox: tensor([[[7],
                            [1]]], requires_grad=True)
        """

    @abc.abstractmethod
    def T(self):
        """Returns the transpose of the tensor."""

    @abc.abstractmethod
    def take(self, indices, axis=None):
        """Gather elements from a tensor.

        Note that ``tensorbox.take(indices, axis=3)`` is equivalent
        to ``tensor[:, :, :, indices, ...]`` for frameworks that support
        NumPy-like fancy indexing.

        This method is roughly equivalent to ``np.take`` and ``tf.gather``.
        In the case of a 1-dimensional set of indices, it is roughly equivalent
        to ``torch.index_select``, but deviates for multi-dimensional indices.

        Args:
            indices (Sequence[int]): the indices of the values to extract
            axis: The axis over which to select the values. If not provided,
                the tensor is flattened before value extraction.

        **Example**

        >>> x = torch.tensor([[1, 2], [3, 4]])
        >>> y = qml.math.TensorBox(x)
        >>> y.take([[0, 0], [1, 0]], axis=1)
        TensorBox: tensor([[[1, 1],
                 [2, 1]],

                [[3, 3],
                 [4, 3]]])
        """

    @staticmethod
    @abc.abstractmethod
    def where(condition, x, y):
        """Return a tensor of elements selected from ``x`` if the condition is True,
        ``y`` otherwise."""

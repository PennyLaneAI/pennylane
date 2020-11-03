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
"""
Wrappers for common functions that manipulate or create
NumPy, TensorFlow, and Torch data structures.
"""
# pylint: disable=import-outside-toplevel
import abc


class TensorBox(abc.ABC):
    """A container for array-like objects that allows array manipulation to be performed in a
    unified manner for supported linear algebra packages.

    Args:
        tensor (array_like): instantiate the ``TensorBox`` container with an array-like object

    .. warning::

        The :class:`TensorBox` class is designed for internal use **only**, to ensure that
        PennyLane templates, cost functions, and optimizers retain differentiability
        across all supported interfaces.

        For front-facing usage, consider using an established package such as
        `EagerPy <https://github.com/jonasrauber/eagerpy>`_.


    By wrapping array-like objects in a ``TensorBox`` class, array manipulations are
    performed by simply chaining method calls. Under the hood, the method call is dispatched
    to the corresponding linear algebra library based on the wrapped array type, without
    the need to import any external libraries manually. As a result, autodifferentiation is
    preserved where needed.

    Currently, the following linear algebra packages are supported:

    * NumPy
    * Autograd
    * TensorFlow
    * Torch

    **Example**

    While this is an abstract base class, this class may be 'instantiated' directly;
    by overloading ``__new__``, the tensor argument is inspected, and the correct subclass
    is returned:

    >>> x = tf.Variable([0.4, 0.1, 0.5])
    >>> y = TensorBox(x)
    >>> print(y)
    >>> type(y)
    <TensorBox <tf.Variable 'Variable:0' shape=(3,) dtype=float32, numpy=array([0.4, 0.1, 0.5], dtype=float32)>>

    The original tensor is available via the :meth:`~.unbox()` method:

    >>> y.unbox()
    <tf.Variable 'Variable:0' shape=(3,) dtype=float32, numpy=array([0.4, 0.1, 0.5], dtype=float32)>

    In addition, this class defines various abstract methods that all subclasses
    must define. These methods allow for common manipulations and
    linear algebra transformations without the need for importing.

    >>> y.ones_like()
    tf.Tensor([1. 1. 1.], shape=(3,), dtype=float32)

    Unless specified, the returned tensors are also unified tensors, allowing
    for method chaining:

    >>> y.ones_like().expand_dims(0)
    tf.Tensor([[1. 1. 1.]], shape=(1, 3), dtype=float32)
    """

    def __new__(cls, tensor):
        if isinstance(tensor, TensorBox):
            return tensor

        if cls is not TensorBox:
            return super(TensorBox, cls).__new__(cls)

        namespace = tensor.__class__.__module__.split(".")[0]

        if isinstance(tensor, (list, tuple)) or namespace == "numpy":
            from .numpy_box import NumpyBox, np

            return NumpyBox.__new__(NumpyBox, np.array(tensor))

        if namespace in ("pennylane", "autograd"):
            from .autograd_box import AutogradBox

            return AutogradBox.__new__(AutogradBox, tensor)

        if namespace == "tensorflow":
            from .tf_box import TensorFlowBox

            return TensorFlowBox.__new__(TensorFlowBox, tensor)

        if namespace == "torch":
            from .torch_box import TorchBox

            return TorchBox.__new__(TorchBox, tensor)

        raise ValueError(f"Unknown tensor type {type(tensor)}")

    def __init__(self, tensor):
        self._data = tensor

    def __repr__(self):
        return f"<TensorBox {self.unbox().__repr__()}>"

    def __len__(self):
        return len(self.unbox())

    def __mul__(self, other):
        if isinstance(other, TensorBox):
            other = other.unbox()

        return self.__class__(self.unbox() * other)

    def __rmul__(self, other):
        if isinstance(other, TensorBox):
            other = other.unbox()

        return self.__class__(other * self.unbox())

    @staticmethod
    def unbox_list(tensors):
        """Unboxes or unwraps a list of array like objects, converting any :class:`TensorBox`
        objects in the list into raw interface tensors.

        Args:
            tensors (list[array_like]): list of arrays, tensors, or :class:`~.TensorBox` objects

        Returns
            list[array_like]: the input list with all :class:`TensorBox` objects
            unwrapped.

        **Example**

        >>> x = tf.Variable([0.4, 0.1, 0.5])
        >>> y = TensorBox(x)
        >>> z = tf.constant([0.1, 0.2])

        Note that this is a static method, so we must pass the unified tensors itself
        if we would like it to be included.

        >>> res = y.unwrap([y, z])
        >>> res
        [<tf.Variable 'Variable:0' shape=(3,) dtype=float32, numpy=array([0.4, 0.1, 0.5], dtype=float32)>,
         <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.1, 0.2], dtype=float32)>]
        >>> print([type(v) for v in res])
        [<class 'tensorflow.python.ops.resource_variable_ops.ResourceVariable'>,
         <class 'tensorflow.python.framework.ops.EagerTensor'>]
        """
        return [v.unbox() if isinstance(v, TensorBox) else v for v in tensors]

    def unbox(self):
        """Unboxes the ``TensorBox`` container, returning the raw interface tensor."""
        return self._data

    @property
    @abc.abstractmethod
    def interface(self):
        """str, None: The linear algebra package that the :class:`.TensorBox` class
        will dispatch to. The returned strings correspond to those used for PennyLane
        :doc:`interfaces </introduction/interfaces>`."""

    @property
    @abc.abstractmethod
    def shape(self):
        """Returns the shape of the tensor as a tuple of integers."""

    @property
    @abc.abstractmethod
    def T(self):
        """Returns the transpose of the tensor."""

    @abc.abstractmethod
    def expand_dims(self, axis):
        """Expand the shape of the tensor.

        Args:
            axis (int or tuple[int]): the axis or axes where the additional
                dimensions should be inserted
        """

    @abc.abstractmethod
    def numpy(self):
        """Converts the tensor to a standard, non-differentiable NumPy ndarray or Python scalar if
        the tensor is 0-dimensional.

        Returns:
            array:

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

    @staticmethod
    @abc.abstractmethod
    def stack(values, axis=0):
        """Stacks a list of tensors along the specified index.

        Args:
            values (Sequence[array_like]): sequence of arrays/tensors to stack
            axis (int): axis on which to stack

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

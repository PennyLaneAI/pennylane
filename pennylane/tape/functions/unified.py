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
import abc


class UnifiedTensor(abc.ABC):
    """Creates a unified tensor wrapper that represents interface array/tensor datatypes
    using a common API.

    While this is an abstract base class, this class may be 'instantiated' directly;
    by overloading ``__new__``, the tensor argument is inspected, and the correct subclass
    is returned:

    >>> x = tf.Variable([0.4, 0.1, 0.5])
    >>> y = UnifiedTensor(x)
    >>> type(y)
    <class 'pennylane.tape.functions.tf.TensorFlowTensor'>

    The original tensor is available via the ``data`` attribute.

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
        if isinstance(tensor, UnifiedTensor):
            return tensor

        if cls is not UnifiedTensor:
            return super(UnifiedTensor, cls).__new__(cls)

        if isinstance(tensor, (list, tuple)):
            from .autograd import AutogradTensor, np

            return AutogradTensor.__new__(AutogradTensor, np.array(tensor))

        namespace = tensor.__class__.__module__.split(".")[0]

        if namespace in ("pennylane", "autograd", "numpy"):
            from .autograd import AutogradTensor

            return AutogradTensor.__new__(AutogradTensor, tensor)

        if namespace == "tensorflow":
            from .tf import TensorFlowTensor

            return TensorFlowTensor.__new__(TensorFlowTensor, tensor)

        if namespace == "torch":
            from .torch import TorchTensor

            return TorchTensor.__new__(TorchTensor, tensor)

        raise ValueError(f"Unknown tensor type {type(tensor)}")

    def __init__(self, tensor):
        self.data = tensor

    def __repr__(self):
        return self.data.__repr__()

    def __str__(self):
        return self.data.__str__()

    def __len__(self):
        return len(self.data)

    def __mul__(self, other):
        if isinstance(other, UnifiedTensor):
            other = other.data

        return self.__class__(self.data * other)

    def __rmul__(self, other):
        if isinstance(other, UnifiedTensor):
            other = other.data

        return self.__class__(other * self.data)

    @staticmethod
    def unwrap(tensors):
        """Unwraps a list of tensors, converting any :class:`UnifiedTensor`
        objects in the list into raw interface tensors.

        Args:
            tensors (list[array_like]): list of arrays, tensors, or :class:`~.UnifiedTensor` objects

        Returns
            list[array_like]: the input list with all :class:`UnifiedTensor` objects
            unwrapped.

        **Example**

        >>> x = tf.Variable([0.4, 0.1, 0.5])
        >>> y = UnifiedTensor(x)
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
        return [v.data if isinstance(v, UnifiedTensor) else v for v in tensors]

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
        >>> y = UnifiedTensor(x)
        >>> y.numpy()
        array([0.4, 0.1, 0.5], dtype=float32)
        """

    @abc.abstractmethod
    def ones_like(self):
        """Returns a unified tensor of all ones, with the shape and dtype
        of the unified tensor.

        Returns:
            UnifiedTensor: all ones array

        **Example**

        >>> x = tf.Variable([[0.4, 0.1], [0.1, 0.5]])
        >>> y = UnifiedTensor(x)
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
        >>> y = UnifiedTensor(x)
        >>> y.stack([a, y])
        <tf.Tensor: shape=(2, 3), dtype=float32, numpy=
        array([[1. , 2. , 3. ],
               [0.4, 0.1, 0.5]], dtype=float32)>

        Note that this is a static method, so we must pass the unified tensor itself
        if we would like it to be included.
        """

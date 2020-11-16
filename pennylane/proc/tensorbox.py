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
"""This module contains the TensorBox abstract base class."""
# pylint: disable=import-outside-toplevel
import abc

import numpy as np


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

    _initialized = False

    def __new__(cls, tensor):
        if isinstance(tensor, TensorBox):
            return tensor

        if cls is not TensorBox:
            return super(TensorBox, cls).__new__(cls)

        namespace = tensor.__class__.__module__.split(".")[0]

        if isinstance(tensor, (list, tuple)) or namespace == "numpy":
            from .numpy_box import NumpyBox

            return NumpyBox.__new__(NumpyBox, tensor)

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

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """By defining this special method, NumPy ufuncs can act directly
        on the contained tensor, with broadcasting taken into account. For
        more details, see https://numpy.org/devdocs/user/basics.subclassing.html#array-ufunc-for-ufuncs"""
        outputs = [v.data if isinstance(v, TensorBox) else v for v in kwargs.get("out", ())]

        if outputs:
            # Insert the unwrapped outputs into the keyword
            # args dictionary, to be passed to ndarray.__array_ufunc__
            outputs = tuple(outputs)
            kwargs["out"] = outputs
        else:
            # If the ufunc has no outputs, we simply
            # create a tuple containing None for all potential outputs.
            outputs = (None,) * ufunc.nout

        args = [v.data if isinstance(v, TensorBox) else v for v in inputs]
        res = getattr(ufunc, method)(*args, **kwargs)

        if ufunc.nout == 1:
            res = (res,)

        # construct a list of ufunc outputs to return
        ufunc_output = []
        for result, output in zip(res, outputs):
            if output is not None:
                ufunc_output.append(output)
            else:
                if isinstance(result, np.ndarray):
                    if result.ndim == 0 and result.dtype == np.dtype("bool"):
                        ufunc_output.append(result.item())
                    else:
                        ufunc_output.append(self.__class__(result))
                else:
                    ufunc_output.append(result)

        if len(ufunc_output) == 1:
            # the ufunc has a single output so return a single tensor
            return ufunc_output[0]

        # otherwise we must return a tuple of tensors
        return tuple(ufunc_output)

    def __init__(self, tensor):
        if self._initialized:
            return

        self.data = tensor
        self._initialized = True

    def __repr__(self):
        return f"TensorBox: {self.data.__repr__()}"

    def __len__(self):
        return len(self.data)

    def __add__(self, other):
        if isinstance(other, TensorBox):
            other = other.data

        return self.__class__(self.data + other)

    def __sub__(self, other):
        if isinstance(other, TensorBox):
            other = other.data

        return self.__class__(self.data - other)

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
    __rsub__ = __sub__
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

    @staticmethod
    @abc.abstractmethod
    def astensor(tensor):
        """Converts the input to the native tensor type of the TensorBox.

        Args:
            tensor (tensor_like): array to convert
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
    def expand_dims(self, axis):
        """Expand the shape of the tensor.

        Args:
            axis (int or tuple[int]): the axis or axes where the additional
                dimensions should be inserted
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

    @property
    @abc.abstractmethod
    def shape(self):
        """tuple[int]: returns the shape of the tensor as a tuple of integers"""

    @staticmethod
    @abc.abstractmethod
    def stack(values, axis=0):
        """Stacks a list of tensors along the specified index.

        Args:
            values (Sequence[tensor_like]): sequence of arrays/tensors to stack
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

    @property
    @abc.abstractmethod
    def T(self):
        """Returns the transpose of the tensor."""

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
"""
This module provides the PennyLane :class:`~.tensor` class.
"""
import numpy as onp
from autograd import numpy as _np
from autograd.core import VSpace
from autograd.extend import defvjp, primitive
from autograd.numpy.numpy_boxes import ArrayBox
from autograd.numpy.numpy_vspaces import ArrayVSpace, ComplexArrayVSpace
from autograd.tracer import Box

from pennylane.exceptions import NonDifferentiableError

__doc__ = "NumPy with automatic differentiation support, provided by Autograd and PennyLane."


# Hotfix since _np.asarray doesn't have a gradient rule defined.
@primitive
def asarray(vals, *args, **kwargs):
    """Gradient supporting autograd asarray"""
    if isinstance(vals, (onp.ndarray, _np.ndarray)):
        return _np.asarray(vals, *args, **kwargs)
    return _np.array(vals, *args, **kwargs)


def asarray_gradmaker(ans, *args, **kwargs):
    """Gradient maker for asarray"""
    del ans, args, kwargs
    return lambda g: g


defvjp(asarray, asarray_gradmaker, argnums=(0,))


class tensor(_np.ndarray):
    """Constructs a PennyLane tensor for use with Autograd QNodes.

    The ``tensor`` class is a subclass of ``numpy.ndarray``,
    providing the same multidimensional, homogeneous data-structure
    of fixed-size items, with an additional flag to indicate to PennyLane
    whether the contained data is differentiable or not.

    .. warning::

        PennyLane ``tensor`` objects are only used as part of the Autograd QNode
        interface. If using another machine learning library such as PyTorch or
        TensorFlow, use their built-in ``tf.Variable`` and ``torch.tensor`` classes
        instead.

    .. warning::

        Tensors should be constructed using standard array construction functions
        provided as part of PennyLane's NumPy implementation, including
        ``np.array``, ``np.zeros`` or ``np.empty``.

        The parameters given here refer to a low-level class
        for instantiating tensors.


    Args:
        input_array (array_like): Any data structure in any form that can be converted to
            an array. This includes lists, lists of tuples, tuples, tuples of tuples,
            tuples of lists and ndarrays.
        requires_grad (bool): whether the tensor supports differentiation

    **Example**

    The trainability of a tensor can be set on construction via the
    ``requires_grad`` keyword argument,

    >>> from pennylane import numpy as np
    >>> x = np.array([0, 1, 2], requires_grad=True)
    >>> x
    tensor([0, 1, 2], requires_grad=True)

    or in-place by modifying the ``requires_grad`` attribute:

    >>> x.requires_grad = False
    >>> x
    tensor([0, 1, 2], requires_grad=False)

    Since tensors are subclasses of ``np.ndarray``, they can be provided as arguments
    to any PennyLane-wrapped NumPy function:

    >>> np.sin(x)
    tensor([0.        , 0.84147098, 0.90929743], requires_grad=False)

    When composing functions of multiple tensors, if at least one input tensor is differentiable,
    then the output will also be differentiable:

    >>> x = np.array([0, 1, 2], requires_grad=False)
    >>> y = np.zeros([3], requires_grad=True)
    >>> np.vstack([x, y])
    tensor([[0., 1., 2.],
        [0., 0., 0.]], requires_grad=True)
    """

    def __new__(cls, input_array, *args, requires_grad=True, **kwargs):
        obj = asarray(input_array, *args, **kwargs)

        if isinstance(obj, onp.ndarray):
            obj = obj.view(cls)
            obj.requires_grad = requires_grad

        return obj

    def __array_finalize__(self, obj):
        # pylint: disable=attribute-defined-outside-init
        if obj is None:  # pragma: no cover
            return

        self.requires_grad = getattr(obj, "requires_grad", None)

    def __repr__(self):
        string = super().__repr__()
        return string[:-1] + f", requires_grad={self.requires_grad})"

    def __array_wrap__(self, obj):
        out_arr = tensor(obj, requires_grad=self.requires_grad)
        return super().__array_wrap__(out_arr)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):

        # unwrap any outputs the ufunc might have
        outputs = [i.view(onp.ndarray) for i in kwargs.get("out", ())]

        if outputs:
            # Insert the unwrapped outputs into the keyword
            # args dictionary, to be passed to ndarray.__array_ufunc__
            outputs = tuple(outputs)
            kwargs["out"] = outputs
        else:
            # If the ufunc has no ouputs, we simply
            # create a tuple containing None for all potential outputs.
            outputs = (None,) * ufunc.nout

        # unwrap the input arguments to the ufunc
        args = [i.unwrap() if hasattr(i, "unwrap") else i for i in inputs]

        # call the ndarray.__array_ufunc__ method to compute the result
        # of the vectorized ufunc
        res = super().__array_ufunc__(ufunc, method, *args, **kwargs)

        if ufunc.nout == 1:
            res = (res,)

        # construct a list of ufunc outputs to return
        ufunc_output = [
            (onp.asarray(result) if output is None else output)
            for result, output in zip(res, outputs)
        ]

        # if any of the inputs were trainable, the output is also trainable
        requires_grad = any(
            isinstance(x, onp.ndarray) and getattr(x, "requires_grad", True) for x in inputs
        )

        # Iterate through the ufunc outputs and convert each to a PennyLane tensor.
        # We also correctly set the requires_grad attribute.
        for i in range(len(ufunc_output)):  # pylint: disable=consider-using-enumerate
            ufunc_output[i] = tensor(ufunc_output[i], requires_grad=requires_grad)

        if len(ufunc_output) == 1:
            # the ufunc has a single output so return a single tensor
            return ufunc_output[0]

        # otherwise we must return a tuple of tensors
        return tuple(ufunc_output)

    def __getitem__(self, *args, **kwargs):
        item = super().__getitem__(*args, **kwargs)

        if not isinstance(item, tensor):
            item = tensor(item, requires_grad=self.requires_grad)

        return item

    def __hash__(self):
        if self.ndim == 0:
            # Allowing hashing if the tensor is a scalar.
            # We hash both the scalar value *and* the differentiability information,
            # to match the behaviour of PyTorch.
            return hash((self.item(), self.requires_grad))

        raise TypeError("unhashable type: 'numpy.tensor'")

    def __reduce__(self):
        # Called when pickling the object.
        # Numpy ndarray uses __reduce__ instead of __getstate__ to prepare an object for
        # pickling. self.requires_grad needs to be included in the tuple returned by
        # __reduce__ in order to be preserved in the unpickled object.
        reduced_obj = super().__reduce__()
        # The last (2nd) element of this tuple holds the data. Add requires_grad to this:
        full_reduced_data = reduced_obj[2] + (self.requires_grad,)
        return (reduced_obj[0], reduced_obj[1], full_reduced_data)

    def __setstate__(self, reduced_obj) -> None:
        # Called when unpickling the object.
        # Set self.requires_grad with the last element in the tuple returned by __reduce__:
        # pylint: disable=attribute-defined-outside-init
        self.requires_grad = reduced_obj[-1]
        # And call parent's __setstate__ without this element:
        super().__setstate__(reduced_obj[:-1])

    def unwrap(self):
        """Converts the tensor to a standard, non-differentiable NumPy ndarray or Python scalar if
        the tensor is 0-dimensional.

        All information regarding differentiability of the tensor will be lost.

        .. warning::

            The returned array is a new view onto the **same data**. That is,
            the tensor and the returned ``ndarray`` share the same underlying storage.
            Changes to the tensor object will be reflected within the returned array,
            and vice versa.

        **Example**

        >>> from pennylane import numpy as np
        >>> x = np.array([1, 2], requires_grad=True)
        >>> x
        tensor([1, 2], requires_grad=True)
        >>> x.unwrap()
        array([1, 2])

        Zero dimensional array are converted to Python scalars:

        >>> x = np.array(1.543, requires_grad=False)
        >>> x.unwrap()
        1.543
        >>> type(x.unwrap())
        <class 'float'>

        The underlying data is **not** copied:

        >>> x = np.array([1, 2], requires_grad=True)
        >>> y = x.unwrap()
        >>> x[0] = 5
        >>> y
        array([5, 2])
        >>> y[1] = 7
        >>> x
        tensor([5, 7], requires_grad=True)


        To create a copy, the ``copy()`` method can be used:

        >>> x = np.array([1, 2], requires_grad=True)
        >>> y = x.unwrap().copy()
        >>> x[0] = 5
        >>> y
        array([1, 2])
        """
        if self.ndim == 0:
            return self.view(onp.ndarray).item()

        return self.view(onp.ndarray)

    def numpy(self):
        """Converts the tensor to a standard, non-differentiable NumPy ndarray or Python scalar if
        the tensor is 0-dimensional.

        This method is an alias for :meth:`~.unwrap`. See :meth:`~.unwrap` for more details.
        """
        return self.unwrap()


def tensor_to_arraybox(x, *args):
    """Convert a :class:`~.tensor` to an Autograd ``ArrayBox``.

    Args:
        x (array_like): Any data structure in any form that can be converted to
            an array. This includes lists, lists of tuples, tuples, tuples of tuples,
            tuples of lists and ndarrays.

    Returns:
        autograd.numpy.numpy_boxes.ArrayBox: Autograd ArrayBox instance of the array

    Raises:
        NonDifferentiableError: if the provided tensor is non-differentiable
    """
    if isinstance(x, tensor):
        if x.requires_grad:
            return ArrayBox(x, *args)

        raise NonDifferentiableError(
            f"{x} is non-differentiable. Set the requires_grad attribute to True."
        )

    return ArrayBox(x, *args)


Box.type_mappings[tensor] = tensor_to_arraybox
VSpace.mappings[tensor] = lambda x: ComplexArrayVSpace(x) if onp.iscomplexobj(x) else ArrayVSpace(x)

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
This module provides the PennyLane :class:`~.tensor` class.
"""
import numpy as onp

from autograd import numpy as _np

from autograd.tracer import Box
from autograd.numpy.numpy_boxes import ArrayBox
from autograd.numpy.numpy_vspaces import ComplexArrayVSpace, ArrayVSpace
from autograd.core import VSpace


__doc__ = "NumPy with automatic differentiation support, provided by Autograd and PennyLane."


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
    tensor([0, 1, 2], requires_grad=False)

    Since tensors are subclasses of ``np.ndarray``, they can be provided as arguments
    to any PennyLane-wrapped NumPy function:

    >>> np.sin(x)
    tensor([0.        , 0.84147098, 0.90929743], requires_grad=True)

    When composing functions of multiple tensors, if at least one input tensor is differentiable,
    then the output will also be differentiable:

    >>> x = np.array([0, 1, 2], requires_grad=False)
    >>> y = np.zeros([3], requires_grad=True)
    >>> np.vstack([x, y])
    tensor([[0., 1., 2.],
        [0., 0., 0.]], requires_grad=True)
    """

    def __new__(cls, input_array, *args, requires_grad=True, **kwargs):
        obj = _np.array(input_array, *args, **kwargs)

        if isinstance(obj, _np.ndarray):
            obj = obj.view(cls)
            obj.requires_grad = requires_grad

        return obj

    def __array_finalize__(self, obj):
        # pylint: disable=attribute-defined-outside-init
        if obj is None:
            return

        self.requires_grad = getattr(obj, "requires_grad", None)

    def __repr__(self):
        string = super().__repr__()
        return string[:-1] + ", requires_grad={})".format(self.requires_grad)


class NonDifferentiableError(Exception):
    """Exception raised if attempting to differentiate non-trainable
    :class:`~.tensor` using Autograd."""


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
            "{} is non-differentiable. Set the requires_grad attribute to True.".format(x)
        )

    return ArrayBox(x, *args)


Box.type_mappings[tensor] = tensor_to_arraybox
VSpace.mappings[tensor] = lambda x: ComplexArrayVSpace(x) if onp.iscomplexobj(x) else ArrayVSpace(x)

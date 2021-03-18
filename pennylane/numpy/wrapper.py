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
This module provides the PennyLane wrapper functions for modifying NumPy,
such that it accepts the PennyLane :class:`~.tensor` class.
"""
from collections.abc import Sequence
import functools

from autograd import numpy as _np

from .tensor import tensor


def extract_tensors(x):
    """Iterate through an iterable, and extract any PennyLane
    tensors that appear.
    """
    if isinstance(x, tensor):
        # If the item is a tensor, return it
        yield x
    elif isinstance(x, Sequence) and not isinstance(x, (str, bytes)):
        # If the item is a sequence, recursively look through its
        # elements for tensors.
        # NOTE: we choose to branch on Sequence here and not Iterable,
        # as NumPy arrays are not Sequences.
        for item in x:
            yield from extract_tensors(item)


def tensor_wrapper(obj):
    """Decorator that wraps callable objects and classes so that they both accept
    a ``requires_grad`` keyword argument, as well as returning a PennyLane
    :class:`~.tensor`.

    Only if the decorated object returns an ``ndarray`` is the
    output converted to a :class:`~.tensor`; this avoids superfluous conversion
    of scalars and other native-Python types.

    Args:
        obj: a callable object or class
    """

    @functools.wraps(obj)
    def _wrapped(*args, **kwargs):
        """Wrapped NumPy function"""

        tensor_kwargs = {}

        if "requires_grad" in kwargs:
            tensor_kwargs["requires_grad"] = kwargs.pop("requires_grad")
        else:
            tensor_args = list(extract_tensors(args))

            if tensor_args:
                # Unless the user specifies otherwise, if all tensors in the argument
                # list are non-trainable, the output is also non-trainable.
                # Equivalently: if any tensor is trainable, the output is also trainable.
                # NOTE: Use of Python's ``any`` results in an infinite recursion,
                # and I'm not sure why. Using ``np.any`` works fine.
                tensor_kwargs["requires_grad"] = _np.any([i.requires_grad for i in tensor_args])

        # evaluate the original object
        res = obj(*args, **kwargs)

        if isinstance(res, _np.ndarray):
            # only if the output of the object is a ndarray,
            # then convert to a PennyLane tensor
            res = tensor(res, **tensor_kwargs)

        return res

    return _wrapped


def wrap_arrays(old, new):
    """Loop through an object's symbol table,

    wrapping each function with :func:`~.tensor_wrapper`.

    Args:
        old (dict): The symbol table to be wrapped. Note that
            callable classes are ignored; only function are wrapped.
        new (dict): The symbol table that contains the wrapped values.
    """
    for name, obj in old.items():
        if callable(obj) and not isinstance(obj, type):
            new[name] = tensor_wrapper(obj)

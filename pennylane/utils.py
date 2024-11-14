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
This module contains utilities and auxiliary functions which are shared
across the PennyLane submodules.
"""
import functools
import inspect
import numbers

# pylint: disable=protected-access,too-many-branches
from collections.abc import Iterable

import numpy as np

import pennylane as qml


def _flatten(x):
    """Iterate recursively through an arbitrarily nested structure in depth-first order.

    See also :func:`_unflatten`.

    Args:
        x (array, Iterable, Any): each element of an array or an Iterable may itself be any of these types

    Yields:
        Any: elements of x in depth-first order
    """
    if isinstance(x, np.ndarray):
        yield from _flatten(x.flat)  # should we allow object arrays? or just "yield from x.flat"?
    elif isinstance(x, qml.wires.Wires):
        # Reursive calls to flatten `Wires` will cause infinite recursion (`Wires` atoms are `Wires`).
        # Since Wires are always flat, just yield.
        yield from x
    elif isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
        for item in x:
            yield from _flatten(item)
    else:
        yield x


def _unflatten(flat, model):
    """Restores an arbitrary nested structure to a flattened iterable.

    See also :func:`_flatten`.

    Args:
        flat (array): 1D array of items
        model (array, Iterable, Number): model nested structure

    Raises:
        TypeError: if ``model`` contains an object of unsupported type

    Returns:
        Union[array, list, Any], array: first elements of flat arranged into the nested
        structure of model, unused elements of flat
    """
    if isinstance(model, (numbers.Number, str)):
        return flat[0], flat[1:]

    if isinstance(model, np.ndarray):
        idx = model.size
        res = np.array(flat)[:idx].reshape(model.shape)
        return res, flat[idx:]

    if isinstance(model, Iterable):
        res = []
        for x in model:
            val, flat = _unflatten(flat, x)
            res.append(val)
        return res, flat

    raise TypeError(f"Unsupported type in the model: {type(model)}")


def unflatten(flat, model):
    """Wrapper for :func:`_unflatten`.

    Args:
        flat (array): 1D array of items
        model (array, Iterable, Number): model nested structure

    Raises:
        ValueError: if ``flat`` has more elements than ``model``
    """
    # pylint:disable=len-as-condition
    res, tail = _unflatten(np.asarray(flat), model)
    if len(tail) != 0:
        raise ValueError("Flattened iterable has more elements than the model.")
    return res


def _inv_dict(d):
    """Reverse a dictionary mapping.

    Returns multimap where the keys are the former values,
    and values are sets of the former keys.

    Args:
        d (dict[a->b]): mapping to reverse

    Returns:
        dict[b->set[a]]: reversed mapping
    """
    ret = {}
    for k, v in d.items():
        ret.setdefault(v, set()).add(k)
    return ret


def _get_default_args(func):
    """Get the default arguments of a function.

    Args:
        func (callable): a function

    Returns:
        dict[str, tuple]: mapping from argument name to (positional idx, default value)
    """
    signature = inspect.signature(func)
    return {
        k: (idx, v.default)
        for idx, (k, v) in enumerate(signature.parameters.items())
        if v.default is not inspect.Parameter.empty
    }

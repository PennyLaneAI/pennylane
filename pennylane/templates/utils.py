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
r"""
Utility functions used in the templates.
"""
#pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
from collections.abc import Iterable

import numpy as np
from copy import deepcopy
from pennylane.variable import Variable


def _check_no_variable(arg, msg):
    """Checks that `arg` does not represent or contain a :func:`~.pennylane.Variable` object.

    This ensures that the user has not passed `arg` to the qnode as a
    positional argument.

    Args:
        arg: argument to check
        msg (str): error message to display
    """

    if isinstance(arg, Variable):
        raise ValueError(msg)

    if isinstance(arg, Iterable):
        if any([isinstance(a_, Variable) for a_ in arg]):
            raise ValueError(msg)


def _check_wires(wires):
    """Standard checks for the wires argument.

    Args:
        wires (int or list): (subset of) wires of a quantum node, a list of positive integers
                             or a single positive integer

    Return:
        list: list of wire indices

    Raises:
        ValueError: if the wires argument is invalid
    """
    if isinstance(wires, int):
        wires = [wires]

    msg = "wires must be a positive integer or a " \
          "list of positive integers; got {}.".format(wires)
    if not isinstance(wires, Iterable):
        raise ValueError(msg)
    if not all([isinstance(w, int) for w in wires]):
        raise ValueError(msg)
    if any([w < 0 for w in wires]):
        raise ValueError(msg)
    return wires


def _get_shape(inpt):
    """Turn ``inpt`` to an array and return its shape.

    Args:
        inpt (list): input to a qnode

    Returns:
        tuple: shape of ``inpt``
    """

    inpt = deepcopy(inpt)

    if isinstance(inpt, (float, int, complex)):
        shape = ()

    else:
        # turn lists into array to get shape
        if isinstance(inpt, list):
            inpt = np.array(inpt)

        try:
            shape = inpt.shape
        except AttributeError:
            raise ValueError("could not extract shape of object of type {}".format(type(inpt)))

        # turn result into tuple to avoid type TensorShape
        shape = tuple(shape)

    return shape


def _check_shape(inpt, target_shape, msg, bound=None):
    """Checks that the shape of ``inpt`` is equal to the target shape.

    Args:
        inpt (list): input to a qnode
        target_shape (tuple[int]): expected shape of inpt
        msg (str): error message to display if the shapes are different
        bound (str): If 'max' or 'min', the target shape is merely required to be a bound on the input shape

    Raises:
        ValueError
    """

    shape = _get_shape(inpt)

    if bound == 'max':
        if shape > target_shape:
            raise ValueError(msg)
    elif bound == 'min':
        if shape < target_shape:
            raise ValueError(msg)
    else:
        if shape != target_shape:
            raise ValueError(msg)

    return shape


def _check_shapes(inpt_list, target_shapes, msg, bounds=None):
    """Checks that the shape of elements in the ``inpt`` list are equal to the shapes of elements
    in the ``target shape`` list.

    Args:
        inpt_list (list): list of elements of which to check the shape
        target_shapes (list): list of target shapes, of same length as ``inpt_list``
        msg (str): error message to display
        bounds (list): list of 'max' or 'min', indicating the bound that the target shape imposes on the input
            shape

    Raises:
        ValueError
    """

    if bounds is None:
        bounds = [None] * len(inpt_list)

    shape_list = [_check_shape(l, t, bound=b, msg=msg) for l, t, b in zip(inpt_list, target_shapes, bounds)]
    return shape_list


def _check_is_in_options(element, options, msg):
    """Checks that the value of ``element`` is in ``options``.

    Args:
        element: arbitraty variable
        options: possible options for ``element``
        msg (str): error message to display
    """

    if element not in options:
        raise ValueError(msg)


def _check_type(element, types, msg):
    """Checks that the type of ``element'' is one of ``types``.

    Args:
        element: variable to check
        types (list): possible types for ``element``
         msg (str): error message to display
    """

    if not any([isinstance(element, t) for t in types]):
        raise ValueError(msg)


def _check_number_of_layers(list_of_weights):
    """Checks that all weights in ``list_of_weights`` have the same first dimension, which is interpreted
    as the number of layers.

    Args:
        list_of_weights (list): list of all weights to the template

    Returns:
        int: number of layers

    Raises:
        ValueError
    """

    shapes = [_get_shape(weight) for weight in list_of_weights]

    if any(len(s) == 0 for s in shapes):
        raise ValueError("the first dimension of the weight parameters must be the number of layers in the "
                         "template; got scalar weights.")

    first_dimensions = [s[0] for s in shapes]
    different_first_dims = set(first_dimensions)
    n_different_first_dims = len(different_first_dims)

    if n_different_first_dims > 1:
        raise ValueError("the first dimension of the weight parameters must be the number of layers in the "
                         "template; got differing first dimensions: {}.".format(*different_first_dims))

    return first_dimensions[0]

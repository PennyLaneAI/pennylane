# Copyright 2018 Xanadu Quantum Technologies Inc.

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

from pennylane.variable import Variable


def _check_no_variable(arg, msg=None):
    """Checks that `arg` does not contain a pennylane.Variable object.

    Ensures that the user has not passed `arg` to the qnode as a
    positional argument.

    Args:
        arg: argument to check
        msg (str): error message
    """

    if isinstance(arg, Variable):
        raise ValueError(msg)

    if isinstance(arg, Iterable):
        if any([isinstance(a_, Variable) for a_ in arg]):
            raise ValueError(msg)


def _check_wires(wires):
    """Standard checks for the wires argument.

    Args:
        wires (int or list): (subset of) wires of a quantum node, can be list or a single integer

    Return:
        list: list of wires

    Raises:
        ValueError: if the wires argument is invalid
    """
    if isinstance(wires, int):
        wires = [wires]

    msg = "Wires must be a positive integer or a " \
           "list of positive integers; got {}.".format(wires)
    if not isinstance(wires, Iterable):
        raise ValueError(msg)
    if not all([isinstance(w, int) for w in wires]):
        raise ValueError(msg)
    if not all([w >= 0 for w in wires]):
        raise ValueError(msg)
    return wires


def _get_shape(inpt):
    """Turn ``inpt`` to an array and return its shape.

    Args:
        inpt (list): input to a qnode

    Returns:
        tuple: shape of ``inpt``
    """
    inpt = np.array(inpt)

    return inpt.shape


def _check_shape(inpt, target_shape, bound=None, msg=None):
    """Checks that the shape of ``inpt`` is equal to the target shape.
    Args:
        inpt (list): input to a qnode
        target_shape (tuple[int]): expected shape of inpt
        bound (str): If 'max' or 'min', the target shape is merely required to be a bound on the input shape
        msg (str): error message if the shapes are different

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


def _check_shapes(inpt_list, target_shape_list, bound_list=None, msg=None):
    """Same as `_check_shape()`, but for lists.

    Args:
        inpt_list (list): list of elements of which to check the shape
        target_shape_list (list): list of target shapes, of same length as ``inpt_list``
        bound_list (list): list of 'max' or 'min', indicating the bound that the target shape imposes on the input
             shape
        msg (str): error message to display

    Raises:
        ValueError
    """

    if bound_list is None:
        bound_list = [None] * len(inpt_list)

    shape_list = [_check_shape(l, t, bound=b, msg=msg) for l, t, b in zip(inpt_list, target_shape_list, bound_list)]
    return shape_list


def _check_is_in_options(element, options, msg=None):
    """Checks that a hyperparameter is one of the valid options of hyperparameters.

    Args:
        element: any element
        options: possible realizations for ``element``
        msg (str): error message to display
    """
    if msg is None:
        msg = "Hyperparameter {} must be one of '{}'".format(element, *options)

    if element not in options:
        raise ValueError(msg)


def _check_type(element, type_list, msg=None):
    """Checks the type of a hyperparameter.

    Args:
        element: element to check
        type_list (list): possible types for ``element``
         msg (str): error message to display
    """
    if msg is None:
        msg = "Hyperparameter type must be one of {}, got {}".format(type_list, type(element))

    if not any([isinstance(element, t) for t in type_list]):
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
        raise ValueError("The first dimension of the weight parameters must be the number of layers in the "
                         "template. Found scalar weights.")

    first_dimensions = [s[0] for s in shapes]
    different_first_dims = set(first_dimensions)
    n_different_first_dims = len(different_first_dims)

    if n_different_first_dims > 1:
        raise ValueError("The first dimension of the weight parameters must be the number of layers in the "
                         "template. Found different first dimensions: {}.".format(*different_first_dims))

    return first_dimensions[0]

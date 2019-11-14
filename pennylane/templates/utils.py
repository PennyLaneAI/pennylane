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
import numpy as np
from collections.abc import Iterable
from pennylane.qnode import Variable


def _check_no_variable(arg, arg_str, msg=None):
    if msg is None:
        msg = "The argument {} can not be passed as a QNode parameter.".format(arg_str)
    for a, s in zip(arg, arg_str):
        if isinstance(a, Variable):
            raise ValueError(msg)
        if isinstance(a, Iterable):
            if any([isinstance(a_, Variable) for a_ in a]):
                raise ValueError


def _check_wires(wires):
    """Standard checks for the wires argument.

    Args:
        wires (int or list): (subset of) wires of a quantum node

    Return:
        list: wires as a list, int: number of wires
    """
    if isinstance(wires, int):
        wires = [wires]

    msg = "Wires must a positive integer or a " \
           "list of positive integers; got {}.".format(wires)
    if not isinstance(wires, Iterable):
        raise ValueError(msg)
    if not all([isinstance(w, int) for w in wires]):
        raise ValueError(msg)
    if not all([w >= 0 for w in wires]):
        raise ValueError(msg)
    return wires, len(wires)


def _get_shape(inpt):
    """Return the shape of ``inpt``.

    Args:
        inpt (list): input to a qnode

    Returns:
        tuple: shape of ``inpt``

    Raises:
        ValueError
    """
    try:
        inpt = np.array(inpt)
    except:
        raise ValueError("Got a list which fails to be converted to a numpy array."
                         "All arguments to a template should be of rectangular shape.")
    if np.isscalar(inpt):
        shape = ()
    else:
        try:
            shape = tuple(inpt.shape)
        except:
            raise ValueError("Cannot derive shape of template input {}.".format(inpt))

    return shape


def _check_shape(inpt, target_shape, msg=None, bound=None):
    """Checks that the shape of inpt is equal to the target shape.
    Args:
        inpt (list): input to a qnode
        target_shape (tuple[int]): expected shape of inpt

    Keyword Args:
        msg (str): Error message if the shapes are different
        bound (str): If 'max' or 'min', the target shape is merely required to be a bound on the input shape
    """

    shape = _get_shape(inpt)

    if msg is None:
        msg = "Input has shape {}; expected {}.".format(shape, target_shape)

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
    """Same as _check_shape, but for lists."""

    if bound_list is None:
        bound_list = [None] * len(inpt_list)

    shape_list = [_check_shape(l, t, bound=b, msg=msg) for l, t, b in zip(inpt_list, target_shape_list, bound_list)]
    return shape_list


def _check_hyperp_is_in_options(hyperparameter, options):
    """Checks that a hyperparameter is one of the valid options of hyperparameters."""
    if hyperparameter not in options:
        raise ValueError("Hyperparameter {} must be one of {}".format(hyperparameter, *options))


def _check_type(hyperparameter, typ, msg=None):
    """Checks the type of a hyperparameter."""
    if msg is None:
        msg = "Hyperparameter type must be one of {}, got {}".format(typ, type(hyperparameter))

    if not any([isinstance(hyperparameter, t) for t in typ]):
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

    print(shapes)
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


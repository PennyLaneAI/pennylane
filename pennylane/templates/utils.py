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
from pennylane.qnode import QuantumFunctionError
from pennylane.qnode import Variable


def _check_no_variable(arg, arg_str, mssg=None):
    if mssg is None:
        mssg = "The argument {} can not be passed as a QNode parameter.".format(arg_str)
    for a, s in zip(arg, arg_str):
        if isinstance(a, Variable):
            raise QuantumFunctionError(mssg)
        if isinstance(a, Iterable):
            if any([isinstance(a_, Variable) for a_ in a]):
                raise QuantumFunctionError


def _check_wires(wires):
    """Standard checks for the wires argument."""
    if isinstance(wires, int):
        wires = [wires]

    mssg = "Wires must a positive integer or a " \
           "list of positive integers; got {}.".format(wires)
    if not isinstance(wires, Iterable):
        raise QuantumFunctionError(mssg)
    if not all([isinstance(w, int) for w in wires]):
        raise QuantumFunctionError(mssg)
    if not all([w >= 0 for w in wires]):
        raise QuantumFunctionError(mssg)
    return wires, len(wires)


def _check_shape(inpt, target_shp, mssg=None, bound=None):
    """Checks that the shape of inpt is equal to the target shape.
    """
    # If inpt is list of inputs, call this function recursively
    if isinstance(target_shp, list):
        shape = [_check_shape(l, t, mssg=mssg, bound=bound) for l, t in zip(inpt, target_shp)]

    else:
        if isinstance(inpt, list):
            try:
                inpt = np.array(inpt)
            except:
                raise QuantumFunctionError("Got a list as template input, which fails to "
                                           "be converted to a numpy array.")
        if np.isscalar(inpt):
            shape = ()
        else:
            try:
                shape = tuple(inpt.shape)
            except:
                raise QuantumFunctionError("Cannot derive shape of template input {}.".format(inpt))

        if mssg is None:
            mssg = "Input has shape {}; expected {}.".format(shape, target_shp)

        if bound == 'max':
            if shape > target_shp:
                raise QuantumFunctionError(mssg)
        elif bound == 'min':
            if shape < target_shp:
                raise QuantumFunctionError(mssg)
        else:
            if shape != target_shp:
                raise QuantumFunctionError(mssg)

    return shape


def _check_hyperp_is_in_options(hp, options):
    """Checks that a hyperparameter is one of the valid options of hyperparameters."""
    if hp not in options:
        raise QuantumFunctionError("Hyperparameter {} must be one of {}".format(hp, *options))


def _check_type(hp, typ, mssg=None):
    """Checks the type of a hyperparameter."""
    if mssg is None:
        mssg = "Hyperparameter type must be one of {}, got {}".format(typ, type(hp))

    if not any([isinstance(hp, t) for t in typ]):
        raise QuantumFunctionError(mssg)


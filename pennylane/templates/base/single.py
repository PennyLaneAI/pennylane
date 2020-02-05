# Copyright 2018-2019 Xanadu Quantum Technologies Inc.

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
Contains the ``Single`` base template.
"""
#pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
from collections import Sequence

from types import FunctionType
from pennylane.operation import Operation
from pennylane.templates.decorator import template
from pennylane.templates.utils import (_check_wires,
                                       _check_type,
                                       _get_shape)


@template
def Single(unitary, wires, parameters=None):
    """
    Applies ``unitary`` to each wire, feeding it values in ``parameters``.

    The unitary is typically a ``~.pennylane.operation.Operation'' object representing a single qubit gate as
    in the following example:

    .. code-block:: python




    Alternatively, one can use a sequence of gates by creating a template, and feeding it into

    .. code-block:: python

        import pennylane as qml
        from pennylane.templates import template, Single

        @template
        def mytemplate(pars, wires):
            qml.Hadamard(wires=wires)
            qml.RY(pars[0], wires=wires)
            qml.RX(pars[1], wires=wires)


        dev = qml.device('default.qubit', wires=3)

        @qml.qnode(dev)
        def circuit(pars):
            Single(unitary=mytemplate, wires=[0,1,2], parameters=pars)
            return qml.expval(qml.PauliZ(0))

        print(circuit([[1, 2], [2, 1], [0.1, 1]]))



    If the unitary does not take any parameters, ``parameters`` is ``None`` or an empty list.

    If the unit

    Args:
        unitary (qml.Operation):
        parameters (list or None):
        wires (Sequence[int] or int):

    Raises:
        ValueError: if inputs do not have the correct format

    """

    #########
    # Input checks

    wires = _check_wires(wires)

    _check_type(parameters, [list, type(None)], msg="parameters must be either None or a list; "
                                                    "got {}".format(type(parameters)))
    _check_type(unitary, [list, FunctionType], msg="unitary must be a ``~.pennylane.operation.Operation`` "
                                                   "or a template function; got {}".format(type(unitary)))

    if parameters is not None:
        if len(parameters) != len(wires):
            raise ValueError("parameters must contain one entry for each wire; got shape {}"
                             .format(_get_shape(parameters)))

    # # turn operation into list of operations
    # if isinstance(unitary, Operation):
    #     unitary = [unitary]

    # for gate in unitary:
    #     if gate.num_wires != 1:
    #         raise ValueError("gate must act on a single wire")

    # TODO: check the number of parameters per gate?
    # shape = _get_shape(parameters)
    #
    # if shape[0] != len(wires):
    #     raise ValueError("number of parameters {} must be equal to number of wires; got {}"
    #                      .format(len(parameters)), len(wires))
    # if shape[1] != gate.num_params:
    #     raise ValueError("gate {} takes {} parameters; got {}". format(gate, gate.num_params, shape[1]))

    #########

    if parameters is None:
        for w in wires:
            unitary(wires=w)
    else:
        for w, p in zip(wires, parameters):
            unitary(p, wires=w)



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
Contains the ``repeat`` template constructor.
"""
# pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
import inspect
import types

from collections.abc import Iterable

from pennylane.templates.decorator import template
from pennylane.templates.utils import check_type, get_shape
from pennylane.wires import Wires


###################


@template
def repeat(unitary, wires, depth, parameters=None, kwargs=None):
    r"""Repeatedly applies a series of quantum gates or templates.

    Args:
        unitary (function): A function that applies the quantum gates/templates being repeated.
        This function must have a signature of the form ``(parameters, wires, **kwargs)`` or ``(wires, **kwargs)``.
        wires (list): The wires on which ``unitary`` acts
        depth (int): The number of times ``unitary`` is repeatedly applied
        parameters (list): A list of parameters that are passed into each application of ``unitary``
        kwargs (list[dict]): A list of dictionaries of auxiliary parameters for each layer of ``unitaries``

    Raises:
        ValueError: if inputs do not have the correct format

    .. UsageDetails::

        **Repeating Gates**

        To repeat a collection of gates, a function applying each of the gates, as well as the wire(s) on which
        it acts must be specified. For example:

        .. code-block:: python

            import pennylane as qml

            dev = qml.device('default.qubit', wires=2)

            def unitary(wires, **kwargs):
                qml.Hadamard(wires=wires[0])
                qml.CNOT(wires=wires)
                qml.PauliX(wires=wires[1])

            @qml.qnode(dev)
            def circuit():
                qml.repeat(unitary, [0, 1], 3)
                return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]

            circuit()

        creates the following circuit:

        .. code-block:: none

             0: ──H──╭C──X──H──╭C──X──H──╭C──X──┤ ⟨Z⟩
             1: ─────╰X────────╰X────────╰X─────┤ ⟨Z⟩

        **Parameters and Keyword Arguments**

        The ``qml.repeat`` function can accept and pass variational parameters and keyword arguments
        to ``unitary``, with the ``parameters`` argument.
        The :math:`i`-th element of ``parameters`` is the list of parameters passed into the template
        at the :math:`i`-th layer.

        For example:

        .. code-block:: python

            import pennylane as qml

            dev = qml.device("default.qubit", wires=3)

            def unitary(parameters, wires, **kwargs):
                qml.templates.AngleEmbedding(parameters[0], wires)
                qml.RY(parameters[1], wires=wires[0])

                if kwargs['var'] == True:
                    qml.Hadamard(wires=wires[1])

            @qml.qnode(dev)
            def circuit():
                qml.repeat(
                    unitary, [0, 1, 2], 2,
                    [[[0.5, 0.5, 0.5], 0.3], [[0.5, 0.5, 0.5], 0.3]],
                    [{'var' : True}, {'var': False}]
                )
                return [qml.expval(qml.PauliZ(i)) for i in range(3)]

            circuit()

        creates the following circuit:

        .. code-block:: none

             0: ──RX(0.5)──RY(0.3)──RX(0.5)──RY(0.3)──┤ ⟨Z⟩
             1: ──RX(0.5)──H────────RX(0.5)──-────────┤ ⟨Z⟩
             2: ──RX(0.5)──RX(0.5)────────────────────┤ ⟨Z⟩

    """

    ##############
    # Input checks

    wires = [Wires(w) for w in wires]

    if not isinstance(depth, int):
        raise ValueError("'depth' must be of type int, got {}".format(type(depth).__name__))

    check_type(
        parameters,
        [Iterable, type(None)],
        msg="'parameters' must be either of type None or "
        "Iterable; got {}".format(type(parameters)),
    )

    if kwargs is None:
        kwargs = [{} for i in range(0, depth)]

    if not isinstance(kwargs, list):
        raise ValueError("'kwargs' must be a list; got {}".format(type(kwargs)))

    if len(kwargs) != depth:
        raise ValueError("Expected length of 'kwargs' to be {}, got {}".format(depth, len(kwargs)))

    for i in kwargs:
        check_type(
            i, [dict], msg="Elements of 'kwargs' must be dictionaries; got {}".format(type(i)),
        )

    check_type(
        unitary, [types.FunctionType], msg="'unitary' must be a function; got {}".format(type(unitary)),
    )

    if len(inspect.signature(unitary).parameters) not in [2, 3]:
        raise ValueError("Signature of 'unitary' must be of the form (parameters, wires, **kwargs)")

    if parameters is None and inspect.signature(unitary) == 3:
        raise ValueError("Expected 'parameters', got None")

    if parameters is not None and inspect.signature(unitary) == 2:
        raise ValueError("Got 'parameters' with non-parametrized 'unitary'")


    ##############


    if parameters is not None:
        shape = get_shape(parameters)

        if int(shape[0]) != depth:
            raise ValueError("Expected first dimension of 'parameters' to be {}; got {}".format(depth, int(shape[0])))

        for i, param in enumerate(parameters):
            unitary(param, wires, **kwargs[i])

    else:
        for i in range(0, depth):
            unitary(wires, **kwargs[i])

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
from collections.abc import Iterable

from pennylane.templates.decorator import template
from pennylane.templates.utils import check_type, get_shape
from pennylane.wires import Wires


###################


@template
def repeat(unitaries, wires, depth, parameters=None, kwargs=None):
    r"""Repeatedly applies a series of quantum gates or templates.

    Args:
        unitaries (list): A list of quantum gates or templates
        wires (list): The wires on which each gate/template act
        depth (int): The number of times the unitaries are repeated
        parameters (list): A list of parameters that are passed into parametrized elements of ``unitaries``.
            This argument has a shape (depth, N,), where N is the number of parametrized elements in ``unitaries``
        kwargs (dict): A dictionary of auxilliary parameters for ``unitaries``

    Raises:
        ValueError: if inputs do not have the correct format

    .. UsageDetails::

        **Repeating Gates**

        To repeat a collection of gates, the type of gate as well as the wire(s) on which
        it acts must be specified. For example:

        .. code-block:: python

            import pennylane as qml

            dev = qml.device('default.qubit', wires=2)

            @qml.qnode(dev)
            def circuit():
                qml.repeat([qml.Hadamard, qml.CNOT, qml.PauliX], [0, [0, 1], 0], 3)
                return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]

            circuit()

        creates the following circuit:

        .. code-block:: none

             0: ──H──╭C──X──H──╭C──X──H──╭C──X──┤ ⟨Z⟩
             1: ─────╰X────────╰X────────╰X─────┤ ⟨Z⟩

        **Parametrized Gates and Templates**

        The ``qml.repeat`` function can accept and pass varaitional parameters
        to gates as well as PennyLane templates, with the ``parameters`` argument.
        The first and second dimensions of ``parameters``, respectively,
        are the number of repetations and the number of parametrized templates/operations.
        After this, each element is a list of parameters passed into the corresponding
        parametrized template/operation.

        .. note::

            Any template passed into ``qml.repeat`` must have a signature of the form:

            .. code-block:: none

                    template(params, wires, **kwargs)

        For example:

        .. code-block:: python

            import pennylane as qml

            dev = qml.device("default.qubit", wires=3)

            @qml.qnode(dev)
            def circuit():
                qml.repeat(
                    [qml.templates.AngleEmbedding, qml.RY, qml.Hadamard],
                    [[0, 1, 2], 0, 1], 2,
                    parameters=[
                        [ [0.5, 0.5, 0.5], 0.3 ],
                        [ [0.5, 0.5, 0.5], 0.3 ]
                    ]
                )
                return [qml.expval(qml.PauliZ(i)) for i in range(3)]

            circuit()

        creates the following circuit:

        .. code-block:: none

             0: ──RX(0.5)──RY(0.3)──RX(0.5)──RY(0.3)──┤ ⟨Z⟩
             1: ──RX(0.5)──H────────RX(0.5)──H────────┤ ⟨Z⟩
             2: ──RX(0.5)──RX(0.5)────────────────────┤ ⟨Z⟩

    """

    ##############
    # Input checks

    wires = [Wires(w) for w in wires]

    if not isinstance(unitaries, list):
        raise ValueError(
            "'unitaries' must be of type list, got {}".format(type(unitaries).__name__)
        )

    if not isinstance(depth, int):
        raise ValueError("'depth' must be of type int, got {}".format(type(depth).__name__))

    check_type(
        parameters,
        [Iterable, type(None)],
        msg="'parameters' must be either of type None or "
        "Iterable; got {}".format(type(parameters)),
    )

    if kwargs is None:
        kwargs = {}

    check_type(
        kwargs, [dict], msg="'kwargs' must be a dictionary; got {}".format(type(kwargs)),
    )

    if len(wires) != len(unitaries):
        raise ValueError(
            "Expected wires to be length {}, got length {}".format(len(unitaries), len(wires))
        )

    # Checks if dimensions of parameters are correct

    if parameters is not None:
        shape = get_shape(parameters)

        # Gets the expected dimensions of the parameter list
        s = 0
        for u in unitaries:
            if type(u).__name__ == "function":
                s += 1
            elif u.num_params > 0:
                s += 1

        if shape[0] != depth or shape[1] != s:
            raise ValueError(
                "Shape of parameters must be {} got {}".format((depth, s), (shape[0], shape[1]))
            )

        # Checks if elements are lists

        for i in parameters:
            for j in i:
                if not isinstance(j, list):
                    raise ValueError(
                        "Elements of 'parameters[i]' must be of type list, got {}".format(
                            type(j).__name__
                        )
                    )

    ##############

    wires = [w.tolist() for w in wires]  # TODO: Delete once operator takes Wires objects
    for d in range(0, depth):
        c = 0
        for i, u in enumerate(unitaries):
            if type(u).__name__ == "function":
                u(*parameters[d][c], wires=wires[i], **kwargs)
                c += 1
            elif u.num_params > 0:
                u(*parameters[d][c], wires=wires[i], **kwargs)
                c += 1
            else:
                u(wires=wires[i], **kwargs)

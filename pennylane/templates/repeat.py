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
from pennylane.templates.decorator import template

###################


@template
def repeat(circuit, depth, *args, **kwargs):
    r"""Repeatedly applies a circuit containing quantum gates or templates.

    Args:
        circuit (function): A function that applies the quantum gates/templates being repeated.
        depth (int): The number of times the circuit is repeatedly applied.
        *args: Dynamic parameters that are passed into ``circuit`` each time it is
               repeated (see UsageDetails for more information). Each dynamic argument
               must be a list of first dimension equal to ``depth``, with the :math:`j`-th element of the list
               corresponding to the value of the argument the :math:`j`-th time the circuit
               is applied.
        **kwargs: Static parameters that are passed into ``circuit`` each time it is
                  repeated (see UsageDetails for more information).

    Raises:
        ValueError: if inputs do not have the correct format

    .. UsageDetails::

        **Repeating Gates**

        To repeatedly apply a collection of gates/templates, a function applying each of these
        operations must first be defined. For example, we can define the following circuit:

        .. code-block:: python3

            import pennylane as qml
            import numpy as np

            def circuit():
                qml.Hadamard(wires=[0])
                qml.CNOT(wires=[0, 1])
                qml.PauliX(wires=[1])

        and then pass it into the ``qml.repeat`` function. In this instance, we repeat ``circuit`` 3 times:

        .. code-block:: python3

            dev = qml.device('default.qubit', wires=3)

            @qml.qnode(dev)
            def ansatz():
                qml.repeat(circuit, 3)
                return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]


        This creates the following circuit:

        >>> ansatz()
        >>> print(ansatz.draw())
        0: ──H──╭C──X──H──╭C──X──H──╭C──X──┤ ⟨Z⟩
        1: ─────╰X────────╰X────────╰X─────┤ ⟨Z⟩

        **Static Arguments**

        Static arguments are arguments passed into ``circuit`` that don't change with each
        repetition of the circuit. Static parameters are always passed as keyword arguments into ``qml.repeat``.
        For example, consider the following circuit:

        .. code-block:: python3

            def circuit(wires):
                qml.Hadamard(wires=wires[0])
                qml.CNOT(wires=wires)
                qml.PauliX(wires=wires[1])

        We wish to repeat this circuit three times on wires ``1`` and ``2``. Since the wires on which the circuit acts
        don't change with each repetition of the circuit, the ``wires`` parameter is passed as a keyword argument.
        We thus repeat the circuit as follows:

        .. code-block:: python3

            @qml.qnode(dev)
            def ansatz():
                qml.repeat(circuit, 3, wires=[1, 2])
                return [qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2))]

        which yields the following circuit:

        >>> ansatz()
        >>> print(ansatz.draw())
        1: ──H──╭C──X──H──╭C──X──H──╭C──X──┤ ⟨Z⟩
        2: ─────╰X────────╰X────────╰X─────┤ ⟨Z⟩

        **Dynamic Arguments**

        In addition to passing static arguments to ``circuit``, we can also pass *dynamic* arguments.
        These are arguments that change with each repetition of the circuit. They are passed
        as non-keyword arguments to ``qml.repeat``, after ``circuit`` and ``depth``. Each dynamic parameter must
        be a list of length equal to ``depth``. The :math:`j`-th element of the list represents the value of the
        argument used for the :math:`j`-th repetition of the circuit.

        For example, let us define the following variational circuit:

        .. code-block:: python3

            def circuit(params):
                qml.RX(params[0], wires=[0])
                qml.MultiRZ(params[1], wires=[0, 1])
                qml.RY(params[2], wires=[1])

        We wish to repeat this circuit two times, with each layer having different ``params``:

        .. code-block:: python3

            @qml.qnode(dev)
            def ansatz(params):
                qml.repeat(circuit, 2, params)
                return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]

        Since we only have one dynamic argument, ``params``, we pass an array of first-dimension two,
        for the two layers of the ansatz. By looking at ``circuit``, we can see that the ``params`` argument
        supplies three different parameters to three different gates. Thus, we supply an array of size
        (2, 3) as an argument to ``qml.repeat``:

        .. code-block:: python3

            params = np.array([[0.5, 0.5, 0.5], [0.4, 0.4, 0.4]])

        which yields the following circuit:

        >>> ansatz(params)
        >>> print(ansatz.draw())
        0: ──RX(0.5)──╭RZ(0.5)──RX(0.4)──╭RZ(0.4)──RX(0.3)──╭RZ(0.3)───────────┤ ⟨Z⟩
        1: ───────────╰RZ(0.5)──RY(0.5)──╰RZ(0.4)──RY(0.4)──╰RZ(0.3)──RY(0.3)──┤ ⟨Z⟩

        **Passing Multiple Static and Dynamic Arguments**

        It is also possible to pass multiple static and dynamic arguments into the same circuit. Dynamic
        arguments must be ordered in ``qml.repeat`` in the same order in which they are passed into the
        ``circuit``.

        Consider the following circuit:

        .. code-block:: python3

            def circuit(param1, param2, wires, var):
                qml.RX(param1, wires=wires[0])
                qml.MultiRZ(param2, wires=wires)

                if var:
                    qml.Hadamard(wires=wires[1])

        This circuit can be repeated as:

        .. code-block:: python3

            @qml.qnode(dev)
            def ansatz(param1, param2):
                qml.repeat(circuit, 2, param1, param2, wires=[1, 2], var=True)
                return [qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2))]

        We can then run the circuit with a given set of parameters (note that the parameters are
        of size (2, 1), as the circuit is repeated twice, and for each repetition, both ``param1`` and
        ``param2`` are simply real numbers):

        .. code-block:: python3

            param1 = np.array([0.1, 0.2])
            param2 = np.array([0.3, 0.4])

        This gives us the following circuit:

        >>> ansatz(parm1, param2)
        >>> print(ansatz.draw())
        1: ──RX(0.1)──╭RZ(0.3)──RX(0.2)──╭RZ(0.4)─────┤ ⟨Z⟩
        2: ───────────╰RZ(0.3)──H────────╰RZ(0.4)──H──┤ ⟨Z⟩
    """

    ##############
    # Input checks

    if not isinstance(depth, int):
        raise ValueError("'depth' must be of type int, got {}".format(type(depth).__name__))

    for arg in args:
        if len(arg) != depth:
            raise ValueError(
                "Each argument in args must have length matching 'depth'; expected {} got {}".format(
                    depth, len(arg)
                )
            )

    ##############

    for i in range(0, depth):
        arg_params = [k[i] for k in args]
        circuit(*arg_params, **kwargs)

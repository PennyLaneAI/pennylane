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
Contains the ``layer`` template constructor.
"""
# pylint: disable-msg=too-many-branches,too-many-arguments,protected-access

###################


@pennylane.templates.decorator.template
def layer(template, depth, *args, **kwargs):
    r"""Repeatedly applies a unitary a given number of times.

    Args:
        template (callable): The sequence of quantum gates that is being repeated.
                             This could be a single gate, a function of gates, or a "registered"
                             PennyLane template.
        depth (int): The number of times the unitary is repeatedly applied.
        *args: Dynamic parameters that are passed into the unitary each time it is
               repeated. Each dynamic argument must be a list of first dimension equal to
               ``depth``, with the :math:`j`-th element of the list corresponding to the
               value of the argument the :math:`j`-th time the unitary is applied.
        **kwargs: Static parameters that are passed into the unitary each time it is
                  repeated.

    See usage details for more information.

    .. UsageDetails::

        **Layering Gates**

        The layering function can be used to repeatedly apply a function containing quantum operations,
        a template, or a quantum gate.

        For example, we can define the following subroutine:

        .. code-block:: python3

            import pennylane as qml
            import numpy as np

            def subroutine():
                qml.Hadamard(wires=[0])
                qml.CNOT(wires=[0, 1])
                qml.PauliX(wires=[1])

        and then pass it into the ``qml.layer`` function. In this instance, we repeat ``subroutine`` 3 times:

        .. code-block:: python3

            dev = qml.device('default.qubit', wires=3)

            @qml.qnode(dev)
            def circuit():
                qml.repeat(subroutine, 3)
                return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]


        This creates the following circuit:

        >>> circuit()
        >>> print(circuit.draw())
        0: ──H──╭C──X──H──╭C──X──H──╭C──X──┤ ⟨Z⟩
        1: ─────╰X────────╰X────────╰X─────┤ ⟨Z⟩

        **Static Arguments**

        Static arguments are arguments passed into ``template`` that don't change with each
        repetition. Static parameters are always passed as keyword arguments into ``qml.layer``.
        For example, consider the following subroutine:

        .. code-block:: python3

            def subroutine(wires):
                qml.Hadamard(wires=wires[0])
                qml.CNOT(wires=wires)
                qml.PauliX(wires=wires[1])

        We wish to repeat this gate sequence three times on wires ``1`` and ``2``. Since the wires on which the subroutine acts
        don't change with each repetition, the ``wires`` parameter is passed as a keyword argument.
        Therefore, we define a circuit as:

        .. code-block:: python3

            @qml.qnode(dev)
            def circuit():
                qml.repeat(subroutine, 3, wires=[1, 2])
                return [qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2))]

        which yields the following circuit:

        >>> circuit()
        >>> print(circuit.draw())
        1: ──H──╭C──X──H──╭C──X──H──╭C──X──┤ ⟨Z⟩
        2: ─────╰X────────╰X────────╰X─────┤ ⟨Z⟩

        **Dynamic Arguments**

        In addition to passing static arguments to ``template``, we can also pass dynamic arguments.
        These are arguments that change with each repetition of the unitary. They are passed
        as non-keyword arguments to ``qml.layer``, after ``template`` and ``depth``. Each dynamic parameter must
        be a list of length equal to ``depth``. The :math:`j`-th element of the list represents the value of the
        argument used for the :math:`j`-th repetition.

        For example, let us define the following variational ansatz:

        .. code-block:: python3

            def ansatz(params):
                qml.RX(params[0], wires=[0])
                qml.MultiRZ(params[1], wires=[0, 1])
                qml.RY(params[2], wires=[1])

        We wish to repeat this ansatz two times, with each layer having different ``params``:

        .. code-block:: python3

            @qml.qnode(dev)
            def circuit(params):
                qml.repeat(ansatz, 2, params)
                return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]

        Since we only have one dynamic argument, ``params``, we pass an array of first-dimension two,
        for the two layers of the repeated ansatz. We can also see that the ``params`` argument
        supplies three different parameters to three different gates. We therefore supply an array of size
        (2, 3) as an argument to ``qml.layer``:

        .. code-block:: python3

            params = np.array([[0.5, 0.5, 0.5], [0.4, 0.4, 0.4]])

        which yields the following circuit:

        >>> circuit(params)
        >>> print(circuit.draw())
        0: ──RX(0.5)──╭RZ(0.5)──RX(0.4)──╭RZ(0.4)──RX(0.3)──╭RZ(0.3)───────────┤ ⟨Z⟩
        1: ───────────╰RZ(0.5)──RY(0.5)──╰RZ(0.4)──RY(0.4)──╰RZ(0.3)──RY(0.3)──┤ ⟨Z⟩

        **Passing Multiple Static and Dynamic Arguments**

        It is also possible to pass multiple static and dynamic arguments into the same unitary. Dynamic
        arguments must be ordered in ``qml.layer`` in the same order in which they are passed into the
        ``template``.

        Consider the following ansatz:

        .. code-block:: python3

            def ansatz(param1, param2, wires, var):
                qml.RX(param1, wires=wires[0])
                qml.MultiRZ(param2, wires=wires)

                if var:
                    qml.Hadamard(wires=wires[1])

        This circuit can be repeated as:

        .. code-block:: python3

            @qml.qnode(dev)
            def circuit(param1, param2):
                qml.repeat(ansatz, 2, param1, param2, wires=[1, 2], var=True)
                return [qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2))]

        We can then run the circuit with a given set of parameters (note that the parameters are
        of size (2, 1), as the circuit is repeated twice, and for each repetition, both ``param1`` and
        ``param2`` are simply real numbers):

        .. code-block:: python3

            param1 = np.array([0.1, 0.2])
            param2 = np.array([0.3, 0.4])

        This gives us the following circuit:

        >>> circuit(parm1, param2)
        >>> print(circuit.draw())
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
                "Each positional argument must have length matching 'depth'; expected {} got {}".format(
                    depth, len(arg)
                )
            )

    ##############

    for i in range(0, depth):
        arg_params = [k[i] for k in args]
        template(*arg_params, **kwargs)

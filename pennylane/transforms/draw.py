# pylint: disable=too-many-arguments

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
Contains the drawing function.
"""
from functools import wraps

import pennylane as qml


def draw(
    qnode,
    charset="unicode",
    wire_order=None,
    show_all_wires=False,
    max_length=None,
    expansion_strategy=None,
):
    """Create a function that draws the given qnode.

    Args:
        qnode (.QNode): the input QNode that is to be drawn.
        charset (str, optional): The charset that should be used. Currently, "unicode" and
            "ascii" are supported.
        wire_order (Sequence[Any]): the order (from top to bottom) to print the wires of the circuit
        show_all_wires (bool): If True, all wires, including empty wires, are printed.
        max_length (int, optional): Maximum string width (columns) when printing the circuit to the CLI.
        expansion_strategy (str): The strategy to use when circuit expansions or decompositions
            are required.

            - ``gradient``: The QNode will attempt to decompose
              the internal circuit such that all circuit operations are supported by the gradient
              method.

            - ``device``: The QNode will attempt to decompose the internal circuit
              such that all circuit operations are natively supported by the device.


    Returns:
        A function that has the same argument signature as ``qnode``. When called,
        the function will draw the QNode.

    **Example**

    Given the following definition of a QNode,

    .. code-block:: python3

        dev = qml.device('default.qubit', wires=2)

        @qml.qnode(dev)
        def circuit(a, w):
            qml.Hadamard(0)
            qml.CRX(a, wires=[0, 1])
            qml.Rot(*w, wires=[1])
            qml.CRX(-a, wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    We can draw the it like such:

    >>> drawer = qml.draw(circuit)
    >>> print(drawer(a=2.3, w=[1.2, 3.2, 0.7]))
    0: ──H──╭C────────────────────────────╭C─────────╭┤ ⟨Z ⊗ Z⟩
    1: ─────╰RX(2.3)──Rot(1.2, 3.2, 0.7)──╰RX(-2.3)──╰┤ ⟨Z ⊗ Z⟩

    Circuit drawing works with devices with custom wire labels:

    .. code-block:: python3

        dev = qml.device('default.qubit', wires=["a", -1, "q2"])

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=-1)
            qml.CNOT(wires=["a", "q2"])
            qml.RX(0.2, wires="a")
            return qml.expval(qml.PauliX(wires="q2"))

    When printed, the wire order matches the order defined on the device:

    >>> drawer = qml.draw(circuit)
    >>> print(drawer())
      a: ─────╭C──RX(0.2)──┤
     -1: ──H──│────────────┤
     q2: ─────╰X───────────┤ ⟨X⟩

    We can use the ``wire_order`` argument to change the wire order:

    >>> drawer = qml.draw(circuit, wire_order=["q2", "a", -1])
    >>> print(drawer())
     q2: ──╭X───────────┤ ⟨X⟩
      a: ──╰C──RX(0.2)──┤
     -1: ───H───────────┤
    """

    @wraps(qnode)
    def wrapper(*args, **kwargs):
        original_expansion_strategy = getattr(qnode, "expansion_strategy", None)

        try:
            qnode.expansion_strategy = expansion_strategy or original_expansion_strategy
            tapes = qnode.construct(args, kwargs)
        finally:
            qnode.expansion_strategy = original_expansion_strategy

        _wire_order = wire_order or qnode.device.wires
        _wire_order = qml.wires.Wires(_wire_order)

        if show_all_wires and len(_wire_order) < qnode.device.num_wires:
            raise ValueError(
                "When show_all_wires is enabled, the provided wire order must contain all wires on the device."
            )

        if not qnode.device.wires.contains_wires(_wire_order):
            raise ValueError(
                f"Provided wire order {_wire_order.labels} contains wires not contained on the device: {qnode.device.wires}."
            )

        if tapes is not None:
            res = [
                t.draw(
                    charset=charset,
                    wire_order=_wire_order,
                    show_all_wires=show_all_wires,
                    max_length=max_length,
                )
                for t in tapes[0]
            ]
            return "\n".join(res)

        return qnode.qtape.draw(
            charset=charset,
            wire_order=_wire_order,
            show_all_wires=show_all_wires,
            max_length=max_length,
        )

    return wrapper

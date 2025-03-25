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
Code for adding registers to the output of a string drawing.
"""

from collections import defaultdict


def _registers_repr(register: dict, wire_order):

    wire_map = {w: i for i, w in enumerate(wire_order)}
    register = {key: [wire_map[str(w)] for w in wires] for key, wires in register.items()}
    all_wires = list(range(len(wire_order)))

    reversed_registers = defaultdict(list)
    for sub_reg, wires in register.items():
        for w in wires:
            reversed_registers[w].append(sub_reg)

    num_layers = max(len(r) for r in reversed_registers.values())
    sorted_registers = sorted(register, key=lambda r: len(register[r]), reverse=True)

    mat = [["" for _ in range(num_layers)] for _ in wire_order]
    max_layer_for_reg = [0 for _ in wire_order]

    for sub_reg in sorted_registers:
        wires = register[sub_reg]
        biggest_layer = max(max_layer_for_reg[w] for w in wires)
        for w in wires:
            mat[w][biggest_layer] = f"{sub_reg}|"
            max_layer_for_reg[w] = biggest_layer + 1

    max_lens = [max(len(mat[w][layer]) for w in all_wires) for layer in range(num_layers)]

    fillers = [[" " for _ in range(num_layers)] for _ in all_wires]
    for layer in range(num_layers):
        for w in all_wires:
            previous_reg = mat[w][layer]
            above_reg = mat[w - 1][layer] if w > 0 else ""
            below_reg = mat[w + 1][layer] if w < all_wires[-1] else ""

            if fillers[w][layer - 1] == "-":
                filler = "-"
            elif previous_reg == "":
                filler = " "
            elif previous_reg != above_reg or previous_reg != below_reg:
                filler = "-"
            else:
                filler = " "
            fillers[w][layer] = filler

            left_filler = fillers[w][layer - 1] if layer > 0 else " "
            mat[w][layer] = (
                mat[w][layer].rjust(max_lens[layer], left_filler) + 3 * fillers[w][layer]
            )

    return ["".join(ms) for ms in mat]


def draw_registers(register: dict, drawing: str) -> str:
    """Add registers to the start of a string drawing.

    Args:
        register (dict): The registers to draw
        drawing (str): output from ``qml.draw``.

    Returns:
        str: a new string drawing that includes the registers.

    .. code-block:: python

        nested_register = qml.registers(
            {
                "all": {
                    "alice": 3,
                    "bob": {"bob1": {"bob1a": 1, "bob1b": 2}, "bob2": 1},
                    "charlie": 1,
                }
            }
        )

        @qml.qnode(qml.device('default.qubit'))
        def circuit(reps):
            for jj in range(reps):
                for kk in nested_register["alice"]:
                    qml.X(kk)
                for kk in nested_register["bob"]:
                    qml.Y(kk)
                for kk in nested_register["charlie"]:
                    qml.Z(kk)
                for kk in [8,9]:
                    qml.H(kk)
            return qml.state()

        out = qml.draw(circuit)(2)
        print(qml.drawer.draw_registers(nested_register, out))

    .. code-block::

        all|-----alice|--------------------0: ──X──X─┤  State
        all|     alice|                    1: ──X──X─┤  State
        all|     alice|--------------------2: ──X──X─┤  State
        all|       bob|---bob1|---bob1a|---3: ──Y──Y─┤  State
        all|       bob|   bob1|   bob1b|---4: ──Y──Y─┤  State
        all|       bob|   bob1|---bob1b|---5: ──Y──Y─┤  State
        all|       bob|---bob2|------------6: ──Y──Y─┤  State
        all|---charlie|--------------------7: ──Z──Z─┤  State
                                           8: ──H──H─┤  State
                                           9: ──H──H─┤  State

    """

    split_drawing = drawing.split("\n")
    wire_order = [line.split(":")[0] for line in split_drawing]
    reg_rep = _registers_repr(register, wire_order)
    return "\n".join(reg_rep_w + line for reg_rep_w, line in zip(reg_rep, split_drawing))

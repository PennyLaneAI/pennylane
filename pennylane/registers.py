# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
This module contains the :func:`registers` function to build the registers for a given dictionary of registers
"""

from .wires import Wires


def registers(register_dict, _start_wire_index=0):
    """Returns a collection of wire registers when given a dictionary of register names and sizes
    (number of qubits in the register).

    The registers are a dictionary of :class:`~.Wires` objects where the key is the register name
    and the value is the ``Wires`` object. The individual wire labels are integers, where the
    ordering is based on appearance first, then on nestedness.

    Args:
        register_dict (dict): a dictionary of registers where the keys are the names (str) of the
        registers and the values are either nested dictionaries of more registers or integers (int).
        At the most nested level for each register key, the value must be an int.

    Returns:
        dict (Wires): dictionary of Wires objects (value) belonging to register names (keys)

    **Example**

    >>> wire_dict = qml.registers({"alice": 3, "bob": {"nest1": 3, "nest2": 3}, "cleo": 3})
    >>> wire_dict
    {'alice': Wires([0, 1, 2]),
    'nest1': Wires([3, 4, 5]),
    'nest2': Wires([6, 7, 8]),
    'bob': Wires([3, 4, 5, 6, 7, 8]),
    'cleo': Wires([9, 10, 11])}
    """

    all_reg = {}
    for register_name, register_wires in register_dict.items():
        if isinstance(register_wires, dict):
            if len(register_wires) == 0:
                raise ValueError(
                    f"Expected a dictionary but got an empty dictionary '{register_wires}'"
                )
            inner_register_dict = registers(register_wires, _start_wire_index=_start_wire_index)
            wire_vals = []
            inner_register_name = None
            for inner_register_name, inner_register_wires in inner_register_dict.items():
                wire_vals.extend(inner_register_wires.tolist())
                all_reg[inner_register_name] = inner_register_wires

            wires = Wires(range(_start_wire_index, all_reg[inner_register_name].labels[-1] + 1))

            all_reg[register_name] = wires
            _start_wire_index = wire_vals[-1] + 1
        elif isinstance(register_wires, int):
            if register_wires < 1:
                raise ValueError(
                    f"Expected '{register_wires}' to be greater than 0. Please ensure that "
                    "the number of wires for the register is a positive integer"
                )
            wires = Wires(range(_start_wire_index, register_wires + _start_wire_index))

            _start_wire_index += register_wires
            all_reg[register_name] = wires
        else:  # Not a dict nor an int
            raise ValueError(f"Expected '{register_wires}' to be either a dict or an int. ")

    return all_reg

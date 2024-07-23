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
This module contains the :func:`registers` function.
"""

from .wires import Wires


def registers(register_dict):
    """
    Create a dictionary mapping register names to Wires objects.

    This function takes a hierarchical register structure and flattens it into a dictionary
    where each key is a register name and each value is a Wires object representing the
    wires in that register.

    Args:
        register_dict (dict): A dictionary describing the register structure. Keys are
            register names (str) and values are either:
            - int: The number of wires in the register.
            - dict: A nested dictionary representing sub-registers.

    Returns:
        dict: A flattened dictionary where keys are register names (str) and values
        are Wires objects representing the wires in each register.

    Notes:
        - For nested registers, the parent register's Wires object will be the union
          of all its sub-registers' Wires objects.
        - Qubit indices are assigned sequentially, starting from 0, in the order they
          appear in the input dictionary.

    Examples:
        >>> qml.registers({"alice": 1, "bob": {"nest1": 2, "nest2": 1}})
        {
            'alice': Wires([0]),
            'bob': Wires([1, 2, 3]),
            'nest1': Wires([1, 2]),
            'nest2': Wires([3])
        }

        >>> qml.registers({"ancilla": {"sub_ancilla": 2, "sub_ancilla1": 1}})
        {
            'ancilla': Wires([0, 1, 2]),
            'sub_ancilla': Wires([0, 1]),
            'sub_ancilla1': Wires([2])
        }
    """

    def _registers(register_dict, _start_wire_index=0):
        """Recursively builds a dictionary of Wires objects from a nested dictionary of register
        names and sizes.

        Args:
            register_dict (dict): A dictionary describing the register structure. Keys are
                register names (str) and values are either:
                - int: The number of wires in the register.
                - dict: A nested dictionary representing sub-registers.
            _start_wire_index (int): the starting index for the wire labels.

        Returns:
            dict: A flattened dictionary where keys are register names (str) and values
            are Wires objects representing the wires in each register.
        """

        all_reg = {}
        for register_name, register_wires in register_dict.items():
            if isinstance(register_wires, dict):
                if len(register_wires) == 0:
                    raise ValueError(f"Got an empty dictionary '{register_wires}'")
                inner_register_dict = _registers(register_wires, _start_wire_index)
                wire_vals = [w for reg in inner_register_dict.values() for w in reg.tolist()]
                all_reg.update(inner_register_dict)
                all_reg[register_name] = Wires(
                    range(
                        _start_wire_index,
                        all_reg[next(reversed(inner_register_dict))].labels[-1] + 1,
                    )
                )
                _start_wire_index = wire_vals[-1] + 1
            elif isinstance(register_wires, int):
                if register_wires < 1:
                    raise ValueError(
                        f"Expected '{register_wires}' to be greater than 0. Please ensure that "
                        "the number of wires for the register is a positive integer"
                    )
                all_reg[register_name] = Wires(
                    range(_start_wire_index, register_wires + _start_wire_index)
                )
                _start_wire_index += register_wires
            else:  # Not a dict nor an int
                raise ValueError(f"Expected '{register_wires}' to be either a dict or an int. ")

        return all_reg

    return _registers(register_dict)

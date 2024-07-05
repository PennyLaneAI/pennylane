"""
This module contains the :func:`registers` function to build the registers for a given dictionary of registers
"""

from .wires import Wires


def registers(register_dict, _start_wire_index=0):
    """Returns the registers for the given dictionary of registers. The registers are a dictionary
    of Wires objects where the key is the register name and the value is the Wires object. The
    ordering of the Wires objects in the dictionary is based on appearance order first, then on
    nestedness.

    Args:
        register_dict (dict[str, int]): dictionary of registers where the keys are the name of the
        registers and the values are the number of qubits for said register

    Returns:
        dict (Wires): dictionary of Wires objects (value) belonging to registers (keys)

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
            raise ValueError(
                f"Expected '{register_wires}' to be either a dict or an int but got neither. "
                "Please double check and ensure that all objects in the dictionary have values "
                "that are either a dictionary or an integer"
            )

    return all_reg

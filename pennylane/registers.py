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
    Returns a dictionary that maps register names to :class:`~.Wires`.

    This function helps to group qubits and abstract away the finer details of running quantum
    algorithms. Register names and their total number of wires are typically known in advance,
    but managing the specific wire range for each register can be a challenge. The ``qml.registers()``
    function creates a dictionary that maps register names to :class:`~.Wires` objects. Moreover,
    it allows one to input a nested structure where registers contain sub-registers, as illustrated
    in the examples below.

    Args:
        register_dict (dict): a dictionary where keys are register names and values are either
            positive integers indicating the number of qubits or nested dictionaries of more registers

    Returns:
        dict: Dictionary where the keys are the names (str) of the registers, and the
        values are :class:`~.Wires` objects.

    **Example**

    Given flat input dictionary:

    >>> qml.registers({"alice": 2, "bob": 3})
    {'alice': Wires([0, 1]), 'bob': Wires([2, 3, 4])}

    Given nested input dictionary:

    >>> wire_registers = qml.registers({"people": {"alice": 2, "bob": 1}})
    >>> wire_registers
    {'alice': Wires([0, 1]), 'bob': Wires([2]), 'people': Wires([0, 1, 2])}
    >>> wire_registers['bob']
    Wires([2])
    >>> wire_registers['alice'][1]
    1

    A simple example showcasing how to implement the `SWAP <https://en.wikipedia.org/wiki/Swap_test>`_ test:

    .. code-block:: python

        dev = qml.device("default.qubit")
        reg = qml.registers({"aux": 1, "phi": 5, "psi": 5})

        @qml.qnode(dev)
        def circuit():
            for state in ["phi", "psi"]:
                 qml.BasisState([1, 1, 0, 0, 0], reg[state])

            qml.Hadamard(reg["aux"])
            for i in range(len(reg["phi"])):
                qml.CSWAP(reg["aux"] + reg["phi"][i] + reg["psi"][i])
            qml.Hadamard(reg["aux"])

            return qml.expval(qml.Z(wires=reg["aux"]))

    >>> print(circuit())
    0.999...
    """

    def _registers(register_dict, _start_wire_index=0):
        """Recursively builds a dictionary of Wires objects from a nested dictionary of register
        names and sizes.

        Args:
            register_dict (dict): a dictionary where keys are register names and values are either
                positive integers indicating the number of qubits or nested dictionaries of more registers
            _start_wire_index (int): the starting index for the wire labels.

        Returns:
            dict (Wires): dictionary where the keys are the names (str) of the registers, and the
            values are either :class:`~.Wires` objects or other registers
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

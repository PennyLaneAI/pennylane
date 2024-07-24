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
    The ``qml.registers()`` function creates a dictionary that maps register name to :class:`~.Wires`
    object.

    The function takes in a dictionary. The keys for this input dictionary are the names of the
    registers. The values for this dictionary are either integers that represent the number of
    wires or dictionaries that represent sub registers. For example, the key-value pair
    `"ancilla": 3` would represent a register named "ancilla" with 3 wires whereas the key-value
    pair `"ancilla": {"sub_ancilla": 2, "sub_ancilla1": 1}` would represent a register named
    "ancilla" with two sub_registers: "sub_ancilla" and "sub_ancilla1", each with their respective
    number of wires.

    Given input `{"ancilla": {"sub_ancilla": 2, "sub_ancilla1": 1}}`, ``qml.registers()`` creates
    a dictionary with 3 key-value pairs. The keys are the register names found in the input:
    "ancilla", "sub_ancilla" and "sub_ancilla1". The values are the respective :class:`~.Wires`
    objects for each register. For example, "ancilla" has two sub registers "sub_ancilla" and
    "sub_ancilla1". Therefore, the value associated with key "ancilla" is the union of the
    :class:`~.Wires` of its sub registers. Since its sub registers "sub_ancilla" and "sub_ancilla1"
    have :class:`~.Wires` objects `Wires([0, 1])` and Wires([2]) respectively, the key "ancilla"
    has the value `Wires([0, 1, 2])`.

    Args:
        register_dict (dict): a dictionary where keys are register names and values are either
            positive integers indicating the number of qubits or nested dictionaries of more registers.

    Returns:
        dict (Wires): dictionary where the keys are the names (str) of the registers, and the
        values are :class:`~.Wires` objects.

    **Example**

    >>> wire_registers = qml.registers({"alice": 1, "bob": {"nest1": 2, "nest2": 1}})
    >>> wire_dict
    {'alice': Wires([0]), 'nest1': Wires([1, 2]), 'nest2': Wires([3]), 'bob': Wires([1, 2, 3])}
    >>> wire_dict['nest1']
    Wires([1, 2])

    A simple example showcasing how to implement the `SWAP <https://en.wikipedia.org/wiki/Swap_test>`_ test:

    .. code-block::

        dev = qml.device("default.qubit")
        reg =  registers({"aux": 1, "phi": 5, "psi": 5})

        @qml.qnode(dev)
        def circuit():
            for state in ["phi", "psi"]:
                 qml.BasisState([1, 1, 0, 0, 0], reg[state])

            qml.Hadamard(reg["aux"])
            for i in range(len(reg["phi"])):
                qml.CSWAP(reg["aux"] + reg["phi"][i] + reg["psi"][i])
            qml.Hadamard(reg["aux"])

            return qml.expval(qml.Z(wires=reg["aux"]))

        >>> circuit()
        0.9999999999999996
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

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
Contains the ``SWAP Test`` template.
"""
import pennylane as qml
from pennylane.templates.decorator import template
from pennylane.wires import Wires


@template
def SWAPTest(register1, register2, ancilla):

    r""" A controlled-SWAP subroutine that can be used to create a
    `SWAP test <https://en.wikipedia.org/wiki/Swap_test>`__ between
    two registers.

    Args:
        register1 (Iterable or Wires): The first register of wires passed into the template.
        register2 (Iterable or Wires): The second register of wires passed into the template.
        ancilla (int or Wires): The ancilla qubit that controls the CSWAP gates in the template.

    Raises:
        ValueError: if inputs do not have the correct format

    .. UsageDetails::

        The template is used inside a qnode:

        .. code-block:: python

            import pennylane as qml
            from pennylane.templates import SWAPTest
            from math import pi

            n_wires = 5
            dev = qml.device('default.qubit', wires=n_wires)

            @qml.qnode(dev)
            def circuit(register1=None, register2=None, ancilla=None):
                for wire in register1:
                    qml.Hadamard(wires=wire)
                SWAPTest(register1, register2, ancilla)
                return qml.expval(qml.PauliZ(wires=ancilla))

        >>> circuit(register1=[1, 2], register2=[3, 4], ancilla=0)
        0.25
    """

    #############
    # Input checks

    register1 = Wires(register1)
    register2 = Wires(register2)
    ancilla = Wires(ancilla)

    # Register lengths are the same and ancilla length is 1

    if len(register1) != len(register2):
        raise ValueError(
            "Lengths of qubit registers must be the same, got {} and {}".format(
                len(register1), len(register2)
            )
        )

    if len(ancilla) != 1:
        raise ValueError("`ancilla` argument takes one wire index, got {}".format(len(ancilla)))

    # No repeats in the registers

    unique = (Wires.unique_wires([register1, register2, ancilla])).tolist()
    if unique != register1.tolist() + register2.tolist() + ancilla.tolist():
        raise ValueError("Wire indices for both registers and the ancilla must be unique")

    #############

    qml.Hadamard(wires=ancilla)

    for i, reg in enumerate(register1):
        qml.CSWAP(wires=[ancilla, reg, register2[i]])

    qml.Hadamard(wires=ancilla)

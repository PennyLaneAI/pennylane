# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""This module contains the base class for qubit management"""

from typing import Union

import pennylane as qml
from pennylane.queuing import QueuingManager


class QubitManager:
    r"""Contains attributes which help track how auxilliary qubits are used in a circuit
    Args:
        work_wires (int or dict): Number of work wires or a dictionary containing
            number of clean and dirty work wires. All work_wires are assumed to be clean when
            `int` is provided.
        tight_budget (bool): flag to determine whether extra clean qubits are available

    **Example**

    >>> q = QubitManager(
    ...             work_wires={"clean": 2, "dirty": 2},
    ...             tight_budget=False,
    ...     )
    >>> print(q)
    QubitManager(clean=2, dirty=2, logic=0, tight_budget=False)

    """

    def __init__(self, work_wires: Union[int, dict], tight_budget=False) -> None:

        if isinstance(work_wires, dict):
            clean_wires = work_wires["clean"]
            dirty_wires = work_wires["dirty"]
        else:
            clean_wires = work_wires
            dirty_wires = 0

        self.tight_budget = tight_budget
        self._logic_qubit_counts = 0
        self._clean_qubit_counts = clean_wires
        self._dirty_qubit_counts = dirty_wires

    def __str__(self):
        return f"QubitManager(clean qubits={self._clean_qubit_counts}, dirty qubits={self._dirty_qubit_counts}, logic qubits={self._logic_qubit_counts}, tight budget={self.tight_budget})"

    def __repr__(self) -> str:
        return str(self)

    @property
    def clean_qubits(self):
        r"""Returns the number of clean qubits."""
        return self._clean_qubit_counts

    @property
    def dirty_qubits(self):
        r"""Returns the number of dirty qubits."""
        return self._dirty_qubit_counts

    @property
    def algo_qubits(self):
        r"""Returns the number of algorithmic qubits."""
        return self._logic_qubit_counts

    @algo_qubits.setter
    def algo_qubits(self, count: int):  # these get set manually, the rest
        r"""Setter for algorithmic qubits."""
        self._logic_qubit_counts = count  #  are dynamically updated

    def allocate_qubits(self, num_qubits: int):
        r"""Allocates extra clean qubits.

        Args:
            num_qubits(int): number of qubits to be allocated

        """
        self._clean_qubit_counts += num_qubits

    def grab_clean_qubits(self, num_qubits: int):
        r"""Grabs clean qubits.

        Args:
            num_qubits(int) : number of clean qubits to be grabbed

        Raises:
            ValueError: If tight_budget is `True` number of qubits to be grabbed is greater than available clean qubits.

        """
        available_clean = self.clean_qubits

        if num_qubits > available_clean:
            if self.tight_budget:
                raise ValueError(
                    f"Grabbing more qubits than available clean qubits."
                    f"Number of clean qubits available is {available_clean}, while {num_qubits} are being grabbed."
                )
            self._clean_qubit_counts = 0
        else:
            self._clean_qubit_counts -= num_qubits
        self._dirty_qubit_counts += num_qubits

    def free_qubits(self, num_qubits: int):
        r"""Frees dirty qubits and converts them to clean qubits.

        Args:
            num_qubits(int) : number of qubits to be freed

        Raises:
            ValueError: If number of qubits to be freed is greater than available dirty qubits.
        """
        available_dirty = self.dirty_qubits

        if num_qubits > available_dirty:
            raise ValueError(
                f"Freeing more qubits than available dirty qubits."
                f"Number of dirty qubits available is {available_dirty}, while {num_qubits} qubits are being released."
            )

        self._dirty_qubit_counts -= min(available_dirty, num_qubits)
        self._clean_qubit_counts += min(available_dirty, num_qubits)


class GrabWires:
    r"""Allows users to allocate clean work wires

    Args:
        num_wires (int): number of work wires to be allocated
    """

    _queue_category = "_resource_qubit_action"

    def __init__(self, num_wires):
        self.num_wires = num_wires

        if qml.QueuingManager.recording():
            self.queue()

    def __repr__(self) -> str:
        return f"GrabWires({self.num_wires})"

    def queue(self, context=QueuingManager):
        r"""Adds GrabWires object to a queue."""
        context.append(self)
        return self


class FreeWires:
    r"""Allows users to free dirty work wires

    Args:
        num_wires (int): number of dirty work wires to be freed.
    """

    _queue_category = "_resource_qubit_action"

    def __init__(self, num_wires):
        self.num_wires = num_wires

        if qml.QueuingManager.recording():
            self.queue()

    def __repr__(self) -> str:
        return f"FreeWires({self.num_wires})"

    def queue(self, context=QueuingManager):
        r"""Adds FreeWires object to a queue."""
        context.append(self)
        return self

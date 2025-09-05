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


from pennylane.queuing import QueuingManager


# pylint: disable=too-few-public-methods


class WireResourceManager:
    r"""Manages and tracks the auxiliary and algorithmic qubits used in a quantum circuit.

    Args:
        work_wires (int or dict[str, int]): Number of work wires or a dictionary containing
            number of clean and dirty work wires. All ``work_wires`` are assumed to be clean when
            ``int`` is provided.
        algo_wires (int): Number of algorithmic wires, default value is ``0``.
        tight_budget (bool): Determines whether extra clean qubits can be allocated when they
            exceed the available amount. The default is ``False``.

    **Example**

    >>> q = WireResourceManager(
    ...             work_wires={"clean": 2, "dirty": 2},
    ...             tight_budget=False,
    ...     )
    >>> print(q)
    WireResourceManager(clean wires=2, dirty wires=2, algorithmic wires=0, tight budget=False)

    """

    def __init__(self, work_wires: int | dict, algo_wires: int = 0, tight_budget: bool = False) -> None:

        clean_wires, dirty_wires = (
            (work_wires["clean"], work_wires["dirty"])
            if isinstance(work_wires, dict)
            else (work_wires, 0)
        )

        self.tight_budget = tight_budget
        self.algo_wires = algo_wires
        self.clean_wires = clean_wires
        self.dirty_wires = dirty_wires

    def __str__(self):
        return (
            f"WireResourceManager(clean wires={self.clean_wires}, dirty wires={self.dirty_wires}, "
            f"algorithmic wires={self.algo_wires}, tight budget={self.tight_budget})"
        )

    def __repr__(self) -> str:
        work_wires_str = repr({"clean": self.clean_wires, "dirty": self.dirty_wires})
        return (
            f"WireResourceManager(work_wires={work_wires_str}, algo_wires={self.algo_wires}, "
            f"tight_budget={self.tight_budget})"
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, self.__class__)
            and (self.clean_wires == other.clean_wires)
            and (self.dirty_wires == other.dirty_wires)
            and (self.algo_wires == other.algo_wires)
            and (self.tight_budget == other.tight_budget)
        )

    @property
    def algo_qubits(self):
        r"""Returns the number of algorithmic qubits."""
        return self.algo_wires

    @property
    def total_qubits(self):
        r"""Returns the number of total qubits."""
        return self.clean_wires + self.dirty_wires + self.algo_wires

    @algo_qubits.setter
    def algo_qubits(self, count: int):  # these get set manually, the rest are dynamically updated
        r"""Setter for algorithmic qubits."""
        self.algo_wires = count

    def grab_clean_qubits(self, num_qubits: int):
        r"""Grabs clean qubits, and converts them into dirty ones.

        Args:
            num_qubits(int) : number of clean qubits to be grabbed

        Raises:
            ValueError: If tight_budget is `True` and the number of qubits to be grabbed is greater than
            available clean qubits.

        """
        available_clean = self.clean_wires

        if num_qubits > available_clean:
            if self.tight_budget:
                raise ValueError(
                    f"Grabbing more qubits than available clean qubits."
                    f"Number of clean qubits available is {available_clean}, while {num_qubits} are being grabbed."
                )
            self.clean_wires = 0
        else:
            self.clean_wires -= num_qubits
        self.dirty_wires += num_qubits

    def free_qubits(self, num_qubits: int):
        r"""Frees dirty qubits and converts them into clean qubits.

        Args:
            num_qubits(int) : number of qubits to be freed

        Raises:
            ValueError: If number of qubits to be freed is greater than available dirty qubits.
        """

        if num_qubits > self.dirty_wires:
            raise ValueError(
                f"Freeing more qubits than available dirty qubits."
                f"Number of dirty qubits available is {self.dirty_wires}, while {num_qubits} qubits are being released."
            )

        self.dirty_wires -= num_qubits
        self.clean_wires += num_qubits


class _WireAction:
    """Base class for operations that manage qubit resources."""

    def __init__(self, num_wires):
        self.num_wires = num_wires
        if QueuingManager.recording():
            self.queue()

    def queue(self, context=QueuingManager):
        r"""Adds the wire action object to a queue."""
        context.append(self)
        return self

    def __eq__(self, other: "_WireAction") -> bool:
        return isinstance(other, self.__class__) and self.num_wires == other.num_wires

    def __mul__(self, other):
        if isinstance(other, int):
            return self.__class__(self.num_wires * other)
        raise NotImplementedError


class AllocWires(_WireAction):
    r"""Allows users to allocate clean work wires.

    Args:
        num_wires (int): number of work wires to be allocated.
    """

    def __repr__(self) -> str:
        return f"AllocWires({self.num_wires})"


class FreeWires(_WireAction):
    r"""Allows users to free dirty work wires.

    Args:
        num_wires (int): number of dirty work wires to be freed.
    """

    def __repr__(self) -> str:
        return f"FreeWires({self.num_wires})"

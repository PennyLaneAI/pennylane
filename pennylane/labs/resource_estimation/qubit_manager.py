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


import pennylane as qml
from pennylane.exceptions import QuantumFunctionError
from pennylane.wires import Wires

# pylint: disable= too-few-public-methods, too-many-instance-attributes


class _WireResourceManager:
    r"""Manages and tracks the auxiliary and algorithmic qubits used in a quantum circuit.

    Args:
        work_wires (int or Dict[str, int]): Number of work wires or a dictionary containing
            number of clean and dirty work wires. All ``work_wires`` are assumed to be clean when
            `int` is provided.
        algo_wires (int): Number of algorithmic wires, default value is ``0``.
        tight_budget (bool): Determines whether extra clean qubits can be allocated when they
            exceed the available amount. The default is ``False``.

    **Example**

    >>> qm = plre._WireResourceManager(
    ...     work_wires = {"clean": 2, "dirty": 2},
    ...     algo_wires = 10,
    ...     tight_budget = False,
    ... )
    >>> print(qm)
    _WireResourceManager(clean_wires=2, dirty_wires=2, algorithmic_wires=10, tight_budget=False)

    """

    def __init__(self, work_wires: int | dict, algo_wires=0, tight_budget=False) -> None:

        self.tight_budget = tight_budget
        if isinstance(work_wires, dict):
            clean_wires = work_wires["clean"]
            dirty_wires = work_wires["dirty"]
        else:
            clean_wires = work_wires
            dirty_wires = 0

        # Idle:
        self.clean_idle_aux = clean_wires
        self.dirty_idle_aux = dirty_wires

        self.clean_idle_algo = 0
        self.dirty_idle_algo = 0

        # Active:
        self.clean_active_algo = algo_wires  # Only set at the "top-level"

        # -> dirty auxiliary:
        self.dirty_active_aux = 0
        self.return_aux_cleaned = 0
        self.return_aux_dirty = 0

        # -> dirty algorithmic:
        self.dirty_active_algo = 0
        self.return_algo_cleaned = 0
        self.return_algo_dirty = 0

    @property
    def algo_wires(self):
        """The number of algorithmic wires in the circuit."""
        return (
            self.clean_active_algo
            + self.clean_idle_algo
            + self.dirty_active_algo
            + self.dirty_idle_algo
        )

    @property
    def dirty_wires(self):
        """The number of auxiliary wires, in the dirty state, in the circuit."""
        return self.dirty_idle_aux + self.dirty_active_aux

    @property
    def clean_wires(self):
        """The number of auxiliary wires, in the clean state, in the circuit."""
        return (
            self.clean_idle_aux
        )  # no clean_active_aux because any active aux wire is autmatically marked dirty

    @property
    def total_wires(self):
        """All of the wires defined & used in the circuit."""
        return self.clean_wires + self.dirty_wires + self.algo_wires

    def allocate_fresh(self, n):
        """Allocate additional clean, idle, auxiliary qubits."""
        if self.tight_budget:
            raise QuantumFunctionError(
                "Requesting more qubits than availble within the budget. Set `tight_budget` to False."
            )
        self.clean_idle_aux += n

    def borrow_clean(self, n_requested=1):
        """Borrow clean qubits in the context of decomposition"""
        # Borrow existing clean auxiliary qubits first:
        n_available = min(n_requested, self.clean_idle_aux)

        self.clean_idle_aux -= n_available
        self.dirty_active_aux += n_available
        self.return_aux_cleaned += n_available

        n_requested -= n_available
        if n_requested == 0:
            return

        # Borrow idle clean algorithmic qubits next:
        n_available = min(n_requested, self.clean_idle_algo)

        self.clean_idle_algo -= n_available
        self.dirty_active_algo += n_available
        self.return_algo_cleaned += n_available

        n_requested -= n_available
        if n_requested == 0:
            return

        # Finally, if we still need more, allocate fresh qubits
        self.allocate_fresh(n_requested)  # allocate clean auxiliary qubits

        self.clean_idle_aux -= n_requested
        self.dirty_active_aux += n_requested
        self.return_aux_cleaned += n_requested

    def borrow_dirty(self, n_requested=1):
        """Borrow dirty qubits in the context of decomposition"""
        # First borrow from any idle dirty auxiliary qubits:
        n_available = min(n_requested, self.dirty_idle_aux)

        self.dirty_idle_aux -= n_available
        self.dirty_active_aux += n_available
        self.return_aux_dirty += n_available

        n_requested -= n_available
        if n_requested == 0:
            return

        # Borrow idle dirty algorithmic qubits next:
        n_available = min(n_requested, self.dirty_idle_algo)

        self.dirty_idle_algo -= n_available
        self.dirty_active_algo += n_available
        self.return_algo_dirty += n_available

        n_requested -= n_available
        if n_requested == 0:
            return

        # Finally, if we still need more, borrow clean qubits:
        self.borrow_clean(n_requested)

    def return_clean(self, n_requested=1):
        """Return borrowed qubits to the clean state in the context of the decomposition"""
        # Return any borrowed idle algorithmic qubits first
        n_available = min(n_requested, self.return_algo_cleaned)

        if n_available > self.dirty_active_algo:
            raise ValueError

        self.clean_idle_algo += n_available
        self.dirty_active_algo -= n_available
        self.return_algo_cleaned -= n_available

        n_requested -= n_available
        if n_requested == 0:
            return

        # Return the remaining auxiliary qubits:
        n_available = min(n_requested, self.return_aux_cleaned)

        if n_available > self.dirty_active_aux:
            raise ValueError

        self.clean_idle_aux += n_available
        self.dirty_active_aux -= n_available
        self.return_aux_cleaned -= n_available

        n_requested -= n_available
        if n_requested == 0:
            return

        # There shouldn't be anymore qubits to return:
        raise ValueError("Don't have anymore qubits to clean and return")

    def return_dirty(self, n_requested=1):
        """Return borrowed qubits to the dirty state in the context of the decomposition"""
        # Return any borrowed idle algorithmic qubits first
        n_available = min(n_requested, self.return_algo_dirty)

        if n_available > self.dirty_active_algo:
            raise ValueError

        self.dirty_idle_algo += n_available
        self.dirty_active_algo -= n_available
        self.return_algo_dirty -= n_available

        n_requested -= n_available
        if n_requested == 0:
            return

        # Return the remaining auxiliary qubits:
        n_available = min(n_requested, self.return_aux_dirty)

        if n_available > self.dirty_active_aux:
            raise ValueError

        self.dirty_idle_aux += n_available
        self.dirty_active_aux -= n_available
        self.return_aux_dirty -= n_available

        n_requested -= n_available
        if n_requested == 0:
            return

        # If there are more qubits to be returned, we must have
        # allocated fresh qubits to aquire them. Thus we have to
        # return_clean the rest:
        self.return_clean(n_requested)

    def take(self, n_requested=1):
        """Use clean qubits in a manner that is not returnable"""
        n_available = min(n_requested, self.clean_idle_aux)

        self.clean_idle_aux -= n_available
        self.dirty_active_aux += n_available

        n_requested -= n_available
        if n_requested == 0:
            return

        # Finally, if we still need more, allocate fresh qubits
        self.allocate_fresh(n_requested)  # allocate clean auxiliary qubits

        self.clean_idle_aux -= n_requested
        self.dirty_active_aux += n_requested

    def release(self, n_requested=1):
        """Restore dirty auxiliary qubits to the |0> state"""
        if n_requested > self.dirty_idle_aux + self.dirty_active_aux:
            raise QuantumFunctionError("Resetting more qubits than availble to restore.")

        n_available = min(n_requested, self.dirty_active_aux)  # release active qubits first

        self.clean_idle_aux += n_available
        self.dirty_active_aux -= n_available

        n_requested -= n_available
        if n_requested == 0:
            return

        self.clean_idle_aux += n_requested
        self.dirty_idle_aux -= n_requested

    def step_into_decomp(self, num_active_qubits) -> "_WireResourceManager":
        """Creates a new instance of _WireResourceManager which is modified to 
        set idle qubits as active according to which qubits are required by the lower
        decomposition scope."""
        qm = self.__class__(work_wires=0, algo_wires=0)

        # Copy over all idle qubit counts
        qm.clean_idle_aux = self.clean_idle_aux
        qm.dirty_idle_aux = self.dirty_idle_aux
        qm.clean_idle_algo = self.clean_idle_algo
        qm.dirty_idle_algo = self.dirty_idle_algo

        # Map all active qubit counts to idle in the new scope:
        qm.dirty_idle_aux += self.dirty_active_aux
        qm.clean_idle_algo += self.clean_active_algo
        qm.dirty_idle_algo += self.dirty_active_algo

        # Set the new active qubits:
        if (
            qm.clean_idle_aux + qm.clean_idle_algo + qm.dirty_idle_aux + qm.dirty_idle_algo
        ) < num_active_qubits:
            raise QuantumFunctionError("Not enough idle qubits to make active")

        # Use any dirty algorithmic qubits first:
        n_available = min(num_active_qubits, qm.dirty_idle_algo)
        qm.dirty_idle_algo -= n_available
        qm.dirty_active_algo += n_available

        num_active_qubits -= n_available
        if num_active_qubits == 0:
            return qm

        # Use any clean algorithmic qubits next:
        n_available = min(num_active_qubits, qm.clean_idle_algo)
        qm.clean_idle_algo -= n_available
        qm.dirty_active_algo += n_available

        num_active_qubits -= n_available
        if num_active_qubits == 0:
            return qm

        # Use any dirty auxiliary qubits next:
        n_available = min(num_active_qubits, qm.dirty_idle_aux)
        qm.dirty_idle_aux -= n_available
        qm.dirty_active_aux += n_available

        num_active_qubits -= n_available
        if num_active_qubits == 0:
            return qm

        # Finally any clean auxiliary qubits last:
        n_available = min(num_active_qubits, qm.clean_idle_aux)
        qm.clean_idle_aux -= n_available
        qm.dirty_active_aux += n_available

        return qm

    def step_out_decomp(self, subqm: "_WireResourceManager") -> None:
        """Resolves the state of all qubits by merging the current state of the qubits
        with the updates captured by the :code:`subqm` resource manager."""

        # Copy over all idle qubit counts
        self.clean_idle_aux = subqm.clean_idle_aux
        self.dirty_idle_aux = subqm.dirty_idle_aux
        self.clean_idle_algo = subqm.clean_idle_algo
        self.dirty_idle_algo = subqm.dirty_idle_algo

        # Map all active qubit counts to idle in the new scope:
        self.dirty_idle_aux += subqm.dirty_active_aux
        self.clean_idle_algo += subqm.clean_active_algo
        self.dirty_idle_algo += subqm.dirty_active_algo

        # Reset the old active qubits:
        num_old_aux_active_qubits = self.dirty_active_aux
        num_old_algo_active_qubits = self.clean_active_algo + self.dirty_active_algo

        self.dirty_active_aux = 0
        self.dirty_active_algo = 0
        self.clean_active_algo = 0

        # Set any dirty algorithmic qubits to active first:
        n_available = min(num_old_algo_active_qubits, self.dirty_idle_algo)
        self.dirty_idle_algo -= n_available
        self.dirty_active_algo += n_available

        num_old_algo_active_qubits -= n_available

        # Set any clean algorithmic qubits to active next:
        if num_old_algo_active_qubits > 0:

            n_available = min(num_old_algo_active_qubits, self.clean_idle_algo)
            self.clean_idle_algo -= n_available
            self.clean_active_algo += n_available

            n_available -= num_old_algo_active_qubits

        # Finally set the dirty auxiliary qubits to active:
        self.dirty_idle_aux -= num_old_aux_active_qubits
        self.dirty_active_aux += num_old_aux_active_qubits

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, self.__class__)
            and (self.clean_idle_aux == other.clean_idle_aux)
            and (self.dirty_idle_aux == other.dirty_idle_aux)
            and (self.clean_idle_algo == other.clean_idle_algo)
            and (self.dirty_idle_algo == other.dirty_idle_algo)
            and (self.clean_active_algo == other.clean_active_algo)
            and (self.dirty_active_aux == other.dirty_active_aux)
            and (self.return_aux_cleaned == other.return_aux_cleaned)
            and (self.return_aux_dirty == other.return_aux_dirty)
            and (self.dirty_active_algo == other.dirty_active_algo)
            and (self.return_algo_cleaned == other.return_algo_cleaned)
            and (self.return_algo_dirty == other.return_algo_dirty)
            and (self.tight_budget == other.tight_budget)
        )

    def __str__(self):
        return (
            f"_WireResourceManager(clean_wires={self.clean_wires}, dirty_wires={self.dirty_wires}, "
            f"algorithmic_wires={self.algo_wires}, tight_budget={self.tight_budget})"
        )

    def _box_plot(self):
        """Private method for plotting the resources using box diagrams."""
        total_algo_qubits = (
            self.clean_idle_algo
            + self.clean_active_algo
            + self.dirty_active_algo
            + self.dirty_idle_algo
        )
        total_aux_qubits = self.clean_idle_aux + self.dirty_active_aux + self.dirty_idle_aux

        str_rep = f"Algo: {total_algo_qubits}\n"
        str_rep += "  | C , D |\n"
        str_rep += f"I | {self.clean_idle_algo} , {self.dirty_idle_algo} |\n"
        str_rep += f"A | {self.clean_active_algo} , {self.dirty_active_algo} |\n"

        str_rep += f"\nAuxi: {total_aux_qubits}\n"
        str_rep += "  | C , D |\n"
        str_rep += f"I | {self.clean_idle_aux} , {self.dirty_idle_aux} |\n"
        str_rep += f"A | x , {self.dirty_active_aux} |\n"

        return str_rep


class QubitManager:
    r"""Manages and tracks the auxiliary and algorithmic qubits used in a quantum circuit.

    Args:
        work_wires (int or Dict[str, int]): Number of work wires or a dictionary containing
            number of clean and dirty work wires. All ``work_wires`` are assumed to be clean when
            `int` is provided.
        algo_wires (int): Number of algorithmic wires, default value is ``0``.
        tight_budget (bool): Determines whether extra clean qubits can be allocated when they
            exceed the available amount. The default is ``False``.

    **Example**

    >>> q = QubitManager(
    ...             work_wires={"clean": 2, "dirty": 2},
    ...             tight_budget=False,
    ...     )
    >>> print(q)
    QubitManager(clean=2, dirty=2, logic=0, tight_budget=False)

    """

    def __init__(self, work_wires: int | dict, algo_wires=0, tight_budget=False) -> None:

        if isinstance(work_wires, dict):
            clean_wires = work_wires["clean"]
            dirty_wires = work_wires["dirty"]
        else:
            clean_wires = work_wires
            dirty_wires = 0

        self.tight_budget = tight_budget
        self._logic_qubit_counts = algo_wires
        self._clean_qubit_counts = clean_wires
        self._dirty_qubit_counts = dirty_wires

    def __str__(self):
        return (
            f"QubitManager(clean qubits={self._clean_qubit_counts}, dirty qubits={self._dirty_qubit_counts}, "
            f"algorithmic qubits={self._logic_qubit_counts}, tight budget={self.tight_budget})"
        )

    def __repr__(self) -> str:
        work_wires_str = repr(
            {"clean": self._clean_qubit_counts, "dirty": self._dirty_qubit_counts}
        )
        return (
            f"QubitManager(work_wires={work_wires_str}, algo_wires={self._logic_qubit_counts}, "
            f"tight_budget={self.tight_budget})"
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, self.__class__)
            and (self.clean_qubits == other.clean_qubits)
            and (self.dirty_qubits == other.dirty_qubits)
            and (self.algo_qubits == other.algo_qubits)
            and (self.tight_budget == other.tight_budget)
        )

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

    @property
    def total_qubits(self):
        r"""Returns the number of total qubits."""
        return self._clean_qubit_counts + self._dirty_qubit_counts + self.algo_qubits

    @algo_qubits.setter
    def algo_qubits(self, count: int):  # these get set manually, the rest are dynamically updated
        r"""Setter for algorithmic qubits."""
        self._logic_qubit_counts = count

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
            ValueError: If tight_budget is `True` number of qubits to be grabbed is greater than
            available clean qubits.

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

        if num_qubits > self.dirty_qubits:
            raise ValueError(
                f"Freeing more qubits than available dirty qubits."
                f"Number of dirty qubits available is {self.dirty_qubits}, while {num_qubits} qubits are being released."
            )

        self._dirty_qubit_counts -= num_qubits
        self._clean_qubit_counts += num_qubits


class _WireAction:
    """Base class for operations that manage qubit resources."""

    def __init__(self, num_wires, wires=None):
        self.wires = None if wires is None else Wires(wires)
        self.num_wires = num_wires
        if qml.QueuingManager.recording():
            self.queue()

    def queue(self, context=qml.QueuingManager):
        r"""Adds the wire action object to a queue."""
        context.append(self)
        return self

    def __eq__(self, other: "_WireAction") -> bool:
        return (
            self.__class__ == other.__class__
            and (self.num_wires == other.num_wires)
            and (self.wires == other.wires)
        )

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        if self.wires:
            return f"{cls_name}({self.num_wires}, wire_labels={self.wires})"
        return f"{cls_name}({self.num_wires})"


class AllocWires(_WireAction):
    r"""Allows users to allocate clean work wires.

    Args:
        num_wires (int): number of work wires to be allocated.
        wires (Sequence[int], optional): the wire labels of the wires that were allocated
    """


class FreeWires(_WireAction):
    r"""Allows users to free dirty work wires.

    Args:
        num_wires (int): number of dirty work wires to be freed.
        wires (Sequence[int], optional): the wire labels of the wires that were freed
    """


class BorrowWires(_WireAction):
    r"""Allows users to borrow idle wires from the circuit.

    Args:
       num_wires (int): number of work wires to be borrowed.
       clean (bool, optional): borrowing qubits in the zeroed state (default is True)
       wires (Sequence[int], optional): the new wire labels of the wires that were borrowed

    """

    def __init__(self, num_wires, clean=True, wires=None):
        self.clean = clean
        super().__init__(num_wires, wires=wires)

    def __repr__(self) -> str:
        if self.wires:
            return f"BorrowWires({self.num_wires}, clean={self.clean}, wire_labels={self.wires})"
        return f"BorrowWires({self.num_wires}, clean={self.clean})"

    def __eq__(self, other: _WireAction) -> bool:
        return (
            self.__class__ == other.__class__
            and (self.num_wires == other.num_wires)
            and (self.wires == other.wires)
            and (self.clean == other.clean)
        )


class ReturnWires(_WireAction):
    r"""Allows users to return idle wires to the circuit.

    Args:
        num_wires (int): number of work wires to be returned.
        clean (bool, optional): returing qubits in the zeroed state (default is True)
        wires (Sequence[int], optional): the wire labels of the wires that were returned

    """

    def __init__(self, num_wires, clean=True, wires=None):
        self.clean = clean
        super().__init__(num_wires, wires=wires)

    def __repr__(self) -> str:
        if self.wires:
            return f"ReturnWires({self.num_wires}, clean={self.clean}, wire_labels={self.wires})"
        return f"ReturnWires({self.num_wires}, clean={self.clean})"

    def __eq__(self, other: _WireAction) -> bool:
        return (
            self.__class__ == other.__class__
            and (self.num_wires == other.num_wires)
            and (self.wires == other.wires)
            and (self.clean == other.clean)
        )

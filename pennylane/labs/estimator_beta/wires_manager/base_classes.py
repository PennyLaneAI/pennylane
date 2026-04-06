# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains the base class for wire management."""
from typing import Literal

from pennylane.allocation import AllocateState
from pennylane.queuing import QueuingManager
from pennylane.wires import Wires


class Allocate:
    r"""A class used to represent the allocation of auxiliary wires to be used in the resource
    decomposition of a :class:`~.pennylane.estimator.resource_operator.ResourceOperator`.

    Args:
        num_wires (int): the number of wires to be allocated
        state (Literal["any", "zero"] | AllocateState): The quantum state of the wires to be allocated, valid values include "zero" or "any".
        restored (bool): A guarantee that the allocated register will be restored (deallocated) to its
            initial state. If True, this requirement will be enforced programmatically.

    Raises:
        ValueError: `num_wires` must be a positive integer
        ValueError: if `restored` is not a boolean

    **Example**

    >>> import pennylane.labs.estimator_beta as qre
    >>> qre.Allocate(4)
    Allocate(4, state=zero, restored=False)
    >>> qre.Allocate(2, state="any", restored=True)
    Allocate(2, state=any, restored=True)

    """

    def __init__(
        self,
        num_wires,
        state: Literal["any", "zero"] | AllocateState = AllocateState.ZERO,
        restored=False,
    ):
        if not isinstance(num_wires, int) or num_wires <= 0:
            raise ValueError(f"num_wires must be a positive integer, got {num_wires}")

        if not isinstance(restored, bool):
            raise ValueError(f"Expected restored to be True or False, got {restored}")

        self._state = AllocateState(state)
        self._restored = restored
        self._num_wires = num_wires

    def equal(
        self, other: "Allocate"
    ) -> bool:  # We avoid overriding `__eq__` due to concerns with hashing
        """Determine if two instances of the class are equal."""
        if not isinstance(other, self.__class__):
            return False

        return all(
            (
                self.state == other.state,
                self.restored == other.restored,
                self.num_wires == other.num_wires,
            )
        )

    def __repr__(self) -> str:
        return f"Allocate({self.num_wires}, state={self.state}, restored={self.restored})"

    @property
    def state(self):
        """The quantum state of the wires to be allocated, valid values include "zero" or "any"."""
        return self._state

    @state.setter
    def state(self, _):
        """Raise error if users attempt to change values"""
        raise AttributeError("Allocate instances are not mutable")

    @property
    def restored(self):
        """A guarantee that the allocated register will be restored (deallocated) to its
        initial state. If True, this requirement will be enforced programmatically."""
        return self._restored

    @restored.setter
    def restored(self, _):
        """Raise error if users attempt to change values"""
        raise AttributeError("Allocate instances are not mutable")

    @property
    def num_wires(self):
        """The number of wires to be allocated."""
        return self._num_wires

    @num_wires.setter
    def num_wires(self, _):
        """Raise error if users attempt to change values"""
        raise AttributeError("Allocate instances are not mutable")


class Deallocate:
    r"""A class used to represent the deallocation of auxiliary wires that were used in the resource
    decomposition of a :class:`~.pennylane.estimator.resource_operator.ResourceOperator`.

    Args:
        num_wires (int | None): the number of wires to be deallocated
        allocated_register (Allocate | None): the allocated wire register the we wish to deallocate
        state (Literal["any", "zero"] | AllocateState): The quantum state of the wires to be deallocated, valid values include "zero" or "any".
        restored (bool): A guarantee that the allocated register will be restored (deallocated) to its
            initial state. If True, this requirement will be enforced programmatically.

    Raises:
        ValueError: if `num_wires` is not a positive integer
        ValueError: if `restored` is not a boolean

    **Example**

    The simplest way to deallocate a register is to provide the instance of ``Allocate``
    where the register was allocated.

    >>> import pennylane.labs.estimator_beta as qre
    >>> allocate_4 = qre.Allocate(4)  # Allocate 4 qubits
    >>> qre.Deallocate(allocated_register=allocate_4)
    Deallocate(4, state=zero, restored=False)

    We can also manually deallocate a register by specifically providing the details of the register.

    >>> qre.Deallocate(num_wires=4, state="zero", restored=False)
    Deallocate(4, state=zero, restored=False)

    .. note::

        If an ``allocated_register`` is provided along with the other parameters (``num_wires``,
        ``state``, ``restored``) and the two differ, then the details provided in the
        ``allocated_register`` will take precedence.

    If a register was allocated with ``state = "any"`` and ``restored = True``, this can
    only be deallocated by passing that specific instance of ``Allocate`` to deallocate.

    >>> temp_register = qre.Allocate(5, state="any", restored=True)
    >>> qre.Deallocate(allocated_register=temp_register)  # Restore the allocated register
    Deallocate(5, state=any, restored=True)

    """

    def __init__(
        self,
        num_wires=None,
        allocated_register=None,
        state: Literal["any", "zero"] | AllocateState = AllocateState.ZERO,
        restored=False,
    ):
        if allocated_register is not None:
            if not isinstance(allocated_register, Allocate):
                raise ValueError(
                    f"The allocated_register must be an instance of Allocate, got {allocated_register}"
                )

            state = allocated_register.state
            restored = allocated_register.restored
            num_wires = allocated_register.num_wires

        else:  # allocated_register = None
            if num_wires is None:
                raise ValueError(
                    "At least one of `num_wires` and `allocated_register` must be provided"
                )

            if state == AllocateState.ANY and restored:
                raise ValueError(
                    "Must provide the `allocated_register` when deallocating an ANY state register with `restored=True`"
                )

        if not isinstance(num_wires, int) or num_wires <= 0:
            raise ValueError(f"num_wires must be a positive integer, got {num_wires}")

        if not isinstance(restored, bool):
            raise ValueError(f"Expected restored to be True or False, got {restored}")

        self._state = AllocateState(state)
        self._restored = restored
        self._num_wires = num_wires
        self._allocated_register = allocated_register

    def equal(
        self, other: "Deallocate"
    ) -> bool:  # We avoid overriding `__eq__` due to concerns with hashing
        """Determine if two instances of the class are equal."""
        if not isinstance(other, self.__class__):
            return False

        equal_allocated_register = self.allocated_register == other.allocated_register
        if self.allocated_register is not None and other.allocated_register is not None:
            equal_allocated_register = self.allocated_register.equal(other.allocated_register)

        return all(
            (
                self.state == other.state,
                self.restored == other.restored,
                self.num_wires == other.num_wires,
                equal_allocated_register,
            )
        )

    def __repr__(self) -> str:
        return f"Deallocate({self.num_wires}, state={self.state}, restored={self.restored})"

    @property
    def state(self):
        """The quantum state of the wires to be deallocated, valid values include "zero" or "any"."""
        return self._state

    @state.setter
    def state(self, _):
        """Raise error if users attempt to change values"""
        raise AttributeError("Deallocate instances are not mutable")

    @property
    def restored(self):
        """A guarantee that the allocated register will be restored (deallocated) to its
        initial state. If True, this requirement will be enforced programmatically."""
        return self._restored

    @restored.setter
    def restored(self, _):
        """Raise error if users attempt to change values"""
        raise AttributeError("Deallocate instances are not mutable")

    @property
    def num_wires(self):
        """The number of wires to be deallocated."""
        return self._num_wires

    @num_wires.setter
    def num_wires(self, _):
        """Raise error if users attempt to change values"""
        raise AttributeError("Deallocate instances are not mutable")

    @property
    def allocated_register(self):
        """The allocated wire register the we wish to deallocate."""
        return self._allocated_register

    @allocated_register.setter
    def allocated_register(self, _):
        """Raise error if users attempt to change values"""
        raise AttributeError("Deallocate instances are not mutable")


class MarkQubits:
    r"""A base class used to mark the state of certain wire labels.

    This class can be used in quantum circuit (qfunc) to mark the state of certain algorithmic wires.
    Its primary use is to mark the state of algorithmic qubits so that they can be used by other subroutines.

    Args:
        wires (WiresLike): the label(s) of the wires to be marked

    """

    def __init__(self, wires):
        self.wires = Wires(wires) if wires is not None else Wires([])
        if QueuingManager.recording():
            self.queue()

    def queue(self, context=QueuingManager):
        r"""Adds the MarkQubit instance to the active queue."""
        context.append(self)
        return self

    def equal(
        self, other: "MarkQubits"
    ):  # We avoid overriding `__eq__` due to concerns with hashing
        """Check if two MarkQubits instances are equal."""
        return (self.__class__ == other.__class__) and (self.wires.toset() == other.wires.toset())


class MarkClean(MarkQubits):
    r"""A class used to mark that certain wires are in the zero state.

    This class can be used in quantum circuit (qfunc) to mark certain algorithmic wires as being in the zero state.
    Its primary use is to mark the state of algorithmic qubits as clean so that they can be used as auxiliary qubits
    by other subroutines.

    Args:
        wires (WiresLike): the label(s) of the wires to be marked

    **Example**

    >>> import pennylane.labs.estimator_beta as qre
    >>> qre.MarkClean(wires=[0,1,2])
    MarkClean(Wires([0, 1, 2]))

    """

    def __repr__(self) -> str:
        return f"MarkClean({self.wires})"

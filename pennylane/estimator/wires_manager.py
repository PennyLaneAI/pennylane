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
"""This module contains the base class for wire management."""


from pennylane.queuing import QueuingManager

# pylint: disable=too-few-public-methods


class WireResourceManager:
    r"""Manages and tracks the auxiliary and algorithmic wires used in a quantum circuit.

    This class provides a high-level abstraction for managing wire resources within a quantum
    circuit.
    The manager tracks the state of three distinct types of wires:

    * Algorithmic wires: The core wires used by the quantum algorithm.
    * Clean wires: Auxiliary wires that are in the :math:`|0\rangle` state. They are converted
      to dirty wires upon allocation.
    * Dirty wires: Auxiliary wires that are in an unknown state. They are converted to
      clean wires when they are freed.

    Args:
        clean (int): Number of clean work wires.
        dirty (int): Number of dirty work wires, default is ``0``.
        algo_wires (int): Number of algorithmic wires, default value is ``0``.
        tight_budget (bool): Determines whether extra clean wires can be allocated when they
            exceed the available amount. The default is ``False``.

    **Example**

    >>> q = WireResourceManager(
    ...             clean=2,
    ...             dirty=2,
    ...             tight_budget=False,
    ...     )
    >>> print(q)
    WireResourceManager(clean wires=2, dirty wires=2, algorithmic wires=0, tight budget=False)

    """

    def __init__(
        self, clean: int, dirty: int = 0, algo: int = 0, tight_budget: bool = False
    ) -> None:

        self.tight_budget = tight_budget
        self._algo_wires = algo
        self.clean_wires = clean
        self.dirty_wires = dirty

    def __str__(self) -> str:
        return (
            f"WireResourceManager(clean wires={self.clean_wires}, dirty wires={self.dirty_wires}, "
            f"algorithmic wires={self.algo_wires}, tight budget={self.tight_budget})"
        )

    def __repr__(self) -> str:
        return (
            f"WireResourceManager(clean={self.clean_wires}, dirty={self.dirty_wires}, algo={self.algo_wires}, "
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
    def algo_wires(self) -> int: 
        r"""Returns the number of algorithmic wires."""
        return self._algo_wires

    @property
    def total_wires(self) -> int:
        r"""Returns the number of total wires."""
        return self.clean_wires + self.dirty_wires + self.algo_wires

    @algo_wires.setter
    def algo_wires(self, count: int):  # these get set manually, the rest are dynamically updated
        r"""Setter for algorithmic wires."""
        self._algo_wires = count

    def grab_clean_wires(self, num_wires: int) -> None:
        r"""Grabs clean wires, and converts them into dirty ones.

        Args:
            num_wires(int) : number of clean wires to be grabbed

        Raises:
            ValueError: If tight_budget is `True` and the number of wires to be grabbed is greater than
                available clean wires.

        """
        available_clean = self.clean_wires

        if num_wires > available_clean:
            if self.tight_budget:
                raise ValueError(
                    f"Grabbing more wires than available clean wires."
                    f"Number of clean wires available is {available_clean}, while {num_wires} are being grabbed."
                )
            self.clean_wires = 0
        else:
            self.clean_wires -= num_wires
        self.dirty_wires += num_wires

    def free_wires(self, num_wires: int):
        r"""Frees dirty wires and converts them into clean wires.

        Args:
            num_wires(int) : number of wires to be freed

        Raises:
            ValueError: If number of wires to be freed is greater than available dirty wires.
        """

        if num_wires > self.dirty_wires:
            raise ValueError(
                f"Freeing more wires than available dirty wires."
                f"Number of dirty wires available is {self.dirty_wires}, while {num_wires} wires are being released."
            )

        self.dirty_wires -= num_wires
        self.clean_wires += num_wires


class _WireAction:
    """Base class for operations that manage wire resources."""

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


class Allocate(_WireAction):
    r"""Allows users to allocate work wires through :class:`~pennylane.estimator.WireResourceManager`.

    Args:
        num_wires (int): number of work wires to be allocated.


    .. details::
        :title: Usage Details

        The ``Allocate`` class is typically used within a decomposition function to track the
        allocation of auxiliary wires. This allows us to accurately determine the wire overhead of a circuit.
        In this example, we show the decomposition for a
        3-controlled X gate, which requires one work wire.

        First, we define a custom decomposition which doesn't track the extra work wire:

        >>> def resource_decomp(num_ctrl_wires=3, num_ctrl_values=0, **kwargs):
        ...     gate_list = []
        ...
        ...     gate_list.append(GateCount(resource_rep(plre.TempAND), 1))
        ...     gate_list.append(GateCount(resource_rep(plre.Adjoint, {"base_cmpr_op": resource_rep(plre.TempAND)}), 1))
        ...     gate_list.append(GateCount(resource_rep(plre.Toffoli), 1))
        ...
        ...     return gate_list
        >>> config = ResourceConfig()
        >>> config.set_decomp(plre.MultiControlledX, resource_decomp)
        >>> res = plre.estimate(plre.MultiControlledX(3, 0), config)
        >>> print(res.WireResourceManager)
        WireResourceManager(clean wires=0, dirty wires=0, algorithmic wires=4, tight budget=False)

        This decomposition uses a total of ``4`` wires and doesn't track any work wires.

        Now, if we want to track the allocation of wires using the ``Allocate``, the decomposition
        can be redefined as:

        >>> def resource_decomp():
        ...     gate_list = []
        ...     gate_list.append(Allocate(num_wires=1))
        ...
        ...     gate_list.append(GateCount(resource_rep(plre.TempAND), 1))
        ...     gate_list.append(GateCount(resource_rep(plre.Adjoint, {"base_cmpr_op": resource_rep(plre.TempAND)}), 1))
        ...     gate_list.append(GateCount(resource_rep(plre.Toffoli), 1))
        ...
        ...     gate_list.append(Deallocate(num_wires=1))
        ...     return gate_list
        >>> config = ResourceConfig()
        >>> config.set_decomp(plre.MultiControlledX, resource_decomp)
        >>> res = plre.estimate(plre.MultiControlledX(3, 0), config)
        >>> print(res.WireResourceManager)
        WireResourceManager(clean wires=1, dirty wires=0, algorithmic wires=4, tight budget=False)

        Now, the one extra auxiliary wire is being tracked.

    """

    def __repr__(self) -> str:
        return f"Allocate({self.num_wires})"


class Deallocate(_WireAction):
    r"""Allows users to free dirty work wires through :class:`~pennylane.estimator.WireResourceManager`.

    Args:
        num_wires (int): number of dirty work wires to be freed.

    .. details::
        :title: Usage Details

        The Deallocate class is typically used within a decomposition function to track the
        allocation of auxiliary wires. This allows us to accurately determine the wire overhead
        of a circuit. In this example, we show the decomposition for a
        3-controlled X gate, which requires one work wire that is returned in a clean state.

        First, we define a custom decomposition which allocates the work wire but doesn't free it.

        >>> def resource_decomp(num_ctrl_wires=3, num_ctrl_values=0, **kwargs):
        ...     gate_list = []
        ...     gate_list.append(Allocate(num_wires=1))
        ...
        ...     gate_list.append(GateCount(resource_rep(plre.TempAND), 1))
        ...     gate_list.append(GateCount(resource_rep(plre.Adjoint, {"base_cmpr_op": resource_rep(plre.TempAND)}), 1))
        ...     gate_list.append(GateCount(resource_rep(plre.Toffoli), 1))
        ...
        ...     return gate_list
        >>> config = ResourceConfig()
        >>> config.set_decomp(plre.MultiControlledX, resource_decomp)
        >>> res = plre.estimate(plre.MultiControlledX(3, 0), config)
        >>> print(res.WireResourceManager)
        WireResourceManager(clean wires=0, dirty wires=1, algorithmic wires=4, tight budget=False)

        This decomposition uses a total of ``4`` algorithmic wires and ``1`` work wire which is returned in the dirty state.

        We can free this wire using the ``Deallocate``, allowing it to be reused with more operations.
        The decomposition can be redefined as:

        >>> def resource_decomp():
        ...     gate_list = []
        ...     gate_list.append(Allocate(num_wires=1))
        ...
        ...     gate_list.append(GateCount(resource_rep(plre.TempAND), 1))
        ...     gate_list.append(GateCount(resource_rep(plre.Adjoint, {"base_cmpr_op": resource_rep(plre.TempAND)}), 1))
        ...     gate_list.append(GateCount(resource_rep(plre.Toffoli), 1))
        ...
        ...     gate_list.append(Deallocate(num_wires=1))
        ...     return gate_list
        >>> config = ResourceConfig()
        >>> config.set_decomp(plre.MultiControlledX, resource_decomp)
        >>> res = plre.estimate(plre.MultiControlledX(3, 0), config)
        >>> print(res.WireResourceManager)
        WireResourceManager(clean wires=1, dirty wires=0, algorithmic wires=4, tight budget=False)

        Now, auxiliary wire is freed and is returned in the clean state after the decomposition, and can
        be used for other operators which require auxiliary wires.

    """

    def __repr__(self) -> str:
        return f"Deallocate({self.num_wires})"

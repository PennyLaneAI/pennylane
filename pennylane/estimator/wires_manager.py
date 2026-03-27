# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY_STATE KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains the base class for wire management."""

from pennylane.queuing import QueuingManager


class WireResourceManager:
    r"""Manages and tracks the auxiliary and algorithmic wires used in a quantum circuit.

    This class provides a high-level abstraction for managing wire resources within a quantum
    circuit.
    The manager tracks the state of three distinct types of wires:

    * Zeroed state wires: Auxiliary wires that are in the :math:`|0\rangle` state. They are converted
      to an unknown state upon allocation.
    * Any state wires: Auxiliary wires that are in an unknown state. They are converted to
      zeroed wires when they are freed.
    * Algorithmic wires: The core wires used by the quantum algorithm.

    Args:
        zeroed (int): Number of zeroed state work wires.
        any_state (int): Number of work wires in an unknown state, default is ``0``.
        algo_wires (int): Number of algorithmic wires, default value is ``0``.
        tight_budget (bool): Determines whether extra zeroed state wires can be allocated when they
            exceed the available amount. The default is ``False``.

    **Example**

    >>> import pennylane.estimator as qre
    >>> q = qre.WireResourceManager(
    ...     zeroed=2,
    ...     any_state=2,
    ...     tight_budget=False,
    ... )
    >>> print(q)
    WireResourceManager(zeroed wires=2, any_state wires=2, algorithmic wires=0, tight budget=False)

    """

    def __init__(
        self, zeroed: int, any_state: int = 0, algo_wires: int = 0, tight_budget: bool = False
    ) -> None:
        self.tight_budget = tight_budget
        self._algo_wires = algo_wires
        self.zeroed = zeroed
        self.any_state = any_state

    def __str__(self) -> str:
        return (
            f"WireResourceManager(zeroed wires={self.zeroed}, any_state wires={self.any_state}, "
            f"algorithmic wires={self.algo_wires}, tight budget={self.tight_budget})"
        )

    def __repr__(self) -> str:
        return (
            f"WireResourceManager(zeroed={self.zeroed}, any_state={self.any_state}, algo_wires={self.algo_wires}, "
            f"tight_budget={self.tight_budget})"
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, self.__class__)
            and (self.zeroed == other.zeroed)
            and (self.any_state == other.any_state)
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
        return self.zeroed + self.any_state + self.algo_wires

    @algo_wires.setter
    def algo_wires(self, count: int):  # these get set manually, the rest are dynamically updated
        r"""Setter for algorithmic wires."""
        self._algo_wires = count

    def grab_zeroed(self, num_wires: int) -> None:
        r"""Grabs zeroed wires, and moves them to an arbitrary state; incrementing the number of any_state wires.

        Args:
            num_wires(int) : number of zeroed wires to be grabbed

        Raises:
            ValueError: If tight_budget is `True` and the number of wires to be grabbed is greater than
                available zeroed wires.

        """
        available_zeroed = self.zeroed

        if num_wires > available_zeroed:
            if self.tight_budget:
                raise ValueError(
                    f"Grabbing more wires than available zeroed wires. "
                    f"Number of zeroed wires available is {available_zeroed}, while {num_wires} are being grabbed."
                )
            self.zeroed = 0
        else:
            self.zeroed -= num_wires
        self.any_state += num_wires

    def free_wires(self, num_wires: int) -> None:
        r"""Frees any_state wires and converts them into zeroed wires.

        Args:
            num_wires(int) : number of wires to be freed

        Raises:
            ValueError: If number of wires to be freed is greater than available any_state wires.
        """

        if num_wires > self.any_state:
            raise ValueError(
                f"Freeing more wires than available any_state wires. "
                f"Number of any_state wires available is {self.any_state}, while {num_wires} wires are being released."
            )

        self.any_state -= num_wires
        self.zeroed += num_wires


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
    r"""Allows allocation of work wires through :class:`~pennylane.estimator.WireResourceManager`.

    Args:
        num_wires (int): number of work wires to be allocated


    .. details::
        :title: Usage Details

        The ``Allocate`` class is typically used within a decomposition function to track the
        allocation of auxiliary wires. This allows determination of a circuit's wire overhead.
        In this example, we show the decomposition for a
        3-controlled ``X`` gate, which requires one work wire.

        First, we define a custom decomposition which doesn't track the extra work wire:

        >>> import pennylane.estimator as qre
        >>> from pennylane.estimator import GateCount, resource_rep
        >>> def resource_decomp(num_ctrl_wires=3, num_zero_ctrl=0, **kwargs):
        ...     gate_list = []
        ...     gate_list.append(GateCount(resource_rep(qre.TemporaryAND), 1))
        ...     gate_list.append(GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(qre.TemporaryAND)}), 1))
        ...     gate_list.append(GateCount(resource_rep(qre.Toffoli), 1))
        ...     return gate_list
        >>> config = qre.ResourceConfig()
        >>> config.set_decomp(qre.MultiControlledX, resource_decomp)
        >>> res = qre.estimate(qre.MultiControlledX(3, 0), config=config)
        >>> print(res.algo_wires, res.zeroed_wires, res.any_state_wires)
        4 0 0

        This decomposition uses a total of ``4`` wires and doesn't track the work wires.

        Now, if we want to track the allocation of wires using ``Allocate``, the decomposition
        can be redefined as:

        >>> import pennylane.estimator as qre
        >>> from pennylane.estimator import GateCount, resource_rep
        >>> def resource_decomp(num_ctrl_wires=3, num_zero_ctrl=0, **kwargs):
        ...     gate_list = []
        ...     gate_list.append(qre.Allocate(num_wires=1))
        ...     gate_list.append(GateCount(resource_rep(qre.TemporaryAND), 1))
        ...     gate_list.append(GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(qre.TemporaryAND)}), 1))
        ...     gate_list.append(GateCount(resource_rep(qre.Toffoli), 1))
        ...     gate_list.append(qre.Deallocate(num_wires=1))
        ...     return gate_list
        >>> config = qre.ResourceConfig()
        >>> config.set_decomp(qre.MultiControlledX, resource_decomp)
        >>> res = qre.estimate(qre.MultiControlledX(3, 0), config=config)
        >>> print(res.algo_wires, res.zeroed_wires, res.any_state_wires)
        4 1 0

        Now, the one extra auxiliary wire is being tracked.

    """

    def __repr__(self) -> str:
        return f"Allocate({self.num_wires})"


class Deallocate(_WireAction):
    r"""Allows freeing ``any_state`` work wires through :class:`~pennylane.estimator.WireResourceManager`.

    Args:
        num_wires (int): number of ``any_state`` work wires to be freed.

    .. details::
        :title: Usage Details

        The ``Deallocate`` class is typically used within a decomposition function to track the
        allocation of auxiliary wires. This allows to accurately determine the wire overhead
        of a circuit. In this example, we show the decomposition for a
        3-controlled ``X`` gate, which requires one work wire that is returned in a zeroed state.

        First, we define a custom decomposition which allocates the work wire but doesn't free it.

        >>> import pennylane.estimator as qre
        >>> from pennylane.estimator import GateCount, resource_rep
        >>> def resource_decomp(num_ctrl_wires=3, num_zero_ctrl=0, **kwargs):
        ...     gate_list = []
        ...     gate_list.append(qre.Allocate(num_wires=1))
        ...     gate_list.append(GateCount(resource_rep(qre.TemporaryAND), 1))
        ...     gate_list.append(GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(qre.TemporaryAND)}), 1))
        ...     gate_list.append(GateCount(resource_rep(qre.Toffoli), 1))
        ...     return gate_list
        >>> config = qre.ResourceConfig()
        >>> config.set_decomp(qre.MultiControlledX, resource_decomp)
        >>> res = qre.estimate(qre.MultiControlledX(3, 0), config=config)
        >>> print(res.algo_wires, res.zeroed_wires, res.any_state_wires)
        4 0 1

        This decomposition uses a total of ``4`` algorithmic wires and ``1`` work wire which is returned in an arbitrary state.

        We can free this wire using ``Deallocate``, allowing it to be reused with more operations.
        The decomposition can be redefined as:

        >>> import pennylane.estimator as qre
        >>> from pennylane.estimator import GateCount, resource_rep
        >>> def resource_decomp(num_ctrl_wires=3, num_zero_ctrl=0, **kwargs):
        ...     gate_list = []
        ...     gate_list.append(qre.Allocate(num_wires=1))
        ...     gate_list.append(GateCount(resource_rep(qre.TemporaryAND), 1))
        ...     gate_list.append(GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(qre.TemporaryAND)}), 1))
        ...     gate_list.append(GateCount(resource_rep(qre.Toffoli), 1))
        ...     gate_list.append(qre.Deallocate(num_wires=1))
        ...     return gate_list
        >>> config = qre.ResourceConfig()
        >>> config.set_decomp(qre.MultiControlledX, resource_decomp)
        >>> res = qre.estimate(qre.MultiControlledX(3, 0), config=config)
        >>> print(res.algo_wires, res.zeroed_wires, res.any_state_wires)
        4 1 0

        Now, the auxiliary wire is freed, meaning that it is described as being in the zeroed state
        after the decomposition, and that it can now be used for other operators which require zeroed auxiliary wires.

    """

    def __repr__(self) -> str:
        return f"Deallocate({self.num_wires})"

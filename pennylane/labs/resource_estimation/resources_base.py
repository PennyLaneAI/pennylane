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
r"""Base class for storing resources."""
from __future__ import annotations

import copy
from collections import defaultdict
from decimal import Decimal

from pennylane.labs.resource_estimation.qubit_manager import QubitManager


class Resources:
    r"""A container to track and update the resources used throughout a quantum circuit.
    The resources tracked include number of gates, number of wires, and gate types.

    Args:
        qubit_manager (QubitManager): A qubit tracker class which contains the number of available
            work wires, categorized as clean, dirty or algorithmic wires.
        gate_types (dict): A dictionary storing operations (ResourceOperator) as keys and the number
            of times they are used in the circuit (int) as values.

    **Example**

    The resources can be accessed as class attributes. Additionally, the :class:`~.Resources`
    instance can be nicely displayed in the console.

    >>> H = re.resource_rep(re.ResourceHadamard)
    >>> X = re.resource_rep(re.ResourceX)
    >>> Y = re.resource_rep(re.ResourceY)
    >>> qm = re.QubitManager(work_wires=3)
    >>> gt = defaultdict(int, {H: 10, X:7, Y:3})
    >>>
    >>> res = re.Resources(qubit_manager=qm, gate_types=gt)
    >>> print(res)
    --- Resources: ---
    Total qubits: 3
    Total gates : 20
    Qubit breakdown:
    clean qubits: 3, dirty qubits: 0, algorithmic qubits: 0
    Gate breakdown:
    {'Hadamard': 10, 'X': 7, 'Y': 3}

    .. details::
        :title: Usage Details

        The :class:`Resources` object supports arithmetic operations which allow for quick addition
        and multiplication of resources. When combining resources, we can make a simplifying
        assumption about how they are applied in a quantum circuit (in series or in parallel).

        When assuming the circuits were executed in series, the number of algorithmic qubits add
        together. When assuming the circuits were executed in parallel, the maximum of each set of
        algorithmic qubits is used. The clean auxiliary qubits can be reused between the circuits,
        and thus we always use the maximum of each set when combining the resources. Finally, the
        dirty qubits cannot be reused between circuits, thus we always add them together.

        .. code-block::

            from collections import defaultdict

            # Resource reps for each operator:
            H = re.resource_rep(re.ResourceHadamard)
            X = re.resource_rep(re.ResourceX)
            Z = re.resource_rep(re.ResourceZ)
            CNOT = re.resource_rep(re.ResourceCNOT)

            # state of qubits:
            qm1 = re.QubitManager(work_wires={"clean":2, "dirty":1}, algo_wires=3)
            qm2 = re.QubitManager(work_wires={"clean":1, "dirty":2}, algo_wires=4)

            # state of gates:
            gt1 = defaultdict(int, {H: 10, X:5, CNOT:2})
            gt2 = defaultdict(int, {H: 15, Z:5, CNOT:4})

            # resources:
            res1 = re.Resources(qubit_manager=qm1, gate_types=gt1)
            res2 = re.Resources(qubit_manager=qm2, gate_types=gt2)


        .. code-block:: pycon

            >>> print(res1)
            --- Resources: ---
            Total qubits: 6
            Total gates : 17
            Qubit breakdown:
            clean qubits: 2, dirty qubits: 1, algorithmic qubits: 3
            Gate breakdown:
            {'Hadamard': 10, 'X': 5, 'CNOT': 2}

            >>> print(res2)
            --- Resources: ---
            Total qubits: 7
            Total gates : 24
            Qubit breakdown:
            clean qubits: 1, dirty qubits: 2, algorithmic qubits: 4
            Gate breakdown:
            {'Hadamard': 15, 'Z': 5, 'CNOT': 4}

        Specifically, users can add together two instances of resources using the :code:`+` and
        :code:`&` operators. These represent combining the resources assuming the circuits were
        executed in series or parallel respectively.

        .. code-block:: pycon

            >>> res_in_series = res1 + res2
            >>> print(res_in_series)
            --- Resources: ---
            Total qubits: 9
            Total gates : 41
            Qubit breakdown:
            clean qubits: 2, dirty qubits: 3, algorithmic qubits: 4
            Gate breakdown:
            {'Hadamard': 25, 'X': 5, 'CNOT': 6, 'Z': 5}

            >>> res_in_parallel = res1 & res2
            >>> print(res_in_parallel)
            --- Resources: ---
            Total qubits: 12
            Total gates : 41
            Qubit breakdown:
            clean qubits: 2, dirty qubits: 3, algorithmic qubits: 7
            Gate breakdown:
            {'Hadamard': 25, 'X': 5, 'CNOT': 6, 'Z': 5}

        Similarly, users can scale up the resources for an operator by some integer factor using
        the :code:`*` and :code:`@` operators. These represent scaling the resources assuming the
        circuits were executed in series or parallel respectively.

        .. code-block:: pycon

            >>> res_in_series = 5 * res1
            >>> print(res_in_series)
            --- Resources: ---
            Total qubits: 10
            Total gates : 85
            Qubit breakdown:
            clean qubits: 2, dirty qubits: 5, algorithmic qubits: 3
            Gate breakdown:
            {'Hadamard': 50, 'X': 25, 'CNOT': 10}

            >>> res_in_parallel = 5 @ res1
            >>> print(res_in_parallel)
            --- Resources: ---
            Total qubits: 22
            Total gates : 85
            Qubit breakdown:
            clean qubits: 2, dirty qubits: 5, algorithmic qubits: 15
            Gate breakdown:
            {'Hadamard': 50, 'X': 25, 'CNOT': 10}

    """

    def __init__(self, qubit_manager: QubitManager, gate_types: dict = None):
        """Initialize the Resources class."""
        gate_types = gate_types or {}

        self.qubit_manager = qubit_manager
        self.gate_types = (
            gate_types
            if (isinstance(gate_types, defaultdict) and isinstance(gate_types.default_factory, int))
            else defaultdict(int, gate_types)
        )

    def __add__(self, other: Resources) -> Resources:
        """Add two resources objects in series"""
        assert isinstance(other, self.__class__)
        return add_in_series(self, other)

    def __and__(self, other: Resources) -> Resources:
        """Add two resources objects in parallel"""
        assert isinstance(other, self.__class__)
        return add_in_parallel(self, other)

    def __eq__(self, other: Resources) -> bool:
        """Determine if two resources objects are equal"""
        return (self.gate_types == other.gate_types) and (self.qubit_manager == other.qubit_manager)

    def __mul__(self, scalar: int) -> Resources:
        """Scale a resources object in series"""
        assert isinstance(scalar, int)
        return mul_in_series(self, scalar)

    def __matmul__(self, scalar: int) -> Resources:
        """Scale a resources object in parallel"""
        assert isinstance(scalar, int)
        return mul_in_parallel(self, scalar)

    __rmul__ = __mul__
    __radd__ = __add__
    __rand__ = __and__
    __rmatmul__ = __matmul__

    @property
    def clean_gate_counts(self):
        r"""Produce a dictionary which stores the gate counts
        using the operator names as keys.

        Returns:
            dict: A dictionary with operator names (str) as keys
                and the number of occurances in the circuit (int) as values.
        """
        clean_gate_counts = defaultdict(int)

        for cmp_res_op, counts in self.gate_types.items():
            clean_gate_counts[cmp_res_op.name] += counts

        return clean_gate_counts

    def __str__(self):
        """Generates a string representation of the Resources object."""
        qm = self.qubit_manager
        total_qubits = qm.total_qubits
        total_gates = sum(self.clean_gate_counts.values())

        total_gates_str = str(total_gates) if total_gates <= 999 else f"{Decimal(total_gates):.3E}"
        total_qubits_str = (
            str(total_qubits) if total_qubits <= 9999 else f"{Decimal(total_qubits):.3E}"
        )

        items = "--- Resources: ---\n"
        items += f" Total qubits: {total_qubits_str}\n"
        items += f" Total gates : {total_gates_str}\n"

        qubit_breakdown_str = f"clean qubits: {qm.clean_qubits}, dirty qubits: {qm.dirty_qubits}, algorithmic qubits: {qm.algo_qubits}"
        items += f" Qubit breakdown:\n  {qubit_breakdown_str}\n"

        gate_type_str = ", ".join(
            [
                f"'{gate_name}': {Decimal(count):.3E}" if count > 999 else f"'{gate_name}': {count}"
                for gate_name, count in self.clean_gate_counts.items()
            ]
        )
        items += " Gate breakdown:\n  {" + gate_type_str + "}"

        return items

    def __repr__(self):
        """Compact string representation of the Resources object"""
        return {
            "qubit_manager": self.qubit_manager,
            "gate_types": self.gate_types,
        }.__repr__()

    def _ipython_display_(self):  # pragma: no cover
        """Displays __str__ in ipython instead of __repr__"""
        print(str(self))


def add_in_series(first: Resources, other) -> Resources:  # +
    r"""Add two resources assuming the circuits are executed in series.

    Args:
        first (Resources): first resource object to combine
        other (Resources): other resource object to combine with

    Returns:
        Resources: combined resources
    """
    qm1, qm2 = (first.qubit_manager, other.qubit_manager)

    new_clean = max(qm1.clean_qubits, qm2.clean_qubits)
    new_dirty = qm1.dirty_qubits + qm2.dirty_qubits
    new_budget = qm1.tight_budget or qm2.tight_budget
    new_logic = max(qm1.algo_qubits, qm2.algo_qubits)

    new_qubit_manager = QubitManager(
        work_wires={"clean": new_clean, "dirty": new_dirty}, tight_budget=new_budget
    )

    new_qubit_manager.algo_qubits = new_logic
    new_gate_types = _combine_dict(first.gate_types, other.gate_types)
    return Resources(new_qubit_manager, new_gate_types)


def add_in_parallel(first: Resources, other) -> Resources:  # &
    r"""Add two resources assuming the circuits are executed in parallel.

    Args:
        first (Resources): first resource object to combine
        other (Resources): other resource object to combine with

    Returns:
        Resources: combined resources
    """
    qm1, qm2 = (first.qubit_manager, other.qubit_manager)

    new_clean = max(qm1.clean_qubits, qm2.clean_qubits)
    new_dirty = qm1.dirty_qubits + qm2.dirty_qubits
    new_budget = qm1.tight_budget or qm2.tight_budget
    new_logic = qm1.algo_qubits + qm2.algo_qubits

    new_qubit_manager = QubitManager(
        work_wires={"clean": new_clean, "dirty": new_dirty},
        tight_budget=new_budget,
    )

    new_qubit_manager.algo_qubits = new_logic
    new_gate_types = _combine_dict(first.gate_types, other.gate_types)
    return Resources(new_qubit_manager, new_gate_types)


def mul_in_series(first: Resources, scalar: int) -> Resources:  # *
    r"""Multiply the resources by a scalar assuming the circuits are executed in series.

    Args:
        first (Resources): first resource object to scale
        scalar (int): integer value to scale the resources by

    Returns:
        Resources: scaled resources
    """
    qm = first.qubit_manager

    new_clean = qm.clean_qubits
    new_dirty = scalar * qm.dirty_qubits
    new_budget = qm.tight_budget
    new_logic = qm.algo_qubits

    new_qubit_manager = QubitManager(
        work_wires={"clean": new_clean, "dirty": new_dirty},
        tight_budget=new_budget,
    )

    new_qubit_manager.algo_qubits = new_logic
    new_gate_types = _scale_dict(first.gate_types, scalar)

    return Resources(new_qubit_manager, new_gate_types)


def mul_in_parallel(first: Resources, scalar: int) -> Resources:  # @
    r"""Multiply the resources by a scalar assuming the circuits are executed in parallel.

    Args:
        first (Resources): first resource object to scale
        scalar (int): integer value to scale the resources by

    Returns:
        Resources: scaled resources
    """
    qm = first.qubit_manager

    new_clean = qm.clean_qubits
    new_dirty = scalar * qm.dirty_qubits
    new_budget = qm.tight_budget
    new_logic = scalar * qm.algo_qubits

    new_qubit_manager = QubitManager(
        work_wires={"clean": new_clean, "dirty": new_dirty},
        tight_budget=new_budget,
    )

    new_qubit_manager.algo_qubits = new_logic
    new_gate_types = _scale_dict(first.gate_types, scalar)

    return Resources(new_qubit_manager, new_gate_types)


def _combine_dict(dict1: defaultdict, dict2: defaultdict) -> defaultdict:
    r"""Private function which combines two dictionaries together."""
    combined_dict = copy.copy(dict1)

    for k, v in dict2.items():
        combined_dict[k] += v

    return combined_dict


def _scale_dict(dict1: defaultdict, scalar: int) -> defaultdict:
    r"""Private function which scales the values in a dictionary."""
    scaled_dict = copy.copy(dict1)

    for k in scaled_dict:
        scaled_dict[k] *= scalar

    return scaled_dict

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
Contains the ``Permute`` template.
"""

import pennylane as qml

from pennylane.templates.decorator import template
from pennylane.wires import Wires


@template
def Permute(permutation, wires):
    r"""Applies a permutation to a set of wires.

    Args:
        permutation (list): A list of wire labels that represents the new ordering of wires
            after the permutation. The list may consist of integers or strings, so long as
            they match the labels of ``wires``.
        wires (Iterable or Wires): Wires that the permutation acts on. Accepts an iterable
            of numbers or strings, or a Wires object.

    Raises:
        ValueError: if inputs do not have the correct format

    **Example**

    .. code-block:: python

        import pennylane as qml

        dev = qml.device('default.qubit', wires=5)

        @qml.qnode(dev)
        def apply_perm():
            # Send contents of wire 4 to wire 0, of wire 2 to wire 1, etc.
            qml.templates.Permute([4, 2, 0, 1, 3], wires=dev.wires)
            return qml.expval(qml.PauliZ(0))

    See "Usage Details" for further examples.

    .. UsageDetails::

        As a simple example, suppose we have a 4-qubit device with wires labeled
        by the integers ``[0, 1, 2, 3]``. We apply a permutation to shuffle the
        order to ``[3, 2, 0, 1]`` (i.e., the qubit state that was previously on
        wire 3 is now on wire 0, the one from 2 is on wire 1, etc.).

        .. code-block:: python

            dev = qml.device('default.qubit', wires=4)

            @qml.qnode(dev)
            def apply_perm():
                qml.Hadamard(wires=0)
                qml.Hadamard(wires=2)
                qml.templates.Permute([3, 2, 0, 1], dev.wires)
                return qml.expval(qml.PauliZ(0))

        >>> apply_perm()
        >>> print(apply_perm.draw())
        0: ──H─────────╭SWAP─────────┤ ⟨Z⟩
        1: ─────╭SWAP──│─────────────┤
        2: ──H──╰SWAP──│──────╭SWAP──┤
        3: ────────────╰SWAP──╰SWAP──┤

        ``Permute`` can also be used with quantum tapes. For example, suppose we
        have a tape with 5 wires ``[0, 1, 2, 3, 4]``, and we'd like to reorder them
        so that wire 4 is moved to the location of wire 0, wire 2 is moved to the
        original location of wire 1, and so on.

        .. code-block:: python

            import pennylane as qml
            from pennylane.templates import Permute

            with qml.tape.QuantumTape() as tape:
                # RZs added to ensure numerical ordering in drawing
                for wire in range(5):
                    qml.RZ(0, wires=wire)
                Permute([4, 2, 0, 1, 3], wires=[0, 1, 2, 3, 4])


        >>> print(tape.draw())
        0: ──RZ(0)─────────╭SWAP────────────────┤
        1: ──RZ(0)──╭SWAP──│────────────────────┤
        2: ──RZ(0)──╰SWAP──│──────╭SWAP─────────┤
        3: ──RZ(0)─────────│──────│──────╭SWAP──┤
        4: ──RZ(0)─────────╰SWAP──╰SWAP──╰SWAP──┤

        ``Permute`` can also be applied to wires with arbitrary labels, like so:

        .. code-block:: python

            wire_labels = [3, 2, "a", 0, "c"]

            with qml.tape.QuantumTape() as tape:
                # RZs added to ensure numerical ordering in drawing
                for wire in range(5):
                    qml.RZ(0, wires=wire)
                Permute(["c", 3, "a", 2, 0], wires=wire_labels)

        The permuted circuit is:

        >>> print(tape.draw())
        3: ──RZ(0)──╭SWAP────────────────┤
        2: ──RZ(0)──│──────╭SWAP─────────┤
        a: ──RZ(0)──│──────│─────────────┤
        0: ──RZ(0)──│──────│──────╭SWAP──┤
        c: ──RZ(0)──╰SWAP──╰SWAP──╰SWAP──┤

        It is also possible to permute a subset of wires by
        specifying a subset of labels. For example,

        .. code-block:: python

           wire_labels = [3, 2, "a", 0, "c"]

            with qml.tape.QuantumTape() as tape:
                # Create 5 wires
                for wire in range(num_wires):
                    qml.RZ(0, wires=wire)

                # Only permute the order of 3 of them
                Permute(["c", 2, 0], wires=[2, 0, "c"])

        will permute only the second, third, and fifth wires as follows:

        >>> print(tape.draw())
        3: ──RZ(0)────────────────┤
        2: ──RZ(0)──╭SWAP─────────┤
        a: ──RZ(0)──│─────────────┤
        0: ──RZ(0)──│──────╭SWAP──┤
        c: ──RZ(0)──╰SWAP──╰SWAP──┤

    """

    wires = Wires(wires)

    if len(permutation) <= 1 or len(wires) <= 1:
        raise ValueError("Permutations must involve at least 2 qubits.")

    # Make sure the lengths of permutation and wires are the same
    if len(permutation) != len(wires):
        raise ValueError("Permutation must specify outcome of all wires.")

    # Permutation order must contain all unique values
    if len(set(permutation)) != len(permutation):
        raise ValueError("Values in a permutation must all be unique.")

    # Make sure everything in the permutation has an associated label in wires
    for label in permutation:
        if label not in wires.labels:
            raise ValueError(f"Cannot permute wire {label} not present in wire set.")

    # Temporary storage to keep track as we permute
    working_order = list(wires.labels)

    # Go through the new order and shuffle things one by one
    for idx_here, here in enumerate(permutation):
        if working_order[idx_here] != here:
            # Where do we need to send the qubit at this location?
            idx_there = working_order.index(permutation[idx_here])

            # SWAP based on the labels of the wires
            qml.SWAP(wires=[wires.labels[idx_here], wires.labels[idx_there]])

            # Update the working order to account for the SWAP
            working_order[idx_here], working_order[idx_there] = (
                working_order[idx_there],
                working_order[idx_here],
            )

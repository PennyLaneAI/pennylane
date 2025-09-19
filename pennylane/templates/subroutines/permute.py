# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
Contains the Permute template.
"""
import copy
from collections import Counter

from pennylane import capture
from pennylane.decomposition import add_decomps, register_resources, resource_rep
from pennylane.operation import Operation
from pennylane.ops import SWAP
from pennylane.wires import Wires

has_jax = True
try:
    from jax import numpy as jnp
except (ModuleNotFoundError, ImportError) as import_error:  # pragma: no cover
    has_jax = False  # pragma: no cover


class Permute(Operation):
    r"""Applies a permutation to a set of wires.

    Args:
        permutation (Sequence): A list of wire labels that represents the new ordering of wires
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
            return qml.expval(qml.Z(0))

    See "Usage Details" for further examples.

    .. details::
        :title: Usage Details

        As a simple example, suppose we have a 4-qubit device with wires labeled
        by the integers ``[0, 1, 2, 3]``. We apply a permutation to shuffle the
        order to ``[3, 2, 0, 1]`` (i.e., the qubit state that was previously on
        wire 3 is now on wire 0, the one from 2 is on wire 1, etc.).

        .. code-block:: python

            dev = qml.device('default.qubit', wires=4)

            @qml.qnode(dev)
            def apply_perm():
                qml.Permute([3, 2, 0, 1], dev.wires)
                return qml.expval(qml.Z(0))

        >>> print(qml.draw(apply_perm, level="device")())
        0: ─╭SWAP─────────────┤  <Z>
        1: ─│─────╭SWAP───────┤
        2: ─│─────╰SWAP─╭SWAP─┤
        3: ─╰SWAP───────╰SWAP─┤

        ``Permute`` can also be used with quantum tapes. For example, suppose we
        have a tape with 5 wires ``[0, 1, 2, 3, 4]``, and we'd like to reorder them
        so that wire 4 is moved to the location of wire 0, wire 2 is moved to the
        original location of wire 1, and so on.

        .. code-block:: python

            import pennylane as qml

            op = qml.Permute([4, 2, 0, 1, 3], wires=[0, 1, 2, 3, 4])
            tape = qml.tape.QuantumTape([op])

        >>> [tape_expanded], _ = qml.transforms.decompose(tape, gate_set={qml.SWAP})
        >>> print(qml.drawer.tape_text(tape_expanded, wire_order=range(5)))
        0: ─╭SWAP───────────────────┤
        1: ─│─────╭SWAP─────────────┤
        2: ─│─────╰SWAP─╭SWAP───────┤
        3: ─│───────────│─────╭SWAP─┤
        4: ─╰SWAP───────╰SWAP─╰SWAP─┤

        ``Permute`` can also be applied to wires with arbitrary labels, like so:

        .. code-block:: python

            wire_labels = [3, 2, "a", 0, "c"]

            dev = qml.device('default.qubit', wires=wire_labels)

            @qml.qnode(dev)
            def circuit():
                qml.Permute(["c", 3,"a",2,0], wires=wire_labels)
                return qml.expval(qml.Z("c"))

        The permuted circuit is:

        >>> print(qml.draw(circuit, level="device")())
        3: ─╭SWAP─────────────┤
        2: ─│─────╭SWAP───────┤
        0: ─│─────│─────╭SWAP─┤
        c: ─╰SWAP─╰SWAP─╰SWAP─┤  <Z>

        It is also possible to permute a subset of wires by
        specifying a subset of labels. For example,

        .. code-block:: python

            wire_labels = [3, 2, "a", 0, "c"]

            dev = qml.device('default.qubit', wires=wire_labels)

            @qml.qnode(dev)
            def circuit():
                # Only permute the order of 3 of them
                qml.Permute(["c", 2, 0], wires=[2, 0, "c"])
                return qml.expval(qml.Z("c"))

        will permute only the second, third, and fifth wires as follows:

        >>> print(qml.draw(circuit, level="device", show_all_wires=True)())
        3: ─────────────┤
        2: ─╭SWAP───────┤
        a: ─│───────────┤
        0: ─│─────╭SWAP─┤
        c: ─╰SWAP─╰SWAP─┤  <Z>

    """

    def __repr__(self):
        return f"Permute({self.hyperparameters['permutation']}, wires={self.wires.tolist()})"

    grad_method = None

    resource_keys = {"wires", "permutation"}

    def __init__(self, permutation, wires, id=None):
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
            if label not in wires:
                raise ValueError(f"Cannot permute wire {label} not present in wire set.")

        self._hyperparameters = {"permutation": tuple(permutation)}
        super().__init__(wires=wires, id=id)

    def map_wires(self, wire_map: dict):
        # pylint: disable=protected-access
        new_op = copy.deepcopy(self)
        new_op._wires = Wires([wire_map.get(wire, wire) for wire in self.wires])
        new_op._hyperparameters["permutation"] = tuple(
            wire_map.get(w, w) for w in new_op._hyperparameters["permutation"]
        )
        return new_op

    @property
    def resource_params(self) -> dict:
        return {
            "permutation": self.hyperparameters["permutation"],
            "wires": self.wires,
        }

    @property
    def num_params(self):
        return 0

    @staticmethod
    def compute_decomposition(wires, permutation):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n.



        .. seealso:: :meth:`~.Permute.decomposition`.

        Args:
            wires (Any or Iterable[Any]): wires that the operator acts on
            permutation (list[Any]): A list of wire labels that represents the new ordering of wires
                after the permutation.

        Returns:
            list[.Operator]: decomposition of the operator
        """
        op_list = []

        # Temporary storage to keep track as we permute
        working_order = wires.tolist()

        # Go through the new order and shuffle things one by one
        for idx_here, here in enumerate(permutation):
            if working_order[idx_here] != here:
                # Where do we need to send the qubit at this location?
                idx_there = working_order.index(permutation[idx_here])

                # SWAP based on the labels of the wires
                op_list.append(SWAP(wires=wires.subset([idx_here, idx_there])))

                # Update the working order to account for the SWAP
                working_order[idx_here], working_order[idx_there] = (
                    working_order[idx_there],
                    working_order[idx_here],
                )
        return op_list


def _permute_resources(wires, permutation):
    resources = Counter()
    working_order = wires.tolist()

    for idx_here, here in enumerate(permutation):
        if working_order[idx_here] != here:
            idx_there = working_order.index(permutation[idx_here])

            resources[resource_rep(SWAP)] += 1

            working_order[idx_here], working_order[idx_there] = (
                working_order[idx_there],
                working_order[idx_here],
            )

    return dict(resources)


@register_resources(_permute_resources)
def _permute_decomposition(wires, permutation):
    # Temporary storage to keep track as we permute
    working_order = wires.tolist()

    if has_jax and capture.enabled():
        working_order = jnp.array(working_order)
        permutation = jnp.array(permutation)

    # Go through the new order and shuffle things one by one
    for idx_here, here in enumerate(permutation):
        if working_order[idx_here] != here:
            # Where do we need to send the qubit at this location?
            idx_there = (
                working_order.index(permutation[idx_here])
                if not (has_jax and capture.enabled())
                else jnp.where(working_order == permutation[idx_here])[0][0]
            )

            # SWAP based on the labels of the wires
            if has_jax and capture.enabled():
                SWAP(wires=[wires[idx_here], wires[idx_there]])
            else:
                SWAP(wires=wires.subset([idx_here, idx_there]))

            # Update the working order to account for the SWAP
            if has_jax and capture.enabled():
                temp = working_order[idx_here]
                working_order = working_order.at[idx_here].set(working_order[idx_there])
                working_order = working_order.at[idx_there].set(temp)
            else:
                working_order[idx_here], working_order[idx_there] = (
                    working_order[idx_there],
                    working_order[idx_here],
                )


add_decomps(Permute, _permute_decomposition)

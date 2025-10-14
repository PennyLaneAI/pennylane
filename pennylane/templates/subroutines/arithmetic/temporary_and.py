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
"""
Contains the TemporaryAND template, which also is known as Elbow.
"""

from functools import lru_cache

from pennylane import math
from pennylane.decomposition import add_decomps, adjoint_resource_rep, register_resources
from pennylane.ops import CNOT, Hadamard, Operation, S, T, X, adjoint
from pennylane.wires import Wires, WiresLike


class TemporaryAND(Operation):
    r"""TemporaryAND(wires, control_values)

    The ``TemporaryAND`` operation is a three-qubit gate equivalent to an ``AND``, or reversible :class:`~pennylane.Toffoli`, gate that leverages extra information
    about the target wire to enable more efficient circuit decompositions. The ``TemporaryAND`` assumes the target qubit
    to be initialized in :math:`|0\rangle`, while the ``Adjoint(TemporaryAND)`` assumes the target output to be :math:`|0\rangle`.
    For more details, see Fig. 4 in `arXiv:1805.03662 <https://arxiv.org/abs/1805.03662>`_.

    .. note::

        For correct usage of this operation, the user must ensure
        that before computation the input state of the target wire is :math:`|0\rangle`,
        and that after uncomputation the output state of the target wire is :math:`|0\rangle`,
        when using ``TemporaryAND`` or ``Adjoint(TemporaryAND)``, respectively.
        Otherwise, behaviour may differ from the expected ``AND``.

    **Details:**

    * Number of wires: 3
    * Number of parameters: 0

    Args:
        wires (Sequence[int]): the subsystem the gate acts on. The first two wires are the control wires and the
            third one is the target wire.
        control_values (tuple[bool or int]): The values on the control wires for which
            the target operator is applied. Integers other than 0 or 1 will be treated as ``int(bool(x))``.
            Default is ``(1,1)``, corresponding to a traditional ``AND`` gate.


    .. seealso:: The alias :class:`~Elbow`.

    **Example**

    .. code-block:: python

        @qml.set_shots(1)
        @qml.qnode(qml.device("default.qubit"))
        def circuit():
            # |0000⟩
            qml.X(0) # |1000⟩
            qml.X(1) # |1100⟩
            # The target wire is in state |0>, so we can apply TemporaryAND
            qml.TemporaryAND([0,1,2]) # |1110⟩
            qml.CNOT([2,3]) # |1111⟩
            # The target wire will be in state |0> after adjoint(TemporaryAND) gate is applied,
            # so we can apply adjoint(TemporaryAND)
            qml.adjoint(qml.TemporaryAND([0,1,2])) # |1101⟩
            return qml.sample(wires=[0,1,2,3])

    >>> print(qml.draw(circuit)())
    0: ──X─╭●─────●╮─┤ ╭Sample
    1: ──X─├●─────●┤─┤ ├Sample
    2: ────╰⊕─╭●──⊕╯─┤ ├Sample
    3: ───────╰X─────┤ ╰Sample
    >>> print(circuit())
    [[1 1 0 1]]
    """

    num_wires = 3
    """int: Number of wires that the operator acts on."""

    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = ()
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    resource_keys = set()

    def __repr__(self):
        cvals = self.hyperparameters["control_values"]
        if all(cvals):
            return f"TemporaryAND(wires={self.wires})"
        return f"TemporaryAND(wires={self.wires}, control_values={cvals})"

    def __init__(self, wires: WiresLike, control_values=(1, 1), id=None):
        wires = Wires(wires)
        self.hyperparameters["control_values"] = tuple(control_values)
        super().__init__(wires=wires, id=id)

    @property
    def resource_params(self) -> dict:
        return {}

    def _flatten(self):
        return tuple(), (self.wires, self.hyperparameters["control_values"])

    @classmethod
    def _unflatten(cls, _, metadata):
        return cls(wires=metadata[0], control_values=metadata[1])

    @staticmethod
    @lru_cache
    def compute_matrix(**kwargs):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        Returns:
            array_like: matrix

        **Example**

        >>> print(qml.TemporaryAND.compute_matrix(control_values = (1,1)))
        [[ 1.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j -0.-1.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  1.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j -0.-1.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  1.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+1.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j -0.-1.j]
         [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  1.+0.j  0.+0.j]]
        """

        control_values = kwargs["control_values"]

        mask = 0

        if control_values[0] == 0:
            mask ^= 4

        if control_values[1] == 0:
            mask ^= 2

        result_matrix = math.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, -1j, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, -1j, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1j, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, -1j],
                [0, 0, 0, 0, 0, 0, 1, 0],
            ],
            dtype=complex,
        )

        perm = math.arange(8) ^ mask
        result_matrix = result_matrix[perm][:, perm]

        return result_matrix

    @staticmethod
    def compute_decomposition(
        wires: WiresLike, control_values=(1, 1)
    ):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.TemporaryAND.decomposition`.

        Args:
            wires (Sequence[int]): the subsystem the gate acts on. The first two wires are the control wires and the
                third one is the target wire.
            control_values (bool or int or list[bool or int]): The value(s) the control wire(s)
                should take. Integers other than 0 or 1 will be treated as ``int(bool(x))``.

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> print(qml.TemporaryAND.compute_decomposition((0,1,2)))
        [H(2),
        T(2),
        CNOT(wires=[1, 2]),
        Adjoint(T(2)),
        CNOT(wires=[0, 2]),
        T(2),
        CNOT(wires=[1, 2]),
        Adjoint(T(2)),
        H(2),
        Adjoint(S(2))]
        """

        list_decomp = []

        list_decomp.extend([X(wires[idx]) for idx in [0, 1] if control_values[idx] == 0])

        list_decomp += [
            Hadamard(wires=wires[2]),
            T(wires=wires[2]),
            CNOT(wires=[wires[1], wires[2]]),
            adjoint(T(wires=wires[2])),
            CNOT(wires=[wires[0], wires[2]]),
            T(wires=wires[2]),
            CNOT(wires=[wires[1], wires[2]]),
            adjoint(T(wires=wires[2])),
            Hadamard(wires=wires[2]),
            adjoint(S(wires=wires[2])),
        ]

        list_decomp.extend([X(wires[idx]) for idx in [0, 1] if control_values[idx] == 0])

        return list_decomp


def _temporary_and_resources():
    number_xs = 4  # worst case scenario
    return {
        X: number_xs,
        Hadamard: 2,
        CNOT: 3,
        T: 2,
        adjoint_resource_rep(T, {}): 2,
        adjoint_resource_rep(S, {}): 1,
    }


@register_resources(_temporary_and_resources)
def _temporary_and(wires: WiresLike, **kwargs):
    control_values = kwargs["control_values"]
    if control_values[0] == 0:
        X(wires[0])

    if control_values[1] == 0:
        X(wires[1])

    Hadamard(wires=wires[2])
    T(wires=wires[2])
    CNOT(wires=[wires[1], wires[2]])
    adjoint(T(wires=wires[2]))
    CNOT(wires=[wires[0], wires[2]])
    T(wires=wires[2])
    CNOT(wires=[wires[1], wires[2]])
    adjoint(T(wires=wires[2]))
    Hadamard(wires=wires[2])
    adjoint(S(wires=wires[2]))

    if control_values[0] == 0:
        X(wires[0])

    if control_values[1] == 0:
        X(wires[1])


add_decomps(TemporaryAND, _temporary_and)
# TODO: add add_decomps("Adjoint(TemporaryAND)", _adjoint_TemporaryAND) when MCMs supported by the pipeline

Elbow = TemporaryAND
r"""Elbow(wire, control_values)
The Elbow, or :class:`~TemporaryAND` operator.

.. seealso:: The alias :class:`~TemporaryAND` for more details.

**Details:**

* Number of wires: 3

Args:
    wires (Sequence[int] or int): the subsystem the gate acts on.
        The first two wires are the control wires and the third one is the target wire.
    control_values (tuple[bool or int]): The values on the control wires for which
        the target operator is applied. Integers other than 0 or 1 will be treated as ``int(bool(x))``.
        Default is ``(1,1)``, corresponding to a traditional ``AND`` gate.
"""

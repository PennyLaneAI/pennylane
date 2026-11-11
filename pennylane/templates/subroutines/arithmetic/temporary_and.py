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

from typing import override

from pennylane import math
from pennylane.core.operator import Operator2
from pennylane.typing import AbstractArray, Bool, Wire
from pennylane.wires import WiresLike


class TemporaryAND(Operator2):
    r"""TemporaryAND(wires, control_values)

    The ``TemporaryAND`` operation is a three-qubit gate equivalent to a reversible ``AND``,
    or :class:`~pennylane.Toffoli`, gate that leverages extra information about the target
    wire to enable more efficient circuit decompositions. ``TemporaryAND`` assumes the target qubit
    to be in the state :math:`|0\rangle`, while ``Adjoint(TemporaryAND)`` assumes the target output
    to be :math:`|0\rangle`. For more details, see Fig. 4
    in `arXiv:1805.03662 <https://arxiv.org/abs/1805.03662>`_.

    .. note::

        For correct usage of this operation, the user must ensure
        that before using ``TemporaryAND`` the input state of the target wire is :math:`|0\rangle`,
        and that after uncomputation, i.e., after using ``Adjoint(TemporaryAND)``, the output
        state of the target wire is :math:`|0\rangle`,
        when using ``TemporaryAND`` or ``Adjoint(TemporaryAND)``, respectively.
        Otherwise, behaviour may differ from the expected ``AND``.

    **Details:**

    * Number of wires: 3
    * Number of parameters: 0

    Args:
        wires (Sequence[int]): the subsystem the gate acts on. The first two wires are the
            control wires and the third one is the target wire.
        control_values (tuple[bool or int]): The values on the control wires for which the target
            operator is applied. Integers other than 0 or 1 will be treated as ``int(bool(x))``.
            Default is ``(1, 1)``, corresponding to a traditional ``AND`` gate.


    .. seealso:: The alias :class:`~Elbow`.

    **Example**

    .. code-block:: python

        import pennylane as qp

        @qp.set_shots(1)
        @qp.qnode(qp.device("default.qubit"))
        def circuit():
            # |0000⟩
            qp.X(0) # |1000⟩
            qp.X(1) # |1100⟩
            # The target wire is in state |0>, so we can apply TemporaryAND
            qp.TemporaryAND([0,1,2]) # |1110⟩
            qp.CNOT([2,3]) # |1111⟩
            # The target wire will be in state |0> after adjoint(TemporaryAND) gate is applied,
            # so we can apply adjoint(TemporaryAND)
            qp.adjoint(qp.TemporaryAND([0,1,2])) # |1101⟩
            return qp.sample(wires=[0,1,2,3])

    >>> print(qp.draw(circuit)())
    0: ──X─╭●─────●╮─┤ ╭Sample
    1: ──X─├●─────●┤─┤ ├Sample
    2: ────╰⊕─╭●──⊕╯─┤ ├Sample
    3: ───────╰X─────┤ ╰Sample
    >>> print(circuit())
    [[1 1 0 1]]

    There is also a decomposition of ``TemporaryAND`` into a standard ``Toffoli`` gate, in order
    to provide a compilation path into gate sets like Clifford + Toffoli:

    .. code-block:: python

        import pennylane as qp

        qp.decomposition.enable_graph()

        @qp.transforms._tape_decompose(gate_set={qp.Toffoli, qp.X})
        def circuit():
            qp.TemporaryAND((0, 1, 2))
            return qp.expval(qp.Z(2))

    >>> print(qp.draw(circuit)())
    0: ─╭●─┤
    1: ─├●─┤
    2: ─╰X─┤  <Z>

    Note that we had to add ``qp.X`` to the gate set passed to ``decompose``, because the
    decomposition may contain bit flips on the control qubits, depending on potentially dynamic
    control values.
    """

    num_wires = 3
    """int: Number of wires that the operator acts on."""

    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    # ndim_params must align with the shape of the dynamic args, but num_params is still 0 because
    # control_values isn't actually meant to be trainable.
    ndim_params = (1,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    dynamic_argnames = ("control_values",)

    arg_specs = {"wires": Wire[3], "control_values": Bool[2]}

    def __init__(self, wires: WiresLike, control_values=None):
        if control_values is None:
            control_values = math.ones(2, dtype=bool)
        interface = math.get_deep_interface(control_values)
        if not isinstance(control_values, AbstractArray):
            control_values = math.cast(math.asarray(control_values, like=interface), bool)
        super().__init__(wires=wires, control_values=control_values)

    def __repr__(self):
        params = [f"wires={self.wires}"]
        ctrl_values = self.control_values
        if isinstance(ctrl_values, AbstractArray) or math.is_tracer(ctrl_values):
            params.append(f"control_values={ctrl_values}")
        elif not all(ctrl_values):
            params.append(f"control_values={ctrl_values.tolist()}")
        return f"TemporaryAND({", ".join(params)})"

    @staticmethod
    @override
    def compute_matrix(wires, control_values):  # pylint: disable=arguments-differ,unused-argument
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        Returns:
            array_like: matrix

        **Example**

        >>> print(qp.TemporaryAND.compute_matrix([0, 1, 2], control_values = (1, 1)))
        [[ 1.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j -0.-1.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  1.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j -0.-1.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  1.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+1.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j -0.-1.j]
         [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  1.+0.j  0.+0.j]]
        """

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

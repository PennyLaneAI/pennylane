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
Contains the TemporaryAnd template.
"""

from functools import lru_cache

import pennylane as qml
from pennylane.decomposition import add_decomps, register_resources
from pennylane.operation import Operation
from pennylane.wires import WiresLike, Wires


class TemporaryAnd(Operation):
    r"""TemporaryAnd(wires)

    The TemporaryAnd operator is a three-qubit gate equivalent to an ``AND``, or reversible :class:`~pennylane.Toffoli`, gate that leverages extra information
    about the target wire to enable more efficient circuit decompositions. The ``TemporaryAnd`` assumes the target qubit
    to be initialized in :math:`|0\rangle`, while the ``Adjoint(TemporaryAnd)`` assumes the target output to be :math:`|0\rangle`.
    For more details, see `Ryan Babbush et al.(2018), Fig 4 <https://arxiv.org/abs/1805.03662>`_.

    .. note::

        For correct usage of the operator, the user must ensure
        that the input before computation is :math:`|0\rangle`,
        and that the output after uncomputation is :math:`|0\rangle`.
        on the target wire when using ``TemporaryAnd`` or ``Adjoint(TemporaryAnd)``, respectively. Otherwise, the behavior could be
        different from the expected ``AND``.

    **Details:**

    * Number of wires: 3
    * Number of parameters: 0

    Args:
        wires (Sequence[int]): the subsystem the gate acts on. The first two wires are the control wires and the
            third one is the target wire.
        control_values (bool or int or list[bool or int]): The value(s) the control wire(s)
            should take. Integers other than 0 or 1 will be treated as ``int(bool(x))``.


    **Example**

    .. code-block::

        dev = qml.device("default.qubit", shots=1)
        @qml.qnode(dev)
        def circuit():
            qml.X(0)
            qml.X(1)
            qml.TemporaryAnd([0,1,2])
            qml.CNOT([2,3])
            qml.adjoint(qml.TemporaryAnd([0,1,2])) # We can apply the adjoint TemporaryAnd because after applying a Toffoli,
                                                   # the target wire would be |0>.

            return qml.sample(wires=[0,1,2,3])

    .. code-block:: pycon

        >>> print(circuit())
        [1 1 0 1]
    """

    num_wires = 3
    """int: Number of wires that the operator acts on."""

    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = ()
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    resource_keys = set(["control_values"])

    def __init__(self, wires: WiresLike, control_values=(1, 1), id=None):
        wires = Wires(wires)
        self.hyperparameters["control_values"] = tuple(control_values)
        super().__init__(wires=wires, id=id)

    @property
    def resource_params(self) -> dict:
        return {"control_values": self.hyperparameters["control_values"]}

    def _flatten(self):
        return tuple(), (self.wires, self.hyperparameters["control_values"])

    @classmethod
    def _unflatten(cls, _, metadata):
        return cls(wires=metadata[0], control_values=metadata[1])

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)

    @staticmethod
    @lru_cache()
    def compute_matrix(**kwargs):
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        Returns:
            array_like: matrix

        **Example**

        >>> print(qml.TemporaryAnd.compute_matrix())
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
        first_qubit = [qml.X(0), qml.I(0)]
        second_qubit = [qml.X(1), qml.I(1)]

        X_matrix = qml.matrix(
            first_qubit[control_values[0]] @ second_qubit[control_values[1]] @ qml.I(2)
        )
        basis_matrix = qml.math.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, -1j, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, -1j, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1j, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, -1j],
                [0, 0, 0, 0, 0, 0, 1, 0],
            ]
        )

        return X_matrix @ basis_matrix @ X_matrix

    @staticmethod
    def compute_decomposition(wires: WiresLike, control_values):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.TemporaryAnd.decomposition`.

        Args:
            wires (Sequence[int]): the subsystem the gate acts on. The first two wires are the control wires and the
                third one is the target wire.
            control_values (bool or int or list[bool or int]): The value(s) the control wire(s)
                should take. Integers other than 0 or 1 will be treated as ``int(bool(x))``.

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.TemporaryAnd.compute_decomposition((0,1,2))
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

        if control_values[0] == 0:
            list_decomp.append(qml.X(wires[0]))

        if control_values[1] == 0:
            list_decomp.append(qml.X(wires[1]))

        list_decomp += [
            qml.Hadamard(wires=wires[2]),
            qml.T(wires=wires[2]),
            qml.CNOT(wires=[wires[1], wires[2]]),
            qml.adjoint(qml.T(wires=wires[2])),
            qml.CNOT(wires=[wires[0], wires[2]]),
            qml.T(wires=wires[2]),
            qml.CNOT(wires=[wires[1], wires[2]]),
            qml.adjoint(qml.T(wires=wires[2])),
            qml.Hadamard(wires=wires[2]),
            qml.adjoint(qml.S(wires=wires[2])),
        ]

        if control_values[0] == 0:
            list_decomp.append(qml.X(wires[0]))

        if control_values[1] == 0:
            list_decomp.append(qml.X(wires[1]))

        return list_decomp


def _temporary_and_resources(control_values):
    number_xs = 4 - 2 * sum(control_values)
    return {
        qml.X: number_xs,
        qml.Hadamard: 2,
        qml.CNOT: 3,
        qml.T: 2,
        qml.decomposition.adjoint_resource_rep(qml.T, {}): 2,
        qml.decomposition.adjoint_resource_rep(qml.S, {}): 1,
    }


@register_resources(_temporary_and_resources)
def _temporary_and(wires: WiresLike, control_values):
    if control_values[0] == 0:
        qml.X(wires[0])

    if control_values[1] == 0:
        qml.X(wires[1])

    qml.Hadamard(wires=wires[2])
    qml.T(wires=wires[2])
    qml.CNOT(wires=[wires[1], wires[2]])
    qml.adjoint(qml.T(wires=wires[2]))
    qml.CNOT(wires=[wires[0], wires[2]])
    qml.T(wires=wires[2])
    qml.CNOT(wires=[wires[1], wires[2]])
    qml.adjoint(qml.T(wires=wires[2]))
    qml.Hadamard(wires=wires[2])
    qml.adjoint(qml.S(wires=wires[2]))

    if control_values[0] == 0:
        qml.X(wires[0])

    if control_values[1] == 0:
        qml.X(wires[1])


add_decomps(TemporaryAnd, _temporary_and)
# TODO: add add_decomps("Adjoint(TemporaryAnd)", _adjoint_TemporaryAnd) when MCMs supported by the pipeline

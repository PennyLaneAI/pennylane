# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
Contains the QSVT template and qsvt wrapper function.
"""
# pylint: disable=too-many-arguments
import pennylane as qml
from pennylane.queuing import QueuingManager
from pennylane.ops import BlockEncode, PCPhase
from pennylane.ops.op_math import adjoint
from pennylane.operation import AnyWires, Operation


def qsvt(A, angles, wires):
    r"""Performs the
    `quantum singular value transformation <https://arxiv.org/abs/1806.01838>`__ using
    :class:`~.BlockEncode` and :class:`~.PCPhase`.

    Args:
        A (tensor_like): the general :math:`(n \times m)` matrix to be encoded.
        angles (tensor_like): a list of angles by which to shift to obtain the desired polynomial.
        wires (Iterable): the wires the template acts on.

    .. details::
        :title: Usage Details

        .. code-block:: python

            dev = qml.device("default.qubit", wires=2)
            A = [[0.1, 0.2], [0.3, 0.4]]
            angles = [0.1, 0.2, 0.3]

            @qml.qnode(dev)
            def example_circuit(A):
                qml.qsvt(A, angles, wires=[0, 1])
                return qml.expval(qml.PauliZ(wires=0))

            print(qml.draw(example_circuit)(A))

            0: ─╭QSVT─┤  <Z>
            1: ─╰QSVT─┤

            print(qml.draw(example_circuit, expansion_strategy="device")(A))

            0: ─╭BlockEncode(M0)─╭∏_ϕ(0.10)─╭BlockEncode(M0)†─╭∏_ϕ(0.20)─╭BlockEncode(M0)─╭∏_ϕ(0.30)─┤  <Z>
            1: ─╰BlockEncode(M0)─╰∏_ϕ(0.10)─╰BlockEncode(M0)†─╰∏_ϕ(0.20)─╰BlockEncode(M0)─╰∏_ϕ(0.30)─┤
    """
    if qml.math.shape(A) == () or qml.math.shape(A) == (1,):
        A = qml.math.reshape(A, [1, 1])

    c, r = qml.math.shape(A)

    UA = BlockEncode(A, wires=wires, do_queue=False)
    projectors = []
    for idx, phi in enumerate(angles):
        if idx % 2 == 0:
            projectors.append(PCPhase(phi, dim=r, wires=wires, do_queue=False))
        else:
            projectors.append(PCPhase(phi, dim=c, wires=wires, do_queue=False))

    return QSVT(UA, projectors, wires=wires)


class QSVT(Operation):
    r"""Implements the
    `quantum singular value transformation <https://arxiv.org/abs/1806.01838>`__ circuit.

    Given a circuit :math:`U(A)`, which block encodes the matrix :math:`A`, and a list of projector-controlled
    phase shifts, this template applies the circuit for quantum singular value transformation.

    .. math::

        \begin{align}
             U_{QSVT}(A, \phi) &=
             \begin{bmatrix}
                Poly^{SV}(A) & \cdot \\
                \cdot & \cdot
            \end{bmatrix}.
        \end{align}

    This circuit can be used to perform the standard quantum singular value transformation algorithm, consisting
    of alternating block encoding and controlled phase shift operations.

    Args:
        UA (Operator): the block encoding circuit, specified as an :class:`~.Operator`
            or quantum function.
        projectors (Sequence[Operator]): a list of projector-controlled phase
            shifts that implement the desired polynomial.
        wires (Iterable): the wires the template acts on.

    Raises:
        ValueError: if the input block encoding is not an operator.

    .. details::
        :title: Usage Details

        .. code-block:: python

            dev = qml.device("default.qubit", wires=2)
            A = [[0.1]]
            blckencode = qml.BlockEncode(A, wires=[0, 1])
            lst_phis = [qml.PCPhase(i + 0.1, dim=1, wires=[0, 1]) for i in range(3)]

            @qml.qnode(dev)
            def example_circuit(A):
                qml.QSVT(blckencode, lst_phis, wires=[0, 1])
                return qml.expval(qml.PauliZ(wires=0))

            print(qml.matrix(example_circuit)(A))

            [[-0.11-0.01j -0.58+0.8j   0.  +0.j    0.  +0.j  ]
             [ 0.45+0.89j -0.11-0.01j  0.  +0.j    0.  +0.j  ]
             [ 0.  +0.j    0.  +0.j    1.  +0.j    0.  +0.j  ]
             [ 0.  +0.j    0.  +0.j    0.  +0.j    1.  +0.j  ]]
    """

    num_wires = AnyWires
    """int: Number of wires that the operator acts on."""

    grad_method = None
    """Gradient computation method."""

    def __init__(self, UA, projectors, wires, do_queue=True, id=None):
        if not isinstance(UA, qml.operation.Operator):
            raise ValueError("Input block encoding must be an Operator")

        self._hyperparameters = {
            "UA": UA,
            "projectors": projectors,
        }

        super().__init__(wires=wires, do_queue=do_queue, id=id)

    @staticmethod
    def compute_decomposition(
        UA, projectors, wires, **hyperparameters
    ):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators.

        The :class:`~.QSVT` is decomposed into alternating block encoding
        and projector-controlled phase shift operators. This is defined by the following
        equations, where :math:`U` is the block encoding operation and both :math:`\Pi_\phi` and
        :math:`\tilde{\Pi}_\phi` are projector-controlled phase shifts with angle :math:`\phi`.
        When the number of projector-controlled phase shifts, :math:`d`, is odd:

        .. math:: U_{QSVT} = \tilde{\Pi}_{\phi_1}U\prod^{(d-1)/2}_{k=1}{\Pi}_{\phi_{2k}}U^{\dagger}\tilde{\Pi}_{\phi_{2k+1}}U,

        and when :math:`d` is even:

        .. math:: U_{QSVT} = \prod^{d/2}_{k=1}{\Pi}_{\phi_{2k-1}}U^{\dagger}\tilde{\Pi}_{\phi_{2k}}U.

        .. seealso:: :meth:`~.QSVT.decomposition`.

        Args:
            UA (Operator): the block encoding circuit, specified as a :class:`~.Operator`
                or quantum function.
            projectors (list[Operator]): a list of projector-controlled phase
                shift circuits that implement the desired polynomial.
            wires (Iterable): wires that the template acts on

        Returns:
            list[.Operator]: decomposition of the operator
        """

        op_list = []
        UA_adj = UA.__copy__()

        for idx, op in enumerate(projectors):
            if idx % 2 == 0:
                qml.apply(UA)
                op_list.append(UA)
            else:
                op_list.append(adjoint(UA_adj))
            qml.apply(op)
            op_list.append(op)
        return op_list

    def label(self, decimals=None, base_label=None, cache=None):
        op_label = base_label or self.__class__.__name__
        return op_label

    def queue(self, context=QueuingManager):
        context.remove(self._hyperparameters["UA"])
        for op in self._hyperparameters["projectors"]:
            context.remove(op)
        context.append(self)
        return self

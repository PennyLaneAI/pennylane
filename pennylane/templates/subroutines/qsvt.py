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
import numpy as np
import pennylane as qml
from pennylane.queuing import QueuingManager
from pennylane.ops import BlockEncode, PCPhase
from pennylane.ops.op_math import adjoint
from pennylane.operation import AnyWires, Operation


def qsvt(A, phi_vect, wires):
    """Performs the
    `quantum singular value transformation <https://arxiv.org/abs/1806.01838>`__ circuit using
    :class:`~.BlockEncode` and :class:`~.PCPhase`.

    Args:
        A (Array): the general :math:`(n \times m)` matrix to be encoded.
        phi_vect (list[Float]): a list of angles by which to shift to obtain the desired polynomial.
        wires (Iterable): the wires the template acts on.

    .. details::
        :title: Usage Details

        .. code-block:: python

            dev = qml.device('default.qubit', wires=2)
            A = [[0.1, 0.2], [0.3, 0.4]]
            lst_phis = [0.1, 0.2, 0.3]
            @qml.qnode(dev)
            def example_circuit(A):
                qml.qsvt(A, lst_phis, wires=[0,1])
                return qml.expval(qml.PauliZ(wires=0))

            qml.draw(example_circuit)(A)
    """
    c, r = qml.math.shape(A)

    U_A = qml.BlockEncode(A, wires=wires, do_queue=False)
    lst_projectors = []
    for idx, phi in enumerate(phi_vect):
        if idx % 2 == 0:
            lst_projectors.append(PCPhase(phi, dim=r, wires=wires, do_queue=False))
        else:
            lst_projectors.append(PCPhase(phi, dim=c, wires=wires, do_queue=False))

    return QSVT(U_A, lst_projectors, wires=wires)


class QSVT(Operation):
    r"""Performs the 
    `quantum singular value transformation <https://arxiv.org/abs/1806.01838>`__ circuit.

    Given a circuit :math:`U(A)`, which block encodes the matrix :math:`A`, and a list of projector-controlled
    phase shifts, this template applies the circuit for quantum singular value transformation.

    .. math::

        \begin{align}
             U_{qsvt}(A, \phi) &=
             \begin{bmatrix}
                Poly^{SV}(A) & \cdot \\
                \cdot & \cdot
            \end{bmatrix}.
        \end{align}

    This circuit can be used to perform the standard quantum singular value transformation algorithm, consisting
    of alternating block encoding and controlled phase shift operations.

    Args:
        U_A (Operator): the block encoding circuit, specified as a :class:`~.Operator` 
            or quantum function.
        lst_projectors (list[Operator] or list[Callable]): a list of projector-controlled phase
            shifts that implement the desired polynomial.
        wires (Iterable): the wires the template acts on.

    Raises:
        QuantumFunctionError: 

    .. details::
        :title: Usage Details

        Consider the matrix corresponding to a rotation from an :class:`~.RX` gate:

        .. code-block:: python

            dev = qml.device('default.qubit', wires=2)
            A = [[0.1]]
            blckencode = qml.BlockEncode(A,wires=[0,1])
            lst_phis = [qml.PCPhase(i+0.1,dim=1, wires=[0,1]) for i in range(3)]
            @qml.qnode(dev)
            def example_circuit(A):
                qml.QSVT(blckencode,lst_phis,wires=[0,1])
                return qml.expval(qml.PauliZ(wires=0))
                
            qml.matrix(example_circuit)(A)

            
        Continue any explanation here.
    """

    num_wires = AnyWires
    """int: Number of wires that the operator acts on."""

    # ndim_params = (0, 1,)
    # """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    grad_method = None
    """Gradient computation method."""

    def __init__(self, U_A, lst_projectors, wires, do_queue=True, id=None):
        if not isinstance(U_A, qml.operation.Operator):
            raise ValueError("Input block encoding must be an Operator")

        self._hyperparameters = {
            "U_A": U_A,
            "lst_projectors": lst_projectors,
        }

        super().__init__(wires=wires, do_queue=do_queue, id=id)

    @staticmethod
    def compute_decomposition(U_A, lst_projectors, wires, **hyperparameters):
        r"""Representation of the operator as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.QSVT.decomposition`.

        Args:
            U_A (Operator or Callable): the block encoding circuit, specified as a :class:`~.Operator`
                or quantum function.
            lst_projectors (list[Operator] or list[Callable]): a list of projector-controlled phase
                shift circuits that implement the desired polynomial.
            wires (Iterable): wires that the template acts on

        Returns:
            list[.Operator]: decomposition of the operator
        """

        op_list = []
        U_Aadj = U_A.__copy__()
        if len(lst_projectors) % 2 == 0:
            for idx, op in enumerate(lst_projectors):
                if idx % 2 == 0:
                    qml.apply(U_A) if idx > 0 else U_A.queue()
                    op_list.append(U_A)
                else:
                    op_list.append(adjoint(U_Aadj))
                qml.apply(op)
                op_list.append(op)

        else:
            for idx, op in enumerate(lst_projectors[:-1]):
                if idx % 2 == 0:
                    qml.apply(U_A) if idx > 0 else U_A.queue()
                    op_list.append(U_A)
                else:
                    op_list.append(adjoint(U_Aadj))
                op.queue()
                op_list.append(op)

            qml.apply(U_A)
            lst_projectors[-1].queue()
            op_list.append(U_A)
            op_list.append(lst_projectors[-1])

        return op_list

    def label(self, decimals=None, base_label=None, cache=None):
        op_label = base_label or self.__class__.__name__
        return op_label

    def queue(self, context=QueuingManager):
        context.remove(self._hyperparameters["U_A"])
        for op in self._hyperparameters["lst_projectors"]:
            context.remove(op)
        context.append(self)
        return self

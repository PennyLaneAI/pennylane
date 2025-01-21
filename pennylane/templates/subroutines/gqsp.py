# Copyright 2024 Xanadu Quantum Technologies Inc.

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
Contains the GQSP template.
"""
# pylint: disable=too-many-arguments

import copy

import pennylane as qml
from pennylane.operation import Operation
from pennylane.queuing import QueuingManager
from pennylane.wires import Wires


class GQSP(Operation):
    r"""
    Implements the generalized quantum signal processing (GQSP) circuit.

    This operation encodes a polynomial transformation of an input unitary operator following the algorithm
    described in `arXiv:2308.01501 <https://arxiv.org/abs/2308.01501>`__ as:

    .. math::
         U
         \xrightarrow{GQSP}
         \begin{pmatrix}
         \text{poly}(U) & * \\
         * & * \\
         \end{pmatrix}

    The implementation requires one control qubit.

    Args:

        unitary (Operator): the operator to be encoded by the GQSP circuit
        angles (tensor[float]): array of angles defining the polynomial transformation. The shape of the array must be `(3, d+1)`, where `d` is the degree of the polynomial.
        control (Union[Wires, int, str]): control qubit used to encode the polynomial transformation

    .. note::

       The  :func:`~.poly_to_angles` function can be used to calculate the angles for a given polynomial.

    Example:

    .. code-block:: python

        # P(x) = 0.1 + 0.2j x + 0.3 x^2
        poly = [0.1, 0.2j, 0.3]

        angles = qml.poly_to_angles(poly, "GQSP")

        @qml.prod # transforms the qfunc into an Operator
        def unitary(wires):
            qml.RX(0.3, wires)

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit(angles):
            qml.GQSP(unitary(wires = 1), angles, control = 0)
            return qml.state()

        matrix = qml.matrix(circuit, wire_order=[0, 1])(angles)

    .. code-block:: pycon

        >>> print(np.round(matrix,3)[:2, :2])
        [[0.387+0.198j 0.03 -0.089j]
        [0.03 -0.089j 0.387+0.198j]]
    """

    grad_method = None

    def __init__(self, unitary, angles, control, id=None):
        total_wires = qml.wires.Wires(control) + unitary.wires

        self._hyperparameters = {"unitary": unitary, "control": qml.wires.Wires(control)}

        super().__init__(angles, *unitary.data, wires=total_wires, id=id)

    def _flatten(self):
        data = self.parameters
        return data, (
            self.hyperparameters["unitary"],
            self.hyperparameters["control"],
        )

    @classmethod
    def _unflatten(cls, data, metadata):
        return cls(unitary=metadata[0], angles=data[0], control=metadata[1])

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)

    def map_wires(self, wire_map: dict):
        # pylint: disable=protected-access
        new_op = copy.deepcopy(self)
        new_op._wires = Wires([wire_map.get(wire, wire) for wire in self.wires])
        new_op._hyperparameters["unitary"] = qml.map_wires(
            new_op._hyperparameters["unitary"], wire_map
        )
        new_op._hyperparameters["control"] = tuple(
            wire_map.get(w, w) for w in new_op._hyperparameters["control"]
        )

        return new_op

    @staticmethod
    def compute_decomposition(*parameters, **hyperparameters):  # pylint: disable=arguments-differ
        r"""
        Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.

        .. seealso:: :meth:`~.Operator.decomposition`.

        Args:
            *parameters (list): trainable parameters of the operator, as stored in the ``parameters`` attribute
            wires (Iterable[Any], Wires): wires that the operator acts on
            **hyperparams (dict): non-trainable hyperparameters of the operator, as stored in the ``hyperparameters`` attribute

        Returns:
            list[Operator]: decomposition of the operator
        """

        unitary = hyperparameters["unitary"]
        control = hyperparameters["control"]

        angles = parameters[0]

        thetas, phis, lambds = angles[0], angles[1], angles[2]

        op_list = []

        # These four gates adapt PennyLane's qml.U3 to the chosen U3 format in the GQSP paper.
        op_list.append(qml.X(control))
        op_list.append(qml.U3(2 * thetas[0], phis[0], lambds[0], wires=control))
        op_list.append(qml.X(control))
        op_list.append(qml.Z(control))

        for theta, phi, lamb in zip(thetas[1:], phis[1:], lambds[1:]):

            op_list.append(qml.ctrl(unitary, control=control, control_values=0))

            op_list.append(qml.X(control))
            op_list.append(qml.U3(2 * theta, phi, lamb, wires=control))
            op_list.append(qml.X(control))
            op_list.append(qml.Z(control))

        return op_list

    def queue(self, context=QueuingManager):
        context.remove(self.hyperparameters["unitary"])
        context.append(self)
        return self

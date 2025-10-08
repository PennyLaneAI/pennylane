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
r"""Resource operators for QubitUnitary operation."""
import pennylane.estimator as qre
from pennylane.estimator.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    resource_rep,
)
from pennylane.estimator.templates import SelectPauliRot
from pennylane.wires import WiresLike

# pylint: disable=arguments-differ


class QubitUnitary(ResourceOperator):
    r"""Resource class for the QubitUnitary template.

    Args:
        num_wires (int | None): the number of qubits the operation acts upon
        precision (Union[float, None], optional): The precision used when preparing the single qubit
            rotations used to synthesize the n-qubit unitary.
        wires (Sequence[int], None): the wires the operation acts on

    Resources:
        The resources are defined by combining the two equalities in `Möttönen and Vartiainen
        (2005), Fig 14 <https://arxiv.org/pdf/quant-ph/0504100>`_ , we can express an :math:`n`
        qubit unitary as four :math:`n - 1` qubit unitaries and three multiplexed rotations
        via (:class:`~.pennylane.estimator.templates.subroutines.SelectPauliRot`). Specifically, the cost
        is given by:

        * 1-qubit unitary, the cost is approximated as a single :code:`RZ` rotation.

        * 2-qubit unitary, the cost is approximated as four single qubit rotations and three :code:`CNOT` gates.

        * 3-qubit unitary or more, the cost is given according to the reference above, recursively.

    .. seealso:: The associated PennyLane operation :class:`~.pennylane.QubitUnitary`.

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.estimator as qre
    >>> qu = qre.QubitUnitary(num_wires=3)
    >>> gate_set =["RZ", "RY", "CNOT"]
    >>> print(qre.estimate(qu, gate_set))
    --- Resources: ---
     Total wires: 3
        algorithmic wires: 3
        allocated wires: 0
             zero state: 0
             any state: 0
     Total gates : 52
      'RZ': 24,
      'RY': 4,
      'CNOT': 24
    """

    resource_keys = {"num_wires", "precision"}

    def __init__(
        self, num_wires: int | None = None, precision: float | None = None, wires: WiresLike = None
    ):
        if num_wires is None:
            if wires is None:
                raise ValueError("Must provide atleast one of `num_wires` and `wires`.")
            num_wires = len(wires)
        self.num_wires = num_wires
        self.precision = precision
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * num_wires (int): the number of qubits the operation acts upon
                * precision (Union[float, None], optional): The precision used when preparing the
                  single qubit rotations used to synthesize the n-qubit unitary.
        """
        return {"num_wires": self.num_wires, "precision": self.precision}

    @classmethod
    def resource_rep(cls, num_wires, precision=None) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Args:
            num_wires (int): the number of qubits the operation acts upon
            precision (Union[float, None], optional): The precision used when preparing the single
                qubit rotations used to synthesize the n-qubit unitary.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        params = {"num_wires": num_wires, "precision": precision}
        return CompressedResourceOp(cls, num_wires, params)

    @classmethod
    def resource_decomp(cls, num_wires, precision=None) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            num_wires (int): the number of qubits the operation acts upon
            precision (Union[float, None], optional): The precision used when preparing the single
                qubit rotations used to synthesize the n-qubit unitary.

        Resources:
            The resources are defined by combining the two equalities in `Möttönen and Vartiainen
            (2005), Fig 14 <https://arxiv.org/pdf/quant-ph/0504100>`_, we can express an :math:`n`-
            qubit unitary as four :math:`n - 1`-qubit unitaries and three multiplexed rotations
            via (:class:`~.pennylane.estimator.templates.subroutines.SelectPauliRot`). Specifically, the cost
            is given by:

            * 1-qubit unitary, the cost is approximated as a single :code:`RZ` rotation.

            * 2-qubit unitary, the cost is approximated as four single qubit rotations and three :code:`CNOT` gates.

            * 3-qubit unitary or more, the cost is given according to the reference above, recursively.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        gate_lst = []

        one_qubit_decomp_cost = [GateCount(resource_rep(qre.RZ, {"precision": precision}))]
        two_qubit_decomp_cost = [
            GateCount(resource_rep(qre.RZ, {"precision": precision}), 4),
            GateCount(resource_rep(qre.CNOT), 3),
        ]

        if num_wires == 1:
            return one_qubit_decomp_cost

        if num_wires == 2:
            return two_qubit_decomp_cost

        for gc in two_qubit_decomp_cost:
            gate_lst.append(4 ** (num_wires - 2) * gc)

        for index in range(2, num_wires):
            multiplex_z = resource_rep(
                SelectPauliRot,
                {
                    "num_ctrl_wires": index,
                    "rot_axis": "Z",
                    "precision": precision,
                },
            )
            multiplex_y = resource_rep(
                SelectPauliRot,
                {
                    "num_ctrl_wires": index,
                    "rot_axis": "Y",
                    "precision": precision,
                },
            )

            gate_lst.append(GateCount(multiplex_z, 2 * 4 ** (num_wires - (1 + index))))
            gate_lst.append(GateCount(multiplex_y, 4 ** (num_wires - (1 + index))))

        return gate_lst

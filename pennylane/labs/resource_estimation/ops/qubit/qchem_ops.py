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
r"""Resource operators for qchem operations."""
import pennylane.labs.resource_estimation as re
from pennylane.labs.resource_estimation.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    resource_rep,
)

# pylint: disable=arguments-differ


class ResourceSingleExcitation(ResourceOperator):
    r"""Resource class for the SingleExcitation gate.

    Args:
        precision (float, optional): error threshold for clifford plus T decomposition of this operation
        wires (Sequence[int], optional): the wires the operation acts on

    Resources:
        The resources are obtained by decomposing the following matrix into fundamental gates.

        .. math:: U(\phi) = \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & \cos(\phi/2) & -\sin(\phi/2) & 0 \\
                    0 & \sin(\phi/2) & \cos(\phi/2) & 0 \\
                    0 & 0 & 0 & 1
                \end{bmatrix}.

        The cost for implementing this transformation is given by:

        .. code-block:: bash

            0: ──T†──H───S─╭X──RZ-─╭X──S†──H──T─┤
            1: ──T†──S†──H─╰●──RY──╰●──H───S──T─┤

    .. seealso:: :class:`~.SingleExcitation`

    **Example**

    The resources for this operation are computed using:

    >>> se = plre.ResourceSingleExcitation()
    >>> print(plre.estimate(se, plre.StandardGateSet))
    --- Resources: ---
    Total qubits: 2
    Total gates : 16
    Qubit breakdown:
     clean qubits: 0, dirty qubits: 0, algorithmic qubits: 2
    Gate breakdown:
     {'Adjoint(T)': 2, 'Hadamard': 4, 'S': 2, 'Adjoint(S)': 2, 'CNOT': 2, 'RZ': 1, 'RY': 1, 'T': 2}

    """

    num_wires = 2
    resource_keys = {"precision"}

    def __init__(self, precision=None, wires=None) -> None:
        self.precision = precision
        super().__init__(wires=wires)

    @classmethod
    def resource_decomp(cls, precision=None, **kwargs):
        r"""Returns a list of GateCount objects representing the operator's resources.

        Args:
            precision (float, optional): error threshold for clifford plus T decomposition of this operation

        Resources:
            The resources are obtained by decomposing the following matrix into fundamental gates.

            .. math:: U(\phi) = \begin{bmatrix}
                        1 & 0 & 0 & 0 \\
                        0 & \cos(\phi/2) & -\sin(\phi/2) & 0 \\
                        0 & \sin(\phi/2) & \cos(\phi/2) & 0 \\
                        0 & 0 & 0 & 1
                    \end{bmatrix}.

            The cost for implementing this transformation is given by:

            .. code-block:: bash

                0: ──T†──H───S─╭X──RZ-─╭X──S†──H──T─┤
                1: ──T†──S†──H─╰●──RY──╰●──H───S──T─┤

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        h = resource_rep(re.ResourceHadamard)
        s = resource_rep(re.ResourceS)
        s_dag = resource_rep(re.ResourceAdjoint, {"base_cmpr_op": s})
        cnot = resource_rep(re.ResourceCNOT)
        rz = resource_rep(re.ResourceRZ, {"precision": precision})
        ry = resource_rep(re.ResourceRY, {"precision": precision})
        t = resource_rep(re.ResourceT)
        t_dag = resource_rep(re.ResourceAdjoint, {"base_cmpr_op": t})

        gate_types = [
            GateCount(t_dag, 2),
            GateCount(h, 4),
            GateCount(s, 2),
            GateCount(s_dag, 2),
            GateCount(cnot, 2),
            GateCount(rz),
            GateCount(ry),
            GateCount(t, 2),
        ]
        return gate_types

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * precision (float): error threshold for clifford plus T decomposition of this operation
        """
        return {"precision": self.precision}

    @classmethod
    def resource_rep(cls, precision=None):
        """Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            precision (float, optional): error threshold for clifford plus T decomposition of this operation

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        return CompressedResourceOp(cls, cls.num_wires, {"precision": precision})

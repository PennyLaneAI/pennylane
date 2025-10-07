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
import pennylane.estimator as qre
from pennylane.estimator.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    resource_rep,
)
from pennylane.wires import Wires, WiresLike

# pylint: disable=arguments-differ


class SingleExcitation(ResourceOperator):
    r"""Resource class for the SingleExcitation gate.

    Args:
        precision (float, optional): error threshold for Clifford + T decomposition of this operation
        wires (Sequence[int], optional): the wires the operation acts on

    Resources:
        The resources are obtained by decomposing the following matrix into fundamental gates.

        .. math:: U(\phi) = \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & \cos(\phi/2) & -\sin(\phi/2) & 0 \\
                    0 & \sin(\phi/2) & \cos(\phi/2) & 0 \\
                    0 & 0 & 0 & 1
                \end{bmatrix}.

        This transformation can be expressed with the following decomposition:

        .. code-block:: bash

            0: ──T†──H───S─╭X──RZ-─╭X──S†──H──T─┤
            1: ──T†──S†──H─╰●──RY──╰●──H───S──T─┤

    .. seealso:: The corresponding PennyLane operation :class:`~.pennylane.SingleExcitation`.

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.estimator as qre
    >>> se = qre.SingleExcitation()
    >>> print(qre.estimate(se))
    --- Resources: ---
     Total wires: 2
        algorithmic wires: 2
        allocated wires: 0
             zero state: 0
             any state: 0
     Total gates : 108
      'T': 92,
      'CNOT': 2,
      'Z': 4,
      'S': 6,
      'Hadamard': 4
    """

    num_wires = 2
    resource_keys = {"precision"}

    def __init__(self, precision: float | None = None, wires: WiresLike = None) -> None:
        if wires is not None and len(Wires(wires)) != self.num_wires:
            raise ValueError(f"Expected {self.num_wires} wires, got {len(Wires(wires))}")
        self.precision = precision
        super().__init__(wires=wires)

    @classmethod
    def resource_decomp(cls, precision=None) -> list[GateCount]:
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
            list[`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        h = resource_rep(qre.Hadamard)
        s = resource_rep(qre.S)
        s_dag = resource_rep(qre.Adjoint, {"base_cmpr_op": s})
        cnot = resource_rep(qre.CNOT)
        rz = resource_rep(qre.RZ, {"precision": precision})
        ry = resource_rep(qre.RY, {"precision": precision})
        t = resource_rep(qre.T)
        t_dag = resource_rep(qre.Adjoint, {"base_cmpr_op": t})

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
    def resource_rep(cls, precision: float | None = None) -> CompressedResourceOp:
        """Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            precision (float, optional): error threshold for clifford plus T decomposition of this operation

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        return CompressedResourceOp(cls, cls.num_wires, {"precision": precision})

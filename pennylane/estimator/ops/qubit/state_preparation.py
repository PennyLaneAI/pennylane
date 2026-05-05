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
r"""Resource operators for state preparation templates."""

import pennylane.estimator as qre
from pennylane.estimator.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    resource_rep,
)

# pylint: disable= arguments-differ


class BasisState(ResourceOperator):
    r"""Resource class for preparing a single basis state.

    Args:
        num_wires (int): the number of wires the operator acts on
        wires (WiresLike, Optional): the wire(s) the operation acts on
    """

    resource_keys = {"num_wires"}

    def __init__(self, num_wires, wires=None):
        if wires and len(wires) != num_wires:
            raise ValueError(f"Expected {num_wires} wires, got {len(wires)}.")
        self.num_wires = num_wires
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * num_wires (int): number of wires the operator acts on
        """
        return {"num_wires": self.num_wires}

    @classmethod
    def resource_rep(cls, num_wires: int) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        return CompressedResourceOp(cls, num_wires, {"num_wires": num_wires})

    @classmethod
    def resource_decomp(cls, num_wires: int) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object in the list represents a gate and the
        number of times it occurs in the circuit.

        Args:
            num_wires (int): the number of wires the operator acts on

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of
            ``GateCount`` objects, where each object represents a specific quantum gate and the
            number of times it appears in the decomposition.
        """
        return [
            GateCount(resource_rep(qre.X), num_wires / 2),
        ]

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
import math

from pennylane import numpy as qnp
from pennylane.labs import resource_estimation as plre
from pennylane.labs.resource_estimation.qubit_manager import AllocWires, FreeWires
from pennylane.labs.resource_estimation.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    resource_rep,
)
from pennylane.queuing import QueuingManager
from pennylane.wires import Wires

class ResourceUniformStatePrep(ResourceOperator):
    r"""Resource class for preparing a uniform superposition:

    ::math:
        \frac{1}{\sqrt{2^{k}L}} \sum_{l=0}^{2^{k}L-1} |l\rangle

    where :math:`L` is odd, starting from the :math:`|0\rangle` state.

    Args:
        num_states (int): the number of states over which the uniform superposition is being prepared
        wires (Sequence[int], optional): the wires the operation acts on

    Resources:
        The resources are obtained from Figure 12 in `Babbush et al. (2018) <https://arxiv.org/pdf/1805.03662>`_.
        The circuit uses amplitude amplification to prepare a uniform superposition over l basis states, where :math:`l = 2^{k}L`.
        The resources are given as :math:`k+ 2logL` qubits, and :math:`8logL` T gates.

    **Example**

    The resources for this operation are computed using:

    >>> unif_state_prep = plre.ResourceUniformStatePrep(10)
    >>> print(plre.estimate_resources(unif_state_prep))
    """

    def __init__(self, num_states, wires=None):
        self.num_states = num_states
        k = (num_states & -num_states).bit_length() - 1
        L = num_states // (2 ** k)
        self.num_wires = k + int(math.ceil(math.log2(L)))
        super().__init__(wires=wires)

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * num_states (int): the number of states over which the uniform superposition is being prepared
        """
        return {"num_states": self.num_states}

    @classmethod
    def resource_rep(cls, num_states: int) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        return CompressedResourceOp(cls, {"num_states": num_states})

    @classmethod
    def default_resource_decomp(cls, num_states, **kwargs):
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Args:
            num_states (int): the number of states over which the uniform superposition is being prepared

        Resources:
            The resources are obtained from Figure 12 in `Babbush et al. (2018) <https://arxiv.org/pdf/1805.03662>`_.
            The circuit uses amplitude amplification to prepare a uniform superposition over l basis states, where :math:`l = 2^{k}L`.
            The resources are given as :math:`k+ 2logL` qubits, and :math:`8logL` T gates.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """

        gate_lst = []
        k = (num_states & -num_states).bit_length() - 1
        L = num_states // (2 ** k)
        logl = int(math.ceil(math.log2(L)))
        gate_lst.append(AllocWires(1))
        gate_lst.append(GateCount(resource_rep(plre.ResourceHadamard), k + 3 * logl))
        gate_lst.append(GateCount(resource_rep(plre.ResourceIntegerComparator, {"val": L, "register_size": logl}), 1))
        gate_lst.append(GateCount(resource_rep(plre.ResourceRZ), 2))
        gate_lst.append(GateCount(resource_rep(plre.ResourceAdjoint, {"base_cmpr_op": resource_rep(plre.ResourceIntegerComparator, {"val":L, "register_size": logl})}), 1))
        gate_lst.append(FreeWires(1))

        return gate_lst
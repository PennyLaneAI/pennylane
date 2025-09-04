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
r"""Resource operators for PennyLane subroutine templates."""
import math

import numpy as np

from pennylane.labs import resource_estimation as plre
from pennylane.labs.resource_estimation.qubit_manager import AllocWires, FreeWires
from pennylane.labs.resource_estimation.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    resource_rep,
)

# pylint: disable=too-many-arguments, arguments-differ


class ResourceQubitizeTHC(ResourceOperator):
    r"""Resource class for qubitization of tensor hypercontracted Hamiltonian.

    .. note::

            This decomposition assumes that an appropriately sized phase gradient state is available.
            Users should ensure that the cost of constructing this state has been accounted for.
            See also :class:`~.pennylane.labs.resource_estimation.ResourcePhaseGradient`.

    Args:
        compact_ham (~pennylane.labs.resource_estimation.CompactHamiltonian): a tensor hypercontracted
            Hamiltonian for which the walk operator is being created
        prep_op (Union[~pennylane.labs.resource_estimation.ResourceOperator, None]): An optional
            resource operator, corresponding to the prepare routine. If :code:`None`, the
            default :class:`~.pennylane.labs.resource_estimation.ResourcePrepTHC` will be used.
        select_op (Union[~pennylane.labs.resource_estimation.ResourceOperator, None]): An optional
            resource operator, corresponding to the select routine. If :code:`None`, the
            default :class:`~.pennylane.labs.resource_estimation.ResourceSelectTHC` will be used.
        wires (list[int] or optional): the wires on which the operator acts

    Resources:
        The resources are calculated based on `arXiv:2011.03494 <https://arxiv.org/abs/2011.03494>`_

    **Example**

    The resources for this operation are computed using:

    >>> compact_ham = plre.CompactHamiltonian.thc(num_orbitals=20, tensor_rank=40)
    >>> prep = plre.ResourcePrepTHC(compact_ham, coeff_precision=20, select_swap_depth=2)
    >>> res = plre.estimate_resources(plre.ResourceQubitizeTHC(compact_ham, prep_op=prep))
    >>> print(res)
    --- Resources: ---
     Total qubits: 381
     Total gates : 5.628E+4
     Qubit breakdown:
      clean qubits: 313, dirty qubits: 0, algorithmic qubits: 68
     Gate breakdown:
      {'Toffoli': 3.504E+3, 'CNOT': 4.138E+4, 'X': 2.071E+3, 'Hadamard': 9.213E+3, 'S': 80, 'Z': 41}
    """

    resource_keys = {"compact_ham", "prep_op", "select_op"}

    def __init__(
        self,
        compact_ham,
        prep_op=None,
        select_op=None,
        wires=None,
    ):
        if compact_ham.method_name != "thc":
            raise TypeError(
                f"Unsupported Hamiltonian representation for ResourceQubitizeTHC."
                f"This method works with thc Hamiltonian, {compact_ham.method_name} provided"
            )

        self.compact_ham = compact_ham
        self.coeff_precision = coeff_precision
        self.rotation_precision = rotation_precision
        self.compare_precision = compare_precision

        num_orb = compact_ham.params["num_orbitals"]
        tensor_rank = compact_ham.params["tensor_rank"]
        self.num_wires = num_orb * 2 + 2 * int(np.ceil(math.log2(2 * tensor_rank + 1))) + 1
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * compact_ham (CompactHamiltonian): a tensor hypercontracted
                  Hamiltonian for which the walk operator is being created
                * prep_op (Union[CompressedResourceOp, None]): An optional compressed
                  resource operator, corresponding to the prepare routine. If :code:`None`, the
                  default :class:`~.pennylane.labs.resource_estimation.ResourcePrepTHC` will be used.
                * select_op (Union[CompressedResourceOp, None]): An optional compressed
                  resource operator, corresponding to the select routine. If :code:`None`, the
                  default :class:`~.pennylane.labs.resource_estimation.ResourceSelectTHC` will be used.
        """
        return {
            "compact_ham": self.compact_ham,
            "prep_op": self.prep_op,
            "select_op": self.select_op,
        }

    @classmethod
    def resource_rep(cls, compact_ham, prep_op=None, select_op=None) -> CompressedResourceOp:
        """Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            compact_ham (~pennylane.labs.resource_estimation.CompactHamiltonian): a tensor hypercontracted
                Hamiltonian for which the walk operator is being created
            prep_op (Union[~pennylane.labs.resource_estimation.CompressedResourceOp, None]): An optional compressed
                resource operator, corresponding to the prepare routine. If :code:`None`, the
                default :class:`~.pennylane.labs.resource_estimation.ResourcePrepTHC` will be used.
            select_op (Union[~pennylane.labs.resource_estimation.CompressedResourceOp, None]): An optional compressed
                resource operator, corresponding to the select routine. If :code:`None`, the
                default :class:`~.pennylane.labs.resource_estimation.ResourceSelectTHC` will be used.

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        num_orb = compact_ham.params["num_orbitals"]
        tensor_rank = compact_ham.params["tensor_rank"]
        num_wires = num_orb * 2 + 2 * int(np.ceil(math.log2(2 * tensor_rank + 1))) + 1

        params = {
            "compact_ham": compact_ham,
            "coeff_precision": coeff_precision,
            "rotation_precision": rotation_precision,
            "compare_precision": compare_precision,
        }
        return CompressedResourceOp(cls, num_wires, params)

    @classmethod
    def default_resource_decomp(
        cls,
        compact_ham,
        prep_op=None,
        select_op=None,
        **kwargs,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object represents a quantum gate
        and the number of times it occurs in the decomposition.

        .. note::

            This decomposition assumes that an appropriately sized phase gradient state is available.
            Users should ensure that the cost of constructing this state has been accounted for.
            See also :class:`~.pennylane.labs.resource_estimation.ResourcePhaseGradient`.

        Args:
            compact_ham (~pennylane.labs.resource_estimation.CompactHamiltonian): a tensor hypercontracted
                Hamiltonian for which the walk operator is being created
            prep_op (Union[~pennylane.labs.resource_estimation.CompressedResourceOp, None]): An optional compressed
                resource operator, corresponding to the prepare routine. If :code:`None`, the
                default :class:`~.pennylane.labs.resource_estimation.ResourcePrepTHC` will be used.
            select_op (Union[~pennylane.labs.resource_estimation.CompressedResourceOp, None]): An optional compressed
                resource operator, corresponding to the select routine. If :code:`None`, the
                default :class:`~.pennylane.labs.resource_estimation.ResourceSelectTHC` will be used.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.

        """
        gate_list = []

        tensor_rank = compact_ham.params["tensor_rank"]
        m_register = int(np.ceil(np.log2(tensor_rank)))

        if not select_op:
            # Select cost from Figure 5 in arXiv:2011.03494
            select_op = resource_rep(
                plre.ResourceSelectTHC,
                {
                    "compact_ham": compact_ham,
                },
            )
        gate_list.append(GateCount(select_op))

        if not prep_op:
            # Prep cost from Figure 3 and 4 in arXiv:2011.03494
            prep_op = resource_rep(
                plre.ResourcePrepTHC,
                {
                    "compact_ham": compact_ham,
                },
            )
        gate_list.append(GateCount(prep_op))
        gate_list.append(GateCount(resource_rep(plre.ResourceAdjoint, {"base_cmpr_op": prep_op})))

        # reflection cost from Eq. 44 in arXiv:2011.03494
        coeff_precision = (
            prep_op.params["coeff_precision"] or kwargs["config"]["qubitization_coeff_precision"]
        )

        toffoli = resource_rep(plre.ResourceToffoli)
        gate_list.append(GateCount(toffoli, 2 * m_register + coeff_precision + 4))

        return gate_list

    @classmethod
    def default_controlled_resource_decomp(
        cls,
        ctrl_num_ctrl_wires,
        ctrl_num_ctrl_values,
        compact_ham,
        prep_op=None,
        select_op=None,
        **kwargs,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for the controlled version of the operator.

        .. note::

            This decomposition assumes that an appropriately sized phase gradient state is available.
            Users should ensure that the cost of constructing this state has been accounted for.
            See also :class:`~.pennylane.labs.resource_estimation.ResourcePhaseGradient`.

        Args:
            ctrl_num_ctrl_wires (int): the number of qubits the operation is controlled on
            ctrl_num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            compact_ham (~pennylane.labs.resource_estimation.CompactHamiltonian): a tensor hypercontracted
                Hamiltonian for which the walk operator is being created
            prep_op (Union[~pennylane.labs.resource_estimation.CompressedResourceOp, None]): An optional compressed
                resource operator, corresponding to the prepare routine. If :code:`None`, the
                default :class:`~.pennylane.labs.resource_estimation.ResourcePrepTHC` will be used.
            select_op (Union[~pennylane.labs.resource_estimation.CompressedResourceOp, None]): An optional compressed
                resource operator, corresponding to the select routine. If :code:`None`, the
                default :class:`~.pennylane.labs.resource_estimation.ResourceSelectTHC` will be used.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.

        """
        gate_list = []

        tensor_rank = compact_ham.params["tensor_rank"]
        m_register = int(np.ceil(np.log2(tensor_rank)))

        if ctrl_num_ctrl_wires > 1:
            mcx = resource_rep(
                plre.ResourceMultiControlledX,
                {
                    "num_ctrl_wires": ctrl_num_ctrl_wires,
                    "num_ctrl_values": ctrl_num_ctrl_values,
                },
            )
            gate_list.append(AllocWires(1))
            gate_list.append(GateCount(mcx, 2))

        if not select_op:
            # Controlled Select cost from Fig 5 in arXiv:2011.03494
            select_op = resource_rep(
                plre.ResourceSelectTHC,
                {
                    "compact_ham": compact_ham,
                },
            )
        gate_list.append(
            GateCount(
                resource_rep(
                    plre.ResourceControlled,
                    {"base_cmpr_op": select_op, "num_ctrl_wires": 1, "num_ctrl_values": 0},
                )
            )
        )

        if not prep_op:
            # Prep cost from Fig 3 and 4 in arXiv:2011.03494
            prep_op = resource_rep(
                plre.ResourcePrepTHC,
                {
                    "compact_ham": compact_ham,
                },
            )
        gate_list.append(GateCount(prep_op))
        gate_list.append(GateCount(resource_rep(plre.ResourceAdjoint, {"base_cmpr_op": prep_op})))

        # reflection cost from Eq. 44 in arXiv:2011.03494s
        coeff_precision = (
            prep_op.params["coeff_precision"] or kwargs["config"]["qubitization_coeff_precision"]
        )
        toffoli = resource_rep(plre.ResourceToffoli)
        gate_list.append(GateCount(toffoli, 2 * m_register + coeff_precision + 4))

        if ctrl_num_ctrl_wires > 1:
            gate_list.append(FreeWires(1))
        elif ctrl_num_ctrl_values > 0:
            gate_list.append(GateCount(resource_rep(plre.ResourceX), 2 * ctrl_num_ctrl_values))

        return gate_list

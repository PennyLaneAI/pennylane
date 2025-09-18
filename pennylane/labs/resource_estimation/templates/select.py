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
r"""Resource operators for select templates."""

import math

import numpy as np

import pennylane.labs.resource_estimation as plre
from pennylane.labs.resource_estimation import AllocWires, FreeWires
from pennylane.labs.resource_estimation.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    resource_rep,
)

# pylint: disable=arguments-differ, too-many-arguments


class ResourceSelectTHC(ResourceOperator):
    r"""Resource class for creating the custom Select operator for tensor hypercontracted Hamiltonian.
    This operator customizes the Select circuit to use the structure of THC Hamiltonian.

    .. note::

            This decomposition assumes that an appropriately sized phase gradient state is available.
            Users should ensure that the cost of constructing this state has been accounted for.
            See also :class:`~.pennylane.labs.resource_estimation.ResourcePhaseGradient`.

    Args:
        compact_ham (~pennylane.labs.resource_estimation.CompactHamiltonian): a tensor hypercontracted
            Hamiltonian on which the select operator is being applied
        rotation_precision (int, optional): The number of bits used to represent the precision for loading
            the rotation angles for basis rotation. If :code:`None` is provided, the default value from the
            :data:`~.pennylane.labs.resource_estimation.resource_tracking.resource_config` is used.
        select_swap_depth (int, optional): A parameter of :class:`~.pennylane.labs.resource_estimation.ResourceQROM`
            used to trade-off extra qubits for reduced circuit depth. Defaults to :code:`None`, which internally determines the optimal depth.
        wires (list[int] or optional): the wires on which the operator acts

    Resources:
        The resources are calculated based on Figure 5 in `arXiv:2011.03494 <https://arxiv.org/abs/2011.03494>`_

    **Example**

    The resources for this operation are computed using:

    >>> compact_ham = plre.CompactHamiltonian.thc(num_orbitals=20, tensor_rank=40)
    >>> res = plre.estimate(plre.ResourceSelectTHC(compact_ham, rotation_precision=15))
    >>> print(res)
    --- Resources: ---
     Total qubits: 371
     Total gates : 1.959E+4
     Qubit breakdown:
      clean qubits: 313, dirty qubits: 0, algorithmic qubits: 58
     Gate breakdown:
      {'Toffoli': 2.219E+3, 'CNOT': 1.058E+4, 'X': 268, 'Hadamard': 6.406E+3, 'S': 80, 'Z': 41}
    """

    resource_keys = {"compact_ham", "rotation_precision", "select_swap_depth"}

    def __init__(self, compact_ham, rotation_precision=None, select_swap_depth=None, wires=None):

        if compact_ham.method_name != "thc":
            raise TypeError(
                f"Unsupported Hamiltonian representation for ResourceSelectTHC."
                f"This method works with thc Hamiltonian, {compact_ham.method_name} provided"
            )

        if not (isinstance(rotation_precision, int) or rotation_precision is None):
            raise TypeError(
                f"`rotation_precision` must be an integer, provided {type(rotation_precision)}."
            )

        self.compact_ham = compact_ham
        self.rotation_precision = rotation_precision
        self.select_swap_depth = select_swap_depth
        num_orb = compact_ham.params["num_orbitals"]
        tensor_rank = compact_ham.params["tensor_rank"]

        # num_orb*2 for state register, 2*log(M) for \mu and \nu registers
        # 6 extras are for 2 spin registers, 1 for rotation on ancilla, 1 flag for success of inequality,
        # 1 flag for one-body vs two-body and 1 to control swap of \mu and \nu registers.
        self.num_wires = num_orb * 2 + 2 * int(np.ceil(math.log2(tensor_rank + 1))) + 6

        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * compact_ham (CompactHamiltonian): a tensor hypercontracted
                  Hamiltonian on which the select operator is being applied
                * rotation_precision (int, optional): The number of bits used to represent the precision for loading
                  the rotation angles for basis rotation. If :code:`None` is provided, the default value from the
                  :data:`~.pennylane.labs.resource_estimation.resource_tracking.resource_config` is used.
                * select_swap_depth (int, optional): A parameter of :class:`~.pennylane.labs.resource_estimation.ResourceQROM`
                  used to trade-off extra qubits for reduced circuit depth. Defaults to :code:`None`, which internally determines the optimal depth.
        """
        return {
            "compact_ham": self.compact_ham,
            "rotation_precision": self.rotation_precision,
            "select_swap_depth": self.select_swap_depth,
        }

    @classmethod
    def resource_rep(
        cls, compact_ham, rotation_precision=None, select_swap_depth=None
    ) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            compact_ham (~pennylane.labs.resource_estimation.CompactHamiltonian): a tensor hypercontracted
                Hamiltonian on which the select operator is being applied
            rotation_precision (int, optional): The number of bits used to represent the precision for loading
                the rotation angles for basis rotation. If :code:`None` is provided, the default value from the
                :data:`~.pennylane.labs.resource_estimation.resource_tracking.resource_config` is used.
            select_swap_depth (int, optional): A parameter of :class:`~.pennylane.labs.resource_estimation.ResourceQROM`
                used to trade-off extra qubits for reduced circuit depth. Defaults to :code:`None`, which internally determines the optimal depth.

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """

        if compact_ham.method_name != "thc":
            raise TypeError(
                f"Unsupported Hamiltonian representation for ResourceSelectTHC."
                f"This method works with thc Hamiltonian, {compact_ham.method_name} provided"
            )

        if not (isinstance(rotation_precision, int) or rotation_precision is None):
            raise TypeError(
                f"`rotation_precision` must be an integer, provided {type(rotation_precision)}."
            )
        num_orb = compact_ham.params["num_orbitals"]
        tensor_rank = compact_ham.params["tensor_rank"]

        # num_orb*2 for state register, 2*log(M) for \mu and \nu registers
        # 6 extras are for 2 spin registers, 1 for rotation on ancilla, 1 flag for success of inequality,
        # 1 flag for one-body vs two-body and 1 to control swap of \mu and \nu registers.
        num_wires = num_orb * 2 + 2 * int(np.ceil(math.log2(tensor_rank + 1))) + 6
        params = {
            "compact_ham": compact_ham,
            "rotation_precision": rotation_precision,
            "select_swap_depth": select_swap_depth,
        }
        return CompressedResourceOp(cls, num_wires, params)

    @classmethod
    def resource_decomp(
        cls, compact_ham, rotation_precision=None, select_swap_depth=None, **kwargs
    ) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object represents a quantum gate
        and the number of times it occurs in the decomposition.

        .. note::

            This decomposition assumes that an appropriately sized phase gradient state is available.
            Users should ensure that the cost of constructing this state has been accounted for.
            See also :class:`~.pennylane.labs.resource_estimation.ResourcePhaseGradient`.

        Args:
            compact_ham (~pennylane.labs.resource_estimation.CompactHamiltonian): a tensor hypercontracted
                Hamiltonian on which the select operator is being applied
            rotation_precision (int, optional): The number of bits used to represent the precision for loading
                the rotation angles for basis rotation. If :code:`None` is provided, the default value from the
                :data:`~.pennylane.labs.resource_estimation.resource_tracking.resource_config` is used.
            select_swap_depth (int, optional): A parameter of :class:`~.pennylane.labs.resource_estimation.ResourceQROM`
                used to trade-off extra qubits for reduced circuit depth. Defaults to :code:`None`, which internally determines the optimal depth.

        Resources:
            The resources are calculated based on Figure 5 in `arXiv:2011.03494 <https://arxiv.org/abs/2011.03494>`_.
            The resources are modified to remove the control from the Select operation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """

        num_orb = compact_ham.params["num_orbitals"]
        tensor_rank = compact_ham.params["tensor_rank"]

        gate_list = []
        # Total select cost from Eq. 43 in arXiv:2011.03494

        # 4 swaps on state registers controlled on spin qubits
        cswap = resource_rep(plre.ResourceCSWAP)
        gate_list.append(GateCount(cswap, 4 * num_orb))

        # Data output for rotations
        gate_list.append(AllocWires(rotation_precision * (num_orb - 1)))

        # QROM to load rotation angles for 2-body integrals
        qrom_twobody = resource_rep(
            plre.ResourceQROM,
            {
                "num_bitstrings": tensor_rank + num_orb,
                "size_bitstring": rotation_precision,
                "clean": False,
                "select_swap_depth": select_swap_depth,
            },
        )
        gate_list.append(GateCount(qrom_twobody))

        # Cost for rotations by adding the rotations into the phase gradient state
        semiadder = resource_rep(
            plre.ResourceControlled,
            {
                "base_cmpr_op": resource_rep(
                    plre.ResourceSemiAdder,
                    {"max_register_size": rotation_precision - 1},
                ),
                "num_ctrl_wires": 1,
                "num_ctrl_values": 0,
            },
        )
        gate_list.append(GateCount(semiadder, num_orb - 1))

        # Adjoint of QROM for 2-body integrals Eq. 34 in arXiv:2011.03494
        gate_list.append(
            GateCount(resource_rep(plre.ResourceAdjoint, {"base_cmpr_op": qrom_twobody}))
        )
        # Adjoint of semiadder for 2-body integrals
        gate_list.append(
            GateCount(resource_rep(plre.ResourceAdjoint, {"base_cmpr_op": semiadder}), num_orb - 1)
        )

        # QROM to load rotation angles for one body integrals
        qrom_onebody = resource_rep(
            plre.ResourceQROM,
            {
                "num_bitstrings": tensor_rank,
                "size_bitstring": rotation_precision,
                "clean": False,
                "select_swap_depth": select_swap_depth,
            },
        )
        gate_list.append(GateCount(qrom_onebody))

        # Cost for rotations by adding the rotations into the phase gradient state
        gate_list.append(GateCount(semiadder, num_orb - 1))

        # Clifford cost for rotations
        h = resource_rep(plre.ResourceHadamard)
        s = resource_rep(plre.ResourceS)
        s_dagg = resource_rep(plre.ResourceAdjoint, {"base_cmpr_op": s})

        gate_list.append(GateCount(h, 4 * (num_orb)))
        gate_list.append(GateCount(s, 2 * num_orb))
        gate_list.append(GateCount(s_dagg, 2 * num_orb))

        # Adjoint of QROM for one body integrals Eq. 35 in arXiv:2011.03494
        gate_list.append(
            GateCount(resource_rep(plre.ResourceAdjoint, {"base_cmpr_op": qrom_onebody}))
        )

        # Adjoint of semiadder for one body integrals
        gate_list.append(
            GateCount(resource_rep(plre.ResourceAdjoint, {"base_cmpr_op": semiadder}), num_orb - 1)
        )

        # Z gate in the center of rotations
        gate_list.append(plre.GateCount(resource_rep(plre.ResourceZ)))

        cz = resource_rep(plre.ResourceCZ)
        gate_list.append(plre.GateCount(cz, 1))

        # 1 cswap between the spin registers
        gate_list.append(plre.GateCount(cswap, 1))
        gate_list.append(FreeWires(rotation_precision * (num_orb - 1)))

        return gate_list

    @classmethod
    def controlled_resource_decomp(
        cls,
        ctrl_num_ctrl_wires,
        ctrl_num_ctrl_values,
        compact_ham,
        rotation_precision=None,
        select_swap_depth=None,
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
                Hamiltonian on which the select operator is being applied
            rotation_precision (int, optional): The number of bits used to represent the precision for loading
                the rotation angles for basis rotation. If :code:`None` is provided, the default value from the
                :data:`~.pennylane.labs.resource_estimation.resource_tracking.resource_config` is used.
            select_swap_depth (int, optional): A parameter of :class:`~.pennylane.labs.resource_estimation.ResourceQROM`
                used to trade-off extra qubits for reduced circuit depth. Defaults to :code:`None`, which internally determines the optimal depth.

        Resources:
            The resources are calculated based on Figure 5 in `arXiv:2011.03494 <https://arxiv.org/abs/2011.03494>`_

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """

        num_orb = compact_ham.params["num_orbitals"]
        tensor_rank = compact_ham.params["tensor_rank"]

        gate_list = []

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

        # 4 swaps on state registers controlled on spin qubits
        cswap = resource_rep(plre.ResourceCSWAP)
        gate_list.append(GateCount(cswap, 4 * num_orb))

        # QROM for loading rotation angles for 2-body integrals
        gate_list.append(AllocWires(rotation_precision * (num_orb - 1)))
        qrom_twobody = resource_rep(
            plre.ResourceQROM,
            {
                "num_bitstrings": tensor_rank + num_orb,
                "size_bitstring": rotation_precision,
                "clean": False,
                "select_swap_depth": select_swap_depth,
            },
        )
        gate_list.append(GateCount(qrom_twobody))

        # Cost for rotations by adding the rotations into the phase gradient state
        semiadder = resource_rep(
            plre.ResourceControlled,
            {
                "base_cmpr_op": resource_rep(
                    plre.ResourceSemiAdder,
                    {"max_register_size": rotation_precision},
                ),
                "num_ctrl_wires": 1,
                "num_ctrl_values": 0,
            },
        )
        gate_list.append(GateCount(semiadder, num_orb - 1))

        # Adjoint of QROM for 2-body integrals Eq. 34 in arXiv:2011.03494
        gate_list.append(
            GateCount(resource_rep(plre.ResourceAdjoint, {"base_cmpr_op": qrom_twobody}))
        )
        # Adjoint of semiadder for 2-body integrals
        gate_list.append(
            GateCount(resource_rep(plre.ResourceAdjoint, {"base_cmpr_op": semiadder}), num_orb - 1)
        )

        # QROM for loading rotation angles for one body integrals
        qrom_onebody = resource_rep(
            plre.ResourceQROM,
            {
                "num_bitstrings": tensor_rank,
                "size_bitstring": rotation_precision,
                "clean": False,
                "select_swap_depth": select_swap_depth,
            },
        )
        gate_list.append(GateCount(qrom_onebody))

        # Cost for rotations by adding the rotations into the phase gradient state
        gate_list.append(GateCount(semiadder, num_orb - 1))

        # Clifford cost for rotations
        h = resource_rep(plre.ResourceHadamard)
        s = resource_rep(plre.ResourceS)
        s_dagg = resource_rep(plre.ResourceAdjoint, {"base_cmpr_op": s})

        gate_list.append(GateCount(h, 4 * (num_orb)))
        gate_list.append(GateCount(s, 2 * num_orb))
        gate_list.append(GateCount(s_dagg, 2 * num_orb))

        # Adjoint of QROM for one body integrals Eq. 35 in arXiv:2011.03494
        gate_list.append(
            GateCount(resource_rep(plre.ResourceAdjoint, {"base_cmpr_op": qrom_onebody}))
        )
        # Adjoint of semiadder for one body integrals
        gate_list.append(
            GateCount(resource_rep(plre.ResourceAdjoint, {"base_cmpr_op": semiadder}), num_orb - 1)
        )

        # Z gate in the center of rotations
        cz = resource_rep(plre.ResourceCZ)
        gate_list.append(plre.GateCount(cz, 1))

        ccz = resource_rep(
            plre.ResourceControlled,
            {
                "base_cmpr_op": plre.ResourceZ.resource_rep(),
                "num_ctrl_wires": 2,
                "num_ctrl_values": 1,
            },
        )
        gate_list.append(plre.GateCount(ccz, 1))

        # 1 cswap between the spin registers
        gate_list.append(plre.GateCount(cswap, 1))

        gate_list.append(FreeWires(rotation_precision * (num_orb - 1)))

        if ctrl_num_ctrl_wires > 1:
            gate_list.append(FreeWires(1))
        elif ctrl_num_ctrl_values > 0:
            gate_list.append(GateCount(resource_rep(plre.ResourceX), 2 * ctrl_num_ctrl_values))

        return gate_list

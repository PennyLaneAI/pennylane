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
from pennylane.wires import Wires

# pylint: disable=too-many-arguments, arguments-differ


class ResourceQubitizeTHC(ResourceOperator):
    r"""Resource class for Qubitization of THC Hamiltonian.

    Args:
        compact_ham (~pennylane.labs.resource_estimation.CompactHamiltonian): a tensor hypercontracted
            Hamiltonian for which the walk operator is being created
        coeff_precision_bits (int, optional): Number of bits used to represent the precision for loading
            the coefficients of the Hamiltonian. If :code:`None` is provided the default value from the
            :code:`resource_config` is used.
        rotation_precision_bits (int, optional): Number of bits used to represent the precision for loading
            the rotation angles for basis rotation. If :code:`None` is provided the default value from the
            :code:`resource_config` is used.
        select_swap_depths (Union[None, int, Iterable(int)],optional): A parameter of :code:`QROM`
            used to trade-off extra qubits for reduced circuit depth. A list can be used to configure the
            ``select_swap_depth`` individually for :code:`ResourcePrepTHC` and :code:`ResourceSelectTHC` circuits,
            respectively.
        wires (list[int] or optional): the wires on which the operator acts

    Resources:
        The resources are calculated based on `arXiv:2011.03494 <https://arxiv.org/abs/2011.03494>`_

    **Example**

    The resources for this operation are computed using:

    >>> compact_ham = plre.CompactHamiltonian.thc(num_orbitals=20, tensor_rank=40)
    >>> res = plre.estimate_resources(plre.ResourceQubitizeTHC(compact_ham))
    >>> print(res)
    --- Resources: ---
     Total qubits: 371
     Total gates : 4.885E+4
     Qubit breakdown:
      clean qubits: 313, dirty qubits: 0, algorithmic qubits: 58
     Gate breakdown:
      {'Toffoli': 3.156E+3, 'CNOT': 3.646E+4, 'X': 1.201E+3, 'Hadamard': 7.913E+3, 'S': 80, 'Z': 41}

    """

    def __init__(
        self,
        compact_ham,
        coeff_precision_bits=None,
        rotation_precision_bits=None,
        select_swap_depths=None,
        wires=None,
    ):
        if compact_ham.method_name != "thc":
            raise TypeError(
                f"Unsupported Hamiltonian representation for ResourceQubitizeTHC."
                f"This method works with thc Hamiltonian, {compact_ham.method_name} provided"
            )

        if isinstance(select_swap_depths, (list, tuple, np.ndarray)):
            if len(select_swap_depths) != 2:
                raise ValueError(
                    f"Expected the length of `select_swap_depths` to be 2, got {len(select_swap_depths)}"
                )
        elif not (isinstance(select_swap_depths, int) or select_swap_depths is None):
            raise TypeError("`select_swap_depths` must be an integer, None or iterable")

        self.compact_ham = compact_ham
        self.coeff_precision_bits = coeff_precision_bits
        self.rotation_precision_bits = rotation_precision_bits
        self.select_swap_depths = select_swap_depths
        if wires is not None:
            self.wires = Wires(wires)
            self.num_wires = len(self.wires)
        else:
            num_orb = compact_ham.params["num_orbitals"]
            tensor_rank = compact_ham.params["tensor_rank"]
            self.num_wires = num_orb * 2 + 2 * int(np.ceil(math.log2(tensor_rank + 1))) + 6
            self.wires = None
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * compact_ham (~pennylane.labs.resource_estimation.CompactHamiltonian): a tensor hypercontracted
                  Hamiltonian for which the walk operator is being created
                * coeff_precision_bits (int, optional): The number of bits used to represent the precision for loading
                  the coefficients of Hamiltonian. If :code:`None` is provided the default value from the
                  :code:`resource_config` is used.
                * rotation_precision_bits (int, optional): The number of bits used to represent the precision for loading
                  the rotation angles for basis rotation. If :code:`None` is provided the default value from the
                  :code:`resource_config` is used.
                * select_swap_depths (Union[None, int, Iterable(int)],optional): A parameter of :code:`QROM`
                    used to trade-off extra qubits for reduced circuit depth. A list can be used to configure the
                    ``select_swap_depth`` individually for :code:`ResourcePrepTHC` and :code:`ResourceSelectTHC` circuits,
                    respectively.
        """
        return {
            "compact_ham": self.compact_ham,
            "coeff_precision_bits": self.coeff_precision_bits,
            "rotation_precision_bits": self.rotation_precision_bits,
            "select_swap_depths": self.select_swap_depths,
        }

    @classmethod
    def resource_rep(
        cls, compact_ham, coeff_precision_bits, rotation_precision_bits, select_swap_depths
    ) -> CompressedResourceOp:
        """Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            compact_ham (~pennylane.labs.resource_estimation.CompactHamiltonian): a tensor hypercontracted
                Hamiltonian for which the walk operator is being created
            coeff_precision_bits (int, optional): The number of bits used to represent the precision for loading
                the coefficients of Hamiltonian. If :code:`None` is provided the default value from the
                :code:`resource_config` is used.
            rotation_precision_bits (int, optional): The number of bits used to represent the precision for loading
                the rotation angles for basis rotation. If :code:`None` is provided the default value from the
                :code:`resource_config` is used.
            select_swap_depths (Union[None, int, Iterable(int)],optional): A parameter of :code:`QROM`
                used to trade-off extra qubits for reduced circuit depth. A list can be used to configure the
                ``select_swap_depth`` individually for :code:`ResourcePrepTHC` and :code:`ResourceSelectTHC` circuits,
                respectively.

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """

        if isinstance(select_swap_depths, (list, tuple, np.ndarray)):
            if len(select_swap_depths) != 2:
                raise ValueError(
                    f"Expected the length of `select_swap_depths` to be 2, got {len(select_swap_depths)}"
                )
        elif not (isinstance(select_swap_depths, int) or select_swap_depths is None):
            raise TypeError("`select_swap_depths` must be an integer, None or iterable")

        params = {
            "compact_ham": compact_ham,
            "coeff_precision_bits": coeff_precision_bits,
            "rotation_precision_bits": rotation_precision_bits,
            "select_swap_depths": select_swap_depths,
        }
        return CompressedResourceOp(cls, params)

    @classmethod
    def default_resource_decomp(
        cls,
        compact_ham,
        coeff_precision_bits=None,
        rotation_precision_bits=None,
        select_swap_depths=None,
        **kwargs,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object represents a quantum gate
        and the number of times it occurs in the decomposition.

        .. note::

            This decomposition assumes an appropriately sized phase gradient state is available.
            Users should ensure the cost of constructing such a state has been accounted for.
            See also :class:`~.pennylane.labs.resource_estimation.ResourcePhaseGradient`.

        Args:
            compact_ham (~pennylane.labs.resource_estimation.CompactHamiltonian): a tensor hypercontracted
                Hamiltonian for which the walk operator is being created
            coeff_precision_bits (int, optional): The number of bits used to represent the precision for loading
                the coefficients of Hamiltonian. If :code:`None` is provided the default value from the
                :code:`resource_config` is used.
            rotation_precision_bits (int, optional): The number of bits used to represent the precision for loading
                the rotation angles for basis rotation. If :code:`None` is provided the default value from the
                :code:`resource_config` is used.
            select_swap_depths (Union[None, int, Iterable(int)],optional): A parameter of :code:`QROM`
                used to trade-off extra qubits for reduced circuit depth. A list can be used to configure the
                ``select_swap_depth`` individually for :code:`ResourcePrepTHC` and :code:`ResourceSelectTHC` circuits,
                respectively.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.

        """
        gate_list = []

        tensor_rank = compact_ham.params["tensor_rank"]
        m_register = int(np.ceil(np.log2(tensor_rank)))
        coeff_precision_bits = coeff_precision_bits or kwargs["config"]["qubitization_coeff_bits"]

        if isinstance(select_swap_depths, int) or select_swap_depths is None:
            select_swap_depths = [select_swap_depths] * 2

        # Select cost from Figure 5 in arXiv:2011.03494
        select = resource_rep(
            plre.ResourceSelectTHC,
            {
                "compact_ham": compact_ham,
                "rotation_precision_bits": rotation_precision_bits,
                "select_swap_depth": select_swap_depths[1],
            },
        )
        gate_list.append(GateCount(select))

        # Prep cost from Figure 3 and 4 in arXiv:2011.03494
        prep = resource_rep(
            plre.ResourcePrepTHC,
            {
                "compact_ham": compact_ham,
                "coeff_precision_bits": coeff_precision_bits,
                "select_swap_depth": select_swap_depths[0],
            },
        )
        gate_list.append(GateCount(prep))
        gate_list.append(GateCount(resource_rep(plre.ResourceAdjoint, {"base_cmpr_op": prep})))

        # reflection cost from Eq. 44 in arXiv:2011.03494
        toffoli = resource_rep(plre.ResourceToffoli)
        gate_list.append(GateCount(toffoli, 2 * m_register + coeff_precision_bits + 4))

        return gate_list

    @classmethod
    def default_controlled_resource_decomp(
        cls,
        ctrl_num_ctrl_wires,
        ctrl_num_ctrl_values,
        compact_ham,
        coeff_precision_bits=None,
        rotation_precision_bits=None,
        select_swap_depths=None,
        **kwargs,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for the controlled version of the operator.

        .. note::

            This decomposition assumes an appropriately sized phase gradient state is available.
            Users should ensure the cost of constructing such a state has been accounted for.
            See also :class:`~.pennylane.labs.resource_estimation.ResourcePhaseGradient`.

        Args:
            ctrl_num_ctrl_wires (int): the number of qubits the operation is controlled on
            ctrl_num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            compact_ham (~pennylane.labs.resource_estimation.CompactHamiltonian): a tensor hypercontracted
                Hamiltonian for which the walk operator is being created
            coeff_precision_bits (int, optional): The number of bits used to represent the precision for loading
                the coefficients of Hamiltonian. If :code:`None` is provided the default value from the
                :code:`resource_config` is used.
            rotation_precision_bits (int, optional): The number of bits used to represent the precision for loading
                the rotation angles for basis rotation. If :code:`None` is provided the default value from the
                :code:`resource_config` is used.
            select_swap_depths (Union[None, int, Iterable(int)],optional): A parameter of :code:`QROM`
                used to trade-off extra qubits for reduced circuit depth. A list can be used to configure the
                ``select_swap_depth`` individually for :code:`ResourcePrepTHC` and :code:`ResourceSelectTHC` circuits,
                respectively.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.

        """
        gate_list = []

        tensor_rank = compact_ham.params["tensor_rank"]
        m_register = int(np.ceil(np.log2(tensor_rank)))
        coeff_precision_bits = coeff_precision_bits or kwargs["config"]["qubitization_coeff_bits"]

        if isinstance(select_swap_depths, int) or select_swap_depths is None:
            select_swap_depths = [select_swap_depths] * 2

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

        # Controlled Select cost from Fig 5 in arXiv:2011.03494
        select = resource_rep(
            plre.ResourceSelectTHC,
            {
                "compact_ham": compact_ham,
                "rotation_precision": rotation_precision_bits,
                "select_swap_depth": select_swap_depths[1],
            },
        )
        gate_list.append(
            GateCount(
                resource_rep(
                    plre.ResourceControlled,
                    {"base_cmpr_op": select, "num_ctrl_wires": 1, "num_ctrl_values": 0},
                )
            )
        )

        # Prep cost from Fig 3 and 4 in arXiv:2011.03494
        prep = resource_rep(
            plre.ResourcePrepTHC,
            {
                "compact_ham": compact_ham,
                "coeff_precision": coeff_precision_bits,
                "select_swap_depth": select_swap_depths[0],
            },
        )
        gate_list.append(GateCount(prep))
        gate_list.append(GateCount(resource_rep(plre.ResourceAdjoint, {"base_cmpr_op": prep})))

        # reflection cost from Eq. 44 in arXiv:2011.03494s
        toffoli = resource_rep(plre.ResourceToffoli)
        gate_list.append(GateCount(toffoli, 2 * m_register + coeff_precision_bits + 4))

        if ctrl_num_ctrl_wires > 1:
            gate_list.append(FreeWires(1))
        elif ctrl_num_ctrl_values > 0:
            gate_list.append(GateCount(resource_rep(plre.ResourceX), 2 * ctrl_num_ctrl_values))

        return gate_list

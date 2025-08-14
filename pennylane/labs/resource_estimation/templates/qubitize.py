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
        coeff_precision (float, optional): precision for loading the coefficients of Hamiltonian
        rotation_precision (float, optional): precision for loading the rotation angles for basis rotation
        select_swap_depths (Union[None, int, Iterable(int)],optional): A parameter of :code:`QROM`
            used to trade-off extra qubits for reduced circuit depth. A list can be used to configure the
            ``select_swap_depth`` individually for ``ResourcePrepTHC` and `ResourceSelectTHC` circuits,
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
     Total qubits: 147.0
     Total gates : 1.631E+5
     Qubit breakdown:
      clean qubits: 10, dirty qubits: 82.0, algorithmic qubits: 55
     Gate breakdown:
      {'X': 2.046E+4, 'CNOT': 8.721E+4, 'Toffoli': 1.584E+4, 'Hadamard': 3.923E+4, 'Z': 160, 'Y': 200}

    """

    def __init__(
        self,
        compact_ham,
        coeff_precision=None,
        rotation_precision=None,
        select_swap_depths=None,
        wires=None,
    ):
        if compact_ham.method_name != "thc":
            raise TypeError(
                f"Unsupported Hamiltonian representation for ResourceQubitizeTHC."
                f"This method works with thc Hamiltonian, {compact_ham.method_name} provided"
            )

        if isinstance(select_swap_depths, int) or select_swap_depths is None:
            pass
        elif isinstance(select_swap_depths, (list, tuple, np.ndarray)):
            if len(select_swap_depths) != 2:
                raise ValueError(
                    f"Expected the length of `select_swap_depths` to be 2, got {len(select_swap_depths)}"
                )
        else:
            raise TypeError("`select_swap_depths` must be an integer, None or iterable")

        self.compact_ham = compact_ham
        self.coeff_precision = coeff_precision
        self.rotation_precision = rotation_precision
        self.select_swap_depths = select_swap_depths
        if wires is not None:
            self.wires = Wires(wires)
            self.num_wires = len(self.wires)
        else:
            num_orb = compact_ham.params["num_orbitals"]
            tensor_rank = compact_ham.params["tensor_rank"]
            self.num_wires = num_orb*2 +  2 * int(np.ceil(math.log2(tensor_rank+1))) + 6
            self.wires = None
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * compact_ham (~pennylane.labs.resource_estimation.CompactHamiltonian): a tensor hypercontracted
                  Hamiltonian for which the walk operator is being created
                * coeff_precision (float, optional): precision for loading the coefficients of Hamiltonian
                * rotation_precision (float, optional): precision for loading the rotation angles for basis rotation
                * select_swap_depths (Union[None, int, Iterable(int)],optional): A parameter of :code:`QROM`
                    used to trade-off extra qubits for reduced circuit depth. A list can be used to configure the
                    ``select_swap_depth`` individually for ``ResourcePrepTHC` and `ResourceSelectTHC` circuits,
                    respectively.
        """
        return {
            "compact_ham": self.compact_ham,
            "coeff_precision": self.coeff_precision,
            "rotation_precision": self.rotation_precision,
            "select_swap_depths": self.select_swap_depths,
        }

    @classmethod
    def resource_rep(
        cls, compact_ham, coeff_precision, rotation_precision, select_swap_depths
    ) -> CompressedResourceOp:
        """Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            compact_ham (~pennylane.labs.resource_estimation.CompactHamiltonian): a tensor hypercontracted
                Hamiltonian for which the walk operator is being created
            coeff_precision (float, optional): precision for loading the coefficients of Hamiltonian
            rotation_precision (float, optional): precision for loading the rotation angles for basis rotation
            select_swap_depths (Union[None, int, Iterable(int)],optional): A parameter of :code:`QROM`
                used to trade-off extra qubits for reduced circuit depth. A list can be used to configure the
                ``select_swap_depth`` individually for ``ResourcePrepTHC` and `ResourceSelectTHC` circuits,
                respectively.

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """

        if isinstance(select_swap_depths, int) or select_swap_depths is None:
            pass
        elif isinstance(select_swap_depths, (list, tuple, np.ndarray)):
            if len(select_swap_depths) != 2:
                raise ValueError(
                    f"Expected the length of `select_swap_depths` to be 2, got {len(select_swap_depths)}"
                )
        else:
            raise TypeError("`select_swap_depths` must be an integer, None or iterable")
        params = {
            "compact_ham": compact_ham,
            "coeff_precision": coeff_precision,
            "rotation_precision": rotation_precision,
            "select_swap_depths": select_swap_depths,
        }
        return CompressedResourceOp(cls, params)

    @classmethod
    def default_resource_decomp(
        cls,
        compact_ham,
        coeff_precision=None,
        rotation_precision=None,
        select_swap_depths=None,
        **kwargs,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object represents a quantum gate
        and the number of times it occurs in the decomposition.

        Args:
            compact_ham (~pennylane.labs.resource_estimation.CompactHamiltonian): a tensor hypercontracted
                Hamiltonian for which the walk operator is being created
            coeff_precision (float, optional): precision for loading the coefficients of Hamiltonian
            rotation_precision (float, optional): precision for loading the rotation angles for basis rotation
            select_swap_depths (Union[None, int, Iterable(int)],optional): A parameter of :code:`QROM`
                used to trade-off extra qubits for reduced circuit depth. A list can be used to configure the
                ``select_swap_depth`` individually for ``ResourcePrepTHC` and `ResourceSelectTHC` circuits,
                respectively.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.

        """
        gate_list = []

        tensor_rank = compact_ham.params["tensor_rank"]
        m_register = int(np.ceil(np.log2(tensor_rank)))
        coeff_precision = coeff_precision or kwargs["config"]["precision_qubitization_prep"]
        coeff_prec_wires = abs(math.floor(math.log2(coeff_precision)))

        if isinstance(select_swap_depths, int) or select_swap_depths is None:
            select_swap_depths = [select_swap_depths] * 2

        select = resource_rep(
            plre.ResourceSelectTHC,
            {
                "compact_ham": compact_ham,
                "rotation_precision": rotation_precision,
                "select_swap_depth": select_swap_depths[1],
            },
        )
        gate_list.append(GateCount(select))

        prep = resource_rep(
            plre.ResourcePrepTHC,
            {
                "compact_ham": compact_ham,
                "coeff_precision": coeff_precision,
                "select_swap_depth": select_swap_depths[0],
            },
        )
        gate_list.append(GateCount(prep))
        gate_list.append(GateCount(resource_rep(plre.ResourceAdjoint, {"base_cmpr_op": prep})))

        # reflection cost
        toffoli = resource_rep(plre.ResourceToffoli)
        gate_list.append(GateCount(toffoli, 2 * m_register + coeff_prec_wires + 4))


        return gate_list

    @classmethod
    def default_controlled_resource_decomp(
        cls,
        ctrl_num_ctrl_wires,
        ctrl_num_ctrl_values,
        compact_ham,
        coeff_precision=None,
        rotation_precision=None,
        select_swap_depths=None,
        **kwargs,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for the controlled version of the operator.

        Args:
            ctrl_num_ctrl_wires (int): the number of qubits the operation is controlled on
            ctrl_num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            compact_ham (~pennylane.labs.resource_estimation.CompactHamiltonian): a tensor hypercontracted
                Hamiltonian for which the walk operator is being created
            coeff_precision (float, optional): precision for loading the coefficients of Hamiltonian
            rotation_precision (float, optional): precision for loading the rotation angles for basis rotation
            select_swap_depths (Union[None, int, Iterable(int)],optional): A parameter of :code:`QROM`
                used to trade-off extra qubits for reduced circuit depth. A list can be used to configure the
                ``select_swap_depth`` individually for ``ResourcePrepTHC` and `ResourceSelectTHC` circuits,
                respectively.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.

        """
        gate_list = []

        tensor_rank = compact_ham.params["tensor_rank"]
        m_register = int(np.ceil(np.log2(tensor_rank)))
        coeff_precision = coeff_precision or kwargs["config"]["precision_qubitization_prep"]
        coeff_prec_wires = abs(math.floor(math.log2(coeff_precision)))

        if isinstance(select_swap_depths, int) or select_swap_depths is None:
            select_swap_depths = [select_swap_depths] * 2

        if ctrl_num_ctrl_wires > 1:
            mcx = resource_rep(
                re.ResourceMultiControlledX,
                {
                    "num_ctrl_wires": ctrl_num_ctrl_wires,
                    "num_ctrl_values": ctrl_num_ctrl_values,
                },
            )
            gate_list.append(AllocWires(1))
            gate_list.append(GateCount(mcx, 2))

        select = resource_rep(
            plre.ResourceSelectTHC,
            {
                "compact_ham": compact_ham,
                "rotation_precision": rotation_precision,
                "select_swap_depth": select_swap_depths[1],
            },
        )
        gate_list.append(GateCount(resource_rep(plre.ResourceControlled, {"base_cmpr_op": select, "num_ctrl_wires":1, "num_ctrl_values":0})))

        prep = resource_rep(
            plre.ResourcePrepTHC,
            {
                "compact_ham": compact_ham,
                "coeff_precision": coeff_precision,
                "select_swap_depth": select_swap_depths[0],
            },
        )
        gate_list.append(GateCount(prep))
        gate_list.append(GateCount(resource_rep(plre.ResourceAdjoint, {"base_cmpr_op": prep})))

        # reflection cost
        toffoli = resource_rep(plre.ResourceToffoli)
        gate_list.append(GateCount(toffoli, 2 * m_register + coeff_prec_wires + 4))

        if ctrl_num_ctrl_wires > 1:
            gate_list.append(FreeWires(1))
        elif ctrl_num_ctrl_values > 0:
            gate_list.append(GateCount(resource_rep(plre.ResourceX), 2 * ctrl_num_ctrl_values))

        return gate_list

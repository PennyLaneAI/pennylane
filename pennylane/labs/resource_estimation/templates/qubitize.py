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
from pennylane.queuing import QueuingManager
from pennylane.wires import Wires

# pylint: disable=arguments-differ, protected-access

class ResourceQubitizeTHC(ResourceOperator):
    r"""Resource class for Qubitization of THC Hamiltonian

    Args:
        compact_ham (~pennylane.resource_estimation.CompactHamiltonian): The tensor hypercontracted
            Hamiltonian we will be creating the walk operator for
        coeff_precision (float, optional): precision for loading the coefficients of Hamiltonian
        rotation_precision (float, optional): precision for loading the rotation angles for basis rotation
        compare_precision (float, optional): precision for comparing two numbers
        wires (list[int] or optional): the wires on which the operator acts

    Resource Parameters:
        * compact_ham:

    Resources:
        The resources are calculated based on `arXiv:2011.03494 <https://arxiv.org/abs/2011.03494>`_

    The resources for this operation are computed using:

    **Example**
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
    def __init__(self, compact_ham, coeff_precision=1e-5, rotation_precision=1e-5, compare_precision=1e-3, wires=None):
        self.compact_ham = compact_ham
        self.coeff_precision = coeff_precision
        self.rotation_precision = rotation_precision
        self.compare_precision = compare_precision


        if wires is not None:
            self.wires = Wires(wires)
            self.num_wires = len(self.wires)
        else:
            num_orb = compact_ham.params["num_orbitals"]
            tensor_rank = compact_ham.params["tensor_rank"]
            self.num_wires = num_orb*2 +  2 * int(np.ceil(math.log2(2*tensor_rank+1))) + 1
            self.wires= None
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * compact_ham (~pennylane.resource_estimation.CompactHamiltonian): The tensor hypercontracted
                  Hamiltonian we will be creating the walk operator for.
                * coeff_precision (float, optional): precision for loading the coefficients of Hamiltonian
                * rotation_precision (float, optional): precision for loading the rotation angles for basis rotation
                * compare_precision (float, optional): precision for comparing two numbers
        """
        return {"compact_ham": self.compact_ham,
                  "coeff_precision": self.coeff_precision,
                  "rotation_precision": self.rotation_precision,
                  "compare_precision": self.compare_precision
                  }

    @classmethod
    def resource_rep(cls, compact_ham, coeff_precision, rotation_precision, compare_precision) -> CompressedResourceOp:
        """Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
        compact_ham (~pennylane.resource_estimation.CompactHamiltonian): The tensor hypercontracted
            Hamiltonian we will be creating the walk operator for
        coeff_precision (float, optional): precision for loading the coefficients of Hamiltonian
        rotation_precision (float, optional): precision for loading the rotation angles for basis rotation
        compare_precision (float, optional): precision for comparing two numbers

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        params = {"compact_ham": compact_ham,
                  "coeff_precision": coeff_precision,
                  "rotation_precision": rotation_precision,
                  "compare_precision": compare_precision
                  }
        return CompressedResourceOp(cls, params)

    @classmethod
    def default_resource_decomp(cls, compact_ham, coeff_precision=1e-5, rotation_precision=1e-5, compare_precision= 1e-3, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object represents a quantum gate
        and the number of times it occurs in the decomposition."""

        num_orb = compact_ham.params["num_orbitals"]
        tensor_rank = compact_ham.params["tensor_rank"]

        rot_prec_wires = abs(np.floor(math.log2(rotation_precision)))
        coeff_wires = abs(int(np.floor(math.log2(coeff_precision))))
        compare_precision = abs(int(np.floor(math.log2(compare_precision))))

        # Number of qubits needed for the integrals tensors
        m_register = int(np.ceil(math.log2(2*tensor_rank+1)))

        gate_list = []

        # Select Circuit Fig. 5 in 	arXiv:2011.03494
        # 1 register of rotation precision for loading the angles,
        # 2 for Resource state, 2 wires for spin registers

        gate_list.append(AllocWires(3*rot_prec_wires + 4))

        # Prepare resource state
        phasegrad = resource_rep(plre.ResourcePhaseGradient, {"num_wires": rot_prec_wires})

        # loading the angles controlled on tensor indices
        qrom_angle = resource_rep(plre.ResourceQROM,
            {
                "num_bitstrings": 2**m_register,
                "size_bitstring": rot_prec_wires,
                "clean": False,
                "select_swap_depth": 1,
            }
        )
        gate_list.append(plre.GateCount(qrom_angle,2*(num_orb)))

        # Cliffords for conversion to RZ rotations
        # cost per rotation times 4 for the number of rotations per angle

        h = resource_rep(plre.ResourceHadamard)
        gate_list.append(plre.GateCount(h, 16*(num_orb)))
        z = resource_rep(plre.ResourceZ)
        gate_list.append(plre.GateCount(z, 8*(num_orb)))
        cnot = resource_rep(plre.ResourceCNOT)
        gate_list.append(plre.GateCount(cnot, 16*(num_orb)))
        x = resource_rep(plre.ResourceX)
        gate_list.append(plre.GateCount(x, 8*(num_orb)))
        y = resource_rep(plre.ResourceY)
        gate_list.append(plre.GateCount(y, 8*(num_orb)))

        # Swap gates for swapping the state register based on the state
        cswap = resource_rep(plre.ResourceCSWAP)
        gate_list.append(plre.GateCount(cswap, 4*num_orb*(num_orb)))

        #Phase gradient technique for rotation
        semiadder = resource_rep(
                    plre.ResourceControlled,
                    {
                        "base_cmpr_op": resource_rep(
                            plre.ResourceSemiAdder,
                            {"max_register_size": rot_prec_wires},
                        ),
                        "num_ctrl_wires": 1,
                        "num_ctrl_values": 0,
                    },
                    )

        # 2 semiadders required per R gate, and there are 4 R gates per angle.
        gate_list.append(plre.GateCount(semiadder, 8*(num_orb)))

        # iZ gate in the center of rotations
        gate_list.append(plre.GateCount(x, 2*num_orb))
        gate_list.append(plre.GateCount(y, 2*num_orb))

        # unloading the angles
        gate_list.append(plre.GateCount(qrom_angle,2*(num_orb)))

        # Free spin and rotation precision wires
        gate_list.append(FreeWires(rot_prec_wires + 2))


        # Prepare and unprepare Circuit
        # This is calculating only the Toffolis from Eq. 33
        d = num_orb + tensor_rank * (2*tensor_rank+1)
        nd = int(np.ceil(math.log2(num_orb + tensor_rank * (2*tensor_rank+1))))
        gate_list.append(AllocWires(2*m_register + nd + 2*coeff_wires))

        m = 2*m_register+2+coeff_wires
        toffoli = resource_rep(plre.ResourceToffoli)
        num_toffoli = 28*m_register + 4*compare_precision - 18 + 2*m_register**2 \
                        + 2*coeff_wires + np.ceil(d/16) + m * (16-1) + np.ceil(d/16) + 16

        gate_list.append(plre.GateCount(toffoli, num_toffoli))

        #reflection
        mcx = resource_rep(plre.ResourceMultiControlledX,
                           {"num_ctrl_wires":nd,
                            "num_ctrl_values":nd})
        gate_list.append(plre.GateCount(h,2))
        gate_list.append(plre.GateCount(mcx))

        return gate_list

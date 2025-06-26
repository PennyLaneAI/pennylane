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
from collections import defaultdict
from typing import Dict
import numpy as np

import pennylane as qml
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

# pylint: disable=arguments-differ, protected-access


class ResourceSelectTHC(ResourceOperator):

    def __init__(self, compact_ham, parallel_rotations=None, rotation_precision= 2e-5, wires=None):

        self.compact_ham = compact_ham
        self.parallel_rotations = parallel_rotations
        self.rotation_precision = rotation_precision
        num_orb = compact_ham.params["num_orbitals"]
        tensor_rank = compact_ham.params["tensor_rank"]
        self.num_wires = num_orb*2 +  2 * int(np.ceil(math.log2(2*tensor_rank+1)))
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        return {"compact_ham": self.compact_ham, "parallel_rotations": self.parallel_rotations, "rotation_precision": self.rotation_precision}

    @classmethod
    def resource_rep(cls, compact_ham, parallel_rotations=None, rotation_precision=2e-5) -> CompressedResourceOp:
        params = {"compact_ham": compact_ham, "parallel_rotations": parallel_rotations,"rotation_precision": rotation_precision}
        return CompressedResourceOp(cls, params)

    @classmethod
    def default_resource_decomp(cls, compact_ham, parallel_rotations= None, rotation_precision=2e-5, **kwargs) -> list[GateCount]:

        num_orb = compact_ham.params["num_orbitals"]
        tensor_rank = compact_ham.params["tensor_rank"]

        rot_prec_wires = abs(math.floor(math.log2(rotation_precision)))
        parallel_rotations = parallel_rotations or kwargs["config"]["parallel_rotations"]

        # Number of qubits needed for the integrals tensors
        m_register = int(np.ceil(math.log2(2*tensor_rank+1)))

        gate_list = []
        # Select Circuit Fig. 5 in arXiv:2011.03494
        # Resource state, 2 wires for spin registers, one for checking between one-body and two-body

        gate_list.append(AllocWires(rot_prec_wires + 3))

        # 1) SWAP gate cost (added both for swap and unswap)

        swap = resource_rep(plre.ResourceCSWAP)
        gate_list.append(plre.GateCount(swap, 2*num_orb))


        # 2) Loading angles and rotations
        # Qubits for loading angles, will change based on parallel rotations
        gate_list.append(AllocWires(rot_prec_wires*parallel_rotations))

        phasegrad = resource_rep(plre.ResourcePhaseGradient, {"num_wires": rot_prec_wires})
        gate_list.append(plre.GateCount(phasegrad,1))

        mult_rot = resource_rep(plre.ResourceMultiplexedRotation, {"num_ctrl_wires": m_register})
        num_rotations = (num_orb-1)/parallel_rotations
        gate_list.append(plre.GateCount(mult_rot, 2*num_rotations))

        # 3) Extra QROM cost for unloading the last angle

        qrom_angle = resource_rep(plre.ResourceQROM,
            {
                "num_bitstrings": 2**m_register,
                "size_bitstring": rot_prec_wires,
                "clean": False,
                "select_swap_depth": 1,
            }
        )

        gate_list.append(plre.GateCount(qrom_angle,1))


        # 4) Z operation
        # Z gate in the center of rotations
        cz = resource_rep(plre.ResourceControlled,
                    {
                       "base_cmpr_op": plre.ResourceZ.resource_rep(),
                        "num_ctrl_wires": 1,
                        "num_ctrl_values": 0,
                    })
        gate_list.append(plre.GateCount(cz, 1))

        ccz = resource_rep(plre.ResourceControlled,
                    {
                       "base_cmpr_op": plre.ResourceZ.resource_rep(),
                        "num_ctrl_wires": 2,
                        "num_ctrl_values": 1,
                    })
        gate_list.append(plre.GateCount(ccz, 1))
        return gate_list


class ResourcePrepTHC(ResourceOperator):

    def __init__(self, compact_ham, coeff_precision= 2e-5, wires=None):

        self.compact_ham = compact_ham
        self.coeff_precision = coeff_precision
        num_orb = compact_ham.params["num_orbitals"]
        tensor_rank = compact_ham.params["tensor_rank"]
        self.num_wires = 2 * int(np.ceil(math.log2(2*tensor_rank+1)))
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        return {"compact_ham": self.compact_ham, "coeff_precision": self.coeff_precision}

    @classmethod
    def resource_rep(cls, compact_ham, coeff_precision=2e-5) -> CompressedResourceOp:
        params = {"compact_ham": compact_ham, "coeff_precision": coeff_precision}
        return CompressedResourceOp(cls, params)

    @classmethod
    def default_resource_decomp(cls, compact_ham, coeff_precision=2e-5, **kwargs) -> list[GateCount]:

        num_orb = compact_ham.params["num_orbitals"]
        tensor_rank = compact_ham.params["tensor_rank"]

        coeff_prec_wires = abs(math.floor(math.log2(coeff_precision)))
        compare_precision_wires = 7 # set from paper

        # Number of qubits needed for the integrals tensors
        num_coeff = num_orb + tensor_rank*(2*tensor_rank+1)
        coeff_register = int(math.ceil(math.log2(num_coeff)))
        m_register = int(np.ceil(math.log2(2*tensor_rank+1)))

        gate_list = []

        # Extra wires
        gate_list.append(AllocWires(coeff_register+2*m_register+4))

        # Figure - 3 cost
        # Comparative circuit cost taken from paper
        toffoli = resource_rep(plre.ResourceToffoli)
        gate_list.append(plre.GateCount(toffoli, 8*m_register-7))

        # br-3 bits of precision
        gate_list.append(plre.GateCount(toffoli, 2*(compare_precision_wires-3)))

        # hadamards
        hadamard = resource_rep(plre.ResourceHadamard)
        gate_list.append(plre.GateCount(hadamard, 2*m_register))

        # rotations
        ry = resource_rep(plre.ResourceRY)
        gate_list.append(plre.GateCount(ry, 2))

        #reflection cost
        gate_list.append(plre.GateCount(toffoli,3))

        # inverting about zero cost
        gate_list.append(plre.GateCount(toffoli, 2*m_register-2))

        return gate_list
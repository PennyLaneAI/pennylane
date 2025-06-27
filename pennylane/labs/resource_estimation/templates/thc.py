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

    def __init__(self, compact_ham, rotation_precision= 2e-5, wires=None):

        self.compact_ham = compact_ham
        self.rotation_precision = rotation_precision
        num_orb = compact_ham.params["num_orbitals"]
        tensor_rank = compact_ham.params["tensor_rank"]
        self.num_wires = num_orb*2 +  2 * int(np.ceil(math.log2(2*tensor_rank+1)))
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        return {"compact_ham": self.compact_ham, "rotation_precision": self.rotation_precision}

    @classmethod
    def resource_rep(cls, compact_ham, rotation_precision=2e-5) -> CompressedResourceOp:
        params = {"compact_ham": compact_ham,"rotation_precision": rotation_precision}
        return CompressedResourceOp(cls, params)

    @classmethod
    def default_resource_decomp(cls, compact_ham, rotation_precision=2e-5, **kwargs) -> list[GateCount]:

        num_orb = compact_ham.params["num_orbitals"]
        tensor_rank = compact_ham.params["tensor_rank"]

        rot_prec_wires = abs(math.floor(math.log2(rotation_precision)))

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
        mult_rots = resource_rep(plre.ResourceParallelMultiplexedRotation, {"num_ctrl_wires": m_register, "total_rotations":num_orb-1})
        gate_list.append(plre.GateCount(mult_rots, 4))

        # 3) Extra QROM cost for unloading the last angle

        qrom_angle = resource_rep(plre.ResourceQROM,
            {
                "num_bitstrings": 2**m_register,
                "size_bitstring": rot_prec_wires,
                "clean": False,
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

        # Figure- 4 cost
        gate_list.append(plre.GateCount(hadamard, 2))

        #Contiguous register cost
        gate_list.append(plre.GateCount(toffoli, m_register**2+m_register-1))

        qrom_coeff = resource_rep(plre.ResourceQROM, {"num_bitstrings": num_coeff, "size_bitstring": 2*m_register+2+coeff_prec_wires})
        gate_list.append(plre.GateCount(qrom_coeff, 1))

        # Comparator
        comparator = resource_rep(plre.ResourceComparator, {"num_wires": coeff_prec_wires})
        gate_list.append(comparator)

        # swap cost
        cswap = resource_rep(plre.ResourceCSWAP)
        gate_list.append(plre.GateCount(cswap, 2*m_register))

        # swap the \mu and \nu registers
        gate_list.append(plre.GateCount(cswap, m_register))
        gate_list.append(plre.GateCount(toffoli, 1))

        return gate_list

class ResourceSelectCDF(ResourceOperator):

    def __init__(self, compact_ham, rotation_precision= 2e-5, wires=None):

        self.compact_ham = compact_ham
        self.rotation_precision = rotation_precision
        num_orb = compact_ham.params["num_orbitals"]
        num_fragments = compact_ham.params["num_fragments"]

        num_i_wires = int(math.ceil(math.log2(num_orb)))
        num_m_wires = int(math.ceil(math.log2(num_fragments)))

        self.num_wires = num_orb*2 + 2 + 2*num_i_wires + num_m_wires
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        return {"compact_ham": self.compact_ham, "rotation_precision": self.rotation_precision}

    @classmethod
    def resource_rep(cls, compact_ham, rotation_precision=2e-5) -> CompressedResourceOp:
        params = {"compact_ham": compact_ham,"rotation_precision": rotation_precision}
        return CompressedResourceOp(cls, params)

    @classmethod
    def default_resource_decomp(cls, compact_ham, rotation_precision=2e-5, **kwargs) -> list[GateCount]:

        num_orb = compact_ham.params["num_orbitals"]
        num_fragments = compact_ham.params["num_fragments"]

        rot_prec_wires = abs(math.floor(math.log2(rotation_precision)))

        # Number of qubits needed for the CDF selects
        i_register = int(math.ceil(math.log2(num_orb)))
        j_register = int(math.ceil(math.log2(num_orb)))
        m_register = int(math.ceil(math.log2(num_fragments)))

        gate_list = []
        # Resource state, one for checking between one-body and two-body

        gate_list.append(AllocWires(rot_prec_wires + 1))

        # 1) SWAP gate cost (added both for swap and unswap)

        swap = resource_rep(plre.ResourceCSWAP)
        gate_list.append(plre.GateCount(swap, 2*num_orb))


        # 2) Loading angles and rotations
        # Qubits for loading angles, will change based on parallel rotations
        mult_rots_mi = resource_rep(plre.ResourceParallelMultiplexedRotation, {"num_ctrl_wires": m_register+i_register, "total_rotations":int(num_orb*(num_orb-1)/2)})
        gate_list.append(plre.GateCount(mult_rots_mi, 2))

        mult_rots_mj = resource_rep(plre.ResourceParallelMultiplexedRotation, {"num_ctrl_wires": m_register+j_register, "total_rotations":int(num_orb*(num_orb-1)/2)})
        gate_list.append(plre.GateCount(mult_rots_mj, 2))

        # 3) Extra QROM cost for unloading the last angle

        qrom_angle = resource_rep(plre.ResourceQROM,
            {
                "num_bitstrings": num_fragments*num_orb,
                "size_bitstring": rot_prec_wires,
                "clean": False,
            }
        )

        gate_list.append(plre.GateCount(qrom_angle,2))


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

class ResourcePrepCDF(ResourceOperator):

    def __init__(self, compact_ham, coeff_precision= 2e-5, wires=None):

        self.compact_ham = compact_ham
        self.coeff_precision = coeff_precision
        num_orb = compact_ham.params["num_orbitals"]
        num_frags = compact_ham.params["num_fragments"]
        self.num_wires = 2 * int(np.ceil(math.log2(num_orb)))+ int(np.ceil(math.log2(num_frags)))
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
        num_frags = compact_ham.params["num_fragments"]

        coeff_prec_wires = abs(math.floor(math.log2(coeff_precision)))

        # Number of qubits needed for the integrals tensors
        num_coeff = num_orb + (num_orb*(2*num_orb+1))*num_frags
        coeff_register = int(math.ceil(math.log2(num_coeff)))
        m_register = int(np.ceil(math.log2(num_frags)))
        n_register = int(np.ceil(math.log2(num_orb)))

        gate_list = []

        # Extra wires
        gate_list.append(AllocWires(coeff_register+2*n_register+m_register+2*coeff_prec_wires+6))

        #Uniform superposition
        unif_state = resource_rep(plre.ResourceUniformStatePrep, {"register_size": coeff_register})
        gate_list.append(plre.GateCount(unif_state, 1))

        hadamard = resource_rep(plre.ResourceHadamard)
        gate_list.append(plre.GateCount(hadamard, coeff_prec_wires+1))

        qrom_coeff = resource_rep(plre.ResourceQROM, {"num_bitstrings": num_coeff, "size_bitstring": 4*n_register+2*m_register+coeff_prec_wires+4})
        gate_list.append(plre.GateCount(qrom_coeff, 1))

        qrom_keep = resource_rep(plre.ResourceQROM, {"num_bitstrings": 2**coeff_prec_wires, "size_bitstring": coeff_prec_wires+1})
        gate_list.append(plre.GateCount(qrom_keep, 1))

        comparator = resource_rep(plre.ResourceComparator, {"num_wires": 1})
        gate_list.append(plre.GateCount(comparator,1))

        # swap cost
        cswap = resource_rep(plre.ResourceCSWAP)
        gate_list.append(plre.GateCount(cswap, 2*n_register))

        toffoli = resource_rep(plre.ResourceToffoli)
        gate_list.append(plre.GateCount(toffoli, 1))

        return gate_list

class ResourceSelectSparsePauli(ResourceOperator):

    def __init__(self, compact_ham, wires=None):

        self.compact_ham = compact_ham
        num_orb = compact_ham.params["num_orbitals"]
        register_size = int(np.ceil(np.log2(num_orb)))
        self.num_wires = 4+4*register_size + 2*num_orb
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        return {"compact_ham": self.compact_ham}

    @classmethod
    def resource_rep(cls, compact_ham) -> CompressedResourceOp:
        params = {"compact_ham": compact_ham}
        return CompressedResourceOp(cls, params)

    @classmethod
    def default_resource_decomp(cls, compact_ham, **kwargs) -> list[GateCount]:

        num_orb = compact_ham.params["num_orbitals"]

        gate_list = []

        hadamard = resource_rep(plre.ResourceHadamard)
        gate_list.append(plre.GateCount(hadamard, 4))

        ops_maj1_Y = resource_rep(plre.ResourceProd, {"cmpr_factors_and_counts": ((resource_rep(plre.ResourceY), 2*num_orb), )})
        ops_maj1_Z = resource_rep(plre.ResourceProd, {"cmpr_factors_and_counts": ((resource_rep(plre.ResourceZ), int((2*num_orb)*(2*num_orb - 1)/2)),)})

        unary_gate_maj1 = resource_rep(plre.ResourceSelect, {"cmpr_ops": [ops_maj1_Y, ops_maj1_Z]})
        controlled_unary_maj1 = resource_rep(plre.ResourceControlled, {"base_cmpr_op": unary_gate_maj1, "num_ctrl_wires":1, "num_ctrl_values":0})
        gate_list.append(plre.GateCount(controlled_unary_maj1, 2))

        ops_maj0_X = resource_rep(plre.ResourceProd, {"cmpr_factors_and_counts": ((resource_rep(plre.ResourceX), 2*num_orb), )})
        ops_maj0_Z = resource_rep(plre.ResourceProd, {"cmpr_factors_and_counts": ((resource_rep(plre.ResourceZ), int((2*num_orb)*(2*num_orb - 1)/2)),)})
        unary_gate_maj0 = resource_rep(plre.ResourceSelect, {"cmpr_ops": [ops_maj0_X, ops_maj0_Z]})
        controlled_unary_maj0 = resource_rep(plre.ResourceControlled, {"base_cmpr_op": unary_gate_maj0, "num_ctrl_wires":1, "num_ctrl_values":0})
        gate_list.append(plre.GateCount(controlled_unary_maj0, 2))

        l_elbow = resource_rep(plre.ResourceTempAND)
        r_elbow = resource_rep(plre.ResourceAdjoint, {"base_cmpr_op": l_elbow})

        gate_list.append(plre.GateCount(l_elbow, 1))
        gate_list.append(plre.GateCount(r_elbow, 1))

        s_gate = resource_rep(plre.ResourceS())
        gate_list.append(plre.GateCount(s_gate, 2))

        return gate_list


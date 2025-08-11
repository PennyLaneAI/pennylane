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

from pennylane.labs import resource_estimation as plre
from pennylane.labs.resource_estimation.qubit_manager import AllocWires
from pennylane.labs.resource_estimation.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    resource_rep,
)


# pylint: disable=arguments-differ, unused-argument
class ResourceUniformStatePrep(ResourceOperator):
    r"""Resource class for preparing a uniform superposition.

    This operation prepares a uniform superposition over a given number of
    basis states. The uniform superposition is defined as:

    .. math::

        \frac{1}{\sqrt{l}} \sum_{i=0}^{l} |i\rangle

    where :math:`l` is the number of states.

    This operation uses ``Hadamard`` gates to create the uniform superposition when
    the number of states is a power of two. If the number of states is not a power of two,
    amplitude amplification technique defined in
    `arXiv:1805.03662 <https://arxiv.org/pdf/1805.03662>`_ is used.

    Args:
        num_states (int): the number of states in the uniform superposition
        wires (Sequence[int], optional): the wires the operation acts on

    Resources:
        The resources are obtained from Figure 12 in `arXiv:1805.03662 <https://arxiv.org/pdf/1805.03662>`_.
        The circuit uses amplitude amplification to prepare a uniform superposition over :math:`l`
        basis states.

    **Example**

    The resources for this operation are computed using:

    >>> unif_state_prep = plre.ResourceUniformStatePrep(10)
    >>> print(plre.estimate_resources(unif_state_prep))
    --- Resources: ---
     Total qubits: 5
     Total gates : 124
     Qubit breakdown:
      clean qubits: 1, dirty qubits: 0, algorithmic qubits: 4
     Gate breakdown:
      {'Hadamard': 16, 'X': 12, 'CNOT': 4, 'Toffoli': 4, 'T': 88}
    """

    resource_keys = {"num_states"}

    def __init__(self, num_states, wires=None):
        self.num_states = num_states
        k = (num_states & -num_states).bit_length() - 1
        L = num_states // (2**k)
        if L == 1:
            self.num_wires = k
        else:
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
        r"""Returns a list representing the resources of the operator. Each object in the list represents a gate and the
        number of times it occurs in the circuit.

        Args:
            num_states (int): the number of states over which the uniform superposition is being prepared

        Resources:
            The resources are obtained from Figure 12 in `arXiv:1805.03662 <https://arxiv.org/pdf/1805.03662>`_.
            The circuit uses amplitude amplification to prepare a uniform superposition over :math:`l` basis states.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """

        gate_lst = []
        k = (num_states & -num_states).bit_length() - 1
        L = num_states // (2**k)
        if L == 1:
            gate_lst.append(GateCount(resource_rep(plre.ResourceHadamard), k))
            return gate_lst

        logl = int(math.ceil(math.log2(L)))
        gate_lst.append(GateCount(resource_rep(plre.ResourceHadamard), k + 3 * logl))
        gate_lst.append(
            GateCount(
                resource_rep(plre.ResourceIntegerComparator, {"value": L, "register_size": logl}), 1
            )
        )
        gate_lst.append(GateCount(resource_rep(plre.ResourceRZ), 2))
        gate_lst.append(
            GateCount(
                resource_rep(
                    plre.ResourceAdjoint,
                    {
                        "base_cmpr_op": resource_rep(
                            plre.ResourceIntegerComparator, {"value": L, "register_size": logl}
                        )
                    },
                ),
                1,
            )
        )

        return gate_lst


class ResourceAliasSampling(ResourceOperator):
    r"""Resource class for preparing a state using coherent alias sampling.

    Args:
        num_coeffs (int): the number of unique coefficients in the state
        precision (float): the precision with which the coefficients are loaded
        wires (Sequence[int], optional): the wires the operation acts on

    Resources:
        The resources are obtained from Section III D in `arXiv:1805.03662 <https://arxiv.org/pdf/1805.03662>`_.
        The circuit uses coherent alias sampling to prepare a state with the given coefficients.

    **Example**

    The resources for this operation are computed using:

    >>> alias_sampling = plre.ResourceAliasSampling(num_coeffs=100)
    >>> print(plre.estimate_resources(alias_sampling))
    --- Resources: ---
     Total qubits: 81
     Total gates : 6.157E+3
     Qubit breakdown:
      clean qubits: 6, dirty qubits: 68, algorithmic qubits: 7
     Gate breakdown:
      {'Hadamard': 730, 'X': 479, 'CNOT': 4.530E+3, 'Toffoli': 330, 'T': 88}
    """

    resource_keys = {"num_coeffs", "precision"}

    def __init__(self, num_coeffs, precision=None, wires=None):
        self.num_coeffs = num_coeffs
        self.precision = precision
        self.num_wires = int(math.ceil(math.log2(num_coeffs)))
        super().__init__(wires=wires)

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * num_coeffs (int): the number of unique coefficients in the state
                * precision (float): the precision with which the coefficients are loaded

        """
        return {"num_coeffs": self.num_coeffs, "precision": self.precision}

    @classmethod
    def resource_rep(cls, num_coeffs, precision=None) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        return CompressedResourceOp(cls, {"num_coeffs": num_coeffs, "precision": precision})

    @classmethod
    def default_resource_decomp(cls, num_coeffs, precision=None, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object in the list represents a gate and the
        number of times it occurs in the circuit.

        Args:
            num_coeffs (int): the number of unique coefficients in the state
            precision (float): the precision with which the coefficients are loaded

        Resources:
            The resources are obtained from Section III D in `arXiv:1805.03662 <https://arxiv.org/pdf/1805.03662>`_.
            The circuit uses coherent alias sampling to prepare a state with the given coefficients.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """

        gate_lst = []

        logl = int(math.ceil(math.log2(num_coeffs)))
        precision = precision or kwargs["config"]["precision_alias_sampling"]

        num_prec_wires = abs(math.floor(math.log2(precision)))

        gate_lst.append(AllocWires(logl + 2 * num_prec_wires + 1))

        gate_lst.append(
            GateCount(resource_rep(plre.ResourceUniformStatePrep, {"num_states": num_coeffs}), 1)
        )
        gate_lst.append(GateCount(resource_rep(plre.ResourceHadamard), num_prec_wires))
        gate_lst.append(
            GateCount(
                resource_rep(
                    plre.ResourceQROM,
                    {"num_bitstrings": num_coeffs, "size_bitstring": logl + num_prec_wires},
                ),
                1,
            )
        )
        gate_lst.append(
            GateCount(
                resource_rep(
                    plre.ResourceRegisterComparator,
                    {"first_register": num_prec_wires, "second_register": num_prec_wires},
                ),
                1,
            )
        )
        gate_lst.append(GateCount(resource_rep(plre.ResourceCSWAP), logl))

        return gate_lst


class ResourcePrepTHC(ResourceOperator):

    def __init__(self, compact_ham, coeff_precision= 2e-5, select_swap_depth=None, wires=None):

        self.compact_ham = compact_ham
        self.coeff_precision = coeff_precision
        self.select_swap_depth = select_swap_depth
        tensor_rank = compact_ham.params["tensor_rank"]
        self.num_wires = 2 * int(np.ceil(math.log2(tensor_rank+1)))
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        return {"compact_ham": self.compact_ham, "coeff_precision": self.coeff_precision, "select_swap_depth": self.select_swap_depth}

    @classmethod
    def resource_rep(cls, compact_ham, coeff_precision=2e-5, select_swap_depth=None) -> CompressedResourceOp:
        params = {"compact_ham": compact_ham, "coeff_precision": coeff_precision, "select_swap_depth":select_swap_depth}
        return CompressedResourceOp(cls, params)

    @classmethod
    def default_resource_decomp(cls, compact_ham, coeff_precision=2e-5, select_swap_depth=None, **kwargs) -> list[GateCount]:

        num_orb = compact_ham.params["num_orbitals"]
        tensor_rank = compact_ham.params["tensor_rank"]

        coeff_prec_wires = abs(math.floor(math.log2(coeff_precision)))
        compare_precision_wires = abs(math.floor(math.log2(coeff_precision)))

        num_coeff = num_orb + tensor_rank*(tensor_rank+1)
        coeff_register = int(math.ceil(math.log2(num_coeff)))
        m_register = int(np.ceil(math.log2(tensor_rank+1)))

        gate_list = []

        gate_list.append(AllocWires(coeff_register+2*m_register+2*compare_precision_wires+6))

        hadamard = resource_rep(plre.ResourceHadamard)
        gate_list.append(plre.GateCount(hadamard, 2*m_register))

        # Figure - 3
        toffoli = resource_rep(plre.ResourceToffoli)
        gate_list.append(plre.GateCount(toffoli, 4*m_register-4))

        ccz = resource_rep(plre.ResourceCCZ)
        gate_list.append(plre.GateCount(resource_rep(plre.ResourceControlled, {"base_cmpr_op": ccz, "num_ctrl_wires":1, "num_ctrl_values":0}), 1))
        gate_list.append(plre.GateCount(toffoli, 2))

        gate_list.append(plre.GateCount(hadamard, 2*m_register))

        gate_list.append(AllocWires(compare_precision_wires))
        gate_list.append(plre.GateCount(toffoli, 2*(compare_precision_wires-3)))
        gate_list.append(FreeWires(compare_precision_wires))

        gate_list.append(plre.GateCount(ccz, 2*m_register-1))

        hadamard = resource_rep(plre.ResourceHadamard)
        gate_list.append(plre.GateCount(hadamard, 2*m_register))

        gate_list.append(plre.GateCount(toffoli, 4*m_register-4))

        mcx = resource_rep(plre.ResourceMultiControlledX, {"num_ctrl_wires": 3, "num_ctrl_values": 0})
        gate_list.append(plre.GateCount(mcx, 1))
        gate_list.append(plre.GateCount(toffoli, 2))

        x = resource_rep(plre.ResourceX)
        gate_list.append(plre.GateCount(x, 2))

        # Figure- 4(Subprepare Circuit)
        gate_list.append(plre.GateCount(hadamard, compare_precision_wires + 1))

        #Contiguous register cost
        gate_list.append(plre.GateCount(toffoli, m_register**2+m_register-1))

        qrom_coeff = resource_rep(plre.ResourceQROM, {"num_bitstrings": num_coeff, "size_bitstring": 2*m_register+2+coeff_prec_wires, "clean": False,"select_swap_depth": select_swap_depth})
        gate_list.append(plre.GateCount(qrom_coeff, 1))

        comparator = resource_rep(plre.ResourceRegisterComparator, {"a_num_qubits": coeff_prec_wires, "b_num_qubits": coeff_prec_wires, "geq":False})
        gate_list.append(plre.GateCount(comparator))

        cz = resource_rep(plre.ResourceCZ)
        gate_list.append(plre.GateCount(cz, 2))
        gate_list.append(plre.GateCount(x, 2))

        cswap = resource_rep(plre.ResourceCSWAP)
        gate_list.append(plre.GateCount(cswap, 2*m_register))

        gate_list.append(plre.GateCount(cswap, m_register))
        gate_list.append(plre.GateCount(toffoli, 1))

        return gate_list

    @classmethod
    def default_adjoint_resource_decomp(cls, compact_ham, coeff_precision=2e-5, select_swap_depth=None, **kwargs) -> list[GateCount]:

        num_orb = compact_ham.params["num_orbitals"]
        tensor_rank = compact_ham.params["tensor_rank"]

        coeff_prec_wires = abs(math.floor(math.log2(coeff_precision)))

        num_coeff = num_orb + tensor_rank*(tensor_rank+1)/2
        coeff_register = int(math.ceil(math.log2(num_coeff)))
        m_register = int(np.ceil(math.log2(tensor_rank+1)))
        gate_list = []

        hadamard = resource_rep(plre.ResourceHadamard)
        gate_list.append(plre.GateCount(hadamard, 2*m_register))

        # Figure - 3
        toffoli = resource_rep(plre.ResourceToffoli)
        gate_list.append(plre.GateCount(toffoli, 4*m_register-4))

        ccz = resource_rep(plre.ResourceCCZ)
        gate_list.append(plre.GateCount(resource_rep(plre.ResourceControlled, {"base_cmpr_op": ccz, "num_ctrl_wires":1, "num_ctrl_values":0}), 1))
        gate_list.append(plre.GateCount(toffoli, 2))

        gate_list.append(plre.GateCount(hadamard, 2*m_register))

        gate_list.append(AllocWires(compare_precision_wires))
        gate_list.append(plre.GateCount(toffoli, 2*(compare_precision_wires-3)))
        gate_list.append(FreeWires(compare_precision_wires))

        gate_list.append(plre.GateCount(ccz, 2*m_register-1))

        hadamard = resource_rep(plre.ResourceHadamard)
        gate_list.append(plre.GateCount(hadamard, 2*m_register))

        gate_list.append(plre.GateCount(toffoli, 4*m_register-4))

        mcx = resource_rep(plre.ResourceMultiControlledX, {"num_ctrl_wires": 3, "num_ctrl_values": 0})
        gate_list.append(plre.GateCount(mcx, 1))
        gate_list.append(plre.GateCount(toffoli, 2))

        x = resource_rep(plre.ResourceX)
        gate_list.append(plre.GateCount(x, 2))

        # Figure- 4 (Subprepare Circuit)
        gate_list.append(plre.GateCount(hadamard, coeff_prec_wires + 1))

        #Contiguous register cost
        gate_list.append(plre.GateCount(toffoli, m_register**2+m_register-1))

        qrom_adj= resource_rep(plre.ResourceAdjoint, {"base_cmpr_op": resource_rep(plre.ResourceQROM, {"num_bitstrings": num_coeff, "size_bitstring": 2*m_register+2+coeff_prec_wires, "clean": False, "select_swap_depth": select_swap_depth})})
        gate_list.append(plre.GateCount(qrom_adj, 1))

        cz = resource_rep(plre.ResourceCZ)
        gate_list.append(plre.GateCount(cz, 2))
        gate_list.append(plre.GateCount(x, 2))

        cswap = resource_rep(plre.ResourceCSWAP)
        gate_list.append(plre.GateCount(cswap, 2*m_register))

        gate_list.append(plre.GateCount(cswap, m_register))
        gate_list.append(plre.GateCount(toffoli, 1))

        # Free Prepare Wires
        gate_list.append(FreeWires(coeff_register+2*m_register+2*coeff_prec_wires+6))

        return gate_list

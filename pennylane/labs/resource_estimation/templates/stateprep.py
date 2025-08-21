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
from pennylane.labs.resource_estimation.qubit_manager import AllocWires, FreeWires
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
    r"""Resource class for preparing the state for THC Hamiltonian.

    Args:
        compact_ham (~pennylane.labs.resource_estimation.CompactHamiltonian): a tensor hypercontracted
            Hamiltonian for which the state is being prepared
        coeff_precision_bits (int, optional): The number of bits used to represent the precision for loading
            the coefficients of Hamiltonian. If :code:`None` is provided the default value from the
            :code:`resource_config` is used.
        select_swap_depth (int, optional): A natural number that determines if data
            will be loaded in parallel by adding more rows following Figure 1.C of `Low et al. (2024) <https://arxiv.org/pdf/1812.00954>`_.
            Defaults to :code:`None`, which internally determines the optimal depth.
        wires (list[int] or optional): the wires on which the operator acts

    Resources:
        The resources are calculated based on Figures 3 and 4 in `arXiv:2011.03494 <https://arxiv.org/abs/2011.03494>`_

    **Example**

    The resources for this operation are computed using:

    >>> compact_ham = plre.CompactHamiltonian.thc(num_orbitals=20, tensor_rank=40)
    >>> res = plre.estimate_resources(plre.ResourcePrepTHC(compact_ham))
    >>> print(res)
    --- Resources: ---
     Total qubits: 185
     Total gates : 1.485E+4
     Qubit breakdown:
      clean qubits: 28, dirty qubits: 145, algorithmic qubits: 12
     Gate breakdown:
      {'Hadamard': 797, 'Toffoli': 467, 'CNOT': 1.307E+4, 'X': 512}

    """

    def __init__(self, compact_ham, coeff_precision_bits=None, select_swap_depth=None, wires=None):

        if compact_ham.method_name != "thc":
            raise TypeError(
                f"Unsupported Hamiltonian representation for ResourcePrepTHC."
                f"This method works with thc Hamiltonian, {compact_ham.method_name} provided"
            )
        self.compact_ham = compact_ham
        self.coeff_precision_bits = coeff_precision_bits
        self.select_swap_depth = select_swap_depth
        tensor_rank = compact_ham.params["tensor_rank"]
        self.num_wires = 2 * int(math.ceil(math.log2(tensor_rank + 1)))
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * compact_ham (~pennylane.labs.resource_estimation.CompactHamiltonian): a tensor hypercontracted
                    Hamiltonian for which the state is being prepared
                * coeff_precision_bits (int, optional): The number of bits used to represent the precision for loading
                    the coefficients of Hamiltonian. If :code:`None` is provided the default value from the
                    :code:`resource_config` is used.
                * select_swap_depth (int, optional): A natural number that determines if data
                    will be loaded in parallel by adding more rows following Figure 1.C of `Low et al. (2024) <https://arxiv.org/pdf/1812.00954>`_.
                    Defaults to :code:`None`, which internally determines the optimal depth.
        """
        return {
            "compact_ham": self.compact_ham,
            "coeff_precision_bits": self.coeff_precision_bits,
            "select_swap_depth": self.select_swap_depth,
        }

    @classmethod
    def resource_rep(
        cls, compact_ham, coeff_precision_bits=None, select_swap_depth=None
    ) -> CompressedResourceOp:
        """Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            compact_ham (~pennylane.labs.resource_estimation.CompactHamiltonian): a tensor hypercontracted
                Hamiltonian for which the state is being prepared
            coeff_precision_bits (int, optional): The number of bits used to represent the precision for loading
                the coefficients of Hamiltonian. If :code:`None` is provided the default value from the
                :code:`resource_config` is used.
            select_swap_depth (int, optional): A natural number that determines if data
                will be loaded in parallel by adding more rows following Figure 1.C of `Low et al. (2024) <https://arxiv.org/pdf/1812.00954>`_.
                Defaults to :code:`None`, which internally determines the optimal depth.

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        params = {
            "compact_ham": compact_ham,
            "coeff_precision_bits": coeff_precision_bits,
            "select_swap_depth": select_swap_depth,
        }
        return CompressedResourceOp(cls, params)

    @classmethod
    def default_resource_decomp(
        cls, compact_ham, coeff_precision_bits=None, select_swap_depth=None, **kwargs
    ) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object represents a quantum gate
        and the number of times it occurs in the decomposition.

        Args:
            compact_ham (~pennylane.labs.resource_estimation.CompactHamiltonian): a tensor hypercontracted
                Hamiltonian for which the walk operator is being created
            coeff_precision_bits (int, optional): The number of bits used to represent the precision for loading
                the coefficients of Hamiltonian. If :code:`None` is provided the default value from the
                :code:`resource_config` is used.
            select_swap_depth (int, optional): A natural number that determines if data
                will be loaded in parallel by adding more rows following Figure 1.C of
                `Low et al. (2024) <https://arxiv.org/pdf/1812.00954>`_.
                Defaults to :code:`None`, which internally determines the optimal depth.

        Resources:
            The resources are calculated based on Figures 3 and 4 in `arXiv:2011.03494 <https://arxiv.org/abs/2011.03494>`_

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.

        """
        num_orb = compact_ham.params["num_orbitals"]
        tensor_rank = compact_ham.params["tensor_rank"]

        coeff_precision_bits = coeff_precision_bits or kwargs["config"]["qubitization_coeff_bits"]

        num_coeff = num_orb + tensor_rank * (tensor_rank + 1) / 2  # N+M(M+1)/2
        coeff_register = int(math.ceil(math.log2(num_coeff)))
        m_register = int(math.ceil(math.log2(tensor_rank + 1)))

        gate_list = []

        # 6 ancillas account for 2 spin registers, 1 for rotation on ancilla, 1 flag for success of inequality,
        # 1 flag for one-body vs two-body and 1 to control swap of \mu and \nu registers.
        gate_list.append(AllocWires(coeff_register + 2 * m_register + 2 * coeff_precision_bits + 6))

        hadamard = resource_rep(plre.ResourceHadamard)

        gate_list.append(plre.GateCount(hadamard, 2 * m_register))

        # Figure - 3

        # Inquality tests
        toffoli = resource_rep(plre.ResourceToffoli)
        gate_list.append(plre.GateCount(toffoli, 4 * m_register - 4))

        # Reflection on 5 registers
        ccz = resource_rep(plre.ResourceCCZ)
        gate_list.append(
            plre.GateCount(
                resource_rep(
                    plre.ResourceControlled,
                    {"base_cmpr_op": ccz, "num_ctrl_wires": 1, "num_ctrl_values": 0},
                ),
                1,
            )
        )
        gate_list.append(plre.GateCount(toffoli, 2))

        gate_list.append(plre.GateCount(hadamard, 2 * m_register))

        # Rotate and invert the rotation of ancilla to obtain amplitude of success
        gate_list.append(AllocWires(coeff_precision_bits))
        gate_list.append(plre.GateCount(toffoli, 2 * (coeff_precision_bits - 3)))
        gate_list.append(FreeWires(coeff_precision_bits))

        # Reflecting about the success amplitude
        gate_list.append(plre.GateCount(ccz, 2 * m_register - 1))

        gate_list.append(plre.GateCount(hadamard, 2 * m_register))

        # Inequality tests
        gate_list.append(plre.GateCount(toffoli, 4 * m_register - 4))

        # Checking that inequality is satisfied
        mcx = resource_rep(
            plre.ResourceMultiControlledX, {"num_ctrl_wires": 3, "num_ctrl_values": 0}
        )
        gate_list.append(plre.GateCount(mcx, 1))
        gate_list.append(plre.GateCount(toffoli, 2))

        x = resource_rep(plre.ResourceX)
        gate_list.append(plre.GateCount(x, 2))

        # Figure- 4(Subprepare Circuit)
        gate_list.append(plre.GateCount(hadamard, coeff_precision_bits + 1))

        # Contiguous register cost Eq.29
        gate_list.append(plre.GateCount(toffoli, m_register**2 + m_register - 1))

        # QROM for keep values Eq.31
        qrom_coeff = resource_rep(
            plre.ResourceQROM,
            {
                "num_bitstrings": num_coeff,
                "size_bitstring": 2 * m_register + 2 + coeff_precision_bits,
                "clean": False,
                "select_swap_depth": select_swap_depth,
            },
        )
        gate_list.append(plre.GateCount(qrom_coeff, 1))

        # Inequality test between alt and keep registers
        comparator = resource_rep(
            plre.ResourceRegisterComparator,
            {
                "first_register": coeff_precision_bits,
                "second_register": coeff_precision_bits,
                "geq": False,
            },
        )
        gate_list.append(plre.GateCount(comparator))

        cz = resource_rep(plre.ResourceCZ)
        gate_list.append(plre.GateCount(cz, 2))
        gate_list.append(plre.GateCount(x, 2))

        # Swap \mu and \nu registers with alt registers
        cswap = resource_rep(plre.ResourceCSWAP)
        gate_list.append(plre.GateCount(cswap, 2 * m_register))

        # Swap \mu and \nu registers controlled on |+> state and success of inequality
        gate_list.append(plre.GateCount(cswap, m_register))
        gate_list.append(plre.GateCount(toffoli, 1))

        return gate_list

    @classmethod
    def default_adjoint_resource_decomp(
        cls, compact_ham, coeff_precision_bits=None, select_swap_depth=None, **kwargs
    ) -> list[GateCount]:
        r"""Returns a list representing the resources of the adjoint of the operator. Each object represents a quantum gate
        and the number of times it occurs in the decomposition.

        Args:
            compact_ham (~pennylane.labs.resource_estimation.CompactHamiltonian): a tensor hypercontracted
                Hamiltonian for which the walk operator is being created
            coeff_precision_bits (int, optional): The number of bits used to represent the precision for loading
                the coefficients of Hamiltonian. If :code:`None` is provided the default value from the
                :code:`resource_config` is used.
            select_swap_depth (int, optional): A natural number that determines if data
                will be loaded in parallel by adding more rows following Figure 1.C of
                `Low et al. (2024) <https://arxiv.org/pdf/1812.00954>`_.
                Defaults to :code:`None`, which internally determines the optimal depth.

        Resources:
            The resources are calculated based on Figures 3 and 4 in `arXiv:2011.03494 <https://arxiv.org/abs/2011.03494>`_

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        num_orb = compact_ham.params["num_orbitals"]
        tensor_rank = compact_ham.params["tensor_rank"]

        coeff_precision_bits = coeff_precision_bits or kwargs["config"]["qubitization_coeff_bits"]

        num_coeff = num_orb + tensor_rank * (tensor_rank + 1) / 2
        coeff_register = int(math.ceil(math.log2(num_coeff)))
        m_register = int(math.ceil(math.log2(tensor_rank + 1)))
        gate_list = []

        hadamard = resource_rep(plre.ResourceHadamard)
        gate_list.append(plre.GateCount(hadamard, 2 * m_register))

        # Figure - 3

        # Inquality tests
        toffoli = resource_rep(plre.ResourceToffoli)
        gate_list.append(plre.GateCount(toffoli, 4 * m_register - 4))

        # Reflection on 5 registers
        ccz = resource_rep(plre.ResourceCCZ)
        gate_list.append(
            plre.GateCount(
                resource_rep(
                    plre.ResourceControlled,
                    {"base_cmpr_op": ccz, "num_ctrl_wires": 1, "num_ctrl_values": 0},
                ),
                1,
            )
        )
        gate_list.append(plre.GateCount(toffoli, 2))

        gate_list.append(plre.GateCount(hadamard, 2 * m_register))

        # Rotate and invert the rotation of ancilla to obtain amplitude of success
        gate_list.append(AllocWires(coeff_precision_bits))
        gate_list.append(plre.GateCount(toffoli, 2 * (coeff_precision_bits - 3)))
        gate_list.append(FreeWires(coeff_precision_bits))

        # Reflecting about the success amplitude
        gate_list.append(plre.GateCount(ccz, 2 * m_register - 1))

        gate_list.append(plre.GateCount(hadamard, 2 * m_register))

        # Inequality tests
        gate_list.append(plre.GateCount(toffoli, 4 * m_register - 4))

        # Checking that inequality is satisfied
        mcx = resource_rep(
            plre.ResourceMultiControlledX, {"num_ctrl_wires": 3, "num_ctrl_values": 0}
        )
        gate_list.append(plre.GateCount(mcx, 1))
        gate_list.append(plre.GateCount(toffoli, 2))

        x = resource_rep(plre.ResourceX)
        gate_list.append(plre.GateCount(x, 2))

        # Figure- 4 (Subprepare Circuit)
        gate_list.append(plre.GateCount(hadamard, coeff_precision_bits + 1))

        # Contiguous register cost
        gate_list.append(plre.GateCount(toffoli, m_register**2 + m_register - 1))

        # Adjoint of QROM for keep values Eq.32
        qrom_adj = resource_rep(
            plre.ResourceAdjoint,
            {
                "base_cmpr_op": resource_rep(
                    plre.ResourceQROM,
                    {
                        "num_bitstrings": num_coeff,
                        "size_bitstring": 2 * m_register + 2 + coeff_precision_bits,
                        "clean": False,
                        "select_swap_depth": select_swap_depth,
                    },
                )
            },
        )
        gate_list.append(plre.GateCount(qrom_adj, 1))

        cz = resource_rep(plre.ResourceCZ)
        gate_list.append(plre.GateCount(cz, 2))
        gate_list.append(plre.GateCount(x, 2))

        # Swap \mu and \nu registers with alt registers
        cswap = resource_rep(plre.ResourceCSWAP)
        gate_list.append(plre.GateCount(cswap, 2 * m_register))

        # Swap \mu and \nu registers controlled on |+> state and success of inequality
        gate_list.append(plre.GateCount(cswap, m_register))
        gate_list.append(plre.GateCount(toffoli, 1))

        # Free Prepare Wires
        # 6 ancillas account for 2 spin registers, 1 for rotation on ancilla, 1 flag for success of inequality,
        # 1 flag for one-body vs two-body and 1 to control swap of \mu and \nu registers.
        gate_list.append(FreeWires(coeff_register + 2 * m_register + 2 * coeff_precision_bits + 6))

        return gate_list

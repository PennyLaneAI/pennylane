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
r"""Compact input classes for resource estimation."""
# pylint: disable=too-many-arguments,
import math

from pennylane.labs import resource_estimation as re


class CompactState:
    r"""A compact representation for the state of a quantum system.

    Args:
        num_qubits (int): number of qubits used to represent the state
        num_coeffs (int): The number of coefficients in the linear combination of
            computational basis states representation of the target state.
        precision (float): a tolerance for approximation when preparing the state
        num_work_wires (int): number of additional work qubits available to prepare state
        num_bit_flips (int): The number of qubits in the :math:`|1\rangle` state for
            preparing the target basis state.
        positive_and_real (bool): A flag which is :code:`True` when all coefficients are
            real and positive for the target state.

    .. details::
        :title: Usage Details

        The :code:`CompactState` class is designed to be an alternative input to preparing
        a full statevector. It should be used in combination with a statepreparation template
        for more efficient state preparation when performing resource estimation.

        .. code-block:: python

            from pennylane.labs import resource_estimation as re

            compact_statevector = re.CompactState.from_state_vector(num_qubits=20, num_coeffs=2**20)

            def circ():
                re.ResourceStatePrep(compact_statevector, wires=range(20))
                return

        The resources can then be extracted as usual:

        >>> res = re.get_resources(circ)()
        >>> print(res)
        wires: 20
        gates: 75497303
        gate_types:
        {'T': 71303083, 'CNOT': 4194220}

    """

    def __init__(
        self,
        num_qubits=None,
        num_coeffs=None,
        precision=None,
        num_work_wires=None,
        num_bit_flips=None,
        positive_and_real=None,
    ):
        self.num_qubits = num_qubits
        self.num_coeffs = num_coeffs
        self.precision = precision
        self.num_work_wires = num_work_wires
        self.num_bit_flips = num_bit_flips
        self.positive_and_real = positive_and_real

    def __eq__(self, other: object) -> bool:
        return all(
            (
                (self.num_qubits == other.num_qubits),
                (self.num_coeffs == other.num_coeffs),
                (self.precision == other.precision),
                (self.num_work_wires == other.num_work_wires),
                (self.num_bit_flips == other.num_bit_flips),
                (self.positive_and_real == other.positive_and_real),
            )
        )

    @classmethod
    def from_mps(cls, num_mps_matrices, max_bond_dim):
        """Instantiate a CompactState representing an MPS.

        Args:
            num_mps_matrices (int): number of tensors in the MPS
            max_bond_dim (int): the maximum bond dimension of the MPS

        Returns:
            CompactState: the compact state representing the MPS
        """
        num_work_wires = math.ceil(math.log2(max_bond_dim))
        return cls(num_qubits=num_mps_matrices, num_work_wires=num_work_wires)

    @classmethod
    def from_bitstring(cls, num_qubits, num_bit_flips):
        r"""Instantiate a CompactState representing a computational basis state.

        Args:
            num_qubits (int): number of qubits used to represent the state
            num_bit_flips (int): The number of qubits in the :math:`|1\rangle` state for
                preparing the target basis state.

        Returns:
            CompactState: the compact state representing the bitstring
        """
        return cls(
            num_qubits=num_qubits,
            num_coeffs=1,
            num_bit_flips=num_bit_flips,
        )

    @classmethod
    def from_state_vector(
        cls,
        num_qubits,
        num_coeffs,
        precision=1e-3,
        num_work_wires=0,
        positive_and_real=False,
    ):
        r"""Instantiate a CompactState representing a statevector (dense or sparse).

        Args:
            num_qubits (int): number of qubits used to represent the state
            num_coeffs (int): The number of coefficients in the linear combination of
                computational basis states representation of the target state.
            precision (float, optional): A tolerance for approximation when preparing
                the state. Defaults to 1e-3.
            num_work_wires (int, optional): The number of additional work qubits available
                to prepare state. Defaults to 0.
            positive_and_real (bool, optional): A flag which is :code:`True` when all
                coefficients are real and positive for the target state. Defaults to False.

        Returns:
            CompactState: the compact state representing the statevector
        """
        return cls(
            num_qubits=num_qubits,
            num_coeffs=num_coeffs,
            precision=precision,
            num_work_wires=num_work_wires,
            positive_and_real=positive_and_real,
        )

class CompactHamiltonian:
    r"""A compact representation for the state of a hamiltonian."""

    def __init__(
        self,
        num_qubits,
        num_terms,
        cost_per_term,
        cost_per_exp_term,
    ):
        return
    
    @classmethod
    def from_pauli_lcu(
        cls,
        num_qubits: int,
        num_terms: int = None,
        k_local: int = None,
        pauli_term_dist: dict = None,
        trotter_fragments: tuple = None,
    ):
        x = re.ResourceX.resource_rep()
        y = re.ResourceY.resource_rep()
        z = re.ResourceZ.resource_rep()

        if pauli_term_dist is None: 
            k_local = k_local or num_qubits
            freq = k_local // 3

            cmpr_factors = [x] * freq + [y] * freq + [z] * (k_local - 2*freq)
            cost_one_term = re.ResourceProd.resource_rep(cmpr_factors)

            cost_per_term = [cost_one_term] * num_terms


        return cls(num_qubits, num_terms)
    
        # if self.lcu_type == "pauli":
        #     cost_per_term = {}
        #     x, y, z = (re.ResourceX.resource_rep(), re.ResourceY.resource_rep(), re.ResourceZ.resource_rep())

        #     freq = self.k_local // 3

        #     cost_per_term[x] = freq
        #     cost_per_term[z] = freq
        #     cost_per_term[y] = self.k_local - 2*freq

        #     avg_pword = freq * "X" + freq * "Z" + (self.k_local - 2*freq) * "Y"
        #     cost_per_exp_term = {re.ResourcePauliRot.resource_rep(avg_pword): 1}

        #     if self.cost_per_term is None:
        #         self.cost_per_term = cost_per_term

        #     if self.cost_per_exp_term is None:
        #         self.cost_per_exp_term = cost_per_exp_term

        # if self.lcu_type == "cdf":
        #     cost_per_term = {}
        #     cost_per_exp_term = {}
        #     cost_per_ctrl_exp_term = {}

        #     basis_rot = re.ResourceBasisRotation.resource_rep(self.num_wires)
        #     adj_basis_rot = re.ResourceAdjoint.resource_rep(
        #         re.ResourceBasisRotation, {"dim_N": self.num_wires}
        #     )
        #     z = re.ResourceZ.resource_rep()
        #     multi_z = re.ResourceMultiRZ.resource_rep(2)
        #     ctrl_multi_z = re.ResourceControlled.resource_rep(
        #         re.ResourceMultiRZ, {"num_wires": 2}, 1, 0, 0
        #     )

        #     cost_per_term[basis_rot] = 2
        #     cost_per_term[adj_basis_rot] = 2
        #     cost_per_term[z] = 2

        #     cost_per_exp_term[basis_rot] = 2
        #     cost_per_exp_term[adj_basis_rot] = 2
        #     cost_per_exp_term[multi_z] = 1

        #     cost_per_ctrl_exp_term[basis_rot] = 2
        #     cost_per_ctrl_exp_term[adj_basis_rot] = 2
        #     cost_per_ctrl_exp_term[ctrl_multi_z] = 1

        #     if self.cost_per_term is None:
        #         self.cost_per_term = cost_per_term

        #     if self.cost_per_exp_term is None:
        #         self.cost_per_exp_term = cost_per_exp_term

        #     if self.cost_per_ctrl_exp_term is None:
        #         self.cost_per_ctrl_exp_term = cost_per_ctrl_exp_term


    @classmethod 
    def from_cdf_lcu(cls, num_qubits, num_terms):
        return cls()


def _process_pauli_key(pauli_key: str, exp_flag=False):
    r"""A function which processes a string representation of a Pauli word.
    
    The pauli_key is expected to be in one of the following formats:

        - "XYZZ"
        - "X:10"
        - "X:2,Z:3"
    """
    op_map = {"X": re.ResourceX.resource_rep(), "Y":re.ResourceY.resource_rep(), "Z":re.ResourceZ.resource_rep()}

    operator_keys = pauli_key.split(',')
    
    if len(operator_keys) == 1:
        char_and_count = operator_keys[0].split(":")
        
        if len(char_and_count) == 1:
            str_ops = char_and_count[0]
        else:
            str_ops = char_and_count[0] * int(char_and_count[1])

    else:
        str_ops = ""

        for operator_key in operator_keys:
            char_and_count = operator_key.split(":")
        
            if len(char_and_count) == 1:
                str_ops += char_and_count[0]
            else:
                str_ops += char_and_count[0] * int(char_and_count[1])

    result_op = re.ResourceProd.resource_rep(
        tuple(
            op_map[char] for char in str_ops         
        ),
    )

    if exp_flag:
        result_op = re.ResourcePauliRot.resource_rep(pauli_string=str_ops)

    return result_op
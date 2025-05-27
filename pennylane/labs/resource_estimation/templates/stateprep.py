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
r"""Resource operators for PennyLane state preparation templates."""
# import math
# from collections import defaultdict
# from typing import Dict

# import pennylane as qml
# from pennylane.labs import resource_estimation as re
# from pennylane.labs.resource_estimation.resource_container import CompressedResourceOp
# from pennylane.labs.resource_estimation.resource_operator import ResourceOperator, GateCount, AddQubits, CutQubits
# from pennylane.labs.resource_estimation.qubit_manager import clean_qubits, tight_qubit_budget
# from pennylane.operation import Operation

# # pylint: disable=arguments-differ, protected-access, non-parent-init-called, too-many-arguments,


# class ResourceStatePrep(ResourceOperator):
#     r"""Resource class for StatePrep.

#     Args:
#         state (array[complex] or csr_matrix): the state vector to prepare
#         wires (Sequence[int] or int): the wire(s) the operation acts on
#         pad_with (float or complex): if not ``None``, ``state`` is padded with this constant to be of size :math:`2^n`, where
#             :math:`n` is the number of wires.
#         normalize (bool): whether to normalize the state vector. To represent a valid quantum state vector, the L2-norm
#             of ``state`` must be one. The argument ``normalize`` can be set to ``True`` to normalize the state automatically.
#         id (str): custom label given to an operator instance,
#             can be useful for some applications where the instance has to be identified
#         validate_norm (bool): whether to validate the norm of the input state

#     Resource Parameters:
#         * num_wires (int): the number of wires that the operation acts on

#     Resources:
#         Uses the resources as defined in the :class:`~.ResourceMottonenStatePreperation` template.

#     .. seealso:: :class:`~.StatePrep`

#     **Example**

#     The resources for this operation are computed using:

#     >>> re.ResourceStatePrep.resources(num_wires=3)
#     {MottonenStatePrep(3): 1}
#     """

#     def __init__(self, num_wires, wires=None):
#         self.num_wires = num_wires
#         super().__init__(wires=wires)

#     @staticmethod
#     def _resource_decomp(num_wires, **kwargs) -> Dict[CompressedResourceOp, int]:
#         r"""Returns a dictionary representing the resources of the operator. The
#         keys are the operators and the associated values are the counts.

#         Args:
#             num_wires (int): the number of wires that the operation acts on

#         Resources:
#             Uses the resources as defined in the :class:`~.ResourceMottonenStatePreperation` template.
#         """
#         return [GateCount(re.ResourceMottonenStatePreparation.resource_rep(num_wires), 1)]

#     @property
#     def resource_params(self) -> dict:
#         r"""Returns a dictionary containing the minimal information needed to compute the resources.

#         Returns:
#             dict: A dictionary containing the resource parameters:
#                 * num_wires (int): the number of wires that the operation acts on
#         """
#         return {"num_wires": self.num_wires}

#     @classmethod
#     def resource_rep(cls, num_wires) -> CompressedResourceOp:
#         r"""Returns a compressed representation containing only the parameters of
#         the Operator that are needed to compute a resource estimation.

#         Args:
#             num_wires (int): the number of wires that the operation acts on

#         Returns:
#             CompressedResourceOp: the operator in a compressed representation
#         """
#         params = {"num_wires": num_wires}
#         return CompressedResourceOp(cls, params)

#     @classmethod
#     def tracking_name(cls, num_wires) -> str:
#         return f"StatePrep({num_wires})"


# class ResourceMottonenStatePreparation(ResourceOperator):
#     """Resource class for the MottonenStatePreparation template.

#     Args:
#         state_vector (tensor_like): Input array of shape ``(2^n,)``, where ``n`` is the number of wires
#             the state preparation acts on. The input array must be normalized.
#         wires (Iterable): wires that the template acts on

#     Resource Parameters:
#         * num_wires(int): the number of wires that the operation acts on

#     Resources:
#         Using the resources as described in `Mottonen et al. (2008) <https://arxiv.org/pdf/quant-ph/0407010>`_.
#         The resources are defined as :math:`2^{N+2} - 5` :class:`~.ResourceRZ` gates and
#         :math:`2^{N+2} - 4N - 4` :class:`~.ResourceCNOT` gates.

#     .. seealso:: :class:`~.MottonenStatePreperation`

#     **Example**

#     The resources for this operation are computed using:

#     >>> re.ResourceMottonenStatePreparation.resources(num_wires=3)
#     {RZ: 27, CNOT: 16}
#     """

#     def __init__(self, num_wires, wires=None):
#         self.num_wires = num_wires
#         super().__init__(wires=wires)

#     @staticmethod
#     def _resource_decomp(num_wires, **kwargs) -> Dict[CompressedResourceOp, int]:
#         r"""Returns a dictionary representing the resources of the operator. The
#         keys are the operators and the associated values are the counts.

#         Args:
#             num_wires(int): the number of wires that the operation acts on

#         Resources:
#             Using the resources as described in `Mottonen et al. (2008) <https://arxiv.org/pdf/quant-ph/0407010>`_.
#             The resources are defined as :math:`2^{N+2} - 5` :class:`~.ResourceRZ` gates and
#             :math:`2^{N+2} - 4N - 4` :class:`~.ResourceCNOT` gates.
#         """
#         gate_lst = []
#         rz = re.ResourceRZ.resource_rep()
#         cnot = re.ResourceCNOT.resource_rep()

#         r_count = 2 ** (num_wires + 2) - 5
#         cnot_count = 2 ** (num_wires + 2) - 4 * num_wires - 4

#         if r_count:
#             gate_lst.append(GateCount(rz, r_count))
#         if cnot_count:
#             gate_lst.append(GateCount(cnot, cnot_count))
#         return gate_lst

#     @property
#     def resource_params(self) -> dict:
#         r"""Returns a dictionary containing the minimal information needed to compute the resources.

#         Returns:
#             dict: A dictionary containing the resource parameters:
#                 * num_wires(int): the number of wires that the operation acts on
#         """
#         return {"num_wires": self.num_wires}

#     @classmethod
#     def resource_rep(cls, num_wires) -> CompressedResourceOp:
#         r"""Returns a compressed representation containing only the parameters of
#         the Operator that are needed to compute a resource estimation.

#         Args:
#             num_wires(int): the number of wires that the operation acts on

#         Returns:
#             CompressedResourceOp: the operator in a compressed representation
#         """
#         params = {"num_wires": num_wires}
#         return CompressedResourceOp(cls, params)

#     @classmethod
#     def tracking_name(cls, num_wires) -> str:
#         return f"MottonenStatePrep({num_wires})"


# class ResourceSuperposition(qml.Superposition, ResourceOperator):
#     """Resource class for the Superposition template.

#     Args:
#         coeffs (tensor-like[float]): normalized coefficients of the superposition
#         bases (tensor-like[int]): basis states of the superposition
#         wires (Sequence[int]): wires that the operator acts on
#         work_wire (Union[Wires, int, str]): the auxiliary wire used for the permutation

#     Resource Parameters:
#         * num_stateprep_wires (int): the number of wires used for the operation
#         * num_basis_states (int): the number of basis states of the superposition
#         * size_basis_state (int): the size of each basis state

#     Resources:
#         The resources are computed following the PennyLane decomposition of
#         the class :class:`~.Superposition`.

#         We use the following (somewhat naive) assumptions to approximate the
#         resources:

#         -   The MottonenStatePreparation routine is assumed for the state prep
#             component.
#         -   The permutation block requires 2 multi-controlled X gates and a
#             series of CNOT gates. On average we will be controlling on and flipping
#             half the number of bits in :code:`size_basis`. (i.e for any given basis
#             state, half will be ones and half will be zeros).
#         -   If the number of basis states provided spans the set of all basis states,
#             then we don't need to permute. In general, there is a probability associated
#             with not needing to permute wires if the basis states happen to match, we
#             estimate this quantity aswell.

#     .. seealso:: :class:`~.Superposition`

#     **Example**

#     The resources for this operation are computed using:

#     >>> re.ResourceSuperposition.resources(num_stateprep_wires=3, num_basis_states=3, size_basis_state=3)
#     {MottonenStatePrep(3): 1, CNOT: 2, MultiControlledX: 4}
#     """

#     def __init__(
#         self, coeffs=None, bases=None, wires=None, work_wire=None, state_vect=None, id=None
#     ):
#         # Overriding the default init method to allow for CompactState as an input.

#         if isinstance(state_vect, re.CompactState):
#             self.compact_state = state_vect
#             all_wires = qml.wires.Wires(wires) + qml.wires.Wires(work_wire)
#             Operation.__init__(self, state_vect, wires=all_wires)
#             return

#         self.compact_state = None
#         super().__init__(coeffs, bases, wires, work_wire, id)

#     @staticmethod
#     def _resource_decomp(
#         num_stateprep_wires, num_basis_states, size_basis_state, **kwargs
#     ) -> Dict[CompressedResourceOp, int]:
#         r"""Returns a dictionary representing the resources of the operator. The
#         keys are the operators and the associated values are the counts.

#         Args:
#             num_stateprep_wires (int): the number of wires used for the operation
#             num_basis_states (int): the number of basis states of the superposition
#             size_basis_state (int): the size of each basis state

#         Resources:
#             The resources are computed following the PennyLane decomposition of
#             the class :class:`~.Superposition`.

#             We use the following (somewhat naive) assumptions to approximate the
#             resources:

#             -   The MottonenStatePreparation routine is assumed for the state prep
#                 component.
#             -   The permutation block requires 2 multi-controlled X gates and a
#                 series of CNOT gates. On average we will be controlling on and flipping
#                 half the number of bits in :code:`size_basis`. (i.e for any given basis
#                 state, half will be ones and half will be zeros).
#             -   If the number of basis states provided spans the set of all basis states,
#                 then we don't need to permute. In general, there is a probability associated
#                 with not needing to permute wires if the basis states happen to match, we
#                 estimate this quantity aswell.

#         """
#         gate_types = {}
#         msp = re.ResourceMottonenStatePreparation.resource_rep(num_stateprep_wires)
#         gate_types[msp] = 1

#         cnot = re.ResourceCNOT.resource_rep()
#         num_zero_ctrls = size_basis_state // 2
#         multi_x = re.ResourceMultiControlledX.resource_rep(
#             num_ctrl_wires=size_basis_state,
#             num_ctrl_values=num_zero_ctrls,
#         )

#         basis_size = 2**size_basis_state
#         prob_matching_basis_states = num_basis_states / basis_size
#         num_permutes = round(num_basis_states * (1 - prob_matching_basis_states))

#         if num_permutes:
#             gate_types[cnot] = num_permutes * (
#                 size_basis_state // 2
#             )  # average number of bits to flip
#             gate_types[multi_x] = 2 * num_permutes  # for compute and uncompute

#         return gate_types

#     @property
#     def resource_params(self) -> Dict:
#         r"""Returns a dictionary containing the minimal information needed to compute the resources.

#         Returns:
#             dict: A dictionary containing the resource parameters:
#                 * num_stateprep_wires (int): the number of wires used for the operation
#                 * num_basis_states (int): the number of basis states of the superposition
#                 * size_basis_state (int): the size of each basis state
#         """
#         if self.compact_state:
#             num_basis_states = self.compact_state.num_coeffs
#             size_basis_state = self.compact_state.num_qubits
#             num_stateprep_wires = math.ceil(math.log2(num_basis_states))

#         else:
#             bases = self.hyperparameters["bases"]
#             num_basis_states = len(bases)
#             size_basis_state = len(bases[0])  # assuming they are all the same size
#             num_stateprep_wires = math.ceil(math.log2(len(self.coeffs)))

#         return {
#             "num_stateprep_wires": num_stateprep_wires,
#             "num_basis_states": num_basis_states,
#             "size_basis_state": size_basis_state,
#         }

#     @classmethod
#     def resource_rep(
#         cls, num_stateprep_wires, num_basis_states, size_basis_state
#     ) -> CompressedResourceOp:
#         r"""Returns a compressed representation containing only the parameters of
#         the Operator that are needed to compute a resource estimation.

#         Args:
#             num_stateprep_wires (int): the number of wires used for the operation
#             num_basis_states (int): the number of basis states of the superposition
#             size_basis_state (int): the size of each basis state

#         Returns:
#             CompressedResourceOp: the operator in a compressed representation
#         """
#         params = {
#             "num_stateprep_wires": num_stateprep_wires,
#             "num_basis_states": num_basis_states,
#             "size_basis_state": size_basis_state,
#         }
#         return CompressedResourceOp(cls, params)


# class ResourceBasisState(ResourceOperator):
#     r"""Resource class for the BasisState template.

#     Args:
#         state (tensor_like): Binary input of shape ``(len(wires), )``. For example, if ``state=np.array([0, 1, 0])`` or ``state=2`` (equivalent to 010 in binary), the quantum system will be prepared in the state :math:`|010 \rangle`.

#         wires (Sequence[int] or int): the wire(s) the operation acts on
#         id (str): Custom label given to an operator instance. Can be useful for some applications where the instance has to be identified.

#     Resource Parameters:
#         * num_bit_flips (int): number of qubits in the :math:`|1\rangle` state

#     Resources:
#         The resources for BasisState are according to the decomposition found in qml.BasisState.

#     .. seealso:: :class:`~.BasisState`

#     **Example**

#     The resources for this operation are computed using:

#     >>> re.ResourceBasisState.resources(num_bit_flips = 6)
#     {X: 6}
#     """

#     def __init__(self, num_wires, num_bit_flips, wires=None):
#         # Overriding the default init method to allow for CompactState as an input.
#         self.num_wires = num_wires
#         self.num_bit_flips = num_bit_flips
#         super().__init__(wires=wires)

#     @staticmethod
#     def _resource_decomp(
#         num_bit_flips,
#         **kwargs,
#     ) -> Dict[CompressedResourceOp, int]:
#         r"""Returns a dictionary representing the resources of the operator. The
#         keys are the operators and the associated values are the counts.

#         Args:
#             num_bit_flips (int): number of qubits in the :math:`|1\rangle` state

#         Resources:
#             The resources for BasisState are according to the decomposition found in qml.BasisState.
#         """
#         x = re.ResourceX.resource_rep()
#         return [GateCount(x, num_bit_flips)]

#     @property
#     def resource_params(self) -> Dict:
#         r"""Returns a dictionary containing the minimal information needed to compute the resources.

#         Returns:
#             dict: A dictionary containing the resource parameters:
#                 * num_bit_flips (int): number of qubits in the :math:`|1\rangle` state
#         """
#         return {"num_bit_flips": self.num_bit_flips}

#     @classmethod
#     def resource_rep(cls, num_bit_flips) -> CompressedResourceOp:
#         r"""Returns a compressed representation containing only the parameters of
#         the Operator that are needed to compute a resource estimation.

#         Args:
#             num_bit_flips (int): number of qubits in the :math:`|1\rangle` state

#         Returns:
#             CompressedResourceOp: the operator in a compressed representation
#         """
#         params = {"num_bit_flips": num_bit_flips}
#         return CompressedResourceOp(cls, params)

#     @classmethod
#     def tracking_name(cls, num_bit_flips) -> str:
#         return f"BasisState({num_bit_flips})"


# class ResourceMPSPrep(ResourceOperator):
#     r"""Resource class for the MPSPrep template.

#     Args:
#         mps (list[TensorLike]):  list of arrays of rank-3 and rank-2 tensors representing an MPS state
#             as a product of site matrices. See the usage details section for more information.
#         wires (Sequence[int]): wires that the template acts on. It should match the number of MPS tensors.
#         work_wires (Sequence[int]): list of extra qubits needed in the decomposition. If the maximum dimension
#             of the MPS tensors is :math:`2^k`, then :math:`k` ``work_wires`` will be needed. If no ``work_wires`` are given,
#             this operator can only be executed on the ``lightning.tensor`` device. Default is ``None``.
#         right_canonicalize (bool): indicates whether a conversion to right-canonical form should be performed to the MPS.
#             Default is ``False``.

#     Resource Parameters:
#         * num_wires (int): number of qubits corresponding to the state preparation register
#         * num_work_wires (int): number of additional qubits matching the bond dimension of the MPS.

#     Resources:
#         The resources for MPSPrep are according to the decomposition, which uses generic :class:`~.QubitUnitary`.
#         The decomposition is based on the routine described in `Fomichev et al. (2024) <https://arxiv.org/pdf/2310.18410>`_.

#     .. seealso:: :class:`~.MPSPrep`

#     **Example**

#     The resources for this operation are computed using:

#     >>> re.ResourceMPSPrep.resources(num_wires=5, num_work_wires=2)
#     defaultdict(<class 'int'>, {QubitUnitary(2): 2, QubitUnitary(3): 3})
#     """

#     def __init__(self, num_mps_matrices, max_bond_dim, wires=None):
#         self.num_wires = num_mps_matrices
#         self.max_bond_dim = max_bond_dim
#         super().__init__(wires=wires)

#     @staticmethod
#     def _resource_decomp(
#         num_wires,
#         max_bond_dim,
#         **kwargs,
#     ) -> Dict[CompressedResourceOp, int]:
#         r"""Returns a dictionary representing the resources of the operator. The
#         keys are the operators and the associated values are the counts.

#         Args:
#             num_wires (int): number of qubits corresponding to the state preparation register
#             num_work_wires (int): number of additional qubits matching the bond dimension of the MPS.

#         Resources:
#         The resources for MPSPrep are according to the decomposition, which uses generic :class:`~.QubitUnitary`.
#         The decomposition is based on the routine described in `Fomichev et al. (2024) <https://arxiv.org/pdf/2310.18410>`_.
#         """
#         num_work_wires = math.ceil(math.log2(max_bond_dim))
#         log2_chi = min(num_work_wires, math.ceil(num_wires / 2))

#         gate_lst = [AddQubits(num_work_wires)]

#         for index in range(1, num_wires + 1):
#             qubit_unitary_wires = min(index + 1, log2_chi + 1, (num_wires - index) + 2)
#             qubit_unitary = re.ResourceQubitUnitary.resource_rep(num_wires=qubit_unitary_wires)
#             gate_lst.append(GateCount(qubit_unitary))

#         gate_lst.append(CutQubits(num_work_wires))
#         return gate_lst

#     @property
#     def resource_params(self) -> Dict:
#         r"""Returns a dictionary containing the minimal information needed to compute the resources.

#         Returns:
#             dict: A dictionary containing the resource parameters:
#                 * num_wires (int): number of qubits corresponding to the state preparation register
#                 * num_work_wires (int): number of additional qubits matching the bond dimension of the MPS.
#         """
#         return {"num_wires": self.num_wires, "max_bond_dim": self.max_bond_dim}

#     @classmethod
#     def resource_rep(cls, num_wires, max_bond_dim) -> CompressedResourceOp:
#         r"""Returns a compressed representation containing only the parameters of
#         the Operator that are needed to compute a resource estimation.

#         Args:
#             num_wires (int): number of qubits corresponding to the state preparation register
#             num_work_wires (int): number of additional qubits matching the bond dimension of the MPS.

#         Returns:
#             CompressedResourceOp: the operator in a compressed representation
#         """
#         params = {"num_wires": num_wires, "max_bond_dim": max_bond_dim}
#         return CompressedResourceOp(cls, params)

#     @classmethod
#     def tracking_name(cls, num_wires, max_bond_dim) -> str:
#         return f"MPSPrep({num_wires}, {max_bond_dim})"


# class ResourceQROMStatePreparation(ResourceOperator):
#     r"""Resource class for the QROMStatePreparation template.

#     This operation implements the state preparation method described
#     in `arXiv:0208112 <https://arxiv.org/abs/quant-ph/0208112>`_.

#     Args:
#         state_vector (tensor_like): The state vector of length :math:`2^n` to be prepared on :math:`n` wires.
#         wires (Sequence[int]): The wires on which to prepare the state.
#         precision_wires (Sequence[int]): The wires allocated for storing the binary representations of the
#             rotation angles utilized in the template.
#         work_wires (Sequence[int], optional):  The work wires used for the QROM operations. Defaults to ``None``.

#     Resource Parameters:
#         * num_state_qubits (int): number of qubits required to represent the state-vector
#         * num_precision_wires (int): number of qubits that specify the precision of the rotation angles
#         * num_work_wires (int): additional qubits which optimize the implementation
#         * positive_and_real (bool): flag that the coefficients of the statevector are all real and positive.

#     Resources:
#         The resources for QROMStatePreparation are according to the decomposition as described
#         in `arXiv:0208112 <https://arxiv.org/abs/quant-ph/0208112>`_.

#     .. seealso:: :class:`~.QROMStatePreparation`

#     **Example**

#     The resources for this operation are computed using:

#     >>> re.ResourceQROMStatePreparation.resources(
#     ...     num_state_qubits=5,
#     ...     num_precision_wires=3,
#     ...     num_work_wires=3,
#     ...     positive_and_real=True,
#     ... )
#     defaultdict(<class 'int'>, {QROM: 1, Adjoint(QROM): 1,
#     QROM: 1, Adjoint(QROM): 1, QROM: 1, Adjoint(QROM): 1,
#     QROM: 1, Adjoint(QROM): 1, QROM: 1, Adjoint(QROM): 1, CRY: 15})
#     """

#     def __init__(self, num_state_qubits, precision=None, positive_and_real=False, wires=None):
#         # Overriding the default init method to allow for CompactState as an input.
#         self.num_wires = num_state_qubits
#         self.precision = precision
#         self.positive_and_real = positive_and_real
#         super().__init__(wires=wires)

#     @staticmethod
#     def _resource_decomp(
#         num_state_qubits,
#         precision,
#         positive_and_real,
#         **kwargs,
#     ):
#         r"""Returns a dictionary representing the resources of the operator. The
#         keys are the operators and the associated values are the counts.

#         Args:
#             num_state_qubits (int): number of qubits required to represent the state-vector
#             num_precision_wires (int): number of qubits that specify the precision of the rotation angles
#             num_work_wires (int): additional qubits which optimize the implementation
#             num_phase_gradient_wires (int): number of qubits where the phase gradient state is stored. Must be equal
#                 to ``num_precision_wires``
#             positive_and_real (bool): flag that the coefficients of the statevector are all real and positive.

#         Resources:
#             The resources for QROMStatePreparation are according to the decomposition as described
#             in `arXiv:0208112 <https://arxiv.org/abs/quant-ph/0208112>`_.
#         """
#         gate_counts = []
#         precision = precision or kwargs["config"]["precision_qrom_state_prep"]
#         num_precision_wires = abs(math.floor(math.log2(precision)))

#         gate_counts.append(AddQubits(num_precision_wires))

#         for j in range(1, num_state_qubits):
#             num_bitstrings = 2**j
#             num_bit_flips = max(2 ** (j - 1), 1)

#             gate_counts.append(
#                 GateCount(
#                     re.ResourceQROM.resource_rep(
#                         num_bitstrings,
#                         num_bit_flips,
#                         num_precision_wires,
#                         clean=False,
#                     )
#                 )
#             )

#             gate_counts.append(
#                 GateCount(
#                     re.ResourceAdjoint.resource_rep(
#                         base_class=re.ResourceQROM,
#                         base_params={
#                             "num_bitstrings": num_bitstrings,
#                             "num_bit_flips": num_bit_flips,
#                             "size_bitstring": num_precision_wires,
#                             "clean": False,
#                         },
#                     )
#                 )
#             )

#         t = re.ResourceT.resource_rep()
#         h = re.ResourceHadamard.resource_rep()

#         # SemiAdder T-cost estimation. Deduce based in image 1 and non-simetrics cnots: https://arxiv.org/pdf/1709.06648
#         # TODO: Update once we have qml.SemiAdder
#         gate_counts.append(
#             GateCount(
#                 t,
#                 (2 * (2 * (num_precision_wires - 1)) + 4 * (2 * num_precision_wires - 1)) * num_state_qubits,
#             )
#         )

#         gate_counts.append(GateCount(h, 2 * num_state_qubits))

#         if not positive_and_real:
#             gate_counts.append(
#                 GateCount(
#                     re.ResourceQROM.resource_rep(
#                         2**num_state_qubits,
#                         2 ** (num_state_qubits - 1),
#                         num_precision_wires,
#                         clean=False,
#                     )
#                 )
#             )

#             # SemiAdder T-cost estimation. Deduce based in image 1 and non-simetrics cnots: https://arxiv.org/pdf/1709.06648
#             # TODO: Update once we have qml.SemiAdder
#             gate_counts.append(
#                 GateCount(
#                     t,
#                     2 * (2 * (num_precision_wires - 1)) + 4 * (2 * num_precision_wires - 1),
#                 )
#             )

#             gate_counts.append(
#                 GateCount(
#                     re.ResourceAdjoint.resource_rep(
#                         base_class=re.ResourceQROM,
#                         base_params={
#                             "num_bitstrings": 2**num_state_qubits,
#                             "num_bit_flips": 2 ** (num_state_qubits - 1),
#                             "size_bitstring": num_precision_wires,
#                             "clean": False,
#                         },
#                     )
#                 )
#             )

#         gate_counts.append(CutQubits(num_precision_wires))
#         return gate_counts

#     @staticmethod
#     def optimized_ww_decomp(
#         num_state_qubits,
#         precision,
#         positive_and_real,
#         **kwargs,
#     ):
#         r"""Returns a dictionary representing the resources of the operator. The
#         keys are the operators and the associated values are the counts.

#         Args:
#             num_state_qubits (int): number of qubits required to represent the state-vector
#             num_precision_wires (int): number of qubits that specify the precision of the rotation angles
#             num_work_wires (int): additional qubits which optimize the implementation
#             num_phase_gradient_wires (int): number of qubits where the phase gradient state is stored. Must be equal
#                 to ``num_precision_wires``
#             positive_and_real (bool): flag that the coefficients of the statevector are all real and positive.

#         Resources:
#             The resources for QROMStatePreparation are according to the decomposition as described
#             in `arXiv:0208112 <https://arxiv.org/abs/quant-ph/0208112>`_.
#         """
#         gate_counts = []
#         precision = precision or kwargs["config"]["precision_qrom_state_prep"]
#         num_precision_wires = abs(math.floor(math.log2(precision)))

#         gate_counts.append(AddQubits(num_precision_wires))
#         available_wires = clean_qubits() - num_precision_wires

#         for j in range(1, num_state_qubits):
#             num_bitstrings = 2**j
#             num_bit_flips = max(2 ** (j - 1), 1)

#             W_opt = re.ResourceQROM._t_optimized_select_swap_width(num_bitstrings, num_precision_wires)
#             l = math.ceil(math.log2(math.ceil(num_bitstrings / W_opt)))
#             l_new = l
#             if tight_qubit_budget() and available_wires < ((W_opt - 1) * num_precision_wires + (l - 1)):
#                 for p in range(0, int(math.log2(W_opt))+ 1):
#                 # for W_opt_new in range(1, W_opt):
#                     W_opt_new = 2**p
#                     l_new = math.ceil(math.log2(math.ceil(num_bitstrings / W_opt_new)))
#                     if available_wires < ((W_opt_new - 1) * num_precision_wires + (l_new - 1)):
#                         break
#                     W_opt = W_opt_new

#             gate_counts.append(
#                 GateCount(
#                     re.ResourceQROM.resource_rep(
#                         num_bitstrings,
#                         num_bit_flips,
#                         num_precision_wires,
#                         select_swap_depth=W_opt,
#                         clean=False,
#                     )
#                 )
#             )

#             gate_counts.append(
#                 GateCount(
#                     re.ResourceAdjoint.resource_rep(
#                         base_class=re.ResourceQROM,
#                         base_params={
#                             "num_bitstrings": num_bitstrings,
#                             "num_bit_flips": num_bit_flips,
#                             "size_bitstring": num_precision_wires,
#                             "select_swap_depth": W_opt,
#                             "clean": False,
#                         },
#                     )
#                 )
#             )

#         t = re.ResourceT.resource_rep()
#         h = re.ResourceHadamard.resource_rep()

#         # SemiAdder T-cost estimation. Deduce based in image 1 and non-simetrics cnots: https://arxiv.org/pdf/1709.06648
#         # TODO: Update once we have qml.SemiAdder
#         gate_counts.append(
#             GateCount(
#                 t,
#                 (2 * (2 * (num_precision_wires - 1)) + 4 * (2 * num_precision_wires - 1)) * num_state_qubits,
#             )
#         )

#         gate_counts.append(GateCount(h, 2 * num_state_qubits))

#         if not positive_and_real:
#             num_bitstrings = 2**num_state_qubits
#             W_opt = re.ResourceQROM._t_optimized_select_swap_width(num_bitstrings, num_precision_wires)
#             l = math.ceil(math.log2(math.ceil(num_bitstrings / W_opt)))

#             if tight_qubit_budget() and available_wires < ((W_opt - 1) * num_precision_wires + (l - 1)):
#                 for p in range(0, int(math.log2(W_opt)) + 1):
#                 # for W_opt_new in range(1, W_opt):
#                     W_opt_new = 2**p
#                     l_new = math.ceil(math.log2(math.ceil(num_bitstrings / W_opt_new)))
#                     if available_wires < ((W_opt_new - 1) * num_precision_wires + (l_new - 1)):
#                         break
#                     W_opt = W_opt_new

#             gate_counts.append(
#                 GateCount(
#                     re.ResourceQROM.resource_rep(
#                         2**num_state_qubits,
#                         2 ** (num_state_qubits - 1),
#                         num_precision_wires,
#                         select_swap_depth=W_opt,
#                         clean=False,
#                     )
#                 )
#             )

#             # SemiAdder T-cost estimation. Deduce based in image 1 and non-simetrics cnots: https://arxiv.org/pdf/1709.06648
#             # TODO: Update once we have qml.SemiAdder
#             gate_counts.append(
#                 GateCount(
#                     t,
#                     2 * (2 * (num_precision_wires - 1)) + 4 * (2 * num_precision_wires - 1),
#                 )
#             )

#             gate_counts.append(
#                 GateCount(
#                     re.ResourceAdjoint.resource_rep(
#                         base_class=re.ResourceQROM,
#                         base_params={
#                             "num_bitstrings": 2**num_state_qubits,
#                             "num_bit_flips": 2 ** (num_state_qubits - 1),
#                             "size_bitstring": num_precision_wires,
#                             "select_swap_depth": W_opt,
#                             "clean": False,
#                         },
#                     )
#                 )
#             )

#         gate_counts.append(CutQubits(num_precision_wires))
#         return gate_counts

#     @staticmethod
#     def zero_swap_w_decomp(
#         num_state_qubits,
#         precision,
#         positive_and_real,
#         **kwargs,
#     ):
#         r"""Returns a dictionary representing the resources of the operator. The
#         keys are the operators and the associated values are the counts.

#         Args:
#             num_state_qubits (int): number of qubits required to represent the state-vector
#             num_precision_wires (int): number of qubits that specify the precision of the rotation angles
#             num_work_wires (int): additional qubits which optimize the implementation
#             num_phase_gradient_wires (int): number of qubits where the phase gradient state is stored. Must be equal
#                 to ``num_precision_wires``
#             positive_and_real (bool): flag that the coefficients of the statevector are all real and positive.

#         Resources:
#             The resources for QROMStatePreparation are according to the decomposition as described
#             in `arXiv:0208112 <https://arxiv.org/abs/quant-ph/0208112>`_.
#         """
#         gate_counts = []
#         precision = precision or kwargs["config"]["precision_qrom_state_prep"]
#         num_precision_wires = abs(math.floor(math.log2(precision)))

#         gate_counts.append(AddQubits(num_precision_wires))

#         for j in range(1, num_state_qubits):
#             num_bitstrings = 2**j
#             num_bit_flips = max(2 ** (j - 1), 1)

#             gate_counts.append(
#                 GateCount(
#                     re.ResourceQROM.resource_rep(
#                         num_bitstrings,
#                         num_bit_flips,
#                         num_precision_wires,
#                         select_swap_depth=1,
#                         clean=False,
#                     )
#                 )
#             )

#             gate_counts.append(
#                 GateCount(
#                     re.ResourceAdjoint.resource_rep(
#                         base_class=re.ResourceQROM,
#                         base_params={
#                             "num_bitstrings": num_bitstrings,
#                             "num_bit_flips": num_bit_flips,
#                             "size_bitstring": num_precision_wires,
#                             "select_swap_depth": 1,
#                             "clean": False,
#                         },
#                     )
#                 )
#             )

#         t = re.ResourceT.resource_rep()
#         h = re.ResourceHadamard.resource_rep()

#         # SemiAdder T-cost estimation. Deduce based in image 1 and non-simetrics cnots: https://arxiv.org/pdf/1709.06648
#         # TODO: Update once we have qml.SemiAdder
#         gate_counts.append(
#             GateCount(
#                 t,
#                 (2 * (2 * (num_precision_wires - 1)) + 4 * (2 * num_precision_wires - 1)) * num_state_qubits,
#             )
#         )

#         gate_counts.append(GateCount(h, 2 * num_state_qubits))

#         if not positive_and_real:
#             gate_counts.append(
#                 GateCount(
#                     re.ResourceQROM.resource_rep(
#                         2**num_state_qubits,
#                         2 ** (num_state_qubits - 1),
#                         num_precision_wires,
#                         select_swap_depth=1,
#                         clean=False,
#                     )
#                 )
#             )

#             # SemiAdder T-cost estimation. Deduce based in image 1 and non-simetrics cnots: https://arxiv.org/pdf/1709.06648
#             # TODO: Update once we have qml.SemiAdder
#             gate_counts.append(
#                 GateCount(
#                     t,
#                     2 * (2 * (num_precision_wires - 1)) + 4 * (2 * num_precision_wires - 1),
#                 )
#             )

#             gate_counts.append(
#                 GateCount(
#                     re.ResourceAdjoint.resource_rep(
#                         base_class=re.ResourceQROM,
#                         base_params={
#                             "num_bitstrings": 2**num_state_qubits,
#                             "num_bit_flips": 2 ** (num_state_qubits - 1),
#                             "size_bitstring": num_precision_wires,
#                             "select_swap_depth": 1,
#                             "clean": False,
#                         },
#                     )
#                 )
#             )

#         gate_counts.append(CutQubits(num_precision_wires))
#         return gate_counts

#     @property
#     def resource_params(self) -> dict:
#         r"""Returns a dictionary containing the minimal information needed to compute the resources.

#         Returns:
#             dict: A dictionary containing the resource parameters:
#                 * num_state_qubits (int): number of qubits required to represent the state-vector
#                 * num_precision_wires (int): number of qubits that specify the precision of the rotation angles
#                 * num_work_wires (int): additional qubits which optimize the implementation
#                 * positive_and_real (bool): flag that the coefficients of the statevector are all real and positive.
#         """

#         return {
#             "num_state_qubits": self.num_wires,
#             "precision": self.precision,
#             "positive_and_real": self.positive_and_real,
#         }

#     @classmethod
#     def resource_rep(cls, num_state_qubits, precision=None, positive_and_real=False):
#         r"""Returns a compressed representation containing only the parameters of
#         the Operator that are needed to compute a resource estimation.

#         Args:
#             num_state_qubits (int): number of qubits required to represent the state-vector
#             num_precision_wires (int): number of qubits that specify the precision of the rotation angles
#             num_work_wires (int): additional qubits which optimize the implementation
#             positive_and_real (bool): flag that the coefficients of the statevector are all real and positive.

#         Returns:
#             CompressedResourceOp: the operator in a compressed representation
#         """
#         params = {
#             "num_state_qubits": num_state_qubits,
#             "precision": precision,
#             "positive_and_real": positive_and_real,
#         }
#         return CompressedResourceOp(cls, params)

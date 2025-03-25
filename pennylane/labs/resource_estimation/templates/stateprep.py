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
import math
from typing import Dict

import pennylane as qml
from pennylane.labs import resource_estimation as re
from pennylane.labs.resource_estimation import CompressedResourceOp, ResourceOperator

# pylint: disable=arguments-differ, protected-access


class ResourceStatePrep(qml.StatePrep, ResourceOperator):
    """Resource class for StatePrep.

    Args:
        state (array[complex] or csr_matrix): the state vector to prepare
        wires (Sequence[int] or int): the wire(s) the operation acts on
        pad_with (float or complex): if not ``None``, ``state`` is padded with this constant to be of size :math:`2^n`, where
            :math:`n` is the number of wires.
        normalize (bool): whether to normalize the state vector. To represent a valid quantum state vector, the L2-norm
            of ``state`` must be one. The argument ``normalize`` can be set to ``True`` to normalize the state automatically.
        id (str): custom label given to an operator instance,
            can be useful for some applications where the instance has to be identified
        validate_norm (bool): whether to validate the norm of the input state

    Resource Parameters:
        * num_wires (int): the number of wires that the operation acts on

    Resources:
        Uses the resources as defined in the :class:`~.ResourceMottonenStatePreperation` template.

    .. seealso:: :class:`~.StatePrep`

    **Example**

    The resources for this operation are computed using:

    >>> re.ResourceStatePrep.resources(num_wires=3)
    {MottonenStatePrep(3): 1}
    """

    @staticmethod
    def _resource_decomp(num_wires, **kwargs) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Args:
            num_wires (int): the number of wires that the operation acts on

        Resources:
            Uses the resources as defined in the :class:`~.ResourceMottonenStatePreperation` template.
        """
        return {re.ResourceMottonenStatePreparation.resource_rep(num_wires): 1}

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * num_wires (int): the number of wires that the operation acts on
        """
        return {"num_wires": len(self.wires)}

    @classmethod
    def resource_rep(cls, num_wires) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            num_wires (int): the number of wires that the operation acts on

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        params = {"num_wires": num_wires}
        return CompressedResourceOp(cls, params)

    @classmethod
    def tracking_name(cls, num_wires) -> str:
        return f"StatePrep({num_wires})"


class ResourceMottonenStatePreparation(qml.MottonenStatePreparation, ResourceOperator):
    """Resource class for the MottonenStatePreparation template.

    Args:
        state_vector (tensor_like): Input array of shape ``(2^n,)``, where ``n`` is the number of wires
            the state preparation acts on. The input array must be normalized.
        wires (Iterable): wires that the template acts on

    Resource Parameters:
        * num_wires(int): the number of wires that the operation acts on

    Resources:
        Using the resources as described in `Mottonen et al. (2008) <https://arxiv.org/pdf/quant-ph/0407010>`_.
        The resources are defined as :math:`2^{N+2} - 5` :class:`~.ResourceRZ` gates and
        :math:`2^{N+2} - 4N - 4` :class:`~.ResourceCNOT` gates.

    .. seealso:: :class:`~.MottonenStatePreperation`

    **Example**

    The resources for this operation are computed using:

    >>> re.ResourceMottonenStatePreparation.resources(num_wires=3)
    {RZ: 27, CNOT: 16}
    """

    @staticmethod
    def _resource_decomp(num_wires, **kwargs) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Args:
            num_wires(int): the number of wires that the operation acts on

        Resources:
            Using the resources as described in `Mottonen et al. (2008) <https://arxiv.org/pdf/quant-ph/0407010>`_.
            The resources are defined as :math:`2^{N+2} - 5` :class:`~.ResourceRZ` gates and
            :math:`2^{N+2} - 4N - 4` :class:`~.ResourceCNOT` gates.
        """
        gate_types = {}
        rz = re.ResourceRZ.resource_rep()
        cnot = re.ResourceCNOT.resource_rep()

        r_count = 2 ** (num_wires + 2) - 5
        cnot_count = 2 ** (num_wires + 2) - 4 * num_wires - 4

        if r_count:
            gate_types[rz] = r_count

        if cnot_count:
            gate_types[cnot] = cnot_count
        return gate_types

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * num_wires(int): the number of wires that the operation acts on
        """
        return {"num_wires": len(self.wires)}

    @classmethod
    def resource_rep(cls, num_wires) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            num_wires(int): the number of wires that the operation acts on

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        params = {"num_wires": num_wires}
        return CompressedResourceOp(cls, params)

    @classmethod
    def tracking_name(cls, num_wires) -> str:
        return f"MottonenStatePrep({num_wires})"


class ResourceSuperposition(qml.Superposition, ResourceOperator):
    """Resource class for the Superposition template.

    Args:
        coeffs (tensor-like[float]): normalized coefficients of the superposition
        bases (tensor-like[int]): basis states of the superposition
        wires (Sequence[int]): wires that the operator acts on
        work_wire (Union[Wires, int, str]): the auxiliary wire used for the permutation

    Resource Parameters:
        * num_stateprep_wires (int): the number of wires used for the operation
        * num_basis_states (int): the number of basis states of the superposition
        * size_basis_state (int): the size of each basis state

    Resources:
        The resources are computed following the PennyLane decomposition of
        the class :class:`~.Superposition`.

        We use the following (somewhat naive) assumptions to approximate the
        resources:

        -   The MottonenStatePreparation routine is assumed for the state prep
            component.
        -   The permutation block requires 2 multi-controlled X gates and a
            series of CNOT gates. On average we will be controlling on and flipping
            half the number of bits in :code:`size_basis`. (i.e for any given basis
            state, half will be ones and half will be zeros).
        -   If the number of basis states provided spans the set of all basis states,
            then we don't need to permute. In general, there is a probability associated
            with not needing to permute wires if the basis states happen to match, we
            estimate this quantity aswell.

    .. seealso:: :class:`~.Superposition`

    **Example**

    The resources for this operation are computed using:

    >>> re.ResourceSuperposition.resources(num_stateprep_wires=3, num_basis_states=3, size_basis_state=3)
    {MottonenStatePrep(3): 1, CNOT: 2, MultiControlledX: 4}
    """

    @staticmethod
    def _resource_decomp(
        num_stateprep_wires, num_basis_states, size_basis_state, **kwargs
    ) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Args:
            num_stateprep_wires (int): the number of wires used for the operation
            num_basis_states (int): the number of basis states of the superposition
            size_basis_state (int): the size of each basis state

        Resources:
            The resources are computed following the PennyLane decomposition of
            the class :class:`~.Superposition`.

            We use the following (somewhat naive) assumptions to approximate the
            resources:

            -   The MottonenStatePreparation routine is assumed for the state prep
                component.
            -   The permutation block requires 2 multi-controlled X gates and a
                series of CNOT gates. On average we will be controlling on and flipping
                half the number of bits in :code:`size_basis`. (i.e for any given basis
                state, half will be ones and half will be zeros).
            -   If the number of basis states provided spans the set of all basis states,
                then we don't need to permute. In general, there is a probability associated
                with not needing to permute wires if the basis states happen to match, we
                estimate this quantity aswell.

        """
        gate_types = {}
        msp = re.ResourceMottonenStatePreparation.resource_rep(num_stateprep_wires)
        gate_types[msp] = 1

        cnot = re.ResourceCNOT.resource_rep()
        num_zero_ctrls = size_basis_state // 2
        multi_x = re.ResourceMultiControlledX.resource_rep(
            num_ctrl_wires=size_basis_state,
            num_ctrl_values=num_zero_ctrls,
            num_work_wires=0,
        )

        basis_size = 2**size_basis_state
        prob_matching_basis_states = num_basis_states / basis_size
        num_permutes = round(num_basis_states * (1 - prob_matching_basis_states))

        if num_permutes:
            gate_types[cnot] = num_permutes * (
                size_basis_state // 2
            )  # average number of bits to flip
            gate_types[multi_x] = 2 * num_permutes  # for compute and uncompute

        return gate_types

    @property
    def resource_params(self) -> Dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * num_stateprep_wires (int): the number of wires used for the operation
                * num_basis_states (int): the number of basis states of the superposition
                * size_basis_state (int): the size of each basis state
        """
        bases = self.hyperparameters["bases"]
        num_basis_states = len(bases)
        size_basis_state = len(bases[0])  # assuming they are all the same size
        num_stateprep_wires = math.ceil(math.log2(len(self.coeffs)))

        return {
            "num_stateprep_wires": num_stateprep_wires,
            "num_basis_states": num_basis_states,
            "size_basis_state": size_basis_state,
        }

    @classmethod
    def resource_rep(
        cls, num_stateprep_wires, num_basis_states, size_basis_state
    ) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            num_stateprep_wires (int): the number of wires used for the operation
            num_basis_states (int): the number of basis states of the superposition
            size_basis_state (int): the size of each basis state

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        params = {
            "num_stateprep_wires": num_stateprep_wires,
            "num_basis_states": num_basis_states,
            "size_basis_state": size_basis_state,
        }
        return CompressedResourceOp(cls, params)


class ResourceBasisState(qml.BasisState, ResourceOperator):
    r"""Resource class for the BasisState template.

    Args:
        state (tensor_like): Binary input of shape ``(len(wires), )``. For example, if ``state=np.array([0, 1, 0])`` or ``state=2`` (equivalent to 010 in binary), the quantum system will be prepared in the state :math:`|010 \rangle`.

        wires (Sequence[int] or int): the wire(s) the operation acts on
        id (str): Custom label given to an operator instance. Can be useful for some applications where the instance has to be identified.

    Resource Parameters:
        * num_bit_flips (int): number of qubits in the :math:`|1\rangle` state

    Resources:
        The resources for BasisState are according to the decomposition found in qml.BasisState.

    .. seealso:: :class:`~.BasisState`

    **Example**

    The resources for this operation are computed using:

    >>> re.ResourceBasisState.resources(num_bit_flips = 6)
    {X: 6}
    """

    @staticmethod
    def _resource_decomp(
        num_bit_flips,
        **kwargs,
    ) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Args:
            num_bit_flips (int): number of qubits in the :math:`|1\rangle` state

        Resources:
            The resources for BasisState are according to the decomposition found in qml.BasisState.
        """
        gate_types = {}
        x = re.ResourceX.resource_rep()
        gate_types[x] = num_bit_flips

        return gate_types

    @property
    def resource_params(self) -> Dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * num_bit_flips (int): number of qubits in the :math:`|1\rangle` state
        """
        num_bit_flips = sum(self.parameters[0])
        return {"num_bit_flips": num_bit_flips}

    @classmethod
    def resource_rep(cls, num_bit_flips) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            num_bit_flips (int): number of qubits in the :math:`|1\rangle` state

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """

        params = {"num_bit_flips": num_bit_flips}
        return CompressedResourceOp(cls, params)

    @classmethod
    def tracking_name(cls, num_bit_flips) -> str:
        return f"BasisState({num_bit_flips})"

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
        num_wires(int): the number of wires that StatePrep acts on

    Resources:
        Uses the resources as defined in the ResourceMottonenStatePreperation template.
    """

    @staticmethod
    def _resource_decomp(num_wires, **kwargs) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Args:
            num_wires(int): the number of wires that StatePrep acts on

        Resources:
            Uses the resources as defined in the ResourceMottonenStatePreperation template.
        """
        return {re.ResourceMottonenStatePreparation.resource_rep(num_wires): 1}

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Resource parameters:
            num_wires(int): the number of wires that StatePrep acts on

        Returns:
            dict: dictionary containing the resource parameters
        """
        return {"num_wires": len(self.wires)}

    @classmethod
    def resource_rep(cls, num_wires) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            num_wires(int): the number of wires that StatePrep acts on

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
        num_wires(int): the number of wires that StatePrep acts on

    Resources:
        Using the resources as described in https://arxiv.org/pdf/quant-ph/0407010.
    """

    @staticmethod
    def _resource_decomp(num_wires, **kwargs) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Args:
            num_wires(int): the number of wires that StatePrep acts on

        Resources:
            Using the resources as described in https://arxiv.org/pdf/quant-ph/0407010.
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

        Resource parameters:
            num_wires(int): the number of wires that StatePrep acts on

        Returns:
            dict: dictionary containing the resource parameters
        """
        return {"num_wires": len(self.wires)}

    @classmethod
    def resource_rep(cls, num_wires) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            num_wires(int): the number of wires that StatePrep acts on

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
        num_stateprep_wires (int): the number of wires used for the operation
        num_basis_states (int): the number of basis states of the superposition
        size_basis_state (int): the size of each basis state

    Resources:
        The resources are computed following the PennyLane decomposition of
        the class. This class was designed based on the method described in
        https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.5.040339.

        We use the following (somewhat naive) assumptions to approximate the
        resources:

        -  The MottonenStatePreparation routine is assumed for the state prep
        component.
        -  The permutation block requires 2 multi-controlled X gates and a
        series of CNOT gates. On average we will be controlling on and flipping
        half the number of bits in :code:`size_basis`. (i.e for any given basis
        state, half will be ones and half will be zeros).
        -  If the number of basis states provided spans the set of all basis states,
        then we don't need to permute. In general, there is a probability associated
        with not needing to permute wires if the basis states happen to match, we
        estimate this quantity aswell.

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
            the class. This class was designed based on the method described in
            https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.5.040339.

            We use the following (somewhat naive) assumptions to approximate the
            resources:

            -  The MottonenStatePreparation routine is assumed for the state prep
            component.
            -  The permutation block requires 2 multi-controlled X gates and a
            series of CNOT gates. On average we will be controlling on and flipping
            half the number of bits in :code:`size_basis`. (i.e for any given basis
            state, half will be ones and half will be zeros).
            -  If the number of basis states provided spans the set of all basis states,
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

        Resource parameters:
            num_stateprep_wires (int): the number of wires used for the operation
            num_basis_states (int): the number of basis states of the superposition
            size_basis_state (int): the size of each basis state

        Returns:
            dict: dictionary containing the resource parameters
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
        state (list): Binary input of shape ``(len(wires), )``. For example, if
        ``state=np.array([0, 1, 0])``, the quantum system will be prepared in the state
        :math:`|010 \rangle`.

    Resources:
        The resources for BasisState are according to the decomposition found in qml.BasisState.
    """

    @staticmethod
    def _resource_decomp(
        state,
        **kwargs,
    ) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Args:
            state (list): Binary input of shape ``(len(wires), )``. For example, if
            ``state=np.array([0, 1, 0])``, the quantum system will be prepared in the state
            :math:`|010 \rangle`.

        Resources:
            The resources for BasisState are according to the decomposition found in qml.BasisState.
        """

        gate_types = {}

        rx = re.ResourceRX.resource_rep()

        gate_types[rx] = sum(1 for basis in state if basis == 1)

        return gate_types

    @property
    def resource_params(self) -> Dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Resource parameters:
            state (list): Binary input of shape ``(len(wires), )``. For example, if
            ``state=np.array([0, 1, 0])``, the quantum system will be prepared in the state
            :math:`|010 \rangle`.

        Returns:
            dict: dictionary containing the resource parameters
        """
        state = self.parameters[0]
        return {"state": state}

    @classmethod
    def resource_rep(cls, state) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            state (list): Binary input of shape ``(len(wires), )``. For example, if
            ``state=np.array([0, 1, 0])``, the quantum system will be prepared in the state
            :math:`|010 \rangle`.

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """

        params = {"state": state}
        return CompressedResourceOp(cls, params)

    @classmethod
    def tracking_name(cls, state) -> str:
        return f"BasisState({state})"

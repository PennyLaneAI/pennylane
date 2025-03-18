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

    Resources:
        Uses the resources as defined in the ResourceMottonenStatePreperation template.
    """

    @staticmethod
    def _resource_decomp(num_wires, **kwargs) -> Dict[CompressedResourceOp, int]:
        return {re.ResourceMottonenStatePreparation.resource_rep(num_wires): 1}

    @property
    def resource_params(self) -> dict:
        return {"num_wires": len(self.wires)}

    @classmethod
    def resource_rep(cls, num_wires) -> CompressedResourceOp:
        params = {"num_wires": num_wires}
        return CompressedResourceOp(cls, params)

    @classmethod
    def tracking_name(cls, num_wires) -> str:
        return f"StatePrep({num_wires})"


class ResourceMottonenStatePreparation(qml.MottonenStatePreparation, ResourceOperator):
    """Resource class for the MottonenStatePreparation template.

    Using the resources as described in https://arxiv.org/pdf/quant-ph/0407010.
    """

    @staticmethod
    def _resource_decomp(num_wires, **kwargs) -> Dict[CompressedResourceOp, int]:
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
        return {"num_wires": len(self.wires)}

    @classmethod
    def resource_rep(cls, num_wires) -> CompressedResourceOp:
        params = {"num_wires": num_wires}
        return CompressedResourceOp(cls, params)

    @classmethod
    def tracking_name(cls, num_wires) -> str:
        return f"MottonenStatePrep({num_wires})"


class ResourceSuperposition(qml.Superposition, ResourceOperator):
    """Resource class for the Superposition template."""

    @staticmethod
    def _resource_decomp(
        num_stateprep_wires, num_basis_states, size_basis_state, **kwargs
    ) -> Dict[CompressedResourceOp, int]:
        r"""The resources are computed following the PennyLane decomposition of
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
        params = {
            "num_stateprep_wires": num_stateprep_wires,
            "num_basis_states": num_basis_states,
            "size_basis_state": size_basis_state,
        }
        return CompressedResourceOp(cls, params)


class ResourceBasisState(qml.BasisState, ResourceOperator):
    """Resource class for the BasisState template."""

    @staticmethod
    def _resource_decomp(
        num_wires,
        **kwargs,
    ) -> Dict[CompressedResourceOp, int]:
        r"""The resources for BasisState are according to the decomposition found
        in qml.BasisState.
        """
        gate_types = {}

        rx = re.ResourceRX.resource_rep()
        phase_shift = re.ResourcePhaseShift.resource_rep()

        gate_types[rx] = num_wires
        gate_types[phase_shift] = num_wires * 2

        return gate_types

    @property
    def resource_params(self) -> Dict:
        return {"num_wires": len(self.wires)}

    @classmethod
    def resource_rep(cls, wires) -> CompressedResourceOp:
        params = {"num_wires": wires}
        return CompressedResourceOp(cls, params)

    @classmethod
    def tracking_name(cls, num_wires) -> str:
        return f"BasisState({num_wires})"

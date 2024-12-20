# Copyright 2024 Xanadu Quantum Technologies Inc.

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
from collections import defaultdict
from typing import Dict

import pennylane as qml
from pennylane import numpy as qnp
from pennylane.labs import resource_estimation as re
from pennylane.labs.resource_estimation import CompressedResourceOp, ResourceOperator

# pylint: disable=arguments-differ


class ResourceQFT(qml.QFT, ResourceOperator):
    """Resource class for QFT.

    Resources:
        The resources are obtained from the standard decomposition of QFT as presented
        in (chapter 5) `Nielsen, M.A. and Chuang, I.L. (2011) Quantum Computation and Quantum Information
        <https://www.cambridge.org/highereducation/books/quantum-computation-and-quantum-information/01E10196D0A682A6AEFFEA52D53BE9AE#overview>`_.
    """

    @staticmethod
    def _resource_decomp(num_wires, **kwargs) -> Dict[CompressedResourceOp, int]:
        gate_types = {}

        hadamard = re.ResourceHadamard.resource_rep()
        swap = re.ResourceSWAP.resource_rep()
        ctrl_phase_shift = re.ResourceControlledPhaseShift.resource_rep()

        gate_types[hadamard] = num_wires
        gate_types[swap] = num_wires // 2
        gate_types[ctrl_phase_shift] = num_wires * (num_wires - 1) // 2

        return gate_types

    def resource_params(self) -> dict:
        return {"num_wires": len(self.wires)}

    @classmethod
    def resource_rep(cls, num_wires) -> CompressedResourceOp:
        params = {"num_wires": num_wires}
        return CompressedResourceOp(cls, params)


class ResourceQuantumPhaseEstimation(qml.QuantumPhaseEstimation, ResourceOperator):
    """Resource class for QPE"""

    # TODO: Add a secondary resource decomp which falls back to op.pow_resource_decomp

    @staticmethod
    def _resource_decomp(
        base_class, base_params, num_estimation_wires, **kwargs
    ) -> Dict[CompressedResourceOp, int]:
        gate_types = {}

        hadamard = re.ResourceHadamard.resource_rep()
        adj_qft = re.ResourceAdjoint.resource_rep(ResourceQFT, {"num_wires": num_estimation_wires})
        ctrl_op = re.ResourceControlled.resource_rep(base_class, base_params, 1, 0, 0)

        gate_types[hadamard] = num_estimation_wires
        gate_types[adj_qft] = 1
        gate_types[ctrl_op] = (2**num_estimation_wires) - 1

        return gate_types

    def resource_params(self) -> dict:
        op = self.hyperparameters["unitary"]
        num_estimation_wires = len(self.hyperparameters["estimation_wires"])

        return {
            "base_class": type(op),
            "base_params": op.resource_params(),
            "num_estimation_wires": num_estimation_wires,
        }

    @classmethod
    def resource_rep(
        cls, base_class, base_params, num_estimation_wires, **kwargs
    ) -> CompressedResourceOp:
        params = {
            "base_class": base_class,
            "base_params": base_params,
            "num_estimation_wires": num_estimation_wires,
        }
        return CompressedResourceOp(cls, params)

    @staticmethod
    def tracking_name(base_class, base_params, num_estimation_wires, **kwargs) -> str:
        return f"QuantumPhaseEstimation({base_class}, {num_estimation_wires})"


ResourceQPE = ResourceQuantumPhaseEstimation  # Alias for ease of typing


class ResourceStatePrep(qml.StatePrep, ResourceOperator):
    """Resource class for StatePrep.

    Resources:
        TODO: add the resources here
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

    def resource_params(self) -> dict:
        return {"num_wires": len(self.wires)}

    @classmethod
    def resource_rep(cls, num_wires) -> CompressedResourceOp:
        params = {"num_wires": num_wires}
        return CompressedResourceOp(cls, params)


class ResourceBasisRotation(qml.BasisRotation, ResourceOperator):

    @staticmethod
    def _resource_decomp(dim_N, **kargs) -> Dict[CompressedResourceOp, int]:
        gate_types = {}
        phase_shift = re.ResourcePhaseShift.resource_rep()
        single_excitation = re.ResourceSingleExcitation.resource_rep()

        se_count = dim_N * (dim_N - 1) / 2
        ps_count = dim_N + se_count

        gate_types[phase_shift] = ps_count
        gate_types[single_excitation] = se_count
        return gate_types

    def resource_params(self) -> dict:
        unitary_matrix = self.hyperparameters["unitary_matrix"]
        return {"dim_N": qml.math.shape(unitary_matrix)[0]}

    @classmethod
    def resource_rep(cls, dim_N) -> CompressedResourceOp:
        params = {"dim_N": dim_N}
        return CompressedResourceOp(cls, params)


class ResourceSelect(qml.Select, ResourceOperator):
    """Resource class for the Select operation"""

    @staticmethod
    def _resource_decomp(cmpr_ops, **kwargs) -> Dict[CompressedResourceOp, int]:
        gate_types = defaultdict(int)
        x = re.ResourceX.resource_rep()

        num_ops = len(cmpr_ops)
        num_ctrl_wires = int(qnp.log2(num_ops))
        num_total_ctrl_possibilities = num_ctrl_wires * (2**num_ctrl_wires)  # n * 2^n

        num_zero_controls = num_total_ctrl_possibilities // 2
        gate_types[x] = num_zero_controls * 2  # conjugate 0 controls

        for cmp_rep in cmpr_ops:
            ctrl_op = re.ResourceControlled.resource_rep(
                cmp_rep.op_type, cmp_rep.params, num_ctrl_wires, 0, 0
            )
            gate_types[ctrl_op] += 1

        return gate_types

    def resource_params(self) -> dict:
        ops = self.hyperparameters["ops"]
        cmpr_ops = tuple(op.resource_rep_from_op() for op in ops)
        return {"cmpr_ops": cmpr_ops}

    @classmethod
    def resource_rep(cls, cmpr_ops) -> CompressedResourceOp:
        params = {"cmpr_ops": cmpr_ops}
        return CompressedResourceOp(cls, params)


class ResourcePrepSelPrep(qml.PrepSelPrep, ResourceOperator):

    @staticmethod
    def _resource_decomp(cmpr_ops, **kwargs) -> Dict[CompressedResourceOp, int]:
        gate_types = {}

        num_ops = len(cmpr_ops)
        num_wires = int(qnp.log2(num_ops))

        prep = ResourceStatePrep.resource_rep(num_wires)
        sel = ResourceSelect.resource_rep(cmpr_ops)
        prep_dag = re.ResourceAdjoint.resource_rep(ResourceStatePrep, {"num_wires": num_wires})

        gate_types[prep] = 1
        gate_types[sel] = 1
        gate_types[prep_dag] = 1
        return gate_types

    def resource_params(self) -> dict:
        ops = self.hyperparameters["ops"]
        cmpr_ops = tuple(op.resource_rep_from_op() for op in ops)
        return {"cmpr_ops": cmpr_ops}

    @classmethod
    def resource_rep(cls, cmpr_ops) -> CompressedResourceOp:
        params = {"cmpr_ops": cmpr_ops}
        return CompressedResourceOp(cls, params)

    @classmethod
    def adjoint_resource_decomp(cls, cmpr_ops, **kwargs) -> Dict[CompressedResourceOp, int]:
        """Returns a compressed representation of the adjoint of the operator"""
        raise {cls.resource_rep(cmpr_ops): 1}

    @classmethod
    def controlled_resource_decomp(
        cls, num_ctrl_wires, num_ctrl_values, num_work_wires, cmpr_ops, **kwargs
    ) -> Dict[CompressedResourceOp, int]:
        """Returns a compressed representation of the controlled version of the operator"""
        gate_types = {}

        num_ops = len(cmpr_ops)
        num_wires = int(qnp.log2(num_ops))

        prep = ResourceStatePrep.resource_rep(num_wires)
        ctrl_sel = re.ResourceControlled.resource_rep(
            ResourceSelect, {"cmpr_ops": cmpr_ops}, num_ctrl_wires, num_ctrl_values, num_work_wires
        )
        prep_dag = re.ResourceAdjoint.resource_rep(ResourceStatePrep, {"num_wires": num_wires})

        gate_types[prep] = 1
        gate_types[ctrl_sel] = 1
        gate_types[prep_dag] = 1
        return gate_types

    @classmethod
    def pow_resource_decomp(cls, z, cmpr_ops, **kwargs) -> Dict[CompressedResourceOp, int]:
        """Returns a compressed representation of the operator raised to a power"""
        gate_types = {}

        num_ops = len(cmpr_ops)
        num_wires = int(qnp.log2(num_ops))

        prep = ResourceStatePrep.resource_rep(num_wires)
        pow_sel = re.ResourcePow.resource_rep(ResourceSelect, z, {"cmpr_ops": cmpr_ops})
        prep_dag = re.ResourceAdjoint.resource_rep(ResourceStatePrep, {"num_wires": num_wires})

        gate_types[prep] = 1
        gate_types[pow_sel] = 1
        gate_types[prep_dag] = 1
        return gate_types


class ResourceReflection(qml.Reflection, ResourceOperator):

    @staticmethod
    def _resource_decomp(base, num_ref_wires, **kwargs) -> Dict[CompressedResourceOp, int]:
        gate_types = {}

        x = re.ResourceX.resource_rep()
        gp = re.ResourceGlobalPhase.resource_rep()
        adj_base = re.ResourceAdjoint.resource_rep(base.op_type, base.params)
        ps = (
            re.ResourceControlled.resource_rep(re.ResourcePhaseShift, {}, num_ref_wires - 1, 0, 0)
            if num_ref_wires > 1
            else re.ResourcePhaseShift.resource_rep()
        )

        gate_types[x] = 2
        gate_types[gp] = 1
        gate_types[base] = 1
        gate_types[adj_base] = 1
        gate_types[ps] = 1

        return gate_types

    def resource_params(self) -> dict:
        base_cmpr_rep = self.hyperparameters["base"].resource_rep_from_op()
        num_ref_wires = len(self.hyperparameters["reflection_wires"])

        return {"base": base_cmpr_rep, "num_ref_wires": num_ref_wires}

    @classmethod
    def resource_rep(cls, base, num_ref_wires) -> CompressedResourceOp:
        params = {"base": base, "num_ref_wires": num_ref_wires}
        return CompressedResourceOp(cls, params)


class ResourceQubitization(qml.Qubitization, ResourceOperator):
    @staticmethod
    def _resource_decomp(cmpr_ops, num_ctrl_wires, **kwargs) -> Dict[CompressedResourceOp, int]:
        gate_types = {}
        ref = ResourceReflection.resource_rep(re.ResourceIdentity.resource_rep(), num_ctrl_wires)
        psp = ResourcePrepSelPrep.resource_rep(cmpr_ops)

        gate_types[ref] = 1
        gate_types[psp] = 1
        return gate_types

    def resource_params(self) -> dict:
        lcu = self.hyperparameters["hamiltonian"]
        _, ops = lcu.terms()

        cmpr_ops = tuple(op.resource_rep_from_op() for op in ops)
        num_ctrl_wires = len(self.hyperparameters["control"])
        return {"cmpr_ops": cmpr_ops, "num_ctrl_wires": num_ctrl_wires}

    @classmethod
    def resource_rep(cls, cmpr_ops, num_ctrl_wires) -> CompressedResourceOp:
        params = {"cmpr_ops": cmpr_ops, "num_ctrl_wires": num_ctrl_wires}
        return CompressedResourceOp(cls, params)

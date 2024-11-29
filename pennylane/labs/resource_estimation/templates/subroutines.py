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

from typing import Dict

import pennylane as qml
import pennylane.labs.resource_estimation as re

# pylint: disable=arguments-differ, protected-access


class ResourceQFT(qml.QFT, re.ResourceOperator):
    """Resource class for the QFT template.

    Resources:
        The resources are obtained from the standard decomposition of QFT as presented
        in (chapter 5) `Nielsen, M.A. and Chuang, I.L. (2011) Quantum Computation and Quantum Information
        <https://www.cambridge.org/highereducation/books/quantum-computation-and-quantum-information/01E10196D0A682A6AEFFEA52D53BE9AE#overview>`_.
    """

    @staticmethod
    def _resource_decomp(num_wires, **kwargs) -> Dict[re.CompressedResourceOp, int]:
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
    def resource_rep(cls, num_wires) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {"num_wires": num_wires})

    @staticmethod
    def tracking_name(num_wires) -> str:
        return f"QFT({num_wires})"


class ResourceControlledSequence(qml.ControlledSequence, re.ResourceOperator):
    """Resource class for the ControlledSequence template."""

    @staticmethod
    def _resource_decomp(
        base_class, base_params, num_ctrl_wires, **kwargs
    ) -> Dict[re.CompressedResourceOp, int]:
        return {
            re.ResourceControlled.resource_rep(base_class, base_params, 1, 0, 0): 2**num_ctrl_wires
            - 1
        }

    def resource_params(self) -> dict:
        return {
            "base_class": type(self.base),
            "base_params": self.base.resource_params(),
            "num_ctrl_wires": len(self.control_wires),
        }

    @classmethod
    def resource_rep(cls, base_class, base_params, num_ctrl_wires) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(
            cls,
            {
                "base_class": base_class,
                "base_params": base_params,
                "num_ctrl_wires": num_ctrl_wires,
            },
        )

    @staticmethod
    def tracking_name(base_class, base_params, num_ctrl_wires) -> str:
        base_name = base_class.tracking_name(**base_params)
        return f"ControlledSequence({base_name}, {num_ctrl_wires})"


class ResourcePhaseAdder(qml.PhaseAdder, re.ResourceOperator):
    """Resource class for the PhaseAdder template."""

    @staticmethod
    def _resource_decomp(mod, num_x_wires, **kwargs) -> Dict[re.CompressedResourceOp, int]:
        if mod == 2**num_x_wires:
            return {re.ResourcePhaseShift.resource_rep(): num_x_wires}

        qft = ResourceQFT.resource_rep(num_x_wires)
        qft_dag = re.ResourceAdjoint.resource_rep(
            ResourceQFT,
            {"num_wires": num_x_wires},
        )

        phase_shift = re.ResourcePhaseShift.resource_rep()
        phase_shift_dag = re.ResourceAdjoint.resource_rep(
            re.ResourcePhaseShift,
            {},
        )
        ctrl_phase_shift = re.ResourceControlled.resource_rep(
            re.ResourcePhaseShift,
            {},
            1,
            0,
            0,
        )

        cnot = re.ResourceCNOT.resource_rep()
        multix = re.ResourceMultiControlledX.resource_rep(1, 0, 1)

        gate_types = {}
        gate_types[qft] = 2
        gate_types[qft_dag] = 2
        gate_types[phase_shift] = 2 * num_x_wires
        gate_types[phase_shift_dag] = 2 * num_x_wires
        gate_types[ctrl_phase_shift] = num_x_wires
        gate_types[cnot] = 1
        gate_types[multix] = 1

        return gate_types

    def resource_params(self) -> dict:
        return {
            "mod": self.hyperparameters["mod"],
            "num_x_wires": len(self.hyperparameters["x_wires"]),
        }

    @classmethod
    def resource_rep(cls, mod, num_x_wires) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {"mod": mod, "num_x_wires": num_x_wires})


class ResourceMultiplier(qml.Multiplier, re.ResourceOperator):
    """Resource class for the Multiplier template."""

    @staticmethod
    def _resource_decomp(
        mod, num_work_wires, num_x_wires, **kwargs
    ) -> Dict[re.CompressedResourceOp, int]:
        if mod == 2**num_x_wires:
            num_aux_wires = num_x_wires
            num_aux_swap = num_x_wires
        else:
            num_aux_wires = num_work_wires - 1
            num_aux_swap = num_aux_wires - 1

        qft = ResourceQFT.resource_rep(num_aux_wires)
        qft_dag = re.ResourceAdjoint.resource_rep(
            ResourceQFT,
            {"num_wires": num_aux_wires},
        )

        sequence = ResourceControlledSequence.resource_rep(
            ResourcePhaseAdder,
            {},
            num_x_wires,
        )

        sequence_dag = re.ResourceAdjoint.resource_rep(
            ResourceControlledSequence,
            {
                "base_class": ResourcePhaseAdder,
                "base_params": {},
                "num_ctrl_wires": num_x_wires,
            },
        )

        cnot = re.ResourceCNOT.resource_rep()

        gate_types = {}
        gate_types[qft] = 2
        gate_types[qft_dag] = 2
        gate_types[sequence] = 1
        gate_types[sequence_dag] = 1
        gate_types[cnot] = min(num_x_wires, num_aux_swap)

        return gate_types

    def resource_params(self) -> dict:
        return {
            "mod": self.hyperparameters["mod"],
            "num_work_wires": len(self.hyperparameters["work_wires"]),
            "num_x_wires": len(self.hyperparameters["x_wires"]),
        }

    @classmethod
    def resource_rep(cls, mod, num_work_wires, num_x_wires) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(
            cls, {"mod": mod, "num_work_wires": num_work_wires, "num_x_wires": num_x_wires}
        )


class ResourceModExp(qml.ModExp, re.ResourceOperator):
    """Resource class for the ModExp template."""

    @staticmethod
    def _resource_decomp(
        mod, num_output_wires, num_work_wires, num_x_wires, **kwargs
    ) -> Dict[re.CompressedResourceOp, int]:
        mult_resources = ResourceMultiplier._resource_decomp(mod, num_work_wires, num_output_wires)
        gate_types = {}

        for comp_rep in mult_resources.items():
            new_rep = re.ResourceControlled.resource_rep(comp_rep.op_type, comp_rep.params, 1, 0, 0)

            # cancel out QFTs from consecutive Multipliers
            if comp_rep._name in ("QFT", "Adjoint(QFT)"):
                gate_types[new_rep] = 1
            else:
                gate_types[new_rep] = mult_resources[comp_rep] * ((2**num_x_wires) - 1)

        return gate_types

    def resource_params(self) -> dict:
        return {
            "mod": self.hyperparameters["mod"],
            "num_output_wires": len(self.hyperparameters["output_wires"]),
            "num_work_wires": len(self.hyperparameters["work_wires"]),
            "num_x_wires": len(self.hyperparameters["x_wires"]),
        }

    @classmethod
    def resource_rep(
        cls, mod, num_output_wires, num_work_wires, num_x_wires
    ) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(
            cls,
            {
                "mod": mod,
                "num_output_wires": num_output_wires,
                "num_work_wires": num_work_wires,
                "num_x_wires": num_x_wires,
            },
        )

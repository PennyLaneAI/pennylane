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

import pennylane as qml
import pennylane.labs.resource_estimation as re

# pylint: disable=arguments-differ


class ResourceQFT(qml.QFT, re.ResourceOperator):
    """Resource class for the QFT template.

    Resources:
        The resources are obtained from the standard decomposition of QFT as presented
        in (chapter 5) `Nielsen, M.A. and Chuang, I.L. (2011) Quantum Computation and Quantum Information
        <https://www.cambridge.org/highereducation/books/quantum-computation-and-quantum-information/01E10196D0A682A6AEFFEA52D53BE9AE#overview>`_.
    """

    @staticmethod
    def _resource_decomp(num_wires, **kwargs):
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
    def resource_rep(cls, num_wires):
        params = {"num_wires": num_wires}
        return re.CompressedResourceOp(cls, params)

class ResourceControlledSequence(qml.ControlledSequence, re.ResourceOperator):
    """Resource class for the ControlledSequence template."""
    @staticmethod
    def _resource_decomp(base_class, base_params, num_ctrl_wires):
        return {
            re.ResourcePow.resource_rep(
                re.ResourceControlled,
                {
                    base_class,
                    base_params,
                    1,
                    0,
                },
                2**num_ctrl_wires - 1,
            )
        }

    def resource_params(self):
        return {
            "base_class": type(self.base),
            "base_params": self.base.resource_params(),
            "num_ctrl_wires": len(self.control_wires),
        }

    @classmethod
    def resource_rep(cls, base_class, base_params, num_ctrl_wires):
        return re.CompressedResourceOp(cls, {"base_class": base_class, "base_params": base_params, "num_ctrl_wires": num_ctrl_wires})

class ResourceModExp(qml.ModExp, re.ResourceOperator):
    """Resource class for the ModExp template."""
    @staticmethod
    def _resource_decomp():
        pass

    def resource_params(self):
        pass

    @classmethod
    def resource_rep(cls):
        pass

class ResourceMultiplier(qml.Multiplier, re.ResourceOperator):
    """Resource class for the Multiplier template."""
    @staticmethod
    def _resource_decomp():
        pass

    def resource_params(self):
        pass

    @classmethod
    def resource_rep(cls):
        pass

class ResourcePhaseAdder(qml.PhaseAdder, re.ResourceOperator):
    """Resource class for the PhaseAdder template."""
    @staticmethod
    def _resource_decomp():
        pass

    def resource_params(self):
        pass

    @classmethod
    def resource_rep(cls):
        pass

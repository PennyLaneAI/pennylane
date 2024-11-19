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
r"""Resource operators for parametric single qubit operations."""
from typing import Dict

import numpy as np

import pennylane as qml
import pennylane.labs.resource_estimation as re

# pylint: disable=arguments-differ


def _rotation_resources(epsilon=10e-3):
    """An estimate on the number of T gates needed to implement a Pauli rotation. The estimate is taken from https://arxiv.org/abs/1404.5320."""
    gate_types = {}

    num_gates = round(1.149 * np.log2(1 / epsilon) + 9.2)
    t = re.ResourceT.resource_rep()
    gate_types[t] = num_gates

    return gate_types


class ResourceRZ(qml.RZ, re.ResourceOperator):
    r"""Resource class for the RZ gate.

    Resources:
        The resources are estimated by approximating the gate with a series of T gates.
        The estimate is taken from https://arxiv.org/abs/1404.5320.
    """

    @staticmethod
    def _resource_decomp(config) -> Dict[re.CompressedResourceOp, int]:
        return _rotation_resources(epsilon=config["error_rz"])

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})

    @classmethod
    def adjoint_resource_decomp(cls, config):
        return cls.resources(config)

    @classmethod
    def pow_resource_decomp(cls, z, config):
        return cls.resources(config)

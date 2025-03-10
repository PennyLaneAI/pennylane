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

"""Utility methods and dummy classes for testing the decomposition graph."""

# pylint: disable=too-few-public-methods

from collections import defaultdict

import pennylane as qml
from pennylane.operation import Operation

decompositions = defaultdict(list)


class CustomHadamard(Operation):
    r"""Hadamard(wires)
    The Hadamard operator"""

    resource_param_keys = ()

    name = "Hadamard"

    @property
    def resource_params(self) -> dict:
        return {}


class CustomRX(Operation):
    r"""
    The single qubit X rotation
    """

    resource_param_keys = ()

    name = "RX"

    @property
    def resource_params(self) -> dict:
        return {}


class CustomRY(Operation):
    r"""
    The single qubit Y rotation
    """

    resource_param_keys = ()

    name = "RY"

    @property
    def resource_params(self) -> dict:
        return {}


class CustomRZ(Operation):
    r"""
    The single qubit Z rotation
    """

    resource_param_keys = ()

    name = "RZ"

    @property
    def resource_params(self) -> dict:
        return {}


class CustomRot(Operation):
    r"""
    The single qubit arbitrary rotation
    """

    resource_param_keys = ()

    name = "Rot"

    @property
    def resource_params(self) -> dict:
        return {}


class CustomGlobalPhase(Operation):
    r"""Multiplies all components of the state by :math:`e^{-i \phi}`."""

    resource_param_keys = ()

    name = "GlobalPhase"

    @property
    def resource_params(self) -> dict:
        return {}


class CustomPhaseShift(Operation):
    """Phase shift gate."""

    resource_param_keys = ()

    name = "PhaseShift"

    @property
    def resource_params(self) -> dict:
        return {}


@qml.register_resources({CustomRZ: 2, CustomRX: 1, CustomGlobalPhase: 1})
def _hadamard_to_rz_rx(*_, **__):
    raise NotImplementedError


@qml.register_resources({CustomRZ: 1, CustomRY: 1, CustomGlobalPhase: 1})
def _hadamard_to_rz_ry(*_, **__):
    raise NotImplementedError


decompositions[CustomHadamard] = [_hadamard_to_rz_rx, _hadamard_to_rz_ry]


@qml.register_resources({CustomRX: 1, CustomRZ: 2})
def _ry_to_rx_rz(*_, **__):
    raise NotImplementedError


decompositions[CustomRY] = [_ry_to_rx_rz]

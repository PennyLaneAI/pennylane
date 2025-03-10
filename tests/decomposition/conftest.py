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

from collections import defaultdict

import pennylane as qml
from pennylane.operation import Operation


decompositions = defaultdict(list)


class Hadamard(Operation):
    r"""Hadamard(wires)
    The Hadamard operator"""

    resource_param_keys = ()

    @property
    def name(self) -> str:
        return "Hadamard"

    @property
    def resource_params(self) -> dict:
        return {}


class RX(Operation):
    r"""
    The single qubit X rotation
    """

    resource_param_keys = ()

    @property
    def resource_params(self) -> dict:
        return {}


class RY(Operation):
    r"""
    The single qubit Y rotation
    """

    resource_param_keys = ()

    @property
    def resource_params(self) -> dict:
        return {}


class RZ(Operation):
    r"""
    The single qubit Z rotation
    """

    resource_param_keys = ()

    @property
    def resource_params(self) -> dict:
        return {}


class Rot(Operation):
    r"""
    The single qubit arbitrary rotation
    """

    resource_param_keys = ()

    @property
    def resource_params(self) -> dict:
        return {}


class GlobalPhase(Operation):
    r"""Multiplies all components of the state by :math:`e^{-i \phi}`."""

    resource_param_keys = ()

    @property
    def resource_params(self) -> dict:
        return {}


@qml.register_resources({RZ: 2, RX: 1, GlobalPhase: 1})
def _hadamard_to_rz_rx(*_, **__):
    raise NotImplementedError


@qml.register_resources({RZ: 1, RY: 1, GlobalPhase: 1})
def _hadamard_to_rz_ry(*_, **__):
    raise NotImplementedError


decompositions[Hadamard] = [_hadamard_to_rz_rx, _hadamard_to_rz_ry]


@qml.register_resources({RX: 1, RZ: 2})
def _ry_to_rx_rz(*_, **__):
    raise NotImplementedError


decompositions[RY] = [_ry_to_rx_rz]

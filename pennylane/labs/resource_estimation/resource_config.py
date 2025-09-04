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
r"""This module contains the base class for qubit management"""

from __future__ import annotations

from collections.abc import Callable
from pennylane.labs.resource_estimation.resource_operator import ResourceOperator


class ResourceConfig:
    r"""A container to track the configuration for errors, precisions, and custom decompositions for the
    resource estimation pipeline.
    """

    def __init__(self) -> None:
        self.conf = {
            "error_rx": 1e-9,
            "error_ry": 1e-9,
            "error_rz": 1e-9,
            "precision_select_pauli_rot": 1e-9,
            "precision_qubit_unitary": 1e-9,
            "precision_qrom_state_prep": 1e-9,
            "precision_mps_prep": 1e-9,
            "precision_alias_sampling": 1e-9,
        }
        self._decomp_tracker = {}
        self._adj_decomp_tracker = {}
        self._ctrl_decomp_tracker = {}
        self._pow_decomp_tracker = {}

    def __str__(self):
        return f"ResourceConfig(conf = {self.conf}, decomps = {self._decomp_tracker}, {self._adj_decomp_tracker}, {self._ctrl_decomp_tracker}, {self._pow_decomp_tracker})"

    def __repr__(self) -> str:
        return f"ResourceConfig(conf = {self.conf}), decomps = {self._decomp_tracker}, {self._adj_decomp_tracker}, {self._ctrl_decomp_tracker}, {self._pow_decomp_tracker}"

    def set_decomp(
        self, op_type: type[ResourceOperator], decomp_func: Callable, type: str = None
    ) -> None:
        if type == "adj":
            self._adj_decomp_tracker[op_type] = decomp_func
        elif type == "ctrl":
            self._ctrl_decomp_tracker[op_type] = decomp_func
        elif type == "pow":
            self._pow_decomp_tracker[op_type] = decomp_func
        else:
            self._decomp_tracker[op_type] = decomp_func

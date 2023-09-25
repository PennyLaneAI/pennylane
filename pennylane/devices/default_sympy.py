# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains the SymPy device"""
from typing import Sequence, Union

import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane.devices import Device, DefaultExecutionConfig, ExecutionConfig


class DefaultSympy(Device):
    """TODO."""

    name: str = "default.sympy"

    def execute(
        self,
        circuits: Union[QuantumTape, Sequence[QuantumTape]],
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        return 0.0 if isinstance(circuits, qml.tape.QuantumScript) else tuple(0.0 for c in circuits)

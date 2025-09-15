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
"""
This module contains functions to import circuits and objects from external frameworks into PennyLane.
"""

from .io import (
    from_pyquil,
    from_qasm,
    from_qiskit,
    from_qiskit_noise,
    from_qiskit_op,
    from_quil,
    from_quil_file,
    plugin_converters,
    from_qasm3,
)
from .qualtran_io import FromBloq, bloq_registers, to_bloq, ToBloq
from .to_openqasm import to_openqasm

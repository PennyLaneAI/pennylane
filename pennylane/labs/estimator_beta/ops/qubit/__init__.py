# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""This module contains alternate decompositions for single qubit operations."""

from .non_parametric_ops import (
    hadamard_controlled_resource_decomp,
    hadamard_toffoli_based_controlled_decomp,
)
from .parametric_ops_multi_qubit import paulirot_controlled_resource_decomp

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
This submodule offers all the non-operator/ measurement custom primitives
created in the ftqc module.
"""

from .parametric_midmeasure import _create_parametrized_mid_measure_primitive

measure_in_basis_prim = _create_parametrized_mid_measure_primitive()

__all__ = [
    "measure_in_basis_prim",
]

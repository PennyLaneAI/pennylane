# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
Condensed vibrational spectroscopy module for Pennylane.

This module extends the standard vibrational spectroscopy capabilities to include
condensed phase effects using QMM (Quantum Mechanical/Molecular Mechanical) methods.
"""

from .pes_driver import condensed_vibrational_pes

__all__ = [
    "condensed_vibrational_pes"
]

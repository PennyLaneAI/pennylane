# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The PennyLane quantum chemistry package. Supports OpenFermion, PySCF,
and Psi4 for quantum chemistry calculations using PennyLane."""
from .structure import *
from .obs import *

__all__ = [
    "read_structure",
    "meanfield",
    "active_space",
    "decompose",
    "convert_observable",
    "molecular_hamiltonian",
    "hf_state",
    "excitations",
    "excitations_to_wires",
    "_qubit_operator_to_terms",
    "_terms_to_qubit_operator",
    "_qubit_operators_equivalent",
    "obs",
]

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
r"""
This subpackage contains experimental PennyLane transforms and their building blocks.

.. currentmodule:: pennylane.labs.transforms

Transforms
~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~rot_to_phase_gradient
    ~select_pauli_rot_phase_gradient

Custom decomposition rules
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~make_rz_to_phase_gradient_decomp
    ~make_selectpaulirot_to_phase_gradient_decomp

"""

from .select_pauli_rot_phase_gradient import select_pauli_rot_phase_gradient
from .decomp_rz_phase_gradient import make_rz_to_phase_gradient_decomp
from .decomp_selectpaulirot_phase_gradient import make_selectpaulirot_to_phase_gradient_decomp

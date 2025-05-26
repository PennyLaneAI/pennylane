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
Quantum circuit intermediate representations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Intermediate representations (IRs) are alternative representations of quantum circuits, typically offering a more efficient classical description for special classes of circuits.

.. currentmodule:: pennylane.labs.intermediate_reps

.. autosummary::
    :toctree: api

    ~parity_matrix
    ~phase_polynomial


"""

from .parity_matrix import parity_matrix
from .phase_polynomial import phase_polynomial

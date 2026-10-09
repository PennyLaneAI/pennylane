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
"""
This module contains utilities for testing PennyLane objects, such as operators,
decomposition rules, and captured programs.

Operators
^^^^^^^^^

.. currentmodule:: pennylane.testing

.. autosummary::
    :toctree: api

    ~assert_valid

Decompositions
^^^^^^^^^^^^^^

.. currentmodule:: pennylane.testing

.. autosummary::
    :toctree: api

    ~assert_valid_decomp_rule
    ~decomp_rule_to_tape

Program capture
^^^^^^^^^^^^^^^

.. currentmodule:: pennylane.testing

.. autosummary::
    :toctree: api

    ~plxpr_to_tape
    ~assert_eqn_matches_op
    ~extract_all_primitives
    ~find_eqns

"""

from .capture import assert_eqn_matches_op, extract_all_primitives, find_eqns, plxpr_to_tape
from .decompositions import assert_valid_decomp_rule, decomp_rule_to_tape
from .operators import assert_valid

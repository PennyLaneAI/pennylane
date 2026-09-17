# Copyright 2018-2026 Xanadu Quantum Technologies Inc.

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
This module contains templates, which are pre-coded routines that can be used in a quantum node.

.. currentmodule:: pennylane.labs.templates

.. autosummary::
    :toctree: api

    ~half_signed_out_multiplier
    ~SumOfSlatersPrep2
    ~uniform_prep_ops
    ~select_thc
    ~select_thc_wires
    ~qubitization_thc
    ~qubitization_thc_wires
    ~one_body_walk
    ~one_body_walk_wires
"""

from .half_signed_out_multiplier import half_signed_out_multiplier
from .sum_of_slaters2 import SumOfSlatersPrep2
from .select_thc import select_thc, select_thc_wires
from .qubitization_thc import qubitization_thc, qubitization_thc_wires
from .one_body_walk import one_body_walk, one_body_walk_wires

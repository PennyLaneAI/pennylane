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
"""A module containing utility functions and mappings for working with bosonic operators.

Overview
--------

This module contains functions and classes for creating and manipulating bosonic operators.


BoseWord and BoseSentence
---------------------------

.. currentmodule:: pennylane.bose

.. autosummary::
    :toctree: api

    ~BoseWord
    ~BoseSentence

Mapping to qubit operators
--------------------------

.. currentmodule:: pennylane.bose

.. autosummary::
    :toctree: api

    ~binary_mapping
    ~unary_mapping
    ~christiansen_mapping

"""

from .bosonic import BoseSentence, BoseWord
from .bosonic_mapping import binary_mapping, christiansen_mapping, unary_mapping

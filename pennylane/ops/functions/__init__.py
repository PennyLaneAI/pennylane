# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
This module contains functions that act on operators and tapes.

.. currentmodule:: pennylane

.. autosummary::
    :toctree: api

    ~ops.functions.bind_new_parameters
    ~dot
    ~eigvals
    ~equal
    ~evolve
    ~generator
    ~is_commuting
    ~is_hermitian
    ~is_unitary
    ~map_wires
    ~matrix
    ~simplify

"""
from .bind_new_parameters import bind_new_parameters
from .dot import dot
from .eigvals import eigvals
from .equal import equal
from .evolve import evolve
from .generator import generator
from .is_commuting import is_commuting
from .is_hermitian import is_hermitian
from .is_unitary import is_unitary
from .map_wires import map_wires
from .matrix import matrix
from .simplify import simplify

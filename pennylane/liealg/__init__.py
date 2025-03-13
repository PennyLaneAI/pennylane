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
Overview
--------

This module contains Lie algebra functionality.

Functions
^^^^^^^^^

.. currentmodule:: pennylane.liealg

.. autosummary::
    :toctree: api

    ~lie_closure
    ~structure_constants
    ~center

Relevant demos
--------------

Lie algebra functionality in PennyLane.
Check out the following demos to learn more about Lie algebras in the context of quantum computation:

* :doc:`Introducing (dynamical) Lie algebras for quantum practitioners <demos/tutorial_liealgebra>`
* :doc:`g-sim: Lie-algebraic classical simulations for variational quantum computing <demos/tutorial_liesim>`
* :doc:`(g + P)-sim: Extending g-sim by non-DLA observables and gates <demos/tutorial_liesim_extension>`
* :doc:`Fixed depth Hamiltonian simulation via Cartan decomposition <demos/tutorial_fixed_depth_hamiltonian_simulation_via_cartan_decomposition>`
* :doc:`The KAK decomposition <demos/tutorial_kak_decomposition>`



"""

from .structure_constants import structure_constants
from .center import center
from .lie_closure import lie_closure

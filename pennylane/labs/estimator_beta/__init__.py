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
r"""
This module contains experimental features for
resource estimation.

.. warning::

    This module is experimental. Frequent changes will occur,
    with no guarantees of stability or backwards compatibility.

.. currentmodule:: pennylane.labs.estimator_beta

Alternate Decompositions
------------------------

.. autosummary::
    :toctree: api

    ~hadamard_controlled_resource_decomp
    ~hadamard_toffoli_based_controlled_decomp
    ~ch_resource_decomp
    ~ch_toffoli_based_resource_decomp

"""

from pennylane.estimator import *
from .ops import (
    hadamard_controlled_resource_decomp,
    hadamard_toffoli_based_controlled_decomp,
    ch_resource_decomp,
    ch_toffoli_based_resource_decomp,
)

# Monkey Patching the resource decomposition methods onto the relevant classes.
Hadamard.controlled_resource_decomp = classmethod(hadamard_controlled_resource_decomp)
Hadamard.toffoli_based_controlled_decomp = classmethod(hadamard_toffoli_based_controlled_decomp)
CH.resource_decomp = classmethod(ch_resource_decomp)
CH.toffoli_based_resource_decomp = classmethod(ch_toffoli_based_resource_decomp)

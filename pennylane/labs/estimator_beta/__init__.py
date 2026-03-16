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

.. currentmodule:: pennylane.labs.estimator_beta.ops

.. autosummary::
    :toctree: api

    ~pauliRot_controlled_resource_decomp
    ~selectPauliRot_toffoli_based_controlled_decom


"""


from pennylane.estimator import *
from .ops import pauliRot_controlled_resource_decomp
from .templates import selectPauliRot_controlled_resource_decomp


# Monkey Patching the resource decomposition methods onto the relevant classes.

PauliRot.controlled_resource_decomp = classmethod(pauliRot_controlled_resource_decomp)
SelectPauliRot.controlled_resource_decomp = classmethod(selectPauliRot_controlled_resource_decomp)

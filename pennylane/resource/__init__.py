# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

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
The ``resource`` module provides classes and functionality to track the quantum resources
(number of qubits, circuit depth, etc.) required to implement advanced quantum algorithms.

.. seealso::
    The :mod:`~.estimator` module for higher level resource estimation of quantum programs.

Circuit Specifications (specs)
------------------------------

.. currentmodule:: pennylane

.. autosummary::
    :toctree: api

    ~specs

Circuit Specification Classes and Utilities
-------------------------------------------

.. currentmodule:: pennylane.resource

.. autosummary::
    :toctree: api

    ~CircuitSpecs
    ~Expression
    ~SpecsResources
    ~SymbolicSpecsResources

    ~resources_from_tape
"""

from .resource import (
    SpecsResources,
    SymbolicSpecsResources,
    CircuitSpecs,
    resources_from_tape,
)
from .expression import Expression
from .specs import specs

__all__ = [
    "SpecsResources",
    "SymbolicSpecsResources",
    "CircuitSpecs",
    "resources_from_tape",
    "Expression",
    "specs",
]

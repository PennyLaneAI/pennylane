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
"""The core abstractions of PennyLane.

Operator Types
~~~~~~~~~~~~~~

.. currentmodule:: pennylane.core.operator
.. autosummary::
    :toctree: api

    ~Operator
    ~Operation
    ~CV
    ~CVObservable
    ~CVOperation
    ~Channel
    ~StatePrepBase

.. currentmodule:: pennylane.core.operator

.. inheritance-diagram:: Operator Operation Channel CV CVObservable CVOperation StatePrepBase
    :parts: 1


"""

from .operator import (
    CV,
    Channel,
    CVObservable,
    CVOperation,
    Operation,
    Operator,
    Operator2,
    StatePrepBase,
)

__all__ = [
    "Operator",
    "Operator2",
    "Operation",
    "Channel",
    "CV",
    "CVOperation",
    "CVObservable",
    "StatePrepBase",
]

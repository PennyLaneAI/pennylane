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

r"""Built-in decomposition rules, organized like their operator and template modules.

The modules directly beneath this package mirror :mod:`pennylane.ops`, and the
``templates`` branch mirrors :mod:`pennylane.templates`.
The drawing-specific ``LabelledOp`` rule remains in :mod:`pennylane.drawer.label`.
Rules are loaded automatically when PennyLane is imported.
Use :func:`pennylane.list_decomps` to retrieve the registered rules for an operator.
Import operator-specific rules and helpers from their corresponding rule modules.

.. currentmodule:: pennylane.decomposition_rules

.. rubric:: Built-in rules

.. autosummary::

    qubit
    common
    null_decomp
    identity
    meta
    templates

.. rubric:: Decomposition algorithms

.. autosummary::

    one_qubit_decomposition
    two_qubit_decomposition
    multi_qubit_decomposition
"""

from .common import null_decomp
from .qubit.matrix_ops import (
    one_qubit_decomposition,
    two_qubit_decomposition,
    multi_qubit_decomposition,
)
from . import identity, meta, qubit, templates

__all__ = [
    "common",
    "identity",
    "meta",
    "null_decomp",
    "qubit",
    "templates",
    "one_qubit_decomposition",
    "two_qubit_decomposition",
    "multi_qubit_decomposition",
]

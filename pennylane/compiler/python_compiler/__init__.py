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
This submodule contains the API for the integration of PennyLane and Catalyst with xDSL.

.. currentmodule:: pennylane.compiler.python_compiler

.. warning::

    This module is currently experimental and will not maintain API stability between releases.

.. automodapi:: pennylane.compiler.python_compiler
    :no-heading:
    :include-all-objects:

Available Transforms
--------------------

.. automodapi:: pennylane.compiler.python_compiler.transforms
    :no-heading:
    :include-all-objects:

Transforms Core API
-------------------

.. automodapi:: pennylane.compiler.python_compiler.transforms.api
    :no-heading:
    :include-all-objects:

JAX Utilities
-------------

.. automodapi:: pennylane.compiler.python_compiler.jax_utils
    :no-heading:
    :include-all-objects:

"""

from .compiler import Compiler
from .quantum_dialect import QuantumDialect as Quantum

__all__ = [
    "Compiler",
    "Quantum",
]

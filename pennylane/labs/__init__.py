# Copyright 2024 Xanadu Quantum Technologies Inc.

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
.. currentmodule:: pennylane

This module contains experimental features enabling
advanced quantum computing research.

.. warning::

    This module is experimental. Frequent changes will occur,
    with no guarantees of stability or backwards compatibility.

<<<<<<< HEAD
.. currentmodule:: pennylane.labs.resource_estimation

Resource Estimation
~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~Resources
    ~CompressedResourceOp
    ~ResourceConstructor

Errors
~~~~~~

.. autosummary::
    :toctree: api

    ~ResourcesNotDefined
=======
>>>>>>> f628f21863b996c5ad05e524a90c0bb2c2472396
"""

from .resource_estimation import (
    Resources,
    CompressedResourceOp,
    ResourceConstructor,
    ResourcesNotDefined,
)

__all__ = ["Resources", "CompressedResourceOp"]

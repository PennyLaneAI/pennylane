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
.. currentmodule:: pennylane

This module contains experimental features for supporting fault-tolerant workloads in PennyLane

.. currentmodule:: pennylane.ftqc

Modules
~~~~~~~

.. autosummary::
    :toctree: api

"""
from warnings import warn
from pennylane import ExperimentalWarning

warn(
    ExperimentalWarning(
        "This module is currently experimental and will not maintain API stability between releases."
    )
)

__all__ = []

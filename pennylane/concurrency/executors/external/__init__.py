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
Submodule for concurrent executors relying on 3rd-party packages.

.. currentmodule:: pennylane.concurrency.executor

All executor functionality in this module is implemented using external packages to handle execution and orchestration.

.. currentmodule:: pennylane.concurrency.executors.external

.. autosummary::
    :toctree: api

    ~dask.DaskExec
    ~mpi.MPICommExec
    ~mpi.MPIPoolExec

"""

from .dask import DaskExec
from .mpi import MPICommExec, MPIPoolExec

__all__ = [
    "DaskExec",
    "MPICommExec",
    "MPIPoolExec",
]

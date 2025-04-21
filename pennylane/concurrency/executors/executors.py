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
"""
Contains concurrent executor abstractions for task-based workloads.
"""

from enum import Enum

from .dask import DaskExec
from .mpi import MPICommExec, MPIPoolExec
from .native import MPPoolExec, ProcPoolExec, SerialExec, ThreadPoolExec


class ExecBackends(Enum):
    """
    Supported executor backends.
    """

    MP_Pool = MPPoolExec
    CF_ProcPool = ProcPoolExec
    CF_ThreadPool = ThreadPoolExec
    Serial = SerialExec
    Dask = DaskExec
    MPI_PoolEx = MPIPoolExec
    MPI_CommEx = MPICommExec


def get_executor(backend: ExecBackends = ExecBackends.MP_Pool):
    """
    Return the associated class type from the provided enumerated backends.
    """
    return backend.value


def create_executor(backend: ExecBackends = ExecBackends.MP_Pool, **kwargs):
    """
    Create an instance of the specified executor backend with forwarded keyword arguments
    """
    return backend.value(**kwargs)

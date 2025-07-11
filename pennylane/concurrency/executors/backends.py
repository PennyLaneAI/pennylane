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
Contains concurrent executor abstractions for task-based workloads.
"""

from enum import Enum

from .external import DaskExec, MPICommExec, MPIPoolExec
from .native import MPPoolExec, ProcPoolExec, SerialExec, ThreadPoolExec


class ExecBackends(Enum):
    """
    Supported executor backends.

    The enumerated options provide a mapping to the implementation-defined classes for task-based executor backends.

    .. note::
        Not all backends are guaranteed to be instantiable without additional package installations.
    """

    MP_Pool = MPPoolExec
    CF_ProcPool = ProcPoolExec
    CF_ThreadPool = ThreadPoolExec
    Serial = SerialExec
    Dask = DaskExec
    MPI_PoolEx = MPIPoolExec
    MPI_CommEx = MPICommExec


_ExecBackendsMap = {
    "mp_pool": MPPoolExec,
    "cf_procpool": ProcPoolExec,
    "cf_threadpool": ThreadPoolExec,
    "serial": SerialExec,
    "dask": DaskExec,
    "mpi4py_pool": MPIPoolExec,
    "mpi4py_comm": MPICommExec,
}


def get_supported_backends():
    """
    Return the list of backends with implementation support.

    .. note::
        Not all backends are guaranteed to be instantiable without additional package installations.
    """
    return _ExecBackendsMap


def get_executor(backend: ExecBackends | str = ExecBackends.MP_Pool):
    """
    Return the associated class type from the provided enumerated backends.
    """
    if isinstance(backend, ExecBackends):
        return backend.value
    return _ExecBackendsMap[backend]


def create_executor(backend: ExecBackends | str = ExecBackends.MP_Pool, **kwargs):
    """
    Create an instance of the specified executor backend with forwarded keyword arguments
    """
    if isinstance(backend, ExecBackends):

        return backend.value(**kwargs)
    return _ExecBackendsMap[backend](**kwargs)

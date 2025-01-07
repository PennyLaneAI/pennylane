# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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

import abc
import os
import sys
from collections.abc import Callable, Sequence
from functools import singledispatchmethod
from types import NoneType


class RemoteExecABC(abc.ABC):
    """
    Abstract base class for defining a task-based parallel executor backend for running python functions
    """

    def __init__(self, *args, **kwargs): ...
    @abc.abstractmethod
    def __call__(self, fn: Callable, data: Sequence):
        """
        fn:     the callable function to run on the executor backend
        data:   is a sequence where each work-item is a packaged chunk for execution
        """
        ...

    @property
    def size(self):
        return self._size


class IntExecABC(RemoteExecABC, abc.ABC):
    """
    Executor class for native Python library concurrency support
    """

    pass


class ExtExecABC(RemoteExecABC, abc.ABC):
    """
    Executor class for external package provided concurrency support
    """

    pass


class MPIPoolExec(ExtExecABC):
    """
    MPIPoolExecutor abstraction class functor.
    """

    def __init__(self, max_workers=None):
        from mpi4py import MPI  # Required to call MPI_Init
        from mpi4py.futures import MPIPoolExecutor as executor

        self._exec_backend = executor
        self._size = MPI.COMM_WORLD.Get_size() if max_workers is None else max_workers

    def __call__(self, fn: Callable, data: Sequence):
        kwargs = {"use_pkl5": True}
        chunksize = max(len(data) // self._size, 1)
        with self._exec_backend(max_workers=self.size, **kwargs) as executor:
            output_f = executor.map(fn, data, chunksize=chunksize)
        return output_f

    @property
    def size(self):
        return self._size


class MPICommExec(ExtExecABC):
    """
    MPICommExecutor abstraction class functor. To be used if dynamic process spawning required by MPIPoolExec is unsupported by the MPI implementation.
    """

    def __init__(self):
        from mpi4py import MPI  # Required to call MPI_Init
        from mpi4py.futures import MPICommExecutor as executor

        self._exec_backend = executor
        self._comm = MPI.COMM_WORLD
        self._size = MPI.COMM_WORLD.Get_size()

    def __call__(self, fn: Callable, data: Sequence):
        kwargs = {"use_pkl5": True}
        chunksize = max(len(data) // self._size, 1)

        with self._exec_backend(self._comm, root=0) as executor:
            if executor is not None:
                output_f = executor.map(fn, data, chunksize=chunksize)
            else:
                raise RuntimeError(f"Failed to start executor {self._exec_backend}")
        return output_f

    @property
    def size(self):
        return self._size


class DaskExec(ExtExecABC):
    """
    Dask distributed abstraction class functor.
    """

    try:
        from dask.distributed.deploy import Cluster
    except:
        Cluster = None

    def __init__(self, max_workers=4, client_provider=None):
        from dask.distributed import Client, LocalCluster

        if client_provider is None:
            cluster = LocalCluster(n_workers=max_workers, processes=True)
            self._exec_backend = Client(cluster)

        # Note: urllib does not validate
        # (see https://docs.python.org/3/library/urllib.parse.html#url-parsing-security),
        # so branch on str as URL
        elif isinstance(client_provider, str):
            self._exec_backend = Client(client_provider)

        elif isinstance(client_provider, Cluster):
            self._exec_backend = client_provider.get_client()

        self._size = len(self._exec_backend.scheduler_info()["workers"])

    def __call__(self, fn: Callable, data: Sequence):
        output_f = self._exec_backend.map(fn, data)
        return [o.result() for o in output_f]

    @property
    def size(self):
        return self._size


class PyNativeExecABC(IntExecABC, abc.ABC):
    def __init__(self, max_workers=None):
        if max_workers:
            self._size = max_workers
        elif sys.version_info.minor >= 13:
            self._size = os.process_cpu_count()
        else:
            self._size = os.cpu_count()

    def __call__(self, fn: Callable, data: Sequence):
        exec_cls = self._exec_backend()
        chunksize = max(len(data) // self._size, 1)
        with exec_cls(self._size) as executor:
            output_f = executor.map(fn, data, chunksize=chunksize)
        return output_f

    @property
    def size(self):
        return self._size

    @classmethod
    @abc.abstractmethod
    def _exec_backend(cls): ...


class MPPoolExec(PyNativeExecABC):
    """
    concurrent.futures.ProcessPoolExecutor abstraction class functor.
    """

    @classmethod
    def _exec_backend(cls):
        from multiprocessing import Pool as exec

        return exec


class ProcPoolExec(PyNativeExecABC):
    """
    concurrent.futures.ProcessPoolExecutor abstraction class functor.
    """

    @classmethod
    def _exec_backend(cls):
        from concurrent.futures import ProcessPoolExecutor as exec

        return exec


class ThreadPoolExec(PyNativeExecABC):
    """
    concurrent.futures.ThreadPoolExecutor abstraction class functor.
    """

    @classmethod
    def _exec_backend(cls):
        from concurrent.futures import ThreadPoolExecutor as exec

        return exec


class RayExec(ExtExecABC):
    """
    Ray abstraction class functor.
    """

    pass

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
Contains concurrent executor abstractions for task-based workloads backed by mpi4py.
"""

from collections.abc import Callable, Sequence

from .base import ExtExecABC


class MPIPoolExec(ExtExecABC):
    """
    MPIPoolExecutor abstraction class functor.
    """

    def __init__(self, max_workers=None, **kwargs):
        super().__init__(max_workers=max_workers, **kwargs)

        # Imports will initialise the MPI environment.
        # Handle set-up upon object creation only.
        from mpi4py import MPI
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
    MPICommExecutor abstraction class functor. To be used if dynamic process spawning
    required by MPIPoolExec is unsupported by the MPI implementation.
    """

    def __init__(self, max_workers=None, **kwargs):
        super().__init__(max_workers=max_workers, **kwargs)

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

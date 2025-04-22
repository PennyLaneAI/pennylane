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
from typing import Any

from .base import ExecBackendConfig, ExtExec


class MPIPoolExec(ExtExec):
    """
    MPIPoolExecutor abstraction class executor.

    This executor wraps the mpi4py ``mpi4py.futures.MPIPoolExecutor`` API, and provides support for execution using multiple processes launched using MPI.
    For an example script ``my_script.py``, and an installed mpi4py library with the active MPI environment, the executor can be used as follows:

    .. code-block:: console

        $ mpirun -n 4 -m mpi4py.futures my_script.py

    All calls to the executor as synchronous, and do not currently support the use of futures as a return object.
    """

    def __init__(self, max_workers=None, **kwargs):
        super().__init__(max_workers=max_workers, **kwargs)

        # Imports will initialise the MPI environment.
        # Handle set-up upon object creation only.
        from mpi4py import MPI
        from mpi4py.futures import MPIPoolExecutor as executor

        self._exec_backend = executor
        self._size = MPI.COMM_WORLD.Get_size() if max_workers is None else max_workers

        self._cfg = ExecBackendConfig(
            submit_fn="submit",
            map_fn="map",
            starmap_fn="starmap",
            shutdown_fn="shutdown",
            submit_unpack=True,
            map_unpack=True,
            blocking=False,
        )

    def __call__(self, fn: Callable, data: Sequence):
        kwargs = {"use_pkl5": True}
        chunksize = max(len(data) // self._size, 1)
        with self._exec_backend(max_workers=self.size, **kwargs) as executor:
            output_f = executor.map(fn, data, chunksize=chunksize)
        return output_f

    @property
    def size(self):
        return self._size

    def submit(self, fn: Callable, *args, **kwargs):
        exec_args = {"use_pkl5": True}

        with self._exec_backend(max_workers=self.size, **exec_args) as executor:
            output_f = executor.submit(fn, *args, **kwargs)
        return output_f.result()

    def map(self, fn: Callable, *args: Sequence[Any], **kwargs):
        exec_args = {"use_pkl5": True}
        chunksize = max(len(args) // self._size, 1)

        with self._exec_backend(max_workers=self.size, **exec_args) as executor:
            output_f = executor.map(fn, *args, chunksize=chunksize, **kwargs)
        return list(output_f)

    def starmap(self, fn: Callable, args: Sequence[tuple], **kwargs):
        exec_args = {"use_pkl5": True}
        chunksize = max(len(args) // self._size, 1)

        with self._exec_backend(max_workers=self.size, **exec_args) as executor:
            output_f = executor.starmap(fn, args, chunksize=chunksize, **kwargs)
        return list(output_f)

    def shutdown(self):
        "Shutdown the executor backend, if valid."
        if self._persist:
            self._shutdown_fn(self._backend)()
            self._backend = None

    def __del__(self):
        self.shutdown()


class MPICommExec(ExtExec):
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

        self._cfg = ExecBackendConfig(
            submit_fn="submit",
            map_fn="map",
            starmap_fn="starmap",
            shutdown_fn="shutdown",
            submit_unpack=True,
            map_unpack=True,
            blocking=False,
        )

    def __call__(self, fn: Callable, data: Sequence):
        exec_args = {"use_pkl5": True}
        chunksize = max(len(data) // self._size, 1)

        with self._exec_backend(self._comm, root=0, **exec_args) as executor:
            if executor is not None:
                output_f = executor.map(fn, data, chunksize=chunksize)
            else:
                raise RuntimeError(f"Failed to start executor {self._exec_backend}")
        return output_f

    def submit(self, fn: Callable, *args, **kwargs):
        exec_args = {"use_pkl5": True}

        with self._exec_backend(max_workers=self.size, **exec_args) as executor:
            output_f = executor.submit(fn, *args, **kwargs)
        return output_f.result()

    def map(self, fn: Callable, *args: Sequence[Any], **kwargs):
        exec_args = {"use_pkl5": True}
        chunksize = max(len(args) // self._size, 1)

        with self._exec_backend(max_workers=self.size, **exec_args) as executor:
            output_f = executor.map(fn, *args, chunksize=chunksize, **kwargs)
        return list(output_f)

    def starmap(self, fn: Callable, args: Sequence[tuple], **kwargs):
        exec_args = {"use_pkl5": True}
        chunksize = max(len(args) // self._size, 1)

        with self._exec_backend(max_workers=self.size, **exec_args) as executor:
            output_f = executor.starmap(fn, args, chunksize=chunksize, **kwargs)
        return list(output_f)

    @property
    def size(self):
        return self._size

    def shutdown(self):
        "Shutdown the executor backend, if valid."
        if self._persist:
            self._shutdown_fn(self._backend)()
            self._backend = None

    def __del__(self):
        self.shutdown()

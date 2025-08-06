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
.. currentmodule:: pennylane.concurrency.executors.external.mpi

Contains concurrent executor abstractions for task-based workloads backed by mpi4py.
"""

from collections.abc import Callable, Sequence
from typing import Any
import time


from ..base import ExecBackendConfig, ExtExec


# pylint: disable=import-outside-toplevel
class MPIPoolExec(ExtExec):  # pragma: no cover
    r"""
    MPIPoolExecutor abstraction class executor.

    This executor wraps the `mpi4py.futures.MPIPoolExecutor <https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html#mpipoolexecutor>`_ class, and provides support for execution using multiple processes launched using MPI.
    For an example script ``my_script.py``, and an installed mpi4py library with the active MPI environment, the executor can be used as follows:

    .. code-block:: console

        $ mpirun -n 4 -m mpi4py.futures my_script.py

    See `mpi4py.futures - Command line <https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html#command-line>`_ for additional details on launching jobs.

    .. note::
        All calls to the executor are synchronous, and do not currently support the use of futures as a return object.

    Args:
        max_workers (int, optional): Maximum number of worker processes to use. If None, defaults to the number of available MPI processes.
        persist (bool): Whether to persist the executor. Currently not supported and will raise an error if True.
        use_pkl5 (bool): Whether to use the pkl5 protocol for serialization. Defaults to True
        profile (bool): Whether to enable profiling of function execution. Defaults to False.
        **kwargs: Additional keyword arguments to pass through to the executor backend.


    """

    def __init__(self, max_workers: int | None = None, persist: bool = False, **kwargs):
        if persist:
            raise RuntimeError("The MPIPoolExec backend does not currently support persistence.")
        super().__init__(max_workers=max_workers, **kwargs)

        # Imports will initialise the MPI environment.
        # Handle set-up upon object creation only.
        from mpi4py import MPI

        self._size = max_workers
        self._comm = MPI.COMM_WORLD

        self._cfg = ExecBackendConfig(
            submit_fn="submit",
            map_fn="map",
            starmap_fn="starmap",
            shutdown_fn="shutdown",
            submit_unpack=True,
            map_unpack=True,
            blocking=True,
        )
        
        self._profile_fn = kwargs.pop("profile", False)
        self._use_pkl5 = kwargs.pop("use_pkl5", True) 

    def __call__(self, dispatch: str, fn: Callable, *args, **kwargs):
        r"""
        dispatch:   the named method to pass the function parameters
        fn:         the callable function to run on the executor backend
        args:       the arguments to pass to ``fn``
        kwargs:     the keyword arguments to pass to ``fn``
        """
        kwargs.update({"use_pkl5": self._use_pkl5})
        return super().__call__(
            dispatch,
            fn,
            *args,
            **kwargs,
        )

    @property
    def size(self):
        return self._size

    def submit(self, fn: Callable, *args, **kwargs):
        with self._exec_backend()(max_workers=self.size, use_pkl5=self._use_pkl5) as executor:
            output_f = executor.submit(fn, *args, **kwargs)
        return output_f.result()

    def map(self, fn: Callable, *args: Sequence[Any], **kwargs):
        chunksize = max(len(args) // self._size, 1)

        if self._profile_fn:
            
            self.call_fn = fn
            
            with self._exec_backend()(max_workers=self.size, use_pkl5=self._use_pkl5) as executor:
                output_f = executor.map(self.profile_fn, *args, chunksize=chunksize, **kwargs)

            return list(output_f)
            
        with self._exec_backend()(max_workers=self.size, use_pkl5=self._use_pkl5) as executor:
            output_f = executor.map(fn, *args, chunksize=chunksize, **kwargs)
        return list(output_f)

    def starmap(self, fn: Callable, args: Sequence[tuple], **kwargs):
        chunksize = max(len(args) // self._size, 1)

        if self._profile_fn:
            
            self.call_fn = fn
            
            with self._exec_backend()(max_workers=self.size, use_pkl5=self._use_pkl5) as executor:
                output_f = executor.starmap(self.profile_fn, args, chunksize=chunksize, **kwargs)

            return list(output_f)
            
        with self._exec_backend()(max_workers=self.size, use_pkl5=self._use_pkl5) as executor:
            output_f = executor.starmap(fn, args, chunksize=chunksize, **kwargs)
        return list(output_f)

    def shutdown(self):
        "Shutdown the executor backend, if valid."

    def __del__(self):
        self.shutdown()
        
    def profile_fn(self, *args, **kwargs):
        """
        A function to profile the execution of the executor.
        This is a placeholder for any profiling logic that may be added in the future.
        """
        # print("Profiling function execution...")  # Placeholder for profiling logic
        from mpi4py import MPI

        rank = MPI.COMM_WORLD.Get_rank()
        size = MPI.COMM_WORLD.Get_size()
        
        start = time.perf_counter()
        results = self.call_fn(*args, **kwargs)  # Call the function to be profiled
        end = time.perf_counter()


        print(f"Rank {rank:<3}/ {size:<3} | finished: {self.call_fn.__name__} | time: {end - start:.4f} seconds",flush=True)
        return results

    @classmethod
    def _exec_backend(cls):
        from mpi4py.futures import MPIPoolExecutor

        return MPIPoolExecutor


# pylint: disable=import-outside-toplevel
class MPICommExec(ExtExec):  # pragma: no cover
    r"""
    MPICommExecutor abstraction class functor. To be used if dynamic process spawning
    required by MPIPoolExec is unsupported by the MPI implementation.

    This executor wraps the `mpi4py.futures.MPICommExecutor <https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html#mpicommexecutor>`_ class, and provides support for execution using multiple processes launched using MPI.
    For an example script ``my_script.py``, and an installed mpi4py library with the active MPI environment, the executor can be used as follows:

    .. code-block:: console

        $ mpirun -n 4 -m mpi4py.futures my_script.py

    See `mpi4py.futures - Command line <https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html#command-line>`_ for additional details on launching jobs.

    .. note::
        All calls to the executor are synchronous, and do not currently support the use of futures as a return object.

    Args:
        max_workers (int, optional): Maximum number of worker processes to use. If None, defaults to the number of available MPI processes.
        persist (bool): Whether to persist the executor. Currently not supported and will raise an error if True.
        use_pkl5 (bool): Whether to use the pkl5 protocol for serialization. Defaults to True
        profile (bool): Whether to enable profiling of function execution. Defaults to False.
        **kwargs: Additional keyword arguments to pass through to the executor backend.
    """

    def __init__(self, max_workers=None, persist: bool = False, **kwargs):
        if persist:
            raise RuntimeError("The MPIPoolExec backend does not currently support persistence.")

        super().__init__(max_workers=max_workers, **kwargs)

        from mpi4py import MPI  # Required to call MPI_Init

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
        
        self._profile_fn = kwargs.pop("profile", False)
        self._use_pkl5 = kwargs.pop("use_pkl5", True) 
        

    def __call__(self, dispatch: str, fn: Callable, *args, **kwargs):
        r"""
        dispatch:   the named method to pass the function parameters
        fn:         the callable function to run on the executor backend
        args:       the arguments to pass to ``fn``
        kwargs:     the keyword arguments to pass to ``fn``
        """
        kwargs.update({"use_pkl5": self._use_pkl5})
        return super().__call__(
            dispatch,
            fn,
            *args,
            **kwargs,
        )

    def submit(self, fn: Callable, *args, **kwargs):
        with self._exec_backend()(max_workers=self.size, use_pkl5=self._use_pkl5) as executor:
            output_f = executor.submit(fn, *args, **kwargs)
        return output_f.result()

    def map(self, fn: Callable, *args: Sequence[Any], **kwargs):
        chunksize = max(len(args) // self._size, 1)
        
        if self._profile_fn:
            print("Using profile function for execution...")  # Debugging statement
            self.call_fn = fn
            
            with self._exec_backend()(max_workers=self.size, use_pkl5=self._use_pkl5) as executor:
                output_f = executor.map(self.profile_fn, *args, chunksize=chunksize, **kwargs)

            return list(output_f)

        with self._exec_backend()(max_workers=self.size, use_pkl5=self._use_pkl5) as executor:
            output_f = executor.map(fn, *args, chunksize=chunksize, **kwargs)
        return list(output_f)

    def starmap(self, fn: Callable, args: Sequence[tuple], **kwargs):
        chunksize = max(len(args) // self._size, 1)
        
        if self._profile_fn:
            print("Using profile function for execution...")  # Debugging statement
            self.call_fn = fn
            
            with self._exec_backend()(max_workers=self.size, use_pkl5=self._use_pkl5) as executor:
                output_f = executor.starmap(self.profile_fn, args, chunksize=chunksize, **kwargs)

            return list(output_f)

        with self._exec_backend()(max_workers=self.size, use_pkl5=self._use_pkl5) as executor:
            output_f = executor.starmap(fn, args, chunksize=chunksize, **kwargs)
        return list(output_f)

    @property
    def size(self):
        return self._size

    def shutdown(self):
        "Shutdown the executor backend, if valid."

    def __del__(self):
        self.shutdown()

    def profile_fn(self, *args, **kwargs):
        """
        A function to profile the execution of the executor.
        This is a placeholder for any profiling logic that may be added in the future.
        """
        # print("Profiling function execution...")  # Placeholder for profiling logic
        from mpi4py import MPI

        rank = MPI.COMM_WORLD.Get_rank()
        size = MPI.COMM_WORLD.Get_size()
        
        start = time.perf_counter()
        results = self.call_fn(*args, **kwargs)  # Call the function to be profiled
        end = time.perf_counter()

        print(f"Rank {rank:<3}/ {size:<3} | finished: {self.call_fn.__name__} | time: {end - start:.4f} seconds",flush=True)
        return results

    @classmethod
    def _exec_backend(cls):
        from mpi4py.futures import MPICommExecutor

        return MPICommExecutor

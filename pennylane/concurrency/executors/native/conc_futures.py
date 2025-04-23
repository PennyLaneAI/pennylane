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
This module provides abstractions around the Python ``concurrent.futures`` library and interface. This module directly offloads to the in-built executors for both multithreaded and multiprocess function execution.
"""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from ..base import ExecBackendConfig
from .api import PyNativeExec


class ProcPoolExec(PyNativeExec):
    """
    concurrent.futures.ProcessPoolExecutor class executor.

    This executor wraps Python standard library ``concurrent.futures.ProcessPoolExecutor`` API, and provides support for execution using multiple processes.
    All calls to the executor are synchronous, and do not currently support the use of futures as a return object.

    Args:
        *args: non keyword arguments to pass through to the executor backend.
        **kwargs: keyword arguments to pass through to the executor backend.

    """

    @classmethod
    def _exec_backend(cls):
        return ProcessPoolExecutor

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cfg = ExecBackendConfig(
            submit_fn="submit",
            map_fn="map",
            starmap_fn="starmap",
            shutdown_fn="shutdown",
            submit_unpack=True,
            map_unpack=True,
            blocking=False,
        )


class ThreadPoolExec(PyNativeExec):
    """
    concurrent.futures.ThreadPoolExecutor class executor.

    This executor wraps Python standard library ``concurrent.futures.ThreadPoolExecutor`` API, and provides support for execution using multiple threads.
    Due to the presence of the GIL in most currently supported releases of CPython, the threading executor may not provide execution speed-ups for tasks.
    All calls to the executor are synchronous, and do not currently support the use of futures as a return object.

    Args:
        *args: non keyword arguments to pass through to the executor backend.
        **kwargs: keyword arguments to pass through to the executor backend.

    """

    @classmethod
    def _exec_backend(cls):
        return ThreadPoolExecutor

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cfg = ExecBackendConfig(
            submit_fn="submit",
            map_fn="map",
            starmap_fn="starmap",
            shutdown_fn="shutdown",
            submit_unpack=True,
            map_unpack=True,
            blocking=False,
        )

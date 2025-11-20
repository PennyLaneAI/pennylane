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
.. currentmodule:: pennylane.concurrency.executors.native.conc_futures

This module provides abstractions around the Python ``concurrent.futures`` library and interface. This module directly offloads to the in-built executors for both multithreaded and multiprocess function execution.
"""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from multiprocessing import get_context

from pennylane.concurrency.executors.base import ExecBackendConfig

from .api import PyNativeExec


class ProcPoolExec(PyNativeExec):
    r"""
    concurrent.futures.ProcessPoolExecutor class executor.

    This executor wraps Python standard library `concurrent.futures.ProcessPoolExecutor <https://docs.python.org/3/library/concurrent.futures.html#processpoolexecutor>`_ interface, and provides support for execution using multiple processes.

    .. note::
        All calls to the executor are synchronous, and do not currently support the use of futures as a return object.

    Args:
        max_workers: the maximum number of concurrent units (processes) to use
        persist: allow the executor backend to persist between executions. True avoids
                    potentially costly set-up and tear-down, where supported.
                    Explicit calls to ``shutdown`` will set this to False.
        **kwargs: Keyword arguments to pass-through to the executor backend.

    """

    @classmethod
    def _exec_backend(cls):
        return partial(ProcessPoolExecutor, mp_context=get_context("spawn"))

    def __init__(self, max_workers: int | None = None, persist: bool = False, **kwargs):
        super().__init__(max_workers=max_workers, persist=persist, **kwargs)
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
    r"""
    concurrent.futures.ThreadPoolExecutor class executor.

    This executor wraps Python standard library `concurrent.futures.ThreadPoolExecutor <https://docs.python.org/3/library/concurrent.futures.html#threadpoolexecutor>`_ interface, and provides support for execution using multiple threads.
    The threading executor may not provide execution speed-ups for tasks when using a GIL-enabled Python.

    .. note::
        All calls to the executor are synchronous, and do not currently support the use of futures as a return object.

    Args:
        max_workers: the maximum number of concurrent units (threads) to use
        persist: allow the executor backend to persist between executions. True avoids
                    potentially costly set-up and tear-down, where supported.
                    Explicit calls to ``shutdown`` will set this to False.
        **kwargs: Keyword arguments to pass-through to the executor backend.

    """

    @classmethod
    def _exec_backend(cls):
        return ThreadPoolExecutor

    def __init__(self, max_workers: int | None = None, persist: bool = False, **kwargs):
        super().__init__(max_workers=max_workers, persist=persist, **kwargs)
        self._cfg = ExecBackendConfig(
            submit_fn="submit",
            map_fn="map",
            starmap_fn="starmap",
            shutdown_fn="shutdown",
            submit_unpack=True,
            map_unpack=True,
            blocking=False,
        )

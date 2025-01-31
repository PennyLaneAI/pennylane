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
Contains concurrent executor abstractions for task-based workloads based on
support provided by the Python standard library.
"""
import abc
import os
import sys
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor as exec_pp
from concurrent.futures import ThreadPoolExecutor as exec_tp
from multiprocessing import Pool as exec_mp

from .base import IntExecABC


class PyNativeExecABC(IntExecABC, abc.ABC):
    """
    Python standard library backed ABC for executor API.
    """

    def __init__(self, max_workers: int = None, persist: bool = False, **kwargs):
        """
        max_workers:    the maximum number of concurrent units (threads, processes) to use
        persist:        allow the executor backend to persist between executions. True avoids
                            potentially costly set-up and tear-down, where supported.
        """
        super().__init__(max_workers=max_workers, **kwargs)
        if max_workers:
            self._size = max_workers
        elif sys.version_info.minor >= 13:
            self._size = os.process_cpu_count()
        else:
            self._size = os.cpu_count()
        self._persist = persist
        if self._persist:
            self._backend = self._exec_backend()(self._size)

    def __call__(self, fn: Callable, data: Sequence):
        exec_cls = self._exec_backend()
        chunksize = max(len(data) // self._size, 1)
        if not self._persist:
            with exec_cls(self._size) as executor:
                output_f = executor.map(fn, data, chunksize=chunksize)
            return output_f
        return self._backend.map(fn, data, chunksize=chunksize)

    @property
    def size(self):
        return self._size

    def starmap(self, fn: Callable, data: Sequence[tuple]):
        if not hasattr(self._exec_backend(), "starmap"):
            return super().starmap(fn, data)
        if self._persist:
            return list(self._backend.starmap(fn, data))
        return list(self._exec_backend()(self._size).starmap(fn, data))

    def map(self, fn: Callable, data: Sequence):
        if self._persist:
            return list(self._backend.map(fn, data))
        return list(self._exec_backend()(self._size).map(fn, data))

    @classmethod
    @abc.abstractmethod
    def _exec_backend(cls): ...


class MPPoolExec(PyNativeExecABC):
    """
    multiprocessing.Pool class functor.
    """

    @classmethod
    def _exec_backend(cls):
        return exec_mp


class ProcPoolExec(PyNativeExecABC):
    """
    concurrent.futures.ProcessPoolExecutor class functor.
    """

    @classmethod
    def _exec_backend(cls):
        return exec_pp


class ThreadPoolExec(PyNativeExecABC):
    """
    concurrent.futures.ThreadPoolExecutor class functor.
    """

    @classmethod
    def _exec_backend(cls):
        return exec_tp

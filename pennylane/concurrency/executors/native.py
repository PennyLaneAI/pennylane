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
import inspect
import os
import sys
import weakref
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor as exec_pp
from concurrent.futures import ThreadPoolExecutor as exec_tp
from dataclasses import dataclass
from itertools import starmap
from multiprocessing import Pool as exec_mp
from typing import Any, Optional

from .base import IntExec


class PyNativeExec(IntExec, abc.ABC):
    """
    Python standard library backed ABC for executor API.
    """

    def __init__(self, max_workers: int = 1, persist: bool = False, **kwargs):
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

        self._cfg = self.LocalConfig()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._persist:
            self.shutdown()

    def shutdown(self):
        "Shutdown the executor backend, if valid."
        if self._persist:
            self._close_fn(self._backend)()
            self._backend = None

    def __del__(self):
        self.shutdown()

    @property
    def size(self):
        return self._size

    def submit(self, fn: Callable, *args, **kwargs):
        exec_be = self._get_backend()
        output = None
        if self._cfg.submit_unpack:
            output = self._submit_fn(exec_be)(fn, *args, **kwargs)
        else:
            output = self._submit_fn(exec_be)(fn, args, **kwargs)
        if self._cfg.blocking:
            return output
        return output.result()

    def map(self, fn: Callable, *args: Sequence[Any]):
        exec_be = self._get_backend()
        output = None
        if self._cfg.map_unpack and len(inspect.signature(fn).parameters) > 1:
            output = self._map_fn(exec_be)(fn, *args)
        else:
            output = self._map_fn(exec_be)(fn, args)

        return list(output)

    def starmap(self, fn: Callable, data: Sequence[tuple]):
        exec_be = self._get_backend()
        if not hasattr(exec_be, "starmap"):
            return super().starmap(fn, data)
        return list(exec_be.starmap(fn, data))

    @classmethod
    @abc.abstractmethod
    def _exec_backend(cls):
        raise NotImplementedError("{cls} does not currently support execution")


class SerialExec(PyNativeExec):
    """
    Serial Python standard library executor.
    """

    class StdLibWrapper:
        "Internal utility class for use with the executor API"

        def __init__(self, *args, **kwargs):
            pass

        def submit(self, fn: Callable, *args, **kwargs):
            results = []
            for t_args in zip(*args):
                results.append(fn(*t_args, **kwargs))
            return results

        def map(self, fn: Callable, *args: Sequence[Any]):
            return list(map(fn, *args))

        def starmap(self, fn: Callable, data: Sequence[tuple]):
            return list(starmap(fn, data))

    def __init__(self, max_workers: int = 1, persist: bool = False, **kwargs):
        super().__init__(max_workers=max_workers, persist=persist, **kwargs)

    def _exec_backend(cls):
        return SerialExec.StdLibWrapper


class MPPoolExec(PyNativeExec):
    """
    multiprocessing.Pool class functor.
    """

    @classmethod
    def _exec_backend(cls):
        return exec_mp

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cfg = self.LocalConfig(
            submit_fn="apply",
            map_fn="map",
            starmap_fn="starmap",
            close_fn="close",
            submit_unpack=False,
            map_unpack=False,
            blocking=True,
        )

    def map(self, fn: Callable, *args: Sequence[Any]):
        if len(inspect.signature(fn).parameters) > 1:
            try:
                # attempt to recover by offloading to starmap
                return self.starmap(fn, zip(*args))
            except:
                raise ValueError(
                    "Python's `multiprocessing.Pool` does not support `map` calls with multiple arguments. Consider a different backend, or use `starmap` instead."
                )
        return super().map(fn, *args)


class ProcPoolExec(PyNativeExec):
    """
    concurrent.futures.ProcessPoolExecutor class functor.
    """

    @classmethod
    def _exec_backend(cls):
        return exec_pp

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cfg = self.LocalConfig(
            submit_fn="submit",
            map_fn="map",
            starmap_fn="starmap",
            close_fn="shutdown",
            submit_unpack=True,
            map_unpack=True,
            blocking=False,
        )


class ThreadPoolExec(PyNativeExec):
    """
    concurrent.futures.ThreadPoolExecutor class functor.
    """

    @classmethod
    def _exec_backend(cls):
        return exec_tp

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cfg = self.LocalConfig(
            submit_fn="submit",
            map_fn="map",
            starmap_fn="starmap",
            close_fn="shutdown",
            submit_unpack=True,
            map_unpack=True,
            blocking=False,
        )

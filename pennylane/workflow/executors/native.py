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
import weakref
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor as exec_pp
from concurrent.futures import ThreadPoolExecutor as exec_tp
from dataclasses import dataclass
from multiprocessing import Pool as exec_mp
from typing import Optional

from .base import IntExecABC


class PyNativeExecABC(IntExecABC, abc.ABC):
    """
    Python standard library backed ABC for executor API.
    """

    @dataclass
    class LocalConfig:
        "Python native executor configuration data-class"

        submit_fn: Optional[str] = None
        map_fn: Optional[str] = None
        starmap_fn: Optional[str] = None
        submit_unpack: Optional[bool] = None
        map_unpack: Optional[bool] = None
        blocking: Optional[bool] = None

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

        self._cfg = self.LocalConfig()

    def __call__(self, fn: Callable, *args):
        exec_cls = self._exec_backend()
        chunksize = max(len(data) // self._size, 1)
        if not self._persist:
            with exec_cls(self._size) as executor:
                output_f = executor.map(fn, data, chunksize=chunksize)
            return output_f
        return self._backend.map(fn, data, chunksize=chunksize)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._persist:
            self.shutdown()

    def shutdown(self):
        if self._persist:
            self._backend.close()
            self._backend = None

    def __del__(self):
        self.shutdown()

    @property
    def size(self):
        return self._size

    def submit(self, fn: Callable, *args, **kwargs):
        if self._persist:
            return self._submit_fn(self._backend)(fn, *args, **kwargs)
        exec_be = self._exec_backend()(self._size)
        # from IPython import embed; embed()
        output = None

        if self._cfg.submit_unpack:
            output = self._submit_fn(exec_be)(fn, *args, **kwargs)
        else:
            # try:
            output = self._submit_fn(exec_be)(fn, args, **kwargs)
            # except:
            #    from IPython import embed; embed()

        if self._cfg.blocking:
            return output
        return output.result()

    def map(self, fn: Callable, *args: Sequence[Sequence]):
        if self._persist:
            return list(self._map_fn(self._backend)(fn, data))
        exec_be = self._exec_backend()(self._size)
        return list(self._map_fn(exec_be)(fn, data))

    def starmap(self, fn: Callable, data: Sequence[tuple]):
        if not hasattr(self._exec_backend(), "starmap"):
            return super().starmap(fn, data)
        if self._persist:
            return list(self._backend.starmap(fn, data))
        return list(self._exec_backend()(self._size).starmap(fn, data))

    @classmethod
    @abc.abstractmethod
    def _exec_backend(cls):
        raise NotImplementedError("{cls} does not currently support execution")

    def _submit_fn(self, backend):
        return getattr(backend, self._cfg.submit_fn)

    def _map_fn(self, backend):
        return getattr(backend, self._cfg.map_fn)

    def _starmap_fn(self, backend):
        return getattr(backend, self._cfg.starmap_fn)


class MPPoolExec(PyNativeExecABC):
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
            submit_unpack=False,
            map_unpack=False,
            blocking=True,
        )


class ProcPoolExec(PyNativeExecABC):
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
            submit_unpack=True,
            map_unpack=False,
            blocking=False,
        )


class ThreadPoolExec(PyNativeExecABC):
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
            submit_unpack=True,
            map_unpack=False,
            blocking=False,
        )

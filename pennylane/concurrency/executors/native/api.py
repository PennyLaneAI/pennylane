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
.. currentmodule:: pennylane.concurrency.executors.native.api

Base API for defining an executor relying on native Python standard library implementations.
"""
import abc
import inspect
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

from pennylane.concurrency.executors.base import ExecBackendConfig, IntExec


class PyNativeExec(IntExec, abc.ABC):
    r"""
    Python standard library backed ABC for executor API.

    This class abstracts single-machine environments and unifies the standard-library backed executor backend to support the same API call structure.

    Args:
        max_workers: the maximum number of concurrent units (threads, processes) to use
        persist: allow the executor backend to persist between executions. True avoids
                    potentially costly set-up and tear-down, where supported.
                    Explicit calls to ``shutdown`` will set this to False.
        **kwargs: Keyword arguments to pass-through to the executor backend.

    """

    def __init__(self, max_workers: int | None = None, persist: bool = False, **kwargs):
        super().__init__(max_workers=max_workers, persist=persist, **kwargs)
        if max_workers:
            self._size = max_workers
        else:
            self._size = self._get_system_core_count()
        self._persist = persist
        if self._persist:
            self._persistent_backend = self._exec_backend()(self._size)

        self._cfg = ExecBackendConfig()

    def shutdown(self):
        "Shutdown the executor backend, if valid."
        if self._persist:
            self._shutdown_fn(self._persistent_backend)()
            self._persistent_backend = None
            self._persist = False

    def __del__(self):
        self.shutdown()

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

    def map(self, fn: Callable, *args: Sequence[Any], **kwargs):
        exec_be = self._get_backend()
        output = None
        fn_p = partial(fn, **kwargs)
        if self._cfg.map_unpack and len(inspect.signature(fn).parameters) > 1:
            output = self._map_fn(exec_be)(fn_p, *args)
        else:
            output = self._map_fn(exec_be)(fn_p, args)

        return list(output)

    def starmap(self, fn: Callable, args: Sequence[tuple], **kwargs):
        exec_be = self._get_backend()
        if not hasattr(exec_be, "starmap"):
            return list(self.map(fn, *list(zip(*args))), **kwargs)
        fn_p = partial(fn, **kwargs)
        return list(exec_be.starmap(fn_p, args))

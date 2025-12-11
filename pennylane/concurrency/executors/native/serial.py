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
.. currentmodule:: pennylane.concurrency.executors.native.serial

This module provides single-threaded, local executor support for function execution. All operations are directed to built-ins, and use direct function execution.
"""

from collections.abc import Callable, Sequence
from functools import partial
from itertools import starmap
from typing import Any

from pennylane.concurrency.executors.base import ExecBackendConfig

from .api import PyNativeExec


class StdLibBackend:
    r"""
    Internal utility class for use with the executor API.
    All execution is local within the calling Python process.

    Args:
        *args: non-keyword arguments for passthrough to the executor backend. All values here will be ignored.
        **kwargs: keyword-arguments for passthrough to the executor backend. All values here will be ignored.

    """

    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def submit(cls, fn: Callable, *args, **kwargs):
        "Directly execute the function as fn(*args, **kwargs)"
        return fn(*args, **kwargs)

    @classmethod
    def map(cls, fn: Callable, *args: Sequence[Any], **kwargs):
        "Offload execution of the function to map and return the results as a list."
        fn_p = partial(fn, **kwargs)
        return list(map(fn_p, *args))

    @classmethod
    def starmap(cls, fn: Callable, data: Sequence[tuple], **kwargs):
        "Offload to itertools.starmap for execution, and return results as a list."
        fn_p = partial(fn, **kwargs)
        return list(starmap(fn_p, data))

    def shutdown(self):
        "No-op close shutdown"


class SerialExec(PyNativeExec):
    r"""
    Serial Python standard library executor class.

    This executor wraps Python standard library calls without support for multithreaded or multiprocess execution. Any calls to external libraries that utilize threads, such as BLAS through numpy, can still use multithreaded calls at that layer.

    Args:
        max_workers:    the maximum number of concurrent units (threads, processes) to use. The serial backend defaults to 1 and will return a ``RuntimeError`` if more are requested.
        persist:        allow the executor backend to persist between executions. This is ignored for the serial backend.
        **kwargs:   Keyword arguments to pass-through to the executor backend. This is ignored for the serial backend.

    """

    def __init__(self, max_workers: int | None = 1, persist: bool = False, **kwargs):
        super().__init__(max_workers=max_workers, persist=persist, **kwargs)
        if max_workers > 1:  # pragma: no cover
            raise RuntimeError("The serial executor backend cannot have more than 1 worker.")
        self._cfg = ExecBackendConfig(
            submit_fn="submit",
            map_fn="map",
            starmap_fn="starmap",
            shutdown_fn="shutdown",
            submit_unpack=True,
            map_unpack=True,
            blocking=True,
        )

    @classmethod
    def _exec_backend(cls):
        return StdLibBackend

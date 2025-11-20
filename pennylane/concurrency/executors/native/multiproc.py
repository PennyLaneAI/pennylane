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
.. currentmodule:: pennylane.concurrency.executors.native.multiproc

This module provides abstractions around the Python ``multiprocessing`` library, with support for function execution using multiple processes.
"""

import inspect
from collections.abc import Callable, Sequence
from multiprocessing import get_context
from typing import Any

from pennylane.concurrency.executors.base import ExecBackendConfig

from .api import PyNativeExec


class MPPoolExec(PyNativeExec):
    r"""
    Python standard library executor class backed by ``multiprocessing.Pool``.

    This executor wraps Python standard library `multiprocessing.Pool <https://docs.python.org/3/library/multiprocessing.html#module-multiprocessing.pool>`_ interface, and provides support for execution using multiple processes.

    Args:
        max_workers: the maximum number of concurrent units (processes) to use
        persist: allow the executor backend to persist between executions. True avoids
                    potentially costly set-up and tear-down, where supported.
                    Explicit calls to ``shutdown`` will set this to False.
        **kwargs: Keyword arguments to pass-through to the executor backend.

    """

    @classmethod
    def _exec_backend(cls):
        return get_context("spawn").Pool

    def __init__(self, max_workers: int | None = None, persist: bool = False, **kwargs):
        super().__init__(max_workers=max_workers, persist=persist, **kwargs)
        self._cfg = ExecBackendConfig(
            submit_fn="apply",
            map_fn="map",
            starmap_fn="starmap",
            shutdown_fn="close",
            submit_unpack=False,
            map_unpack=False,
            blocking=True,
        )

    def map(self, fn: Callable, *args: Sequence[Any], **kwargs):
        if len(inspect.signature(fn).parameters) > 1:  # pragma: no cover
            try:
                # attempt offloading to starmap
                return self.starmap(fn, zip(*args), **kwargs)
            except Exception as e:
                raise ValueError(
                    "Python's `multiprocessing.Pool` does not support `map` calls with multiple arguments. "
                    "Consider a different backend, or use `starmap` instead."
                ) from e
        return super().map(fn, *args, **kwargs)

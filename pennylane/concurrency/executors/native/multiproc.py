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
This module provides abstractions around the Python ``multiprocessing`` library, with support for function execution using multiple processes.
"""

import inspect
from collections.abc import Callable, Sequence
from multiprocessing import Pool
from typing import Any

from ..base import ExecBackendConfig
from .api import PyNativeExec


class MPPoolExec(PyNativeExec):
    """
    multiprocessing.Pool class executor.

    This executor wraps Python standard library ``multiprocessing.Pool`` API, and provides support for execution using multiple processes.

    Args:
        *args: non keyword arguments to pass through to the executor backend.
        **kwargs: keyword arguments to pass through to the executor backend.

    """

    @classmethod
    def _exec_backend(cls):
        return Pool

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        if len(inspect.signature(fn).parameters) > 1:
            try:
                # attempt offloading to starmap
                return self.starmap(fn, zip(*args), **kwargs)
            except Exception as e:
                raise ValueError(
                    "Python's `multiprocessing.Pool` does not support `map` calls with multiple arguments. "
                    "Consider a different backend, or use `starmap` instead."
                ) from e
        return super().map(fn, *args, **kwargs)

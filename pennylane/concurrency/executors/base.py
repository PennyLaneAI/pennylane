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
Contains concurrent executor abstractions for task-based workloads.
"""

import abc
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import starmap
from typing import Optional


class RemoteExecABC(abc.ABC):
    """
    Abstract base class for defining a task-based parallel executor backend for running python functions
    """

    @dataclass
    class LocalConfig:
        "Python native executor configuration data-class"

        submit_fn: Optional[str] = None
        map_fn: Optional[str] = None
        starmap_fn: Optional[str] = None
        close_fn: Optional[str] = None
        submit_unpack: Optional[bool] = None
        map_unpack: Optional[bool] = None
        blocking: Optional[bool] = None

    def __init__(self, max_workers: int = None, persist: bool = False, *args, **kwargs):
        self._size = max_workers
        self._persist = persist

    def __call__(self, dispatch: str, fn: Callable, *args, **kwargs):
        """
        dispatch:   the named method to pass the function parameters
        fn:         the callable function to run on the executor backend
        args:       the arguments to pass to `fn`
        kwargs:     the keyword arguments to pass to `fn`
        """
        return getattr(self, dispatch)(fn, *args, **kwargs)

    @property
    def size(self):
        return self._size

    @property
    def persist(self):
        return self._persist

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @abc.abstractmethod
    def submit(self, fn: Callable, *args, **kwargs):
        """
        Single function submission for remote execution with provided args.
        """
        ...

    def map(self, fn: Callable, *args, **kwargs):
        """
        Single iterable map for batching execution of fn over data entries.
        Length of every entry in *args must be consistent.
        kwargs are assumed as broadcastable to each function call.
        """
        for a in zip(*args):
            yield self.submit(fn, *a, **kwargs)

    def starmap(self, fn: Callable, args: Sequence):
        """
        Single iterable map for batching execution of fn over data entries, with each entry being a tuple of arguments to fn.
        """
        for a in args:
            yield fn(*a)

    @abc.abstractmethod
    def shutdown(self):
        """
        Disconnect from executor backend and release acquired resources.
        """
        ...

    def _submit_fn(self, backend):
        "Helper utility to return the config-defined submit function for the given backend."
        return getattr(backend, self._cfg.submit_fn)

    def _map_fn(self, backend):
        "Helper utility to return the config-defined map function for the given backend."
        return getattr(backend, self._cfg.map_fn)

    def _starmap_fn(self, backend):
        "Helper utility to return the config-defined starmap function for the given backend."

        return getattr(backend, self._cfg.starmap_fn)

    def _close_fn(self, backend):
        "Helper utility to return the config-defined close function for the given backend."

        return getattr(backend, self._cfg.close_fn)

    def _get_backend(self):
        "Convenience method to return the existing backend if persistence is enabled, or to create a new temporary backend with the defined size if not."
        if self._persist:
            return self._backend
        return self._exec_backend()(self._size)


class IntExecABC(RemoteExecABC, abc.ABC):
    """
    Executor class for native Python library concurrency support
    """

    pass


class ExtExecABC(RemoteExecABC, abc.ABC):
    """
    Executor class for external package provided concurrency support
    """

    pass

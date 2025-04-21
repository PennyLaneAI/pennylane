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


@dataclass
class ExecBackendConfig:
    """
    Executor backend configuration data-class.

    To allow for differences in each executor backend implementation, this class dynamically defines overloads to the main API functions. For explicitly-defined executors, this class is optional, and is provided for convenience with hierarchical inheritance class structures, where subtle differences are best resolved dynamically, rather than with API modifications. All initial values default to ``None``.

    Args:
        submit_fn (str): The backend function that best matches the `submit` API call.
        map_fn (str): The backend function that best matches the `map` API call.
        starmap_fn (str): The backend function that best matches the `starmap` API call.
        shutdown_fn (str): The backend function that best matches the `shutdown` API call.
        submit_unpack (bool): Whether the arguments to ``submit`` are to be unpacked (*args) or directly passed (args) to ``submit_fn``.
        map_unpack (bool): Whether the arguments to ``map`` are to be unpacked (*args) or directly passed (args) to ``map_unpack``.
        blocking (bool): Whether the return values from ``submit``, ``map`` and `starmap`` are blocking (synchronous) or non-blocking (asynchronous).

    """

    submit_fn: Optional[str] = None
    map_fn: Optional[str] = None
    starmap_fn: Optional[str] = None
    shutdown_fn: Optional[str] = None
    submit_unpack: Optional[bool] = None
    map_unpack: Optional[bool] = None
    blocking: Optional[bool] = None


class RemoteExec(abc.ABC):
    """
    Abstract base class for defining a task-based parallel executor backend.

    This ABC is intended to provide the highest-layer abstraction in the inheritance tree.

    Args:
        max_workers (int): The size of the worker pool. This value will directly control (given backend support),
            the number of concurrent executions that the backend can avail of. Generally, this value should match
            the number of physical cores on the executing system, or with the executing remote environment. Defaults
            to ``None``.
        persist (bool): Indicates to the executor backend that the state should persist between calls. If supported,
            this allows a pre-configured device to be reused for several computations but removing the need to
            automatically shutdown. The pool may require manual shutdown upon completion of the work, even if the
            executor goes out-of-scope.
    """

    def __init__(self, max_workers: Optional[int] = None, persist: bool = False, *args, **kwargs):
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
        """
        The size of the worker pool for the given executor.

        A larger worker pool indicates the number of potential executions that can happen concurrently.
        """
        return self._size

    @property
    def persist(self):
        """
        Indicates whether the executor will maintain its configured state between calls.
        """
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

    def _submit_fn(self, backend):
        "Helper utility to return the config-defined submit function for the given backend."
        return getattr(backend, self._cfg.submit_fn)

    def _map_fn(self, backend):
        "Helper utility to return the config-defined map function for the given backend."
        return getattr(backend, self._cfg.map_fn)

    def _starmap_fn(self, backend):
        "Helper utility to return the config-defined starmap function for the given backend."

        return getattr(backend, self._cfg.starmap_fn)

    def _shutdown_fn(self, backend):
        "Helper utility to return the config-defined shutdown function for the given backend."

        return getattr(backend, self._cfg.shutdown_fn)

    def _get_backend(self):
        "Convenience method to return the existing backend if persistence is enabled, or to create a new temporary backend with the defined size if not."
        if self._persist:
            return self._backend
        return self._exec_backend()(self._size)


class IntExec(RemoteExec, abc.ABC):
    """
    Executor class for native Python library concurrency support
    """

    pass


class ExtExec(RemoteExec, abc.ABC):
    """
    Executor class for external package provided concurrency support
    """

    pass

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
Contains concurrent executor abstractions for task-based workloads.

All of the base abstractions for building an executor follow a simplified `concurrent.futures.Executor <https://docs.python.org/3/library/concurrent.futures.html#executor-objects>`_ interface. Given the differences observed in support for ``*args`` and ``**kwargs`` in various modes of execution, the abstractions provide a fixed API to interface with each backend, performing function and argument transformations, where necessary.

To build a new executor backend, the following classes provide scaffolding to simplify abstracting the function call signatures between each backend interface layer.

.. currentmodule:: pennylane.concurrency.executors.base

.. autosummary::
    :toctree: api

    ExecBackendConfig
    RemoteExec
    IntExec
    ExtExec

"""

import abc
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass
class ExecBackendConfig:
    r"""
    Executor backend configuration data-class.

    To allow for differences in each executor backend implementation, this class dynamically defines overloads to the main API functions. For explicitly-defined executors, this class is optional, and is provided for convenience with hierarchical inheritance class structures, where subtle differences are best resolved dynamically, rather than with API modifications. All initial values default to ``None``.

    Args:
        submit_fn (str, None): The backend function that best matches the ``submit`` API call.
        map_fn (str): The backend function that best matches the ``map`` API call.
        starmap_fn (str, None): The backend function that best matches the ``starmap`` API call.
        shutdown_fn (str, None): The backend function that best matches the ``shutdown`` API call.
        submit_unpack (bool, None): Whether the arguments to ``submit`` are to be unpacked (``*args``) or directly passed (``args``) to ``submit_fn``.
        map_unpack (bool): Whether the arguments to ``map`` are to be unpacked (``*args``) or directly passed (``args``) to ``map_unpack``.
        blocking (bool, None): Whether the return values from ``submit``, ``map`` and ``starmap`` are blocking (synchronous) or non-blocking (asynchronous).

    """

    submit_fn: str | None = None
    map_fn: str | None = None
    starmap_fn: str | None = None
    shutdown_fn: str | None = None
    submit_unpack: bool | None = None
    map_unpack: bool | None = None
    blocking: bool | None = None


class RemoteExec(abc.ABC):
    r"""
    Abstract base class for defining a task-based parallel executor backend.

    This ABC is intended to provide the highest-layer abstraction in the inheritance tree.

    Args:
        max_workers (int): The size of the worker pool. This value will directly control (given backend support)
            the number of concurrent executions that the backend can avail of. Generally, this value should match
            the number of physical cores on the executing system, or with the executing remote environment. Defaults
            to ``None``, which defers to support provided by the child class.
        persist (bool): Indicates to the executor backend that the state should persist between
            calls. If supported, this allows a pre-configured device to be reused for several
            computations but removing the need to automatically shutdown. The pool may require
            manual shutdown upon completion of the work, even if the executor goes out-of-scope.
        *args: Non keyword arguments to pass through to executor backend.
        **kwargs: Keyword arguments to pass through to executor backend.

    """

    def __init__(self, max_workers: int | None = None, persist: bool = False, **kwargs):
        self._size = max_workers
        self._persist = persist
        self._inputs = kwargs
        self._cfg = ExecBackendConfig()
        self._persistent_backend = None

    def __call__(self, dispatch: str, fn: Callable, *args, **kwargs):
        r"""
        dispatch:   the named method to pass the function parameters
        fn:         the callable function to run on the executor backend
        args:       the arguments to pass to ``fn``
        kwargs:     the keyword arguments to pass to ``fn``
        """
        return getattr(self, dispatch)(fn, *args, **kwargs)

    @property
    def size(self):
        """
        The size of the worker pool for the given executor.
        """
        return self._size

    @property
    def persist(self):
        """
        Indicates whether the executor will maintain its configured state between calls.
        """
        return self._persist

    def __enter__(self):
        """Context-manager entry point for executor.

        Returns:
            RemoteExec: this instance
        """
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """Context-manager clean-up for executor."""
        if not self._persist:
            self.shutdown()

    @abc.abstractmethod
    def submit(self, fn: Callable, *args, **kwargs):
        """
        Single function submission for remote execution with provided args.
        """

    @abc.abstractmethod
    def map(self, fn: Callable, *args, **kwargs):
        r"""
        Single iterable map for batching execution of fn over data entries.
        Length of every entry in ``*args`` must be consistent.
        kwargs are assumed as broadcastable to each function call.
        """

    @abc.abstractmethod
    def starmap(self, fn: Callable, args: Sequence, **kwargs):
        """
        Single iterable map for batching execution of fn over data entries, with each entry being a tuple of arguments to fn.
        """

    @abc.abstractmethod
    def shutdown(self):
        """
        Disconnect from executor backend and release acquired resources.
        """

    def _submit_fn(self, backend):  # pragma: no cover
        "Helper utility to return the config-defined submit function for the given backend."
        return getattr(backend, self._cfg.submit_fn)

    def _map_fn(self, backend):  # pragma: no cover
        "Helper utility to return the config-defined map function for the given backend."
        return getattr(backend, self._cfg.map_fn)

    def _starmap_fn(self, backend):  # pragma: no cover
        "Helper utility to return the config-defined starmap function for the given backend."

        return getattr(backend, self._cfg.starmap_fn)

    def _shutdown_fn(self, backend):  # pragma: no cover
        "Helper utility to return the config-defined shutdown function for the given backend."

        return getattr(backend, self._cfg.shutdown_fn)

    def _get_backend(self):  # pragma: no cover
        "Convenience method to return the existing backend if persistence is enabled, or to create a new temporary backend with the defined size if not."
        if self._persist:
            return self._persistent_backend
        return self._exec_backend()(self._size)

    @classmethod
    @abc.abstractmethod
    def _exec_backend(cls):
        "Return the class type of the given backend variant."

    @staticmethod
    def _get_system_core_count():  # pragma: no cover
        if sys.version_info.minor >= 13:
            return os.process_cpu_count()  # pylint: disable=no-member
        return os.cpu_count()


class IntExec(RemoteExec, abc.ABC):
    r"""
    Executor class for native Python library concurrency support.

    This class is intended to be used as the parent-class for building Python-native executors, allowing an ease of distinction from the external-based classes implemented using :class:`~.ExtExec`.

    Args:
        max_workers (int): The size of the worker pool. This value will directly control (given backend support)
            the number of concurrent executions that the backend can avail of. Generally, this value should match
            the number of physical cores on the executing system, or with the executing remote environment. Defaults
            to ``None``, leaving interpretation to the child class.
        persist (bool): Indicates to the executor backend that the state should persist between
            calls. If supported, this allows a pre-configured device to be reused for several
            computations but removing the need to automatically shutdown. The pool may require
            manual shutdown upon completion of the work, even if the executor goes out-of-scope.
        *args: Non keyword arguments to pass through to executor backend.
        **kwargs: Keyword arguments to pass through to executor backend.

    """


class ExtExec(RemoteExec, abc.ABC):
    r"""
    Executor class for external packages providing concurrency support.

    This class is intended to be used as the parent-class for building external package-based executors, allowing an ease of distinction from the Python-native classes implemented using :class:`~.IntExec`.

    Args:
        max_workers (int): The size of the worker pool. This value will directly control (given backend support)
            the number of concurrent executions that the backend can avail of. Generally, this value should match
            the number of physical cores on the executing system, or with the executing remote environment. Defaults
            to ``None``, leaving interpretation to the child class.
        persist (bool): Indicates to the executor backend that the state should persist between
            calls. If supported, this allows a pre-configured device to be reused for several
            computations but removing the need to automatically shutdown. The pool may require
            manual shutdown upon completion of the work, even if the executor goes out-of-scope.
        *args: Non keyword arguments to pass through to executor backend.
        **kwargs: Keyword arguments to pass through to executor backend.
    """

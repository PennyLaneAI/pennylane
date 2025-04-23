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
Provides abstractions for task-based parallel workloads within PennyLane using a simplified `concurrent.futures <https://docs.python.org/3/library/concurrent.futures.html>`__ executor-like interface.

Local and remote function execution through an instantiated ``executor`` can be through the following API calls:

.. code-block:: python

    executor = create_executor(...)
    ...
    # Single function execution on the executor backend
    executor.submit(self, fn: Callable, *args, **kwargs)

    # Map provided function to all iterables provided in ``args``,
    # with each index in *args running on the executor backend
    executor.map(self, fn: Callable, *args, **kwargs)

    # Map provided function to all tuples provided in ``args``,
    # with each paired values running on the executor backend
    executor.starmap(self, fn: Callable, args, **kwargs)

.. currentmodule:: pennylane.concurrency.executors
.. autosummary::
    :toctree: api

    backends
    base
    dask
    mpi
    native

"""

from .backends import ExecBackends, create_executor, get_executor, get_supported_backends
from .base import RemoteExec, IntExec, ExtExec
from .native import PyNativeExec, SerialExec, MPPoolExec, ProcPoolExec, ThreadPoolExec

__all__ = [
    "ExecBackends",
    "create_executor",
    "get_executor",
    "get_supported_backends",
    "RemoteExec",
    "IntExec",
    "ExtExec",
    "PyNativeExec",
    "SerialExec",
    "MPPoolExec",
    "ProcPoolExec",
    "ThreadPoolExec",
]

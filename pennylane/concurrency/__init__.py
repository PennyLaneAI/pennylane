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
.. currentmodule:: pennylane.concurrency

Executors
=========

This module contains concurrent execution functionality for PennyLane workloads.

.. automodule:: pennylane.concurrency.executors


Executor implementations in PennyLane build functional abstractions around the following API calls:

.. code-block:: python

    # Single function execution on the executor backend
    executor.submit(fn: Callable, *args, **kwargs)

    # Map provided function to all iterables provided in ``args``,
    # with each index in *args running on the executor backend
    executor.map(fn: Callable, *args, **kwargs)

    # Map provided function to all tuples provided in ``args``,
    # with each paired values running on the executor backend
    executor.starmap(fn: Callable, args, **kwargs)


These loosely mirror the native Python functions ``map`` and ``itertools.starmap``, whilst also following the design  from the ``concurrent.futures`` interface. To allow for Liskov substitution, we define a uniform signature for functions with ``*args`` and ``**kwargs``, allowing reduced coupling between the caller and the executor backend. This allows an ease of scaling from local execution to remote execution by controlling the executor being instantiated.

Local and remote function execution through an instantiated ``executor`` is available through the above API calls, or via a functor-like dispatch mechanism:

.. code-block:: python

    executor = create_executor(...)
    executor("submit", myfunc, *mydata)


Support functions to query supported backends and initialize them are provided through the following functions.

.. currentmodule:: pennylane.concurrency.executors
.. autosummary::
    :toctree: api

    ~backends.get_supported_backends
    ~backends.create_executor
    ~backends.get_executor

Native Python executors
^^^^^^^^^^^^^^^^^^^^^^^

PennyLane currently has support for a collection of executor implementations targeting local execution within the ``pennylane.concurrency.executors.native`` module, and remote execution within the ``pennylane.concurrency.executors.external`` module

.. currentmodule:: pennylane.concurrency.executors.native
.. autosummary::
    :toctree: api

    ~multiproc.MPPoolExec
    ~conc_futures.ProcPoolExec
    ~conc_futures.ThreadPoolExec
    ~serial.SerialExec


Executor API
^^^^^^^^^^^^

To build a new executor backend, the following abstract base classes provide scaffolding to simplify the creation around the required support and abstractions.

.. currentmodule:: pennylane.concurrency.executors
.. autosummary::
    :toctree: api

    ~base.ExecBackendConfig
    ~base.RemoteExec
    ~base.IntExec
    ~base.ExtExec
    ~native.api.PyNativeExec


"""
from .executors import native, external, backends, base

__all__ = ["native", "external", "backends", "base"]

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
This module contains concurrent execution functionality for PennyLane workloads using a task-based executor abstraction.

.. currentmodule:: pennylane

Executors and backends
**********************

Executor implementations in PennyLane build functional abstractions around the following API calls:

.. code-block:: python3

    # Single function execution on the executor backend
    executor.submit(fn: Callable, *args, **kwargs)

    # Map provided function to all iterables provided in ``args``,
    # with each index in *args running on the executor backend
    executor.map(fn: Callable, *args, **kwargs)

    # Map provided function to all tuples provided in ``args``,
    # with each paired values running on the executor backend
    executor.starmap(fn: Callable, args, **kwargs)


These loosely mirror the native Python functions `map <https://docs.python.org/3/library/functions.html#map>`_ and `itertools.starmap <https://docs.python.org/3/library/itertools.html#itertools.starmap>`_, whilst also following the design from the `concurrent.futures.Executor <https://docs.python.org/3/library/concurrent.futures.html#executor-objects>`_ interface. To allow for `Liskov substitution <https://en.wikipedia.org/wiki/Liskov_substitution_principle>`_, we define a uniform signature for functions with ``*args`` and ``**kwargs``, allowing reduced coupling between the caller and the executor backend. This allows an ease of scaling from local execution to remote execution by controlling the executor being instantiated.

Local and remote function execution through an instantiated ``executor`` is available through the above API calls, or via a functor-like dispatch mechanism:

.. code-block:: python3

    executor = create_executor(...)
    executor("submit", myfunc, *mydata)


Support functions to query supported backends and initialize them are provided through the following functions.


Supported executors
===================

PennyLane currently has support for a collection of executor implementations using local (Python native standard library) execution, and remote (third-party distributed) implemented backends.

Native Python executors
^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: pennylane.concurrency.executors.native
.. automodule:: pennylane.concurrency.executors.native
    :noindex:


External package-backed executors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: pennylane.concurrency.executors.external
.. automodule:: pennylane.concurrency.executors.external
    :noindex:


Executor API
============

.. currentmodule:: pennylane.concurrency.executors.base
.. automodule:: pennylane.concurrency.executors.base
    :noindex:

"""
from .executors import backends, base, external, native

__all__ = ["native", "external", "backends", "base"]

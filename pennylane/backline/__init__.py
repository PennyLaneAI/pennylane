# Copyright 2026 Xanadu Quantum Technologies Inc.

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
.. currentmodule:: pennylane.backline

This module contains experimental features for compilation and execution on heterogeneous devices.
The ``"ftqc.heterogeneous"`` device requires a ``backline`` object, which specifies the placement
of processes (i.e., where each part of the workload runs), and the transport protocol.

.. warning::

    Backline is experimental. Its API may change without notice, and it is only usable through
    the Catalyst compiler.

A ``backline`` placement is built from a :class:`controller <.Controller>` (which drives a QNode and executes through a backend simulator such as ``lightning.qubit`` or ``null.qubit``), zero or more :class:`coprocessors <.Coprocessor>`, and a
:class:`transport <.Transport>`. This is then passed to the ``"ftqc.heterogeneous"`` device:

.. code-block:: python

    import pennylane as qp

    cpu_controller = qp.Controller(
        name="cpu-controller",
        addr="192.168.1.1",
        port="1234",
        triple="aarch64-unknown-linux-gnu",
        remote=True,
    )

    gpu_coprocessor = qp.Coprocessor(
        name="gpu-coprocessor",
        coprocessor_fn="decoder",
        remote=False,
    )

    backline = qp.Backline(
        controller=cpu_controller, coprocessors=(gpu_coprocessor,), transport="rdma"
    )

    dev = qp.device("ftqc.heterogeneous", backline=backline, wires=4)

Executors
~~~~~~~~~

An executor is a node in the backline fabric.

.. autosummary::
    :toctree: api

    ~Controller
    ~Coprocessor
    ~Executor

Coprocessor functions
~~~~~~~~~~~~~~~~~~~~~~~

A coprocessor applies a precompiled function to each message it receives (e.g., decoding a syndrome). Currently, coprocessor functions can be defined directly in C++ as a runtime function, or in Python through helper functions such as ``css_decoder``.

.. autosummary::
    :toctree: api

    ~CoprocessorFunction
    ~css_decoder

Placement
~~~~~~~~~

A placement groups the controller, coprocessors, and transport into a single object.

.. autosummary::
    :toctree: api

    ~Backline

Transports
~~~~~~~~~~

A transport selects, by name, how messages transfer between executors. The implementation
lives in the compiled runtime.

.. autosummary::
    :toctree: api

    ~Transport
    ~get_transport
    ~register_transport
"""

from .functions import CoprocessorFunction, css_decoder
from .placement import Backline, Controller, Coprocessor, Executor
from .transports import Transport, get_transport, register_transport

__all__ = [
    "Executor",
    "Controller",
    "Coprocessor",
    "Backline",
    "CoprocessorFunction",
    "css_decoder",
    "Transport",
    "get_transport",
    "register_transport",
]

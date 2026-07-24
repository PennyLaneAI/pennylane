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
The :func:`~pennylane.backline` function builds a device from a placement, which specifies where
each part of the workload runs and the transport protocol between them.

.. warning::

    Backline is experimental. Its API may change without notice, and it is only usable through
    the Catalyst compiler.

A backline device is built with :func:`~pennylane.backline` from a
:class:`controller <.Controller>` (which wraps the PennyLane device the QNode runs on, such as
``lightning.qubit`` or ``null.qubit``), zero or more :class:`coprocessors <.Coprocessor>`, and a
:class:`transport <.Transport>`. The resulting device is passed into a QNode:

.. code-block:: python

    import pennylane as qp

    cpu_controller = qp.Controller(
        qp.device("lightning.qubit", wires=4),
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

    dev = qp.backline(cpu_controller, gpu_coprocessor, transport="rdma")

    @qp.qjit
    @qp.qnode(dev)
    def circuit():
        # To be updated

Executors
~~~~~~~~~

An executor is a node in the backline fabric. An executor can be a Controller (where the QNode is executed, and issues messages), or a Coprocessor (where the messages are processed and returned).

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

A placement groups the controller, coprocessors, and transport. :func:`~pennylane.backline` assembles
them into a device that can be bound to a QNode.

.. autosummary::
    :toctree: api

    ~backline
    ~Backline

Device
~~~~~~

:func:`~pennylane.backline` returns a device that carries the placement and can be bound directly to
a QNode. It requires the Catalyst compiler for execution.

.. autosummary::
    :toctree: api

    ~HeterogeneousDevice

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

from .device import HeterogeneousDevice, backline
from .functions import CoprocessorFunction, css_decoder
from .placement import Backline, Controller, Coprocessor, Executor
from .transports import Transport, get_transport, register_transport

__all__ = [
    "Executor",
    "Controller",
    "Coprocessor",
    "Backline",
    "backline",
    "HeterogeneousDevice",
    "CoprocessorFunction",
    "css_decoder",
    "Transport",
    "get_transport",
    "register_transport",
]

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
The :func:`~pennylane.backline` function builds a device from a :class:`~.Placement`, which specifies
where each part of the workload runs and the :class:`~.Transport` protocol between them.

.. warning::

    Backline is experimental. Its API may change without notice, and it is only usable through
    the Catalyst compiler.

A backline device is built with :func:`~pennylane.backline` from a
:class:`controller <.Controller>` (which wraps the PennyLane device the QNode runs on, such as
``lightning.qubit`` or ``null.qubit``), zero or more :class:`coprocessors <.Coprocessor>`, and a
:class:`transport <.Transport>`. The resulting :class:`~.HeterogeneousDevice` is passed into a
:func:`~pennylane.qnode`:

.. code-block:: python

    import pennylane as qp

    cpu_controller = qp.Controller(
        label="cpu-controller",
        backend="cpu_verbs",
        remote=True,
        executor_options={"host": "192.0.2.10", "port": 7810},
        init_args={
            "config": "dev=mlx5_0;gid=1",
            "data_path": "cpu_verbs",
            "in_bytes": 8,
            "out_bytes": 8,
        },
    )

    gpu_coprocessor = qp.Coprocessor(
        label="gpu-coprocessor",
        coprocessor_fn="decoder",
        backend="gpu_verbs",
        comm_host="198.51.100.2",
        oob_port=7760,
        remote=True,
        executor_options={"host": "192.0.2.11", "port": 7813},
        init_args={"config": "dev=mlx5_0;gid=3", "data_path": "cpu_verbs"},
    )

    dev = qp.Backline(
        controller=cpu_controller, coprocessors=[gpu_coprocessor], transport="rdma"
    )

    @qp.qjit
    @qp.qnode(dev)
    def circuit():
        ...

Nodes
~~~~~

A node is a participant in the backline fabric. It is either a :class:`~.Controller`, where the QNode
executes and which issues messages, or a :class:`~.Coprocessor`, where those messages are processed
and returned. Both share the options on :class:`~.Node`.

.. autosummary::
    :toctree: api

    ~Controller
    ~Coprocessor
    ~Node

Coprocessor functions
~~~~~~~~~~~~~~~~~~~~~~~

A :class:`~.Coprocessor` applies a precompiled function to each message it receives (e.g., decoding a
syndrome). Coprocessor functions can be defined directly in C++ as a runtime function, or in Python
through helper functions such as :func:`~.css_decoder`. Either way they are referenced by a
:class:`~.CoprocessorFunction`.

.. autosummary::
    :toctree: api

    ~CoprocessorFunction
    ~css_decoder

Placement
~~~~~~~~~

A :class:`~.Placement` groups the :class:`~.Controller`, its :class:`coprocessors <.Coprocessor>`, and
the :class:`~.Transport`. :func:`~pennylane.backline` assembles them into a device that can be bound to
a QNode, so a :class:`~.Placement` is normally created by calling :func:`~pennylane.backline` rather
than constructed directly.

.. autosummary::
    :toctree: api

    ~backline
    ~Placement

Device
~~~~~~

:func:`~pennylane.backline` returns a :class:`~.HeterogeneousDevice` that carries the
:class:`~.Placement` and can be bound directly to a :func:`~pennylane.qnode`. It requires the Catalyst
compiler for execution, and exposes the placement it was built from as
:attr:`~.HeterogeneousDevice.placement`.

.. autosummary::
    :toctree: api

    ~HeterogeneousDevice

Transports
~~~~~~~~~~

A :class:`~.Transport` selects, by name, how messages transfer between nodes. Names are resolved with
:func:`~.get_transport` and new ones added with :func:`~.register_transport`; the implementation itself
lives in the compiled runtime.

.. autosummary::
    :toctree: api

    ~Transport
    ~get_transport
    ~register_transport
"""

from .device import HeterogeneousDevice, backline
from .functions import CoprocessorFunction, css_decoder
from .placement import Controller, Coprocessor, Node, Placement
from .transports import Transport, get_transport, register_transport

__all__ = [
    "Node",
    "Controller",
    "Coprocessor",
    "Placement",
    "backline",
    "HeterogeneousDevice",
    "CoprocessorFunction",
    "css_decoder",
    "Transport",
    "get_transport",
    "register_transport",
]

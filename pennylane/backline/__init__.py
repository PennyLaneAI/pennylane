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
        device=qp.device("lightning.qubit", wires=4),
        label="cpu-controller",
        backend="cpu_verbs",
        remote=True,
        executor_options={"host": "192.168.3.15"},
        init_args={"config": "dev=mlx5_0;gid=1"},
    )

    gpu_coprocessor = qp.Coprocessor(
        label="gpu-coprocessor",
        coprocessor_fn="decoder",
        backend="gpu_verbs",
        comm_host="192.168.1.3",
        oob_port=18590,
        remote=False,
        init_args={"config": "dev=mlx5_0;gid=3"},
    )

    dev = qp.backline(
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
syndrome). A :class:`~.CoprocessorFunction` can reference any compatible precompiled library symbol,
including a custom C++ or Triton function. The :func:`~.triton_decoder` and :func:`~.css_bp_decoder`
helpers compile user-defined Triton decoders and CSS belief-propagation decoders, respectively.

.. autosummary::
    :toctree: api

    ~CoprocessorFunction
    ~css_bp_decoder
    ~triton_decoder

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

from .decode import decode
from .device import HeterogeneousDevice, backline
from .functions import CoprocessorFunction, css_bp_decoder, triton_decoder
from .placement import Controller, Coprocessor, ExecutorSpec, Node, Placement
from .transports import Transport, get_transport, register_transport

backline.decode = decode
backline.css_bp_decoder = css_bp_decoder
backline.triton_decoder = triton_decoder

__all__ = [
    "Node",
    "Controller",
    "Coprocessor",
    "ExecutorSpec",
    "Placement",
    "backline",
    "decode",
    "HeterogeneousDevice",
    "CoprocessorFunction",
    "css_bp_decoder",
    "triton_decoder",
    "Transport",
    "get_transport",
    "register_transport",
]

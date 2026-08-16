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
The :class:`~pennylane.Backline` class builds a device with a :class:`~.Placement`, which specifies
where each part of the workload runs and the :class:`~.Transport` protocol between them.

.. warning::

    Backline is experimental. Its API may change without notice, and it is only usable through
    the Catalyst compiler.

A backline device is built with :class:`~pennylane.Backline` from a
:class:`controller <.Controller>` (which wraps the PennyLane device the QNode runs on, such as
``lightning.qubit`` or ``null.qubit``), zero or more :class:`coprocessors <.Coprocessor>`, and a
:class:`transport <.Transport>`. The resulting device is passed into a
:func:`~pennylane.qnode`:

.. code-block:: python

    import pennylane as qp

    controller = qp.Controller(
        hardware="cpu",
        executor_options={"host": "192.0.2.10", "port": 7810},
        init_args={"config": "dev=mlx5_0;gid=1"},
    )

    coprocessor = qp.Coprocessor(
        coprocessor_fn="decoder",
        hardware="gpu",
        endpoint=qp.Endpoint("198.51.100.2", 7760),
        executor_options={"host": "192.0.2.11", "port": 7813},
        init_args={"config": "dev=mlx5_0;gid=3"},
    )

    dev = qp.Backline(
        controller=controller, coprocessors=[coprocessor], transport="rdma"
    )

    @qp.qjit
    @qp.qnode(dev)
    def circuit(x):
        qp.RX(x, wires=0)
        return qp.expval(qp.Z(0))

.. currentmodule:: pennylane

Nodes
~~~~~

A node is a participant in the backline fabric. It is either a :class:`~.Controller`, where the QNode
executes and which issues messages, or a :class:`~.Coprocessor`, where those messages are processed
and returned. Both share the options on :class:`~.Node`.

.. autosummary::
    :toctree: api

    ~Controller
    ~Coprocessor
    ~Endpoint
    ~Node

.. currentmodule:: pennylane.backline

Coprocessor functions
~~~~~~~~~~~~~~~~~~~~~~~

A :class:`~pennylane.Coprocessor` applies a precompiled function to each message it receives (e.g.,
decoding a syndrome). A :class:`~pennylane.CoprocessorFunction` can reference any compatible
precompiled library symbol, including a custom C++ or Triton function. The
:func:`~.triton_decoder` and :func:`~.css_bp_decoder` helpers compile user-defined Triton decoders
and CSS belief-propagation decoders, respectively.

.. currentmodule:: pennylane

.. autosummary::
    :toctree: api

    ~CoprocessorFunction

.. currentmodule:: pennylane.backline

.. autosummary::
    :toctree: api

    ~css_bp_decoder
    ~triton_decoder

Placement
~~~~~~~~~

A :class:`~.Placement` groups the :class:`~pennylane.Controller`, its
:class:`coprocessors <pennylane.Coprocessor>`, and the :class:`~.Transport`.
:class:`~pennylane.Backline` assembles them into a device that can be bound to a QNode, so a
:class:`~.Placement` is normally created by constructing a :class:`~pennylane.Backline` rather than
directly.

.. autosummary::
    :toctree: api

    ~Placement

Decoding
~~~~~~~~

:func:`~.decode` drives one syndrome->correction round from inside a captured QNode: it stages the
syndrome, posts it to a :class:`coprocessor <pennylane.Coprocessor>`, and returns the correction it
replies with.

.. autosummary::
    :toctree: api

    ~decode

.. currentmodule:: pennylane

Device
~~~~~~

:class:`~pennylane.Backline` carries a :class:`~pennylane.backline.Placement` and can be bound
directly to a :func:`~pennylane.qnode`. It requires the Catalyst compiler for execution, and exposes
the placement it was built from as :attr:`~.Backline.placement`.

.. autosummary::
    :toctree: api

    ~Backline

.. currentmodule:: pennylane.backline

Transports
~~~~~~~~~~

A :class:`~.Transport` selects, by name, how messages transfer between nodes. The compiler combines
it with each node's hardware to choose a concrete runtime backend. The built-in transports are
``"rdma"`` and ``"memcpy"``. Names are resolved with :func:`~.get_transport` and new ones added
with :func:`~.register_transport`; the implementation itself lives in the compiled runtime.

.. autosummary::
    :toctree: api

    ~Transport
    ~get_transport
    ~register_transport

Runtime calls
~~~~~~~~~~~~~

This module provides the functionality to call a runtime entry point directly, by its C symbol
name.

A symbol is declared with :func:`~.runtime_declare` and called with :func:`~.runtime_call` from
inside a compiled program. The call can be dispatched to an executor, which invokes the symbol on
the machine the runtime lives on.

.. currentmodule:: pennylane.backline.runtime

.. autosummary::
    :toctree: api

    ~CSignature
    ~CType

**Example**

Declare a symbol once, then call it:

.. code-block:: python

    import pennylane as qp

    qp.runtime_declare("example_run_rounds", "(ptr, u32) -> u64")

    def program(session):
        return qp.runtime_call("example_run_rounds", session, 100000, address="board:9000")

.. currentmodule:: pennylane.backline
"""

from . import runtime
from .decode import decode
from .device import Backline
from .functions import CoprocessorFunction, css_bp_decoder, triton_decoder
from .placement import Controller, Coprocessor, Endpoint, Node, Placement
from .transports import Transport, get_transport, register_transport

__all__ = [
    "Node",
    "Controller",
    "Coprocessor",
    "Endpoint",
    "Placement",
    "Backline",
    "decode",
    "CoprocessorFunction",
    "css_bp_decoder",
    "triton_decoder",
    "Transport",
    "get_transport",
    "register_transport",
    "runtime",
]

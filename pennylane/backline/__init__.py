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
The :class:`~pennylane.Backline` device is built from a :class:`~.Placement`, which specifies where
each part of the workload runs and the :class:`~.Transport` protocol between them.

.. warning::

    Backline is experimental. Its API may change without notice, and it is only usable through
    the Catalyst compiler.

A backline device is built with :class:`~pennylane.Backline` from a
:class:`controller <.Controller>` (which wraps the PennyLane device the QNode runs on, such as
``lightning.qubit`` or ``null.qubit``), zero or more :class:`coprocessors <.Coprocessor>`, and a
:class:`transport <.Transport>`, selected by name (e.g. ``transport="rdma"``) and resolved to a
:class:`~.Transport`. The resulting device is passed into a :func:`~pennylane.qnode`:

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
    def circuit(x):
        qp.RX(x, wires=0)
        return qp.expval(qp.Z(0))

.. currentmodule:: pennylane

Nodes
~~~~~

A node is a participant in the backline fabric. It is either a :class:`~.Controller`, where the
QNode executes and which issues messages, or a :class:`~.Coprocessor`, where those messages are
processed and returned. Both share the options on :class:`~.Node`: a ``label`` to identify the
node, the transport ``backend`` it uses, whether it runs ``remote``, and how its code is deployed
there. A placement has exactly one controller and zero or more coprocessors, and nodes are never
used on their own --- they are passed to :class:`~pennylane.Backline`, which assembles them into a
device.

.. autosummary::
    :toctree: api

    ~Controller
    ~Coprocessor
    ~Node

.. currentmodule:: pennylane.backline

Coprocessor functions
~~~~~~~~~~~~~~~~~~~~~~~

A :class:`~.Coprocessor` applies a precompiled function to each message it receives, for example
decoding a syndrome into a correction. Because that function runs inside the real-time loop it is
compiled ahead of time rather than traced: it can be written directly in C++ as a runtime function,
or generated from Python by :func:`~.triton_decoder`, which compiles user-defined Triton decoders,
and :func:`~.css_bp_decoder`, which builds a CSS belief-propagation decoder from parity-check
matrices. Either way the coprocessor refers to it through a :class:`~.CoprocessorFunction`, which
names the symbol and, optionally, the shared library it lives in. Passing a plain string as a
coprocessor's ``coprocessor_fn`` builds one for you.

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

A :class:`~.Placement` is the complete declarative description of where the workload runs: the
:class:`~.Controller`, its :class:`coprocessors <.Coprocessor>`, the :class:`~.Transport` between
them, and optionally the ``qec_code`` the circuit is encoded for. It is what the compiler
consumes - everything it contains ends up in the compiled program, and nothing else about the
deployment does.
You normally do not construct one directly: :class:`~pennylane.Backline` takes the same arguments,
builds the placement, and carries it as the device's ``placement`` attribute.

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

:class:`~pennylane.Backline` is a device that is bound to a :func:`~pennylane.qnode` like any other
PennyLane device. Its wires come from the controller's own device, so the QNode is written exactly
as it would be against that device alone - the placement changes where the work runs, not the content
of the circuit. It has no Python execution path: the device carries the placement through to the
Catalyst compiler, so a QNode using it must be :func:`~pennylane.qjit`-compiled.

.. autosummary::
    :toctree: api

    ~Backline

.. currentmodule:: pennylane.backline

Transports
~~~~~~~~~~

A :class:`~.Transport` selects, by name, how messages move between nodes. Passing a string as
the ``transport`` argument of :class:`~pennylane.Backline` resolves it for you, so most code never
calls either function directly.

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
from .placement import Controller, Coprocessor, Node, Placement
from .transports import Transport, get_transport, register_transport

__all__ = [
    "Node",
    "Controller",
    "Coprocessor",
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

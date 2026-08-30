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

This module contains functionality for defining and using backlines in PennyLane.
Backline is an open platform for compilation and low-latency execution
that dynamically connects quantum workloads to heterogenous hardware devices, including
GPUs, CPUs, FPGAs, and QPUs. For examples and tutorials see the
`Backline demo <https://pennylane.ai/demos/backline>`__.

.. warning::

    Backline is experimental and under heavy development. Its API may change without notice, and it
    is only usable through the Catalyst compiler.

.. note::

    Backline requires a recent version of PennyLane, Catalyst, and Lightning. Check out the `installation
    instructions and requirements <https://github.com/PennyLaneAI/backline/tree/readme#installation>`__.

    Note that due to the wide range of system, network, and hardware configurations you can use
    Backline with, there are different installation requirements and steps depending on your
    needs.

Overview
~~~~~~~~

.. currentmodule:: pennylane

.. autosummary::
    :toctree: api

    ~Backline
    ~Controller
    ~Coprocessor
    ~Endpoint
    ~backline.get_transport
    ~backline.Node
    ~backline.Placement
    ~backline.Transport
    ~backline.register_transport

Backline provides the following abstractions for use with PennyLane and Catalyst.

- :class:`.Controller`: The classical hardware :class:`~.Node` (such as a CPU or FPGA) that controls
  the QPU(a quantum hardware or simulator `qp.device`), receives quantum measurement results, and
  initiates data transfers with other hardware devices (*coprocessors*). For example, it might
  perform quantum error correction (QEC) syndrome measurements on the QPU, and send these to a
  coprocessor for decoding.

  .. code-block:: python

      import pennylane as qp

      CPU = qp.Controller(
          hardware="cpu",
          executor_options={"host": "192.0.2.10", "port": 7810},
          init_args={"config": "dev=mlx5_0;gid=1"},
          device=qp.device("lightning.qubit", wires=1)
      )

- :class:`.Coprocessor`: Hardware device :class:`~.Node` (such as CPUs, GPUs, or FPGAs)  that
  receive information from a controller for processing. They run :class:`.CoprocessingFuncion`
  callables, potentially as a persistent kernel, such as a QEC decoder.

  .. code-block:: python

      GPU = qp.Coprocessor(
          coprocessor_fn="decoder",
          hardware="gpu",
          endpoint=qp.Endpoint("198.51.100.2", 7760),
          executor_options={"host": "192.0.2.11", "port": 7813},
          init_args={"config": "dev=mlx5_0;gid=3"},
      )

- :class:`.Backline`: A representation of the complete hardware infrastructure supporting the
   quantum-classical program (:class:`~.Placement`), packaged as a QNode device. The backline
   includes a :class:`.Controller`, zero or more :class:`.Coprocessor` nodes, and a transport
   method (``"rdma"``, ``"memcpy"``, or a :class:`~.Transport` object). A backline object is given
   directly to a QNode in place of a traditional QNode `qp.device`, and orchestrates the remote
   executor and the RDMA network the controllers and coprocessors talk over, separate from the
   network used to log into remote machines.

  .. code-block:: python

      dev = qp.Backline(controller=CPU, coprocessors=[GPU], transport="rdma")

      @qp.qjit
      @qp.qnode(dev)
      def circuit(x):
          qp.RX(x, wires=0)
          return qp.expval(qp.Z(0))

Coprocessing functions and QEC
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pennylane

.. autosummary::
    :toctree: api

    ~CoprocessorFunction
    ~backline.decode
    ~backline.css_bp_decoder
    ~backline.triton_decoder

When using a backline to execute a quantum program, there are multiple ways to incorporate
quantum error correction (QEC) encoding and decoding.

Implicit QEC
************

If the ``qec_code`` argument is provided to the Backline object, e.g.,

>>> qp.Backline(controller=CPU, coprocessors=[GPU], transport="rdma", qec_code="steane")

then the encoding is automatically applied when compiling via Catalyst as an
MLIR compilation pass --- the string provided must correspond to an existing
Catalyst QEC encoding pass.

Here, the circuit defined in the PennyLane frontend for execution represents a **logical** circuit;
QEC decoding will automatically be applied. The decoding function to be executed on the coprocessor
can be specified in multiple ways:

* **A Triton kernel**: The provided function :func:`~.triton_decoder` compiles a Python Triton
  decoder function into a shared library that can be used as a coprocessing function. Alternatively,
  :func:`~.css_bp_decoder` is a convenience function to compile a CSS Tanner graph to a
  belief proagation decoder using Triton.

* **A precompiled library**: :class:`~.CoprocessorFunction` registers a precompiled library symbol
  and (optionally) the library path.

The coprocessor decoding function can then be provided when defining the :class:`~.Coprocessor`.

.. note::

    Catalyst ships with a built-in Steane GPU decoding library. This can be specified via the
    string ``coprocessor_fn="gpu_steane_launcher"``.

Explicit QEC
************

If the ``qec_code`` argument is not provided to the Backline object, then no automatic
encoding or decoding will occur. It will be assumed that the circuit defined in the PennyLane
frontend represents a **physical** circuit; encoding and decoding should be manually defined
in the frontend.

To manually encode the circuit, this can be done explicitly in Python, or by manually applying
an MLIR or xDSL compilation pass.

To manually decode the circuit, the :func:`~.decode` function can be used within the QNode
to call a registered coprocessing function --- providing a measured syndrome as input and returning
a correction.

.. code-block:: python

    qdev = qp.device("lightning.qubit", wires=3)
    CPU1 = qp.Controller(device=qdev)
    CPU2 = qp.Coprocessor(coprocessor_fn=steane_decode)

    dev = qp.Backline(controller=CPU1, coprocessors=[CPU2], transport="rdma")

    @qp.qjit(capture=True, autograph=True)
    @qp.qnode(dev)
    def circuit():
        # logical circuit
        ...

        # measure qubits to extract syndromes
        z_syndrome, x_syndrome = extract_syndromes()

        # manual decoding
        correction_z = qp.backline.decode(x_syndrome, decoder_id=0)

        for q in range(N):
            if correction_x[q]:
                qp.X(wires=q)

        return qp.expval(qp.Z(0))

Runtime calls
~~~~~~~~~~~~~

.. currentmodule:: pennylane

.. autosummary::
    :toctree: api

    ~runtime_call
    ~runtime.declare
    ~backline.runtime.CSignature
    ~backline.runtime.CType

For explicit control of runtime calls, Catalyst additionally provides functionality to call a
runtime entry point directly within the ``qjit`` compiled function, by its C symbol name.

A symbol is declared with :func:`~.runtime_declare` and called with :func:`~.runtime_call` from
inside a compiled program. The call can be dispatched to an executor, which invokes the symbol on
the machine the runtime lives on.
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

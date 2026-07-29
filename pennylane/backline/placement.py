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

"""Placement types and node constructors for backline heterogeneous compilation and execution."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .functions import CoprocessorFunction
from .transports import Transport, get_transport

if TYPE_CHECKING:
    from pennylane.devices import Device

@dataclass(frozen=True)
class ExecutorSpec:
    """Declarative executor configuration, realized by the compiler backend.
    """

    options: dict = field(default_factory=dict)
    """Backend-specific executor options, passed verbatim to the compiler's executor builder."""


@dataclass(frozen=True, kw_only=True)
class Node:
    """A node in a backline fabric, including its name and connection information.

    Base class for :class:`Controller` and :class:`Coprocessor`. It carries information to determine whether the node's code needs to be cross-compiled and dispatched to a remote host or run
    locally. Nodes are assembled into a device with :func:`~pennylane.backline`.

    See the Attributes section to learn more about the available options.
    """

    name: str | None = None
    """The backend device this node maps to, e.g. ``"gpu-libibverbs"`` or
    ``"cpu-libibverbs"``. Defaults to ``None``."""

    addr: str | None = None
    """Host address of the node. Required for remote nodes; may be ``None`` for local ones."""

    port: str | None = None
    """Port the node is reached on."""

    triple: str | None = None
    """Cross-compilation target triple for the node's code."""

    remote: bool = True
    """Whether the node runs on a separate host reached over the network (cross-compiled and
    dispatched) rather than locally."""

    executor: object | None = None
    """The executor this node runs on. Either a launched executor (e.g. a ``catalyst.Executor``,
    duck-typed: exposes ``address`` and ``triple``) or an :class:`ExecutorSpec` the compiler realizes
    into one. When realized, its ``address``/``triple`` drive the node's cross-compile and remote
    dispatch; ``addr``/``port`` remain the data-plane endpoint. Defaults to ``None``."""

    init_args: dict = field(default_factory=dict)
    """Backend-specific initialization arguments; empty by default (never ``None``)."""


@dataclass(frozen=True)
class Controller(Node):
    """The node that controls the QPU and initiates data transfers.

    The controller runs the qnode and is the data-initiator during a decoding step: it sends
    syndromes to the coprocessors and receives corrections back. Pass it to
    :func:`~pennylane.backline` to build a device.

    Args:
        device (Device | None): The PennyLane device the controller executes (e.g. ``qp.device("lightning.qubit")``). Defaults to ``None``, which uses a ``null.qubit`` device.

    See the Attributes section for the connection options inherited from :class:`Node`.
    """

    device: "Device | None" = None
    """The PennyLane device the controller executes, e.g. ``qp.device("lightning.qubit")``. When ``None``, a ``null.qubit`` device is used."""

    def __post_init__(self):
        if self.device is None:
            import pennylane as qp

            object.__setattr__(self, "device", qp.device("null.qubit"))


@dataclass(frozen=True, kw_only=True)
class Coprocessor(Node):
    """The node that runs a coprocessor function per received message.

    A coprocessor receives messages from the controller (e.g., syndromes). The ``coprocessor_fn`` is
    used to process the message, and sends the result back (e.g., corrections). Depending on the
    connection type, a ``coprocessor_fn`` may be a persistent kernel. Pass coprocessors to
    :func:`~pennylane.backline` to build a device.

    See the Attributes section to learn more about the available options.
    """

    coprocessor_fn: str | CoprocessorFunction
    """The function for processing the received message. A string is resolved to a
    :class:`~.CoprocessorFunction` by name."""

    def __post_init__(self):
        if isinstance(self.coprocessor_fn, str):
            object.__setattr__(self, "coprocessor_fn", CoprocessorFunction(self.coprocessor_fn))


@dataclass(frozen=True, kw_only=True)
class Backline:
    """Declarative placement for heterogeneous execution.

    Contains a controller node, any coprocessor nodes, and the transport that carries data between
    them. Use :func:`~pennylane.backline` to assemble a controller, coprocessors, and transport into a device that carries this placement.

    See the Attributes section to learn more about the available options.
    """

    controller: Controller
    """The node running the qnode."""

    coprocessors: tuple = ()
    """Coprocessing accelerators."""

    transport: str | Transport
    """How bytes move between executors, by registry name (e.g. ``"rdma"``) or a
    :class:`~.Transport`. A name is resolved to a :class:`~.Transport` on construction."""

    def __post_init__(self):
        if not isinstance(self.coprocessors, tuple):
            object.__setattr__(self, "coprocessors", tuple(self.coprocessors))
        if isinstance(self.transport, str):
            object.__setattr__(self, "transport", get_transport(self.transport))


def controller(*, device=None, name=None, addr=None, port=None, remote=True, triple=None,
               init_args=None, **executor_kwargs):
    """Construct a :class:`Controller`, recording its executor for the compiler to launch.

    Returns:
        Controller: The controller node carrying its executor spec.
    """
    executor = ExecutorSpec(executor_kwargs) if executor_kwargs else None
    return Controller(device=device, name=name, addr=addr, port=port, remote=remote,
                      triple=triple, init_args=init_args or {}, executor=executor)


def coprocessor(*, coprocessor_fn, name=None, addr=None, port=None, remote=True, triple=None,
                init_args=None, **executor_kwargs):
    """Construct a :class:`Coprocessor`, recording its executor for the compiler to launch.
    
    Returns:
        Coprocessor: The coprocessor node carrying its executor spec.
    """
    executor = ExecutorSpec(executor_kwargs) if executor_kwargs else None
    return Coprocessor(coprocessor_fn=coprocessor_fn, name=name, addr=addr, port=port,
                       remote=remote, triple=triple, init_args=init_args or {}, executor=executor)

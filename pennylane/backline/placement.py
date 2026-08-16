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

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, get_args

from pennylane.devices.device_constructor import device as _make_device

from .functions import CoprocessorFunction
from .transports import Transport, get_transport

if TYPE_CHECKING:
    from pennylane.devices import Device

# Wires given to the ``null.qubit`` device a :class:`~.Controller` falls back to.
DEFAULT_WIRES = 32
DEFAULT_MESSAGE_BYTES = 8
Hardware = Literal["cpu", "gpu", "fpga"]
"""Hardware on which a backline node executes."""
_SUPPORTED_HARDWARE = frozenset(get_args(Hardware))


@dataclass(frozen=True)
class Endpoint:
    """The address the controller dials to bring up a connection to a coprocessor.

    The coprocessor listens on :attr:`port`; the controller connects to :attr:`host`\\ ``:``\\
    :attr:`port`. Some transports, such as ``"rdma"``, require an endpoint on every coprocessor;
    others, such as ``"memcpy"``, do not use a network endpoint. For a coprocessor co-located with
    the controller on a transport that does require one, use localhost (``"127.0.0.1"``).

    .. seealso:: :class:`~.Coprocessor`
    """

    host: str
    """The address the controller connects to, e.g. ``"192.0.2.11"`` or ``"127.0.0.1"``."""

    port: int | None = None
    """The port the coprocessor listens on for the out-of-band connection handshake. This is the
    handshake channel that exchanges the information needed to set up the data path. Defaults to
    ``None``, leaving the choice to the compiled runtime."""

    def __post_init__(self):
        if not isinstance(self.host, str):
            raise TypeError(f"host must be a str, got {type(self.host).__name__}: {self.host!r}")
        if not self.host:
            raise ValueError("host must be a non-empty str")
        if self.port is not None:
            if not isinstance(self.port, int):
                raise TypeError(
                    f"port must be an int, got {type(self.port).__name__}: {self.port!r}"
                )
            if not 1 <= self.port <= 65535:
                raise ValueError(f"port must be in 1..65535, got {self.port}")


@dataclass(frozen=True, kw_only=True)
class Node:
    """A node in a backline fabric.

    Base class for :class:`~.Controller` and :class:`~.Coprocessor`. It carries the node's name and
    hardware, how its code is deployed, and any backend-specific initialization arguments. Nodes
    are assembled into a device with :class:`~pennylane.Backline`.

    See the Attributes section to learn more about the available options.
    """

    name: str | None = None
    """An optional name used to reference this node."""

    hardware: Hardware = "cpu"
    """The hardware this node executes on: ``"cpu"``, ``"gpu"``, or ``"fpga"``. The compiler
    combines this with the placement's :class:`~.Transport` to select the runtime backend."""

    executor_options: dict | None = None
    """Options for the executor to launch for this node, passed to the compiler's executor
    builder. ``None`` (the default) requests no executor, leaving the node in this process; ``{}``
    requests one with all defaults. Asking for an executor is what makes a node :attr:`remote`. The
    launched executor also determines the node's cross-compilation target triple, detecting it on
    the target host when not given explicitly. TODO: add what is recognized here"""

    executor: object | None = None
    """The launched executor this node runs on. Created automatically by the compiler from
    :attr:`executor_options`."""

    init_args: dict = field(default_factory=dict)
    """Backend-specific initialization arguments; empty by default (never ``None``). TODO: add what is recognized here
    """

    def __post_init__(self):
        if self.hardware not in _SUPPORTED_HARDWARE:
            raise ValueError(
                f"hardware must be one of {sorted(_SUPPORTED_HARDWARE)}, got {self.hardware!r}"
            )

    @property
    def remote(self) -> bool:
        """bool: Whether this node's code is dispatched to an executor rather than run in the
        present process, so that the libraries it loads live beside that executor rather than in
        this installation. A node is remote exactly when it has an executor to run on, whether still
        requested through :attr:`executor_options` or already launched."""
        return self.executor_options is not None or self.executor is not None


@dataclass(frozen=True, kw_only=True)
class Controller(Node):
    """The node that controls the QPU and initiates data transfers.

    The controller runs the QNode and is the data-initiator during a decoding step: it sends
    syndromes to the :class:`coprocessors <.Coprocessor>` and receives corrections back. Pass it to
    :class:`~pennylane.Backline` to build a device.

    Args:
        device (pennylane.devices.Device | None): The PennyLane device the controller executes.
            Defaults to ``None``, which uses a ``null.qubit`` device.

    See the Attributes section for the options inherited from :class:`~.Node`.

    .. seealso:: :class:`~.Coprocessor`, :class:`~pennylane.Backline`
    """

    device: "Device | None" = None
    """The PennyLane device the controller executes, e.g. one built with :func:`~pennylane.device`.
    Defaults to ``None``, which builds a ``null.qubit`` over :data:`DEFAULT_WIRES` wires.
    A controller needing more wires or an actual simulation, should pass a device of its own."""

    in_bytes: int = DEFAULT_MESSAGE_BYTES
    """The transport's input-message capacity in bytes."""

    out_bytes: int = DEFAULT_MESSAGE_BYTES
    """The transport's reply-message capacity in bytes."""

    def __post_init__(self):
        super().__post_init__()
        if self.device is None:
            object.__setattr__(self, "device", _make_device("null.qubit", wires=DEFAULT_WIRES))
        for name in ("in_bytes", "out_bytes"):
            value = getattr(self, name)
            if not isinstance(value, int):
                raise TypeError(f"{name} must be an int, got {type(value).__name__}: {value!r}")
            if value < 1:
                raise ValueError(f"{name} must be a positive int, got {value}")


@dataclass(frozen=True, kw_only=True)
class Coprocessor(Node):
    """The node that runs a coprocessor function per received message.

    A coprocessor receives messages from the :class:`controller <.Controller>` (e.g., syndromes). The
    :attr:`coprocessor_fn` is used to process the message, and sends the result back (e.g.,
    corrections). Depending on the connection type, a :attr:`coprocessor_fn` may be a persistent
    kernel. Pass coprocessors to :class:`~pennylane.Backline` to build a device.

    The coprocessor owns the connection :attr:`endpoint`: it listens on :attr:`Endpoint.port`, and
    the controller dials :attr:`Endpoint.host`\\ ``:``\\ :attr:`Endpoint.port` to bring the
    connection up.

    See the Attributes section to learn more about the available options.

    .. seealso:: :class:`~.Controller`, :class:`~.CoprocessorFunction`, :class:`~.Endpoint`,
        :class:`~pennylane.Backline`
    """

    coprocessor_fn: str | CoprocessorFunction
    """The function for processing the received message. A string is resolved to a
    :class:`~.CoprocessorFunction` by name."""

    endpoint: Endpoint | None = None
    """The address the controller dials to reach this coprocessor. Some transports, such as
    ``"rdma"``, require it; others, such as ``"memcpy"``, do not use a network endpoint and may
    leave it unset."""

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.coprocessor_fn, str):
            object.__setattr__(self, "coprocessor_fn", CoprocessorFunction(self.coprocessor_fn))


@dataclass(frozen=True, kw_only=True)
class Placement:
    """Declarative placement for heterogeneous execution.

    Contains a :class:`controller <.Controller>` node, any :class:`coprocessor <.Coprocessor>` nodes,
    and the :class:`transport <.Transport>` that carries data between them. Rather than constructing
    this directly, use :class:`~pennylane.Backline` to assemble a controller, coprocessors, and
    transport into a device; the resulting placement is available as the device's
    :attr:`~.Backline.placement` attribute.

    See the Attributes section to learn more about the available options.

    .. seealso:: :class:`~pennylane.Backline`
    """

    controller: Controller
    """The :class:`~.Controller` running the QNode."""

    coprocessors: Sequence["Coprocessor"] = ()
    """The :class:`coprocessing accelerators <.Coprocessor>`. Any sequence is accepted, and is stored
    as a tuple."""

    transport: str | Transport
    """How bytes move between nodes, by registry name (e.g. ``"rdma"``) or a :class:`~.Transport`. A
    name is resolved to a :class:`~.Transport` on construction with :func:`~.get_transport`."""

    qec_code: str | None = None
    """The quantum error-correcting code the circuit is encoded for, e.g. ``"steane"``. Naming it 
    here lets the compiler encode the circuit, and no separate lowering step is needed. Defaults to ``None``,
    which leaves the circuit unencoded."""

    def __post_init__(self):
        if not isinstance(self.coprocessors, tuple):
            object.__setattr__(self, "coprocessors", tuple(self.coprocessors))
        if isinstance(self.transport, str):
            object.__setattr__(self, "transport", get_transport(self.transport))

        if self.transport.name == "rdma":
            for coprocessor in self.coprocessors:
                # TODO: how should we handle when these fields are provided for `memcpy`
                if coprocessor.endpoint is None:
                    raise ValueError(
                        "transport='rdma' requires every coprocessor to set endpoint; "
                        "memcpy does not require it"
                    )

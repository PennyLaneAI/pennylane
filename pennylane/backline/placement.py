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
from typing import TYPE_CHECKING

from pennylane.devices.device_constructor import device as _make_device

from .functions import CoprocessorFunction
from .transports import Transport, get_transport

if TYPE_CHECKING:
    from pennylane.devices import Device


@dataclass(frozen=True, kw_only=True)
class Node:
    """A node in a backline fabric.

    Base class for :class:`~.Controller` and :class:`~.Coprocessor`. It carries the node's label and
    backend implementation, how its code is deployed, and any backend-specific initialization
    arguments. Nodes are assembled into a device with :func:`~pennylane.backline`.

    See the Attributes section to learn more about the available options.
    """

    label: str | None = None
    """A name for this node, used to identify its transport session and to label its executor's
    logs. Defaults to ``None``, in which case the compiler derives one from the node's role
    (``"controller"``, ``"coprocessor.0"``, ...). This does not select a backend — see
    :attr:`backend`."""

    backend: str | None = None
    """The transport backend this node uses, by name, e.g. ``"cpu_verbs"`` or ``"gpu_verbs"``. The
    compiler resolves the name together with the node's role to the installed backend library, so the
    backend only has to be available to the compiler. Defaults to ``None``, letting the compiler pick
    its default. A ``"backend_lib"`` path in :attr:`init_args` takes precedence."""

    remote: bool = False
    """Whether the node runs on a separate host reached over the network (cross-compiled and
    dispatched) rather than locally. Defaults to ``False``. A remote node needs an executor to
    dispatch its compiled code to, which is created and attached by the compiler
    using :attr:`executor_options`."""

    executor_options: dict | None = None
    """Options for the executor to launch for this node, passed to the compiler's executor
    builder. ``None`` (the default) requests no executor; ``{}`` requests one with all defaults.
    The launched executor also determines the node's cross-compilation target triple, detecting it
    on the target host when not given explicitly. TODO: add what is recognized here"""

    executor: object | None = None
    """The launched executor this node runs on. Created automatically by the compiler from
    :attr:`executor_options`."""

    init_args: dict = field(default_factory=dict)
    """Backend-specific initialization arguments; empty by default (never ``None``). TODO: add what is recognized here
    """

    # Aliases for Catalyst (``name``/``addr``/``port``/``triple``).

    @property
    def name(self) -> "str | None":
        """Alias of :attr:`label`."""
        return self.label

    @property
    def addr(self) -> "str | None":
        """Alias of :attr:`~.Coprocessor.comm_host`, or ``None``."""
        return getattr(self, "comm_host", None)

    @property
    def port(self) -> "int | None":
        """Alias of :attr:`~.Coprocessor.oob_port`, or ``None``."""
        return getattr(self, "oob_port", None)

    @property
    def triple(self) -> "str | None":
        """Cross-compilation triple from :attr:`executor`, or ``None``."""
        return getattr(getattr(self, "executor", None), "triple", None)

    def _fill_backend_lib(self, role: str) -> None:
        """Set ``init_args['backend_lib']`` from :attr:`backend` if not already set."""
        if not self.backend:
            return
        init = self.init_args or {}
        if "backend_lib" in init:
            return
        lib = f"libcatalyst_transport_{self.backend}_{role}.so"
        object.__setattr__(self, "init_args", {**init, "backend_lib": lib})

    def _ensure_executor_spec(self) -> None:
        """Wrap :attr:`executor_options` into an :class:`ExecutorSpec` for remote nodes."""
        if self.remote and self.executor_options is not None and self.executor is None:
            object.__setattr__(self, "executor", ExecutorSpec(options=dict(self.executor_options)))


@dataclass(frozen=True)
class ExecutorSpec:
    """Unrealized executor request held on :attr:`~.Node.executor` until the compiler launches it."""

    options: dict = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class Controller(Node):
    """The node that controls the QPU and initiates data transfers.

    The controller runs the QNode and is the data-initiator during a decoding step: it sends
    syndromes to the :class:`coprocessors <.Coprocessor>` and receives corrections back. Pass it to
    :func:`~pennylane.backline` to build a device.

    Args:
        device (Device | None): The PennyLane device the controller executes. Defaults to ``None``,
            which uses a ``null.qubit`` device.

    See the Attributes section for the options inherited from :class:`~.Node`.

    .. seealso:: :class:`~.Coprocessor`, :func:`~pennylane.backline`
    """

    device: "Device | None" = None
    """The PennyLane device the controller executes, e.g. one built with :func:`~pennylane.device`.
    When ``None``, a ``null.qubit`` device is used."""

    def __post_init__(self):
        if self.device is None:
            object.__setattr__(self, "device", _make_device("null.qubit"))
        self._fill_backend_lib("controller")
        self._ensure_executor_spec()


@dataclass(frozen=True, kw_only=True)
class Coprocessor(Node):
    """The node that runs a coprocessor function per received message.

    A coprocessor receives messages from the :class:`controller <.Controller>` (e.g., syndromes). The
    :attr:`coprocessor_fn` is used to process the message, and sends the result back (e.g.,
    corrections). Depending on the connection type, a :attr:`coprocessor_fn` may be a persistent
    kernel. Pass coprocessors to :func:`~pennylane.backline` to build a device.

    The coprocessor owns the connection endpoint: it listens on :attr:`oob_port`, and the controller
    dials :attr:`comm_host`\\ ``:``\\ :attr:`oob_port` to bring the connection up.

    See the Attributes section to learn more about the available options.

    .. seealso:: :class:`~.Controller`, :class:`~.CoprocessorFunction`, :func:`~pennylane.backline`
    """

    coprocessor_fn: str | CoprocessorFunction
    """The function for processing the received message. A string is resolved to a
    :class:`~.CoprocessorFunction` by name."""

    comm_host: str
    """This coprocessor's address, which the controller connects to in order to bring up the
    connection. Must be reachable from the host the controller runs on, and is required for every
    coprocessor. For one co-located with the controller, use localhost (``"127.0.0.1"``)."""

    oob_port: int | None = None
    """The port this coprocessor listens on for the out-of-band connection handshake. This is the handshake channel that exchanges the information needed to set up the
    data path. Defaults to ``None``, leaving the choice to the compiled runtime."""

    def __post_init__(self):
        if isinstance(self.coprocessor_fn, str):
            object.__setattr__(self, "coprocessor_fn", CoprocessorFunction(self.coprocessor_fn))
        if self.oob_port is not None:
            if not isinstance(self.oob_port, int):
                raise TypeError(
                    f"oob_port must be an int, got {type(self.oob_port).__name__}: "
                    f"{self.oob_port!r}"
                )
            if not 1 <= self.oob_port <= 65535:
                raise ValueError(f"oob_port must be in 1..65535, got {self.oob_port}")
        self._fill_backend_lib("coprocessor")
        self._ensure_executor_spec()


@dataclass(frozen=True, kw_only=True)
class Placement:
    """Declarative placement for heterogeneous execution.

    Contains a :class:`controller <.Controller>` node, any :class:`coprocessor <.Coprocessor>` nodes,
    and the :class:`transport <.Transport>` that carries data between them. Rather than constructing
    this directly, use :func:`~pennylane.backline` to assemble a controller, coprocessors, and
    transport into a device; the resulting placement is available as the device's
    :attr:`~.HeterogeneousDevice.placement` attribute.

    See the Attributes section to learn more about the available options.

    .. seealso:: :func:`~pennylane.backline`, :class:`~.HeterogeneousDevice`
    """

    controller: Controller
    """The :class:`~.Controller` running the QNode."""

    coprocessors: Sequence["Coprocessor"] = ()
    """The :class:`coprocessing accelerators <.Coprocessor>`. Any sequence is accepted, and is stored
    as a tuple."""

    transport: str | Transport
    """How bytes move between nodes, by registry name (e.g. ``"rdma"``) or a :class:`~.Transport`. A
    name is resolved to a :class:`~.Transport` on construction with :func:`~.get_transport`."""

    def __post_init__(self):
        if not isinstance(self.coprocessors, tuple):
            object.__setattr__(self, "coprocessors", tuple(self.coprocessors))
        if isinstance(self.transport, str):
            object.__setattr__(self, "transport", get_transport(self.transport))

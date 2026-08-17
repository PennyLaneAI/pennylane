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

# Wires given to the ``null.qubit`` device a :class:`~.Controller` falls back to.
DEFAULT_WIRES = 32


@dataclass(frozen=True, kw_only=True)
class Node:
    """A node in a backline fabric.

    Base class for :class:`~.Controller` and :class:`~.Coprocessor`. It carries the node's label and
    backend implementation, how its code is deployed, and any backend-specific initialization
    arguments. Nodes are assembled into a device with :class:`~pennylane.Backline`.

    .. warning::

        Backline is experimental. Its API may change without notice, and it is only usable through
        the Catalyst compiler.

    Keyword Args:
        label (str | None): A name identifying this node. Defaults to ``None``, letting the compiler
            derive one from the node's role.
        backend (str | None): The transport backend this node uses, by name. Defaults to ``None``,
            letting the compiler pick its default.
        remote (bool): Whether this node runs on another machine. Defaults to ``False``.
        executor_options (dict | None): Options for the executor to launch for this node.
            Defaults to ``None``, which runs the node in this process. See the
            :attr:`~.Node.executor_options` attribute below for every option it accepts.
        executor (object | None): An already-launched executor to attach. Defaults to ``None``, in
            which case the compiler builds one from ``executor_options``.
        init_args (dict): Backend-specific initialization arguments. Empty by default. See the
            :attr:`~.Node.init_args` attribute below for the keys it accepts.

    .. seealso:: :class:`~.Controller`, :class:`~.Coprocessor`, :class:`~pennylane.Backline`
    """

    label: str | None = None
    """A name for this node, used to identify its transport session and to label its executor's
    logs. Defaults to ``None``, in which case the compiler derives one from the node's role
    (``"controller"``, ``"coprocessor.0"``, ...). This does not select a backend - see
    :attr:`backend`."""

    backend: str | None = None
    """The transport backend this node uses, by name, e.g. ``"cpu_verbs"`` or ``"gpu_verbs"``. The
    compiler resolves the name together with the node's role to the installed backend library, so
    the backend only has to be available to the compiler. Defaults to ``None``, letting the compiler
    pick its default. A ``"backend_lib"`` path in :attr:`init_args` takes precedence."""

    remote: bool = False
    """Whether this node runs on another machine, so that the libraries it loads are the ones
    installed beside it there rather than the ones in this installation. Defaults to ``False``.

    This is about the machine, not the process: it says where the node's libraries are resolved, and
    :attr:`executor_options` independently says whether its code runs in a process of its own. The
    three valid combinations are:

    * ``remote=True`` with :attr:`executor_options`: out-of-process on another machine, reached by
      deploying an executor there. A remote node cannot be reached without one.
    * ``remote=False`` with :attr:`executor_options`: out-of-process on this machine, in an executor
      subprocess whose libraries still resolve from this installation.
    * ``remote=False`` with no :attr:`executor_options`: in-process, the default.
    """

    executor_options: dict | None = None
    """Options for the executor to launch for this node, passed to it as keyword arguments. ``None``
    (the default) requests no executor, leaving the node in this process.

    **Which machine, and how it is reached**

    * ``"host"`` (str) - deploy the executor to this host over ssh. Requires ``"port"``, so that
      the address is known without having to reach the host at compile time.
    * ``"address"`` (str) - ``"host:port"`` of an executor whose lifetime is managed elsewhere.
      Attach to it rather than launching one.
    * ``"port"`` (int) - the port the executor binds on the target, and the local end of the ssh
      tunnel reaching it. Omit it for a local executor and a free port is chosen at launch, so the
      address is only known afterwards.
    * ``"user"`` (str) - the ssh account on the remote host. Defaults to the local username.

    Naming neither ``"host"`` nor ``"address"`` runs the executor as a subprocess on this machine.

    **What is placed on the target**

    * ``"workspace"`` (str) - the directory the executor runs in. Defaults to a generated
      ``catalyst-exec-*`` directory removed on teardown; one named here is left in place.
    * ``"deploy"`` (list[str]) - files and directories copied into the workspace before the
      executor starts. A directory contributes the files inside it, which is how a cross-built set
      of artifacts travels.
    * ``"plugins"`` (list[str]) - shared libraries the executor loads at startup, in order. They
      share one symbol namespace, so the first definition wins. The compiler appends the libraries
      this node implies - a coprocessor's decode function, a controller's device runtime - if
      they are not already listed.
    * ``"executor_bin"`` (str) - the command that starts the executor, for wrapping it in
      something like ``numactl``.

    **How the process runs**

    * ``"env"`` (dict[str, str]) - environment variables for the executor process.
    * ``"sudo"`` (bool) - run it as root, for a target whose devices are not world-accessible.
      Defaults to ``False``.
    * ``"sudo_password"`` (str) - password piped to ``sudo -S``; unnecessary with passwordless
      sudo.
    * ``"ready_timeout"`` (float) - seconds to wait for the executor to report that it bound its
      port. Defaults to ``60.0``.
    * ``"verbose"`` (int) - how much the launcher logs: ``0`` quiet, ``1`` normal, ``2``
      per-command detail.

    **Code generation and identity**

    * ``"triple"`` (str) - the LLVM target triple to compile this node's code for. Omit it and the
      executor detects it from the target's ``uname``, which requires reaching the host.
    * ``"name"`` (str) - the executor's own label, used in its log filenames. Defaults to this
      node's :attr:`label`.

    Note that ``"port"`` here is the executor's, on the channel that ships compiled code. It is
    unrelated to :attr:`~.Coprocessor.oob_port`, which is the transport's handshake port. An
    unrecognized key raises.

    .. warning::

        These options are defined by the Catalyst compiler, and may not be stable.
    """

    executor: object | None = None
    """The launched executor this node runs on. Created automatically by the compiler from
    :attr:`executor_options`; set it directly to attach one that is already launched, in which case
    :attr:`executor_options` is ignored."""

    init_args: dict = field(default_factory=dict)
    """Backend-specific initialization arguments, forwarded to the transport backend. Empty by
    default (never ``None``). The keys the compiler forwards are

    * ``"backend_lib"`` (str) - explicit path to the transport backend library, taking precedence
      over :attr:`backend` and over a :class:`~.CoprocessorFunction`'s ``lib_path``.
    * ``"config"`` (str) - a ``;``-separated ``key=value`` string configuring the backend on this
      machine, e.g. ``"dev=mlx5_0;gid=3"``. ``dev`` and ``gid`` select the RDMA device and GID
      index; the remaining keys are backend-specific (a GPU backend takes ``gpu=``, an FPGA engine
      takes ``sq_mem=``/``data_mem=``/``reply_mem=``).
    * ``"data_path"`` (str) - which wire format carries the data, e.g. ``"cpu_verbs"``.
    * ``"in_bytes"`` / ``"out_bytes"`` (int) - the fixed message sizes exchanged with this node.

    Keys outside this set are dropped rather than rejected, so a misspelling is silent.
    """


@dataclass(frozen=True, kw_only=True)
class Controller(Node):
    """The node that controls the QPU and initiates data transfers.

    The controller runs the QNode and is the data-initiator during a decoding step: it sends
    syndromes to the :class:`coprocessors <.Coprocessor>` and receives corrections back. Pass it to
    :class:`~pennylane.Backline` to build a device.

    .. warning::

        Backline is experimental. Its API may change without notice, and it is only usable through
        the Catalyst compiler.

    Keyword Args:
        device (pennylane.devices.Device | None): The PennyLane device the controller executes.
            Defaults to ``None``, which builds a ``null.qubit``.

    See :class:`~.Node` for the options every node shares.

    .. seealso:: :class:`~.Coprocessor`, :class:`~pennylane.Backline`

    **Example**

    In the simplest case the controller runs in this process, on a default ``null.qubit``:

    >>> con = qp.Controller()
    >>> con.device.name
    'null.qubit'

    Pass a device to run a real simulation, and deployment options to run it on another machine:

    .. code-block:: python

        con = qp.Controller(
            device=qp.device("lightning.qubit", wires=4),
            label="cpu-controller",
            backend="cpu_verbs",
            remote=True,
            executor_options={"host": "192.0.2.10", "port": 7810},
            init_args={"config": "dev=mlx5_0;gid=1", "data_path": "cpu_verbs", "out_bytes": 8},
        )

    Either way, the controller is passed to :class:`~pennylane.Backline` to build a device:

    .. code-block:: python

        dev = qp.Backline(controller=con, transport="rdma")
    """

    device: "Device | None" = None
    """The PennyLane device the controller executes, e.g. one built with :func:`~pennylane.device`.
    Defaults to ``None``, which builds a ``null.qubit`` over :data:`DEFAULT_WIRES` wires. A
    controller needing more wires, or an actual simulation, should pass a device of its own."""

    def __post_init__(self):
        if self.device is None:
            object.__setattr__(self, "device", _make_device("null.qubit", wires=DEFAULT_WIRES))


@dataclass(frozen=True, kw_only=True)
class Coprocessor(Node):
    """The node that runs a coprocessor function per received message.

    A coprocessor receives messages from the :class:`controller <.Controller>` (e.g., syndromes).
    The ``coprocessor_fn`` is used to process the message, and sends the result back (e.g.,
    corrections). Depending on the connection type, a ``coprocessor_fn`` may be a persistent
    kernel. Pass coprocessors to :class:`~pennylane.Backline` to build a device.

    The coprocessor owns the connection endpoint: it listens on ``oob_port``, and the controller
    connects via ``comm_host``\\ ``:``\\ ``oob_port`` to bring the connection up.

    .. warning::

        Backline is experimental. Its API may change without notice, and it is only usable through
        the Catalyst compiler.

    Keyword Args:
        coprocessor_fn (str | CoprocessorFunction): The function that processes each received
            message. A string is resolved to a :class:`~.CoprocessorFunction` by name.
        comm_host (str): This coprocessor's address, which the controller uses to connect. Required.
        oob_port (int | None): The port it listens on for the connection handshake. Must be in
            ``1..65535``. Defaults to ``None``, leaving the choice to the compiled runtime.

    See :class:`~.Node` for the options every node shares.

    .. seealso:: :class:`~.Controller`, :class:`~.CoprocessorFunction`, :class:`~pennylane.Backline`

    **Example**

    A coprocessor co-located with the controller needs only a function and a loopback address:

    >>> coproc = qp.Coprocessor(coprocessor_fn="decoder", comm_host="127.0.0.1")
    >>> coproc.coprocessor_fn.name
    'decoder'

    A remote GPU decoder names its port, its transport backend, and how to deploy to it:

    .. code-block:: python

        coproc = qp.Coprocessor(
            label="gpu-decoder",
            coprocessor_fn="decoder",
            backend="gpu_verbs",
            comm_host="198.51.100.2",
            oob_port=7760,
            remote=True,
            executor_options={"host": "192.0.2.11", "port": 7813},
            init_args={"config": "dev=mlx5_0;gid=3;gpu=0", "data_path": "cpu_verbs"},
        )

    Note that ``comm_host`` and ``executor_options["host"]`` are two addresses for the *same*
    machine: the first is the interface the transport's data path uses, the second is how the
    compiler reaches it to deploy code.

    Coprocessors are passed to :class:`~pennylane.Backline` as a sequence:

    .. code-block:: python

        dev = qp.Backline(controller=qp.Controller(), coprocessors=[coproc], transport="rdma")
    """

    coprocessor_fn: str | CoprocessorFunction
    """The function for processing each received message. A string is resolved to a
    :class:`~.CoprocessorFunction` by name."""

    comm_host: str
    """This coprocessor's address, which the controller connects to in order to bring the connection
    up. Must be reachable from the host the controller runs on, and is required for every
    coprocessor. For one co-located with the controller, use localhost (``"127.0.0.1"``)."""

    oob_port: int | None = None
    """The port this coprocessor listens on for the out-of-band connection handshake - the channel
    that exchanges the information needed to set up the data path. Must be in ``1..65535``. Defaults
    to ``None``, leaving the choice to the compiled runtime."""

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


@dataclass(frozen=True, kw_only=True)
class Placement:
    """Declarative placement for heterogeneous execution.

    Contains a :class:`controller <.Controller>` node, any :class:`coprocessor <.Coprocessor>`
    nodes, and the :class:`transport <.Transport>` that carries data between them. Rather than
    constructing this directly, use :class:`~pennylane.Backline` to assemble a controller,
    coprocessors, and transport into a device; the resulting placement is available as the
    device's ``placement`` attribute.

    .. warning::

        Backline is experimental. Its API may change without notice, and it is only usable through
        the Catalyst compiler.

    Keyword Args:
        controller (Controller): The :class:`~.Controller` running the QNode.
        coprocessors (Sequence[Coprocessor]): The
            :class:`coprocessing accelerators <.Coprocessor>`. Defaults to ``()``.
        transport (str | Transport): How bytes move between nodes, by registry name (e.g.
            ``"rdma"``) or a :class:`~.Transport`.
        qec_code (str | None): The quantum error-correcting code the circuit is encoded for.
            Defaults to ``None``, which leaves the circuit unencoded.

    .. seealso:: :class:`~.Controller`, :class:`~.Coprocessor`, :class:`~.Transport`,
        :class:`~pennylane.Backline`

    **Example**

    A placement is built for you by :class:`~pennylane.Backline` and read back off the device:

    >>> con = qp.Controller(label="cpu-controller")
    >>> coproc = qp.Coprocessor(coprocessor_fn="decoder", comm_host="127.0.0.1")
    >>> dev = qp.Backline(controller=con, coprocessors=[coproc], transport="rdma")
    >>> dev.placement.transport
    Transport(name='rdma')
    >>> dev.placement.controller.label
    'cpu-controller'
    >>> len(dev.placement.coprocessors)
    1

    ``coprocessors`` accepts any sequence and is normalized to a tuple, and ``transport`` accepts
    either a registry name or a :class:`~.Transport`:

    >>> isinstance(dev.placement.coprocessors, tuple)
    True
    """

    controller: Controller
    """The :class:`~.Controller` running the QNode."""

    coprocessors: Sequence["Coprocessor"] = ()
    """The :class:`coprocessing accelerators <.Coprocessor>`. Any sequence is accepted, and is
    stored as a tuple."""

    transport: str | Transport
    """How bytes move between nodes, by registry name (e.g. ``"rdma"``) or a :class:`~.Transport`. A
    name is resolved to a :class:`~.Transport` on construction with :func:`~.get_transport`."""

    qec_code: str | None = None
    """The quantum error-correcting code the circuit is encoded for, e.g. ``"steane"``. Naming it
    here lets the compiler encode the circuit, and no separate lowering step is needed. Defaults to
    ``None``, which leaves the circuit unencoded."""

    def __post_init__(self):
        if not isinstance(self.coprocessors, tuple):
            object.__setattr__(self, "coprocessors", tuple(self.coprocessors))
        if isinstance(self.transport, str):
            object.__setattr__(self, "transport", get_transport(self.transport))

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

    Args:
        host (str): The address the controller connects to, e.g. ``"192.0.2.11"`` or
            ``"127.0.0.1"``.
        port (int): The port the coprocessor listens on for the out-of-band connection handshake.
            This is the handshake channel that exchanges the information needed to set up the data
            path.

    .. seealso:: :class:`~.Coprocessor`

    **Example**

    >>> ep = qp.Endpoint("127.0.0.1", 7760)
    >>> (ep.host, ep.port)
    ('127.0.0.1', 7760)
    """

    host: str
    """The address the controller connects to, e.g. ``"192.0.2.11"`` or ``"127.0.0.1"``."""

    port: int
    """The port the coprocessor listens on for the out-of-band connection handshake. This is the
    handshake channel that exchanges the information needed to set up the data path."""

    def __post_init__(self):
        if not isinstance(self.host, str):
            raise TypeError(f"host must be a str, got {type(self.host).__name__}: {self.host!r}")
        if not self.host:
            raise ValueError("host must be a non-empty str")
        if not isinstance(self.port, int):
            raise TypeError(f"port must be an int, got {type(self.port).__name__}: {self.port!r}")
        if not 1 <= self.port <= 65535:
            raise ValueError(f"port must be in 1..65535, got {self.port}")


@dataclass(frozen=True, kw_only=False)
class Node:
    """A node in a backline fabric.

    Base class for :class:`~.Controller` and :class:`~.Coprocessor`. It carries the node's name and
    hardware, how its code is deployed, and any backend-specific initialization arguments. Nodes
    are assembled into a device with :class:`~pennylane.Backline`.

    .. warning::

        :mod:`Backline <.backline>` is experimental and only usable through the Catalyst
        compiler.

    Keyword Args:
        name (str, None): A name identifying this node. Defaults to ``None``, letting the compiler
            derive one from the node's role.
        hardware (str): The hardware this node executes on. Defaults to ``"cpu"``; other allowed values are
            ``gpu`` and ``fpga`. The compiler
            combines this with the placement's :class:`~.Transport` to select the runtime backend.
        remote (bool): Whether this node runs on another machine. Defaults to ``False``.
        executor_options (dict, None): Options for the executor to launch for this node.
            Defaults to ``None``, which runs the node in this process. See the
            :attr:`~.Node.executor_options` attribute below for every option it accepts.
        executor (object, None): An already-launched executor to attach. Defaults to ``None``, in
            which case the compiler builds one from :attr:`executor_options`.
        init_args (dict): Backend-specific initialization arguments. Empty by default. See the
            :attr:`~.Node.init_args` attribute below for the keys it accepts.

    .. seealso:: :class:`~.Controller`, :class:`~.Coprocessor`, :class:`~pennylane.Backline`
    """

    name: str | None = None
    """An optional name used to reference this node."""

    hardware: Hardware = "cpu"
    """The hardware this node executes on. The compiler combines this with the placement's
    :class:`~.Transport` to select the runtime backend."""

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
      node's :attr:`name`.

    Note that ``"port"`` here is the executor's, on the channel that ships compiled code. It is
    unrelated to :attr:`~.Endpoint.port`, which is the transport's handshake port. An
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
      over :attr:`hardware` and over :attr:`~.CoprocessorFunction.lib_path`.
    * ``"config"`` (str) - a ``;``-separated ``key=value`` string configuring the backend on this
      machine, e.g. ``"dev=mlx5_0;gid=3"``. ``dev`` and ``gid`` select the RDMA device and GID
      index; the remaining keys are backend-specific (a GPU backend takes ``gpu=``, an FPGA engine
      takes ``sq_mem=``/``data_mem=``/``reply_mem=``).


    Keys outside this set are dropped rather than rejected, so a misspelling is silent.
    """

    def __post_init__(self):
        if self.hardware not in _SUPPORTED_HARDWARE:
            raise ValueError(
                f"hardware must be one of {sorted(_SUPPORTED_HARDWARE)}, got {self.hardware!r}"
            )


@dataclass(frozen=True, kw_only=False)
class Controller(Node):
    """The node that controls the QPU and initiates data transfers.

    The controller runs the QNode and is the data-initiator during a decoding step: it sends
    syndromes to the :class:`coprocessors <.Coprocessor>` and receives corrections back. Pass it to
    :class:`~pennylane.Backline` to build a device.

    .. warning::

        Backline is experimental. Its API may change without notice, and it is only usable through
        the Catalyst compiler.

    Keyword Args:
        name (str, None): A name identifying the controller. Defaults to ``None``, letting the compiler
            derive one from the node's role.
        hardware (Hardware): The hardware the controller executes on. Defaults to ``"cpu"``; other allowed
            values are ``"gpu"`` and ``"fpga"``. The compiler
            combines this with the placement's :class:`~.Transport` to select the runtime backend.
        remote (bool): Whether the controller runs on another machine. Defaults to ``False``.
        executor_options (dict, None): Options for the executor to launch for this controller.
            Defaults to ``None``, which runs the node in this process. See the
            :attr:`~.Node.executor_options` attribute below for every option it accepts.
        executor (object, None): An already-launched executor to attach. Defaults to ``None``, in
            which case the compiler builds one from :attr:`executor_options`.
        init_args (dict): Backend-specific initialization arguments. Empty by default. See the
            :attr:`~.Node.init_args` attribute below for the keys it accepts.
        device (pennylane.devices.Device, None): The PennyLane device the controller executes.
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
            name="cpu-controller",
            hardware="cpu",
            remote=True,
            executor_options={"host": "192.0.2.10", "port": 7810},
            init_args={"config": "dev=mlx5_0;gid=1"},
        )

    Either way, the controller is passed to :class:`~pennylane.Backline` to build a device:

    .. code-block:: python

        dev = qp.Backline(controller=con, transport="rdma")
    """

    device: "Device | None" = None
    """The PennyLane device the controller executes. Defaults to ``None``, which builds a
    ``null.qubit`` over :data:`DEFAULT_WIRES` wires. A controller needing more wires, or an actual
    simulation, should pass a device of its own."""

    in_bytes: int = field(default=DEFAULT_MESSAGE_BYTES, init=False, repr=False)
    """The transport's input-message capacity in bytes. Always :data:`DEFAULT_MESSAGE_BYTES`;
    provided for the compiler, not a constructor argument."""

    out_bytes: int = field(default=DEFAULT_MESSAGE_BYTES, init=False, repr=False)
    """The transport's reply-message capacity in bytes. Always :data:`DEFAULT_MESSAGE_BYTES`;
    provided for the compiler, not a constructor argument."""

    def __post_init__(self):
        super().__post_init__()
        if self.device is None:
            object.__setattr__(self, "device", _make_device("null.qubit", wires=DEFAULT_WIRES))


@dataclass(frozen=True, kw_only=False)
class Coprocessor(Node):
    """The node that runs a coprocessor function per received message.

    A coprocessor receives messages from the :class:`controller <.Controller>` (e.g., syndromes).
    The :attr:`coprocessor_fn` is used to process the message, and sends the result back (e.g.,
    corrections). Depending on the connection type, a :attr:`coprocessor_fn` may be a persistent
    kernel. Pass coprocessors to :class:`~pennylane.Backline` to build a device.

    The coprocessor owns the connection :attr:`endpoint`: it listens on :attr:`Endpoint.port`, and
    the controller dials :attr:`Endpoint.host`\\ ``:``\\ :attr:`Endpoint.port` to bring the
    connection up.

    .. warning::

        :mod:`Backline <.backline>` is experimental and only usable through the Catalyst
        compiler.

    Keyword Args:
        name (str, None): A name identifying the controller. Defaults to ``None``, letting the compiler
            derive one from the node's role.
        hardware (Hardware): The hardware the coprocessing function executes on. Defaults to ``"cpu"``; other allowed values are
            ``gpu``. ``fpga`` is not currently accepted. The compiler
            combines this with the placement's :class:`~.Transport` to select the runtime backend.
        remote (bool): Whether the controller runs on another machine. Defaults to ``False``.
        coprocessor_fn (str, CoprocessorFunction): The function that processes each received
            message. A string is wrapped in a :class:`~.CoprocessorFunction` naming that symbol, so
            reading the attribute back always gives a :class:`~.CoprocessorFunction`.
        endpoint (Endpoint, None): The address the controller dials to reach this coprocessor.
            Some transports, such as ``"rdma"``, require it; others, such as ``"memcpy"``, do not
            use a network endpoint and may leave it unset.
        executor_options (dict, None): Options for the executor to launch for this controller.
            Defaults to ``None``, which runs the node in this process. See the
            :attr:`~.Node.executor_options` attribute below for every option it accepts.
        executor (object, None): An already-launched executor to attach. Defaults to ``None``, in
            which case the compiler builds one from :attr:`executor_options`.
        init_args (dict): Backend-specific initialization arguments. Empty by default. See the
            :attr:`~.Node.init_args` attribute below for the keys it accepts.

    See :class:`~.Node` for the options every node shares.

    .. seealso:: :class:`~.Controller`, :class:`~.CoprocessorFunction`, :class:`~.Endpoint`,
        :class:`~pennylane.Backline`

    **Example**

    A coprocessor co-located with the controller needs only a function (and, for ``"rdma"``, a
    loopback endpoint):

    >>> coproc = qp.Coprocessor(coprocessor_fn="decoder", endpoint=qp.Endpoint("127.0.0.1", 7760))
    >>> coproc.coprocessor_fn.name
    'decoder'

    A remote GPU decoder names its endpoint, its hardware, and how to deploy to it:

    .. code-block:: python

        coproc = qp.Coprocessor(
            name="gpu-decoder",
            coprocessor_fn="decoder",
            hardware="gpu",
            endpoint=qp.Endpoint("198.51.100.2", 7760),
            remote=True,
            executor_options={"host": "192.0.2.11", "port": 7813},
            init_args={"config": "dev=mlx5_0;gid=3;gpu=0"},
        )

    Note that :attr:`Endpoint.host` and ``executor_options["host"]`` are two addresses for the
    *same* machine: the first is the interface the transport's data path uses, the second is how
    the compiler reaches it to deploy code.

    Coprocessors are passed to :class:`~pennylane.Backline` as a sequence:

    .. code-block:: python

        dev = qp.Backline(controller=qp.Controller(), coprocessors=[coproc], transport="rdma")
    """

    coprocessor_fn: str | CoprocessorFunction
    """The function for processing each received message. Accepts a
    :class:`~.CoprocessorFunction`, or a string naming the symbol, which is wrapped in one on
    construction - so this attribute always reads back as a :class:`~.CoprocessorFunction`. The
    symbol itself is resolved later, by the runtime."""

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

    Contains a :class:`controller <.Controller>` node, any :class:`coprocessor <.Coprocessor>`
    nodes, and the :class:`transport <.Transport>` that carries data between them. Rather than
    constructing this directly, use :class:`~pennylane.Backline` to assemble a controller,
    coprocessors, and transport into a device; the resulting placement is available as
    :attr:`~pennylane.Backline.placement`.

    .. warning::

        :mod:`Backline <.backline>` is experimental and only usable through the Catalyst
        compiler.

    Keyword Args:
        controller (Controller): The :class:`~.Controller` running the QNode.
        coprocessors (Sequence[Coprocessor]): The
            :class:`coprocessing accelerators <.Coprocessor>`. Defaults to ``()``.
        transport (str | Transport): Which transport carries data between nodes, as a
            :class:`~.Transport` or its registry name (e.g. ``"rdma"``). A name is resolved on
            construction, so this reads back as a :class:`~.Transport`.
        qec_code (str | None): The quantum error-correcting code the circuit is encoded for.
            Defaults to ``None``, which leaves the circuit unencoded.

    .. seealso:: :class:`~.Controller`, :class:`~.Coprocessor`, :class:`~.Transport`,
        :class:`~pennylane.Backline`

    **Example**

    A placement is built for you by :class:`~pennylane.Backline` and read back off the device:

    >>> con = qp.Controller(name="cpu-controller")
    >>> coproc = qp.Coprocessor(coprocessor_fn="decoder", endpoint=qp.Endpoint("127.0.0.1", 7760))
    >>> dev = qp.Backline(controller=con, coprocessors=[coproc], transport="rdma")
    >>> dev.placement.transport
    Transport(name='rdma')
    >>> dev.placement.controller.name
    'cpu-controller'
    >>> len(dev.placement.coprocessors)
    1


    :attr:`coprocessors` accepts any sequence and is normalized to a tuple, and :attr:`transport`
    accepts either a registry name or a :class:`~.Transport`:

    >>> isinstance(dev.placement.coprocessors, tuple)
    True
    """

    controller: Controller
    """The :class:`~.Controller` running the QNode."""

    coprocessors: Sequence["Coprocessor"] = ()
    """The :class:`coprocessing accelerators <.Coprocessor>`. Any sequence is accepted, and is
    stored as a tuple."""

    transport: str | Transport
    """Which transport carries data between nodes. Accepts a :class:`~.Transport`, or its registry
    name (e.g. ``"rdma"``), which is resolved with :func:`~.get_transport` on construction - so
    this attribute always reads back as a :class:`~.Transport`. The transport itself is implemented
    in the compiled runtime."""

    qec_code: str | None = None
    """The quantum error-correcting code for logical qubit encoding, e.g. ``"steane"``. Naming it
    here lets the compiler encode the circuit, and no separate lowering step is needed. Defaults to
    ``None``, which leaves the circuit unencoded."""

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

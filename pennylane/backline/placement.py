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

"""Placement types for backline heterogeneous compilation and execution."""

from dataclasses import dataclass

from .functions import CoprocessorFunction
from .transports import Transport, get_transport


@dataclass(frozen=True, kw_only=True)
class Executor:
    """A node in a backline fabric, including its name and connection information.

    Base class for :class:`Controller` and :class:`Coprocessor`. It carries information to determine
    whether the node's code needs to be cross-compiled and dispatched to a remote host or run locally.

    Args:
        name (str): The backend executor device this executor maps to, e.g. ``"gpu-libibverbs"`` or
            ``"cpu-libibverbs"``.
        addr (str | None): Host address of the executor. Required for remote executors; may be
            ``None`` for local ones. Defaults to ``None``.
        port (str | None): Port the executor is reached on. Defaults to ``None``.
        triple (str | None): Cross-compilation target triple for the executor's code. Defaults to
            ``None``.
        remote (bool): Whether the executor runs on a separate host reached over the network
            (cross-compiled and dispatched) rather than locally. Defaults to ``True``.
        init_args (dict | None): Optional backend-specific initialization arguments. Defaults to
            ``None``.
    """

    name: str
    addr: str | None = None
    port: str | None = None
    triple: str | None = None
    remote: bool = True
    init_args: dict | None = None


@dataclass(frozen=True, kw_only=True)
class Controller(Executor):
    """The executor that controls the QPU and initiates data transfers.

    The controller runs the qnode and is the data-initiator during a decoding step: it sends
    syndromes to the coprocessors and receives corrections back.

    Args:
        simulator (str): The QPU simulator backing the controller, e.g. ``"null.qubit"``. Defaults
            to ``"null.qubit"``.
    """

    simulator: str = "null.qubit"


@dataclass(frozen=True, kw_only=True)
class Coprocessor(Executor):
    """The executor that runs a coprocessor function per received message.

    A coprocessor receives messages from the controller (e.g. syndromes). The
    ``coprocessor_fn`` is used to process the message, and sends the result back (e.g. corrections).

    Depending on the connectino type, a ``coprocessor_fn`` may be a persistent kernel.

    Args:
        coprocessor_fn (str | CoprocessorFunction): The function for processing the received message. A
            string is resolved to a :class:`~.CoprocessorFunction` by name.
    """

    coprocessor_fn: str | CoprocessorFunction

    def __post_init__(self):
        if isinstance(self.coprocessor_fn, str):
            object.__setattr__(self, "coprocessor_fn", CoprocessorFunction(self.coprocessor_fn))


@dataclass(frozen=True)
class Backline:
    """Declarative placement for heterogeneous execution.

    Contains a controller node, any coprocessor nodes, and the transport that carries data between
    them.

    Args:
        controller (Controller): The node controlling the QPU/qnode.
        transport (str | Transport): How bytes move between executors, by registry name (e.g.
            ``"rdma"``) or a :class:`~.Transport`.
        coprocessors (tuple[Coprocessor, ...]): Coprocessing accelerators. Defaults to ``()``.
    """

    controller: Controller
    transport: str | Transport
    coprocessors: tuple = ()

    def __post_init__(self):
        if not isinstance(self.coprocessors, tuple):
            object.__setattr__(self, "coprocessors", tuple(self.coprocessors))
        if isinstance(self.transport, str):
            get_transport(self.transport)


def controller(  # pylint: disable=too-many-arguments
    *,
    name,
    controller_simulator="null.qubit",
    addr=None,
    port=None,
    triple=None,
    remote=True,
    init_args=None,
):
    """Construct a :class:`Controller` executor.

    Args:
        name (str): The backend executor device the controller maps to.
        controller_simulator (str): The QPU simulator backing the controller. Defaults to
            ``"null.qubit"``.
        addr (str | None): Host address. Defaults to ``None``.
        port (str | None): Port. Defaults to ``None``.
        triple (str | None): Cross-compilation target triple. Defaults to ``None``.
        remote (bool): Whether the controller runs remotely. Defaults to ``True``.
        init_args (dict | None): Backend-specific initialization arguments. Defaults to ``None``.

    Returns:
        Controller: The constructed controller.
    """
    return Controller(
        name=name,
        simulator=controller_simulator,
        addr=addr,
        port=port,
        triple=triple,
        remote=remote,
        init_args=init_args,
    )


def coprocessor(  # pylint: disable=too-many-arguments
    *,
    name,
    coprocessor_fn,
    addr=None,
    port=None,
    triple=None,
    remote=True,
    init_args=None,
):
    """Construct a :class:`Coprocessor` executor.

    Args:
        name (str): The backend execturo device the coprocessor maps to.
        coprocessor_fn (str | CoprocessorFunction): The function applied per received message.
        addr (str | None): Host address. Defaults to ``None``.
        port (str | None): Port. Defaults to ``None``.
        triple (str | None): Cross-compilation target triple. Defaults to ``None``.
        remote (bool): Whether the coprocessor runs remotely. Defaults to ``True``.
        init_args (dict | None): Backend-specific initialization arguments. Defaults to ``None``.

    Returns:
        Coprocessor: The constructed coprocessor.
    """
    return Coprocessor(
        name=name,
        coprocessor_fn=coprocessor_fn,
        addr=addr,
        port=port,
        triple=triple,
        remote=remote,
        init_args=init_args,
    )


def backline(controller, transport, coprocessors=()):  # pylint: disable=redefined-outer-name
    """Construct a :class:`Backline` placement."""
    return Backline(controller=controller, transport=transport, coprocessors=tuple(coprocessors))

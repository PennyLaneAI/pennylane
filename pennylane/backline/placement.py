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

    Base class for :class:`Controller` and :class:`Coprocessor`. It carries information to determine whether the node's code needs to be cross-compiled and dispatched to a remote host or run
    locally.

    See the Attributes section to learn more about the available options.
    """

    name: str
    """The backend executor device this executor maps to, e.g. ``"gpu-libibverbs"`` or
    ``"cpu-libibverbs"``."""

    addr: str | None = None
    """Host address of the executor. Required for remote executors; may be ``None`` for local ones."""

    port: str | None = None
    """Port the executor is reached on."""

    triple: str | None = None
    """Cross-compilation target triple for the executor's code."""

    remote: bool = True
    """Whether the executor runs on a separate host reached over the network (cross-compiled and
    dispatched) rather than locally."""

    init_args: dict | None = None
    """Optional backend-specific initialization arguments."""


@dataclass(frozen=True, kw_only=True)
class Controller(Executor):
    """The executor that controls the QPU and initiates data transfers.

    The controller runs the qnode and is the data-initiator during a decoding step: it sends
    syndromes to the coprocessors and receives corrections back.

    See the Attributes section to learn more about the available options.
    """

    simulator: str = "null.qubit"
    """The QPU simulator running on the controller, e.g., ``"null.qubit"``."""


@dataclass(frozen=True, kw_only=True)
class Coprocessor(Executor):
    """The executor that runs a coprocessor function per received message.

    A coprocessor receives messages from the controller (e.g., syndromes). The ``coprocessor_fn`` is
    used to process the message, and sends the result back (e.g., corrections). Depending on the
    connection type, a ``coprocessor_fn`` may be a persistent kernel.

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
    them.

    See the Attributes section to learn more about the available options.
    """

    controller: Controller
    """The node running the qnode."""

    coprocessors: tuple = ()
    """Coprocessing accelerators."""

    transport: str | Transport
    """How bytes move between executors, by registry name (e.g. ``"rdma"``) or a
    :class:`~.Transport`."""

    def __post_init__(self):
        if not isinstance(self.coprocessors, tuple):
            object.__setattr__(self, "coprocessors", tuple(self.coprocessors))
        if isinstance(self.transport, str):
            get_transport(self.transport)

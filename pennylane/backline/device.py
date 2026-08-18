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

"""The heterogeneous device.

A frontend device that carries a backline placement (consisting of controller, coprocessors, transport) for
heterogeneous compilation and execution. This device requires the Catalyst compiler.
"""

from collections.abc import Sequence
from typing import cast

from pennylane.capture import get_tracing_device
from pennylane.devices import Device

from .placement import Controller, Coprocessor, Placement


class Backline(Device):
    """A device for heterogeneous compilation and execution over a backline placement.

    The device stores the :class:`~.Placement` consisting of a :class:`transport <.Transport>`,
    :class:`controller <.Controller>`, and :class:`coprocessors <.Coprocessor>`. This device
    requires the Catalyst compiler.

    Keyword Args:
        controller (Controller): The :class:`~.Controller` that drives the QPU and runs the QNode.
        coprocessors (Sequence[Coprocessor]): Zero or more :class:`~.Coprocessor` accelerators.
        transport (str | Transport): The transfer protocol between nodes, by registry name (e.g.
            ``"rdma"``) or a :class:`~.Transport`.
        shots (int | None): Number of shots. Defaults to ``None`` (analytic); set shots on the
            QNode with :func:`~pennylane.set_shots` instead.

    .. warning::

        Backline is experimental. Its API may change without notice, and it is only usable through
        the Catalyst compiler.

    .. seealso:: :class:`~.Controller`, :class:`~.Coprocessor`, :class:`~.Placement`

    **Example**

    .. code-block:: python

        import pennylane as qp

        con = qp.Controller(
            device=qp.device("null.qubit", wires=4),
            name="cpu-controller",
            executor_options={"host": "192.168.3.15", "port": 7810},
        )
        coproc = qp.Coprocessor(
            coprocessor_fn="decoder",
            name="decoder-0",
            hardware="gpu",
            endpoint=qp.Endpoint("198.51.100.2", 7760),
            executor_options={"host": "192.0.2.11", "port": 7813},
        )

        dev = qp.Backline(controller=con, coprocessors=[coproc], transport="rdma")

        @qp.qjit
        @qp.qnode(dev)
        def circuit(x):
            qp.RX(x, wires=0)
            return qp.expval(qp.Z(0))
    """

    def __init__(
        self,
        *,
        controller: Controller,
        coprocessors: Sequence[Coprocessor] = (),
        transport,
        qec_code: str | None = None,
        shots=None,
    ):
        self._placement = Placement(
            controller=controller,
            coprocessors=coprocessors,
            transport=transport,
            qec_code=qec_code,
        )
        self._device = cast(Device, controller.device)
        super().__init__(wires=self._device.wires, shots=shots)
        self.config_filepath = self._device.config_filepath

    @property
    def placement(self):
        """Placement: The :class:`~.Placement` the device was configured with."""
        return self._placement

    @property
    def backline(self):
        """Placement: Alias of :attr:`placement` for Catalyst (``device.backline``)."""
        return self._placement

    @property
    def transport(self):
        """Transport: The :class:`~.Transport` carrying data between nodes."""
        return self._placement.transport

    @property
    def controller(self):
        """Controller: The :class:`~.Controller` node of the placement."""
        return self._placement.controller

    @property
    def coprocessors(self):
        """tuple[Coprocessor, ...]: The :class:`~.Coprocessor` nodes of the placement."""
        return self._placement.coprocessors

    @property
    def qec_code(self):
        """str | None: The quantum error-correcting code circuits on this device are encoded for."""
        return self._placement.qec_code

    @property
    def name(self):
        return self._device.name

    def preprocess(self, *args, **kwargs):
        return self._device.preprocess(*args, **kwargs)

    def __getattr__(self, item):
        if item == "_device":
            raise AttributeError(item)
        return getattr(self._device, item)

    def execute(self, circuits, execution_config=None):
        """Execution is handled by the Catalyst compiler; there is no Python execution path."""
        raise NotImplementedError(
            "Backline has no Python execution path; execute it via a compiler such as "
            "Catalyst (@qjit)."
        )


def active_placement() -> "Placement | None":
    """The placement an in-circuit call belongs to: the one on the device being traced.

    ``None`` when there is no trace in progress, or when the device being traced did not come from
    :class:`Backline`.

        import pennylane as qp

        con = qp.Controller(
            name="cpu-controller",
            remote=True,
            executor_options={"host": "192.0.2.10", "port": 7810},
            init_args={"config": "dev=mlx5_1;gid=3"},
        )
        coproc = qp.Coprocessor(
            name="decoder-0",
            coprocessor_fn="decoder",
            hardware="gpu",
            endpoint=qp.Endpoint("198.51.100.2", 7760),
            remote=True,
            executor_options={"host": "192.0.2.11", "port": 7813},
            init_args={"config": "dev=mlx5_1;gid=3;gpu=0"},
        )

        dev = qp.Backline(
            controller=con, coprocessors=[coproc], transport="rdma", qec_code="steane"
        )

        @qp.qjit
        @qp.qnode(dev)
        def circuit():
            ...
        
    .. seealso:: :func:`~pennylane.backline.decode`
 
    """
    return getattr(get_tracing_device(), "placement", None)

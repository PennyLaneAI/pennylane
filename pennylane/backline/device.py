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

from pennylane.capture import get_tracing_device
from pennylane.devices import Device

from .placement import Controller, Coprocessor, Placement


class HeterogeneousDevice(Device):
    """A device for heterogeneous compilation and execution over a backline placement.

    Rather than constructing this directly, build one with :func:`~pennylane.backline`::

        cpu_controller = qp.Controller(
            label="cpu-controller",
            backend="cpu_verbs",
            executor_options={"host": "192.0.2.10", "port": 7810},
        )

        gpu_coprocessor = qp.Coprocessor(
            label="gpu-coprocessor",
            coprocessor_fn="decoder",
            backend="gpu_verbs",
            comm_host="198.51.100.2",
            oob_port=7760,
            executor_options={"host": "192.0.2.11", "port": 7813},
        )

        dev = qp.backline(
            controller=cpu_controller, coprocessors=[gpu_coprocessor], transport="rdma"
        )

    The device stores the :class:`~.Placement` consisting of a :class:`transport <.Transport>`,
    :class:`controller <.Controller>`, and :class:`coprocessors <.Coprocessor>`. This device
    requires the Catalyst compiler.

    Args:
        placement (Placement): The :class:`~.Placement` to execute over.
        shots (int | None): Number of shots. Defaults to ``None`` (analytic); set shots on the
            QNode with :func:`~pennylane.set_shots` instead.

    .. seealso:: :func:`~pennylane.backline`, :class:`~.Placement`
    """

    def __init__(self, *, placement, shots=None):
        self._placement = placement
        self._device = placement.controller.device
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
            "HeterogeneousDevice has no Python execution path; execute it via a "
            "compiler such as Catalyst (@qjit)."
        )


def backline(
    *,
    controller: Controller,
    coprocessors: Sequence[Coprocessor] = (),
    transport,
    qec_code: str | None = None,
) -> HeterogeneousDevice:
    """Build a heterogeneous execution device from a backline placement.

    The returned device can be passed straight to a :func:`~pennylane.qnode`. Its wires are taken
    from the controller's device. This device requires the Catalyst compiler.

    .. warning::

        Backline is experimental. Its API may change without notice, and it is only usable through
        the Catalyst compiler.

    Keyword Args:
        controller (Controller): The :class:`~.Controller` that drives the QPU and runs the QNode.
        coprocessors (Sequence[Coprocessor]): Zero or more :class:`~.Coprocessor` accelerators.
            Defaults to ``()``.
        transport (str | Transport): The transfer protocol between nodes, by registry name (e.g.
            ``"rdma"``) or a :class:`~.Transport`.
        qec_code (str | None): The quantum error-correcting code to implicitly encode the circuit.
            Currently the only supported option is ``"steane"``. Defaults to ``None``, leaving the
            circuit unencoded.

    Returns:
        HeterogeneousDevice: A :class:`~.HeterogeneousDevice` carrying the
        :class:`~.Placement`.

    .. seealso:: :class:`~.Controller`, :class:`~.Coprocessor`, :class:`~.Placement`,
        :class:`~.HeterogeneousDevice`

    **Example**

    .. code-block:: python

        import pennylane as qp

        con = qp.Controller(
            label="cpu-controller",
            backend="cpu_verbs",
            executor_options={"host": "192.0.2.10", "port": 7810},
            init_args={
                "config": "dev=mlx5_1;gid=3",
                "data_path": "cpu_verbs",
                "in_bytes": 8,
                "out_bytes": 8,
            },
        )
        coproc = qp.Coprocessor(
            label="decoder-0",
            coprocessor_fn="decoder",
            backend="gpu_verbs",
            comm_host="198.51.100.2",
            oob_port=7760,
            executor_options={"host": "192.0.2.11", "port": 7813},
            init_args={"config": "dev=mlx5_1;gid=3;gpu=0", "data_path": "cpu_verbs"},
        )

        dev = qp.backline(
            controller=con, coprocessors=[coproc], transport="rdma", qec_code="steane"
        )

        @qp.qjit
        @qp.qnode(dev)
        def circuit(x):
            qp.RX(x, wires=0)
            return qp.expval(qp.Z(0))
    """
    placement = Placement(
        controller=controller,
        coprocessors=coprocessors,
        transport=transport,
        qec_code=qec_code,
    )
    return HeterogeneousDevice(placement=placement)


def active_placement() -> "Placement | None":
    """The placement an in-circuit call belongs to: the one on the device being traced.

    ``None`` when there is no trace in progress, or when the device being traced did not come from
    :func:`backline`.

    .. seealso:: :func:`~pennylane.backline.decode`
    """
    return getattr(get_tracing_device(), "placement", None)

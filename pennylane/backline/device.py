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

from pennylane.devices import Device

from .placement import Backline, Controller, Coprocessor


class HeterogeneousDevice(Device):
    """A device for heterogeneous compilation and execution over a backline placement.

    Rather than constructing this directly, build one with :func:`~pennylane.backline`::

        cpu_controller = qp.Controller(
                qp.device("lightning.qubit", wires=4),
                name="cpu-controller",
                addr="192.168.1.1",
                port="1234",
                triple="aarch64-unknown-linux-gnu",
                remote=True,
        )

        gpu_coprocessor = qp.Coprocessor(
            name="gpu-coprocessor",
            coprocessor_fn="decoder",
            remote=False,
        )

        dev = qp.backline(cpu_controller, gpu_coprocessor, transport="rdma")

    The device stores the :class:`~pennylane.backline.Backline` placement consisting of a transport,
    controller, and coprocessors. Its wires are taken from the controller's device. This device
    requires the Catalyst compiler.

    Args:
        backline (Backline): The placement (controller, coprocessors, transport).
        shots (int | None): Number of shots. Defaults to ``None`` (analytic); set shots on the
            QNode with :func:`~pennylane.set_shots` instead.
    """

    def __init__(self, backline, shots=None):
        self._backline = backline
        self._device = backline.controller.device  # the real, compilable device
        super().__init__(wires=self._device.wires, shots=shots)

    @property
    def backline(self):
        """The placement backline the device was configured with."""
        return self._backline

    @property
    def transport(self):
        """The transport that carries data between executors (the backline's transport)."""
        return self._backline.transport

    @property
    def controller(self):
        """Controller: The controller executor of the backline placement."""
        return self._backline.controller

    @property
    def coprocessors(self):
        """tuple[Coprocessor, ...]: The coprocessor executors of the backline placement."""
        return self._backline.coprocessors

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


def backline(controller: Controller, *coprocessors: Coprocessor, transport) -> HeterogeneousDevice:
    """Build a heterogeneous execution device from a backline placement.

    The returned device can be passed straight to a QNode, e.g. ``@qp.qnode(dev)``. Its wires are
    taken from the controller's device. This device requires the Catalyst compiler.

    .. warning::

        Backline is experimental. Its API may change without notice, and it is only usable through
        the Catalyst compiler.

    Args:
        controller (Controller): The executor that drives the QPU and runs the QNode.
        *coprocessors (Coprocessor): Zero or more coprocessing accelerators.

    Keyword Args:
        transport (str | Transport): The transfer protocol between executors, by registry name (e.g.
            ``"rdma"``) or a :class:`~pennylane.backline.Transport`.

    Returns:
        HeterogeneousDevice: A PennyLane device carrying the placement.

    **Example**

    .. code-block:: python

        import pennylane as qp

        con = qp.Controller(qp.device("lightning.qubit", wires=4), addr="192.168.1.1", port="1234")
        coproc = qp.Coprocessor(coprocessor_fn="decoder", name="gpu-libibverbs")

        dev = qp.backline(con, coproc, transport="rdma")

        @qp.qjit
        @qp.qnode(dev)
        def circuit():
            ...
    """
    placement = Backline(controller=controller, coprocessors=coprocessors, transport=transport)
    return HeterogeneousDevice(placement)

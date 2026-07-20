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

"""The ``ftqc.heterogeneous`` device.

A frontend device that carries a backline placement (consisting of controller, coprocessors, transport) for
heterogeneous compilation and execution. This device requires the Catalyst compiler.
"""

from typing import Optional

from pennylane.core.transforms.compile_pipeline import CompilePipeline
from pennylane.devices import Device, ExecutionConfig


class HeterogeneousDevice(Device):
    """A device for heterogeneous compilation and execution over a backline placement.

    Construct via :func:`pennylane.device`::

        dev = qp.device("ftqc.heterogeneous", backline=backline, wires=WIRES)

    The device stores the :class:`~pennylane.backline.Backline` placement consisting of a transport,
    controller, and coprocessors. This device requires the Catalyst compiler.

    Args:
        wires (int | Iterable | None): The device wires. Defaults to ``None``.
        backline (Backline): The placement (controller, coprocessors, transport), constructed via
            :func:`pennylane.backline`.
        shots (int | None): Number of shots. Defaults to ``None``.
    """

    def __init__(self, wires=None, *, backline, shots=None):
        self._backline = backline
        super().__init__(wires=wires, shots=shots)

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

    def preprocess(self, execution_config: Optional[ExecutionConfig] = None):
        """Return the (empty) transform program and the unchanged execution config."""
        if execution_config is None:
            execution_config = ExecutionConfig()
        return CompilePipeline(), execution_config

    def execute(self, circuits, execution_config: Optional[ExecutionConfig] = None):
        """Execution is handled by a downstream compiler; there is no Python execution path."""
        raise NotImplementedError(
            "The ftqc.heterogeneous device has no Python execution path; execute it via a "
            "compiler such as Catalyst (@qjit)."
        )

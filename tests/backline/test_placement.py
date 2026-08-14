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

"""Tests for backline placements and transports."""

from typing import Any

import pytest

import pennylane as qp
from pennylane.backline import get_transport

UNKNOWN_HARDWARE: Any = "tpu"


def test_nodes_default_to_cpu_hardware():
    """Controllers and coprocessors execute on CPUs by default."""
    controller = qp.Controller()
    coprocessor = qp.Coprocessor(coprocessor_fn="decoder", comm_host="127.0.0.1")

    assert controller.hardware == "cpu"
    assert coprocessor.hardware == "cpu"


def test_controller_owns_message_sizes():
    """Controllers provide transport message sizes without backend initialization arguments."""
    controller = qp.Controller()

    assert controller.in_bytes == 8
    assert controller.out_bytes == 8


@pytest.mark.parametrize("hardware", ["cpu", "gpu", "fpga"])
def test_node_accepts_supported_hardware(hardware):
    """Every public hardware kind can be represented by a node."""
    assert qp.Controller(hardware=hardware).hardware == hardware


@pytest.mark.parametrize(
    "node",
    [
        lambda: qp.Controller(hardware=UNKNOWN_HARDWARE),
        lambda: qp.Coprocessor(
            coprocessor_fn="decoder", comm_host="127.0.0.1", hardware=UNKNOWN_HARDWARE
        ),
    ],
)
def test_node_rejects_unknown_hardware(node):
    """Both node types validate their hardware."""
    with pytest.raises(ValueError, match="hardware must be one of"):
        node()
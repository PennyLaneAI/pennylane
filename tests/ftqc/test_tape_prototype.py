# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Basic tests and sanity checks for the tape-based execution pipeline prototype. Not intended to be comprehensive."""

import numpy as np
import pytest

import pennylane as qml
from pennylane.ftqc.ftqc_device import FTQCQubit, LightningQubitBackend, NullQubitBackend
from pennylane.measurements import MidMeasureMP
from pennylane.ops import CZ, Conditional, H


@pytest.mark.parametrize(
    "backend_cls, n_wires, name",
    [(LightningQubitBackend, 25, "lightning"), (NullQubitBackend, 1000, "null")],
)
def test_backend_initializes(backend_cls, n_wires, name):
    """Test that the backends initialize successfully"""

    backend = backend_cls()

    assert isinstance(backend.device, qml.devices.Device)
    assert backend.device.name == f"{name}.qubit"
    assert backend.wires == qml.wires.Wires(range(n_wires))
    assert isinstance(backend.capabilities, qml.devices.DeviceCapabilities)


@pytest.mark.parametrize("backend_cls", [LightningQubitBackend, NullQubitBackend])
def test_ftqc_device_initializes(backend_cls):
    """Test that the ftqc.qubit device initializes as expected"""

    backend = backend_cls()
    dev = FTQCQubit(wires=2, backend=backend)

    assert isinstance(dev, qml.devices.Device)
    assert dev.name == "ftqc.qubit"
    assert dev.wires == qml.wires.Wires([0, 1])
    assert isinstance(dev.capabilities, qml.devices.DeviceCapabilities)


@pytest.mark.parametrize("backend_cls", [LightningQubitBackend, NullQubitBackend])
def test_executing_arbitrary_circuit(backend_cls):
    """Test that an arbitrary circuit is preprocessed to be expressed in the
    MBQC formalism before executing, and executes as expected"""

    qml.decomposition.enable_graph()

    backend = backend_cls()
    dev = FTQCQubit(wires=2, backend=backend)

    def circ():
        qml.RY(1.2, 0)
        qml.RZ(0.2, 0)
        qml.RX(0.45, 1)
        return qml.expval(qml.X(0)), qml.expval(qml.Y(0)), qml.expval(qml.Y(1))

    ftqc_circ = qml.qnode(device=dev)(circ)
    ftqc_circ = qml.set_shots(ftqc_circ, shots=3000)

    ref_circ = qml.qnode(device=qml.device("lightning.qubit", wires=2))(circ)

    # the processed circuit is two tapes (split_non_commuting), returning
    # only samples, and expressed in the MBQC formalism
    tapes, _ = qml.workflow.construct_batch(ftqc_circ, level="device")()
    assert len(tapes) == 2
    for tape in tapes:
        assert all(isinstance(mp, qml.measurements.SampleMP) for mp in tape.measurements)
        assert all(isinstance(op, (Conditional, CZ, H, MidMeasureMP)) for op in tape.operations)

    # circuit executes
    res = ftqc_circ()

    expected_res = (1.0, 1.0, 1.0) if backend_cls is NullQubitBackend else ref_circ()

    assert np.allclose(res, expected_res, atol=0.05)

# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Integration tests for the transform program and the execution pipeline.

Differentiability tests are still in the ml-framework specific files.
"""
from typing import Tuple, Callable
import pytest

import numpy as np

import pennylane as qml


device_suite = (
    qml.device("default.qubit", wires=1),
    qml.devices.experimental.DefaultQubit2(),
    qml.device("lightning.qubit", wires=1),
)


class TestTransformProgram:
    """Non differentiability tests for the transform program keyword argument."""

    @pytest.mark.parametrize("dev", device_suite)
    @pytest.mark.parametrize("interface", (None, "autograd", "jax", "tf", "torch"))
    @pytest.mark.parametrize("old_return", (False, True))
    def test_transform_program_none(self, interface, dev, old_return):
        """Test that if no transform program is provided, null default behavior is used."""

        if old_return:
            qml.disable_return()

        tape0 = qml.tape.QuantumScript([qml.RX(1.0, 0)], [qml.expval(qml.PauliZ(0))])
        tape1 = qml.tape.QuantumScript([qml.RY(2.0, 0)], [qml.state()])

        with dev.tracker as tracker:
            results = qml.execute((tape0, tape1), dev, transform_program=None, interface=interface)

        assert qml.math.allclose(results[0], np.cos(1.0))
        assert qml.math.allclose(results[1], np.array([np.cos(1.0), np.sin(1.0)]))

        # checks on what is passed to the device. Should be exactly what we put in.
        assert tracker.totals["executions"] == 2
        assert tracker.history["resources"][0].gate_types["RX"] == 1
        assert tracker.history["resources"][1].gate_types["RY"] == 1

        qml.enable_return()

    @pytest.mark.parametrize("dev", device_suite)
    @pytest.mark.parametrize("interface", (None, "autograd", "jax", "tf", "torch"))
    @pytest.mark.parametrize("old_return", (False, True))
    def test_transform_program_modifies_circuit(self, interface, dev, old_return):
        """Integration tests for a transform program that modifies the input tapes."""
        if old_return:
            qml.disable_return()

        def null_postprocessing(results):
            return results[0]

        def just_pauli_x_out(
            tape: qml.tape.QuantumTape,
        ) -> (Tuple[qml.tape.QuantumTape], Callable):
            return (
                qml.tape.QuantumScript([qml.PauliX(0)], tape.measurements),
            ), null_postprocessing

        pauli_x_out_container = qml.transforms.core.TransformContainer(just_pauli_x_out)

        transform_program = qml.transforms.core.TransformProgram([pauli_x_out_container])

        tape0 = qml.tape.QuantumScript(
            [qml.Rot(1.2, 2.3, 3.4, wires=0)], [qml.expval(qml.PauliZ(0))]
        )
        tape1 = qml.tape.QuantumScript(
            [qml.Hadamard(0), qml.IsingXX(1.2, wires=(0, 1))], [qml.expval(qml.PauliX(0))]
        )

        with dev.tracker as tracker:
            results = qml.execute(
                (tape0, tape1), dev, transform_program=transform_program, interface=interface
            )

        assert qml.math.allclose(results[0], -1.0)
        assert qml.math.allclose(results[1], 0.0)

        assert tracker.totals["executions"] == 2
        assert tracker.history["resources"][0].gate_types["PauliX"] == 1
        assert tracker.history["resources"][0].num_gates == 1
        assert tracker.history["resources"][1].gate_types["PauliX"] == 1
        assert tracker.history["resources"][1].num_gates == 1

        qml.enable_return()

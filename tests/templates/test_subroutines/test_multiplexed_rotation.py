# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Unit tests for the MultiplexedRotation template.
"""
import numpy as np

# pylint: disable=too-many-arguments,too-few-public-methods
import pytest

import pennylane as qml
from pennylane import numpy as np


class TestMultiplexedRotation:

    def test_standard_validity(self):
        """Check the operation using the assert_valid function."""

        wires = qml.registers({"control_wires": 3, "target_wire": 1})

        op = qml.MultiplexedRotation(
            angles=qml.math.ones(8),
            control_wires=wires["control_wires"],
            target_wire=wires["target_wire"],
            rot_axis="X",
        )

        qml.ops.functions.assert_valid(op)

    @pytest.mark.parametrize(
        ("angles", "rot_axis", "msg_match"),
        [
            (
                np.array([1.0, 2.0, 3.0, 4.0]),
                "K",
                "'rot_axis' can only take the values 'X', 'Y' and 'Z'.",
            ),
            (
                np.array([1.0]),
                "Z",
                "Number of angles must",
            ),
        ],
    )
    def test_MultiplexedRotation_error(self, angles, rot_axis, msg_match):
        """Test that proper errors are raised for MultiplexedRotation"""

        wires = qml.registers({"control_wires": 2, "target_wire": 1})

        with pytest.raises(ValueError, match=msg_match):
            qml.MultiplexedRotation(
                angles=angles,
                control_wires=wires["control_wires"],
                target_wire=wires["target_wire"],
                rot_axis=rot_axis,
            )

    @pytest.mark.parametrize(
        ("angles", "rot_axis"),
        [
            (np.array([1, 2, 3, 4, 5, 6, 7, 8]), "Z"),
            (np.array([12, 2, 3, 4, 5.4, 6, 7, 8]), "X"),
            (np.array([1, 2, -3, 4, 5, 26, 7, 8]), "Y"),
        ],
    )
    def test_correctness(self, angles, rot_axis):
        """Tests the correctness of the MultiplexedRotation template.
        This is done by comparing the results with the naive Select(Rotation) decomposition
        """

        dev = qml.device("default.qubit", wires=4)

        gate = {"Z": qml.RZ, "Y": qml.RY, "X": qml.RX}

        # Check that applying the MultiplexedRotation and adjoint(Select) to an state,
        # does not modify the state
        qs = qml.tape.QuantumScript(
            [
                qml.Hadamard(0),
                qml.Hadamard(1),
                qml.MultiplexedRotation(
                    angles, control_wires=[0, 1, 2], target_wire=3, rot_axis=rot_axis
                ),
                qml.Select([gate[rot_axis](-angle, 3) for angle in angles], control=[0, 1, 2]),
                qml.Hadamard(0),
                qml.Hadamard(1),
            ],
            [qml.state()],
        )

        program, _ = dev.preprocess()
        tape = program([qs])
        output = dev.execute(tape[0])[0]

        assert np.isclose(1.0, output[0])

    def test_decomposition(self):
        """Test that the correct gates are added in the decomposition"""

        decomposition = qml.MultiplexedRotation.compute_decomposition(
            np.array([1, 2, 3, 4, 5, 6, 7, 8]), control_wires=range(3), target_wire=3, rot_axis="Z"
        )

        for gate in decomposition:
            assert gate.name in ["CNOT", "RZ"]

    @pytest.mark.jax
    def test_interface_jax(self):
        """Test that MultiplexedRotation works with jax"""

        from jax import numpy as jnp

        angles = [1, 2, 3, 4]

        wires = qml.registers({"control": 2, "target": 1})
        dev = qml.device("default.qubit", wires=3)

        qs = qml.tape.QuantumScript(
            [
                qml.MultiplexedRotation(
                    jnp.array(angles),
                    control_wires=wires["control"],
                    target_wire=wires["target"],
                    rot_axis="X",
                )
            ],
            [qml.state()],
        )

        program, _ = dev.preprocess()
        tape = program([qs])
        output_jax = dev.execute(tape[0])[0]

        qs = qml.tape.QuantumScript(
            [
                qml.MultiplexedRotation(
                    angles,
                    control_wires=wires["control"],
                    target_wire=wires["target"],
                    rot_axis="X",
                )
            ],
            [qml.state()],
        )

        program, _ = dev.preprocess()
        tape = program([qs])
        output = dev.execute(tape[0])[0]

        assert qml.math.allclose(output, output_jax)

    @pytest.mark.torch
    def test_interface_torch(self):
        """Test that MultiplexedRotation works with torch"""

        import torch

        angles = [1, 2, 3, 4]

        wires = qml.registers({"control": 2, "target": 1})
        dev = qml.device("default.qubit", wires=3)

        qs = qml.tape.QuantumScript(
            [
                qml.MultiplexedRotation(
                    torch.tensor(angles),
                    control_wires=wires["control"],
                    target_wire=wires["target"],
                    rot_axis="Y",
                )
            ],
            [qml.state()],
        )

        program, _ = dev.preprocess()
        tape = program([qs])
        output_torch = dev.execute(tape[0])[0]

        qs = qml.tape.QuantumScript(
            [
                qml.MultiplexedRotation(
                    angles,
                    control_wires=wires["control"],
                    target_wire=wires["target"],
                    rot_axis="Y",
                )
            ],
            [qml.state()],
        )

        program, _ = dev.preprocess()
        tape = program([qs])
        output = dev.execute(tape[0])[0]

        assert qml.math.allclose(output, output_torch)

    @pytest.mark.tf
    def test_interface_tf(self):
        """Test that MultiplexedRotation works with tensorflow"""

        import tensorflow as tf

        angles = [1, 2, 3, 4]

        wires = qml.registers({"control": 2, "target": 1})
        dev = qml.device("default.qubit", wires=3)

        qs = qml.tape.QuantumScript(
            [
                qml.MultiplexedRotation(
                    tf.Variable(angles),
                    control_wires=wires["control"],
                    target_wire=wires["target"],
                    rot_axis="Y",
                )
            ],
            [qml.state()],
        )

        program, _ = dev.preprocess()
        tape = program([qs])
        output_tf = dev.execute(tape[0])[0]

        qs = qml.tape.QuantumScript(
            [
                qml.MultiplexedRotation(
                    angles,
                    control_wires=wires["control"],
                    target_wire=wires["target"],
                    rot_axis="Y",
                )
            ],
            [qml.state()],
        )

        program, _ = dev.preprocess()
        tape = program([qs])
        output = dev.execute(tape[0])[0]

        assert qml.math.allclose(output, output_tf)
